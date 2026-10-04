"""Retrieve P5 reference pairs through shared KG entities.

Candidates are positive training pairs sharing a target, enzyme, transporter,
carrier or pathway with either query drug. Exclude pairs containing query
drugs, rank by the better direct/cross Morgan-Tanimoto match, and retain k.

Each pair-keyed result has ``fewshot_samples`` tuples
``(score, ref_a, ref_b, "1")`` and aligned ``fewshot_metadata`` entries.
Metadata lists ``(entity_name, entity_type)`` under ``shared_QA_CA``,
``shared_QA_CB``, ``shared_QB_CA`` and ``shared_QB_CB``. Fallback entries
include a ``note``; if no disjoint pair exists, it marks unavoidable overlap.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from coldddi.llm.retrieval.fewshot_smiles import _morgan_fingerprint

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset


# Entity-class labels used in few-shot prompts.
_EDGE_TYPE_DISPLAY: dict[str, str] = {
    "target":      "Target",
    "enzyme":      "Enzyme",
    "transporter": "Transporter",
    "carrier":     "Carrier",
    "pathway":     "Pathway",
}


def _build_drug_to_entities(ds: "PairDataset") -> dict[str, set[tuple[str, str]]]:
    """Map drug IDs to (entity name, class) sets using KG name lookups."""
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for edge_type, display in _EDGE_TYPE_DISPLAY.items():
        try:
            nd = ds.kg.name_dict(edge_type)
        except KeyError:
            continue
        for drug_id, names in nd.items():
            for ent in names:
                if ent is None or pd.isna(ent):
                    continue
                ent_s = str(ent).strip()
                if not ent_s or ent_s == "nan":
                    continue
                out[str(drug_id)].add((ent_s, display))
    return out


def build_fewshot_2hop_map(
    ds: "PairDataset",
    *,
    k: int = 3,
    seed: int = 42,
    radius: int = 2,
    nbits: int = 1024,
    max_pool: int | None = 5000,
    pool_pairs: pd.DataFrame | None = None,
    fallback_pool_cap: int = 2000,
) -> dict[tuple[str, str], dict]:
    """Build P5 examples for positive splits and cached static negatives.

    ``pool_pairs`` defaults to all positive training pairs, without a G1 filter.
    ``max_pool`` limits this pool by seeded sampling. ``radius`` and ``nbits``
    set Morgan fingerprints; ``fallback_pool_cap`` limits similarity scoring
    when no shared entity is found. ``seed`` controls sampling and padding.

    Return pair-keyed entries with exactly ``k`` examples and aligned metadata,
    including fallback notes where applicable.
    """
    rng = np.random.default_rng(seed)

    # Training-pair pool
    if pool_pairs is None:
        pool_pairs = ds.splits.train[["drug_a_id", "drug_b_id"]].copy()
    pool_pairs = pool_pairs.copy().reset_index(drop=True)
    if max_pool is not None and len(pool_pairs) > max_pool:
        pool_pairs = pool_pairs.sample(
            n=max_pool, random_state=seed
        ).reset_index(drop=True)
    pool_a = pool_pairs["drug_a_id"].astype(str).to_numpy()
    pool_b = pool_pairs["drug_b_id"].astype(str).to_numpy()
    pool_len = len(pool_pairs)
    if pool_len == 0:
        raise ValueError("Few-shot 2-hop pool is empty (ds.splits.train empty).")

    # Per-drug entities
    drug_to_entities = _build_drug_to_entities(ds)

    # Index candidate pairs by entity and type.
    entity_to_pool_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, (da, db) in enumerate(zip(pool_a, pool_b)):
        ents = drug_to_entities.get(da, set()) | drug_to_entities.get(db, set())
        for item in ents:
            entity_to_pool_idx[item].append(idx)

    # Morgan fingerprints
    smiles_map: dict[str, str] = {}
    if ds.drugs is not None and "smiles" in ds.drugs.columns:
        for did, smi in zip(
            ds.drugs["drugbank_id"].astype(str), ds.drugs["smiles"]
        ):
            if pd.isna(smi):
                continue
            smiles_map[did] = str(smi)

    # Cache fingerprints as drugs are queried.
    fp_cache: dict[str, object] = {}

    def _fp(drug_id: str):
        if drug_id not in fp_cache:
            fp_cache[drug_id] = _morgan_fingerprint(
                smiles_map.get(drug_id, ""), radius, nbits,
            )
        return fp_cache[drug_id]

    # Per-query retrieval
    from rdkit import DataStructs

    # Pad from disjoint pairs with replacement to reach the requested count.
    def _random_fallback(qa: str, qb: str, note: str, need: int = k) -> tuple[list, list]:
        empty_meta = {
            "shared_QA_CA": [], "shared_QA_CB": [],
            "shared_QB_CA": [], "shared_QB_CB": [],
            "note": note,
        }
        # Mask out pool rows that re-use query drugs.
        mask = (
            (pool_a != qa) & (pool_a != qb)
            & (pool_b != qa) & (pool_b != qb)
        )
        safe_idx = np.flatnonzero(mask)
        if safe_idx.size > 0:
            # Sample WITH replacement so we always emit `need` rows.
            chosen = rng.choice(safe_idx, size=need, replace=True)
            leak_note = note
        else:
            # Mark unavoidable overlap when no disjoint candidate exists.
            chosen = rng.integers(0, pool_len, size=need)
            leak_note = (note + "_leaking_unavoidable") if note else "leaking_unavoidable"
        out_samples = [
            (0.0, str(pool_a[i]), str(pool_b[i]), "1") for i in chosen
        ]
        out_meta = [
            dict(empty_meta, note=leak_note) for _ in chosen
        ]
        return out_samples, out_meta

    def _retrieve(qa: str, qb: str) -> tuple[list, list]:
        ents_qa = drug_to_entities.get(qa, set())
        ents_qb = drug_to_entities.get(qb, set())
        q_ents = ents_qa | ents_qb

        cand: set[int] = set()
        for item in q_ents:
            if item in entity_to_pool_idx:
                cand.update(entity_to_pool_idx[item])

        # Self-exclusion: drop any pool pair that re-uses query drugs.
        safe = [
            idx for idx in cand
            if pool_a[idx] != qa and pool_a[idx] != qb
            and pool_b[idx] != qa and pool_b[idx] != qb
        ]
        found_by_protein = len(safe) > 0

        if not found_by_protein:
            # Cap fallback computation; otherwise random fallback at the end.
            safe = list(range(pool_len))
            if len(safe) > fallback_pool_cap:
                safe = rng.choice(safe, fallback_pool_cap, replace=False).tolist()

        fp_qa, fp_qb = _fp(qa), _fp(qb)
        if fp_qa is None or fp_qb is None:
            return _random_fallback(qa, qb, "no_fp")

        # Filter candidates whose FPs are missing or which self-overlap.
        valid_indices: list[int] = []
        valid_fps_a: list = []
        valid_fps_b: list = []
        for idx in safe:
            ca, cb = pool_a[idx], pool_b[idx]
            if ca == qa or ca == qb or cb == qa or cb == qb:
                continue
            fa, fb = _fp(str(ca)), _fp(str(cb))
            if fa is None or fb is None:
                continue
            valid_indices.append(idx)
            valid_fps_a.append(fa)
            valid_fps_b.append(fb)

        if not valid_indices:
            return _random_fallback(qa, qb, "no_valid_candidates")

        s_aa = DataStructs.BulkTanimotoSimilarity(fp_qa, valid_fps_a)
        s_bb = DataStructs.BulkTanimotoSimilarity(fp_qb, valid_fps_b)
        s_ab = DataStructs.BulkTanimotoSimilarity(fp_qa, valid_fps_b)
        s_ba = DataStructs.BulkTanimotoSimilarity(fp_qb, valid_fps_a)
        scores = np.maximum(
            (np.asarray(s_aa) + np.asarray(s_bb)) * 0.5,
            (np.asarray(s_ab) + np.asarray(s_ba)) * 0.5,
        )


        order = np.argsort(-scores)[:k]
        samples: list = []
        metas: list = []
        for j in order:
            pidx = valid_indices[int(j)]
            ca, cb = pool_a[pidx], pool_b[pidx]
            meta = {
                "shared_QA_CA": [], "shared_QA_CB": [],
                "shared_QB_CA": [], "shared_QB_CB": [],
            }
            if found_by_protein:
                ents_ca = drug_to_entities.get(str(ca), set())
                ents_cb = drug_to_entities.get(str(cb), set())
                meta["shared_QA_CA"] = list(ents_qa & ents_ca)
                meta["shared_QA_CB"] = list(ents_qa & ents_cb)
                meta["shared_QB_CA"] = list(ents_qb & ents_ca)
                meta["shared_QB_CB"] = list(ents_qb & ents_cb)
            samples.append((float(scores[int(j)]), str(ca), str(cb), "1"))
            metas.append(meta)

        if len(samples) < k:
            need = k - len(samples)
            fb_s, fb_m = _random_fallback(qa, qb, "padding", need=need)
            samples.extend(fb_s)
            metas.extend(fb_m)
        # Return empty lists when k is zero.
        return samples[:k], metas[:k]

    # Build examples for positives and cached negatives across all splits.
    out: dict[tuple[str, str], dict] = {}
    seen: set[tuple[str, str]] = set()

    def _ingest(df) -> None:
        if df is None or len(df) == 0:
            return
        for a, b in zip(
            df["drug_a_id"].astype(str),
            df["drug_b_id"].astype(str),
        ):
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            samples, metas = _retrieve(a, b)
            out[key] = {
                "fewshot_samples": samples,
                "fewshot_metadata": metas,
            }

    # Reuse cached negatives; per-epoch training negatives are not in this map.
    cached_negs = getattr(ds, "negatives_by_split", {}) or {}
    for split_name, split_df in ds.splits.items():
        _ingest(split_df)
        _ingest(cached_negs.get(split_name))
    return out


__all__ = ["build_fewshot_2hop_map"]
