"""Retrieve P2 examples by Morgan-fingerprint Tanimoto similarity.

Fingerprints cover drugs across all splits. Candidates default to positive
G1 training pairs; similarity ranking excludes pairs containing query drugs.
Random padding draws from the candidate pool without that exclusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset


def _morgan_fingerprint(smiles: str, radius: int, nbits: int):
    """Return an RDKit Morgan-FP bit vector or ``None`` for unparseable SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _all_drug_ids(ds: "PairDataset") -> list[str]:
    """Universe of drug ids that appear anywhere in the dataset."""
    ids: set[str] = set()
    for _, df in ds.splits.items():
        ids.update(df["drug_a_id"].astype(str))
        ids.update(df["drug_b_id"].astype(str))
    if ds.drugs is not None and "drugbank_id" in ds.drugs.columns:
        ids.update(ds.drugs["drugbank_id"].astype(str))
    return sorted(ids)


def build_fewshot_smiles_map(
    ds: "PairDataset",
    *,
    k: int = 3,
    seed: int = 42,
    radius: int = 2,
    nbits: int = 1024,
    pool_pairs: pd.DataFrame | None = None,
) -> dict[tuple[str, str], dict]:
    """Build a pair-keyed map of P2 few-shot examples.

    Candidates default to positive G1 x G1 training pairs unless ``pool_pairs``
    is supplied. ``radius`` and ``nbits`` set Morgan fingerprints; ``seed``
    controls random padding.

    Each entry contains exactly ``k`` ``fewshot_samples`` tuples
    ``(score, ref_a_id, ref_b_id, "1")``. Random padding uses score 0.0001.
    The map covers positive splits and cached static negatives.
    """
    rng = np.random.default_rng(seed)

    # Drug universe and training-pair pool
    universe = _all_drug_ids(ds)
    drug2idx = {d: i for i, d in enumerate(universe)}

    if pool_pairs is None:
        train_pos = ds.splits.train.copy()
        g1 = set(map(str, ds.splits.g1_drugs))
        mask = (
            train_pos["drug_a_id"].astype(str).isin(g1)
            & train_pos["drug_b_id"].astype(str).isin(g1)
        )
        pool_pairs = train_pos.loc[mask, ["drug_a_id", "drug_b_id"]]
    pool_pairs = pool_pairs.copy().reset_index(drop=True)
    pool_a = pool_pairs["drug_a_id"].astype(str).to_numpy()
    pool_b = pool_pairs["drug_b_id"].astype(str).to_numpy()
    pool_len = len(pool_pairs)

    if pool_len == 0:
        raise ValueError(
            "Few-shot SMILES pool is empty — every training pair was "
            "filtered out by the G1×G1 mask. Pass `pool_pairs=` explicitly "
            "or check that ds.splits.g1_drugs is populated."
        )

    # Morgan fingerprints
    smiles_map: dict[str, str] = {}
    if ds.drugs is not None and "smiles" in ds.drugs.columns:
        for did, smi in zip(
            ds.drugs["drugbank_id"].astype(str), ds.drugs["smiles"]
        ):
            if pd.isna(smi):
                continue
            smiles_map[did] = str(smi)

    fps = [_morgan_fingerprint(smiles_map.get(d, ""), radius, nbits)
           for d in universe]

    # Similarities for drugs with valid fingerprints
    from rdkit import DataStructs

    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    n = len(universe)
    sim = np.zeros((n, n), dtype=np.float32)
    if valid_idx:
        valid_fps = [fps[i] for i in valid_idx]
        for k_i, idx_i in enumerate(valid_idx):
            sim[idx_i, valid_idx] = np.asarray(
                DataStructs.BulkTanimotoSimilarity(valid_fps[k_i], valid_fps),
                dtype=np.float32,
            )

    pool_a_global = np.array([drug2idx[d] for d in pool_a], dtype=np.int64)
    pool_b_global = np.array([drug2idx[d] for d in pool_b], dtype=np.int64)

    # Per-query ranking
    def _score_one(qa: str, qb: str) -> list[tuple]:
        qa_i = drug2idx.get(qa, -1)
        qb_i = drug2idx.get(qb, -1)
        if qa_i < 0 or qb_i < 0:
            # Drug not in universe — fall back to k random pool rows.
            return _random_fallback()

        s_aa = sim[qa_i, pool_a_global]
        s_bb = sim[qb_i, pool_b_global]
        s_ab = sim[qa_i, pool_b_global]
        s_ba = sim[qb_i, pool_a_global]
        score = np.maximum((s_aa + s_bb) * 0.5, (s_ab + s_ba) * 0.5)

        # Leakage mask: pool pair shares any drug with the query.
        leak = (
            (pool_a_global == qa_i)
            | (pool_a_global == qb_i)
            | (pool_b_global == qa_i)
            | (pool_b_global == qb_i)
        )
        score = np.where(leak, -1.0, score)

        # Top-k by similarity (descending).
        topk_count = min(k, pool_len)
        if topk_count == 0:
            return _random_fallback()
        top_idx = np.argpartition(-score, kth=topk_count - 1)[:topk_count]
        # Exclude masked pairs; retain zero-similarity candidates.
        top_idx = top_idx[score[top_idx] >= 0]
        top_idx = top_idx[np.argsort(-score[top_idx])]
        out: list[tuple] = []
        for j in top_idx:
            out.append((float(score[j]), str(pool_a[j]), str(pool_b[j]), "1"))
        # Pad with random fallback if Top-k under-filled.
        if len(out) < k:
            out += _random_fallback(needed=k - len(out))
        return out[:k]

    def _random_fallback(needed: int | None = None) -> list[tuple]:
        n_needed = needed if needed is not None else k
        idxs = rng.integers(0, pool_len, size=n_needed)
        return [
            (0.0001, str(pool_a[i]), str(pool_b[i]), "1")
            for i in idxs
        ]

    # Build examples for positives and cached negatives across all splits.
    fewshot_map: dict[tuple[str, str], dict] = {}
    seen: set[tuple[str, str]] = set()

    def _ingest(df: pd.DataFrame) -> None:
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
            fewshot_map[key] = {"fewshot_samples": _score_one(a, b)}

    # Reuse cached negatives without resampling; skip splits with no cache.
    cached_negs = getattr(ds, "negatives_by_split", {}) or {}
    for split_name, split_df in ds.splits.items():
        _ingest(split_df)
        _ingest(cached_negs.get(split_name))
    return fewshot_map


__all__ = ["build_fewshot_smiles_map"]
