"""P2 few-shot retrieval — SMILES Morgan-FP similarity.

Port of ``Version_1_1/Preprocessor/fewshot.fewshot_sequence_retrieval_step``.

Builds a ``fewshot_map: {(drug_a_id, drug_b_id): {...}}`` that
:func:`coldddi.llm.retrieval.llm_view.to_llm_samples` can splice into
each sample's ``fewshot_samples`` field.  Used by the P2 prompt branch
(``Few_Shot_Similarity_SMILES``) at LLM inference / FT time.

Pool design
-----------
* **Universe** (sim matrix scope): every drug appearing in any
  :class:`PairDataset` split.  This guarantees G2 (cold-start) drugs
  also get fingerprints so we can score similarity *against* them.
* **Pool** (candidates we may emit): only positive **G1** training
  pairs.  Cold-start integrity demands the retrieved examples come
  from the train side; cross-pool leakage (sharing a drug with the
  query) is filtered out at scoring time.
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
    """Return a ``{(drug_a_id, drug_b_id): {"fewshot_samples": [...]}}``
    map covering every pair in **all** of ``ds.splits``.

    Parameters
    ----------
    ds
        Loaded :class:`PairDataset`.
    k
        Top-k to keep per query (paper default = 3).
    seed
        RNG seed for the fallback (random) path.
    radius / nbits
        Morgan-FP hyperparameters.
    pool_pairs
        Optional override for the candidate pool.  When ``None``
        defaults to ``ds.splits.train`` filtered to G1×G1 pairs (the
        upstream behaviour).

    Returns
    -------
    ``{(drug_a_id, drug_b_id): {"fewshot_samples":
    [(score, ref_a_id, ref_b_id, "1"), ...]}}``

    The list always has exactly ``k`` entries; if Top-k cannot fill
    that many high-similarity hits, random fallback rows (score=0.0001)
    are appended from the same pool.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Universe + Pool ---------------------------------------------------
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

    # --- 2. Per-drug Morgan FPs ----------------------------------------------
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

    # --- 3. Tanimoto sim matrix on the valid subset --------------------------
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

    # --- 4. Query helper ------------------------------------------------------
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
        # Keep only positive-score hits, sort descending.
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

    # --- 5. Materialise the map over EVERY pair in any split -----------------
    # Iterates BOTH positives (from ``ds.splits.items()``) AND cached
    # static negatives (from ``ds.negatives_by_split``) for each split
    # — without the negatives loop, negative samples used at FT/eval
    # time would silently miss their few-shot pool entry and the
    # prompt would render as zero-shot, breaking the P2 / P5 contract
    # for half of every batch.  We read ``negatives_by_split``
    # directly rather than via ``ds.get_negatives()`` because the
    # latter can raise on synthetic fixtures (exhausted pool capacity)
    # or non-static splits; the cached map is exactly what real
    # workflows feed into FT/inference, so it is also the correct
    # surface for the few-shot pool.
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

    # Cached static negatives for val/test (S0/S1/S2). We read
    # ``negatives_by_split`` directly rather than calling
    # ``ds.get_negatives()`` because the regeneration path can raise
    # (RuntimeError on empty pools, ValueError on non-static splits
    # like "train"); the cached map is what real workflows feed into
    # FT / inference, so it's also the correct surface for the
    # few-shot pool. Splits without cached negatives (e.g. "train" or
    # a synthetic fixture that skipped pre-sampling) are silently
    # ignored.
    cached_negs = getattr(ds, "negatives_by_split", {}) or {}
    for split_name, split_df in ds.splits.items():
        _ingest(split_df)
        _ingest(cached_negs.get(split_name))
    return fewshot_map


__all__ = ["build_fewshot_smiles_map"]
