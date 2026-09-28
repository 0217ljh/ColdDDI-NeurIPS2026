"""
SSP (Structural Similarity Profile) feature extraction for DeepDDI.

For each drug D, SSP(D) = [Tanimoto(Morgan(D), Morgan(D_ref)) for D_ref in reference_set],
then PCA-reduce to a fixed dim. Reference set uses G1 training drugs (no G2 leakage).

Artifacts cached under <output_dir>/ssp_cache/<seed>.pkl:
    {
        "reference_drug_ids": List[str],
        "pca": fitted sklearn.decomposition.PCA,
        "ssp": Dict[drug_id, np.ndarray (n_components,)],
        "drugs_without_smiles": List[str],   # logged, zero-filled
    }
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024
DEFAULT_SSP_DIM = 50


def _smiles_to_morgan(smiles: str) -> Optional[object]:
    """Return ExplicitBitVect (nBits=1024, radius=2) or None if SMILES invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)


def compute_morgan_fps(
    smiles_dict: Dict[str, str]
) -> Tuple[Dict[str, object], List[str]]:
    """Compute Morgan bit-vector FPs for every drug with a parsable SMILES.

    Returns:
        fps: dict[drug_id, ExplicitBitVect]
        missing: list[drug_id] with unparsable SMILES (caller should log + zero-fill)
    """
    fps: Dict[str, object] = {}
    missing: List[str] = []
    for drug_id, smiles in smiles_dict.items():
        fp = _smiles_to_morgan(smiles)
        if fp is None:
            missing.append(str(drug_id))
        else:
            fps[str(drug_id)] = fp
    return fps, missing


def compute_tanimoto_matrix(
    fps: Dict[str, object],
    drug_ids: Sequence[str],
    ref_drug_ids: Sequence[str],
) -> np.ndarray:
    """Return (len(drug_ids), len(ref_drug_ids)) float32 similarity matrix.

    Rows with missing drug (not in fps) are zero-filled.
    """
    n_drugs = len(drug_ids)
    n_ref = len(ref_drug_ids)
    mat = np.zeros((n_drugs, n_ref), dtype=np.float32)
    ref_fps = [fps.get(str(rid)) for rid in ref_drug_ids]
    for i, did in enumerate(drug_ids):
        fp_i = fps.get(str(did))
        if fp_i is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, [r for r in ref_fps if r is not None])
        # Re-align to reference positions: some ref entries may also be missing.
        j = 0
        for k, r in enumerate(ref_fps):
            if r is None:
                mat[i, k] = 0.0
            else:
                mat[i, k] = sims[j]
                j += 1
    return mat


def build_ssp_artifacts(
    smiles_dict: Dict[str, str],
    reference_drug_ids: Sequence[str],
    all_drug_ids: Sequence[str],
    train_drug_ids_for_pca_fit: Sequence[str],
    n_components: int = DEFAULT_SSP_DIM,
    random_state: int = 42,
) -> Dict:
    """End-to-end SSP construction.

    Args:
        smiles_dict: drug_id -> SMILES (from bundle.extra["kb"]["drug_id2smiles"])
        reference_drug_ids: SSP basis drugs (must be G1 only, no G2 leakage)
        all_drug_ids: every drug we need SSP for (train + val + test)
        train_drug_ids_for_pca_fit: rows used to fit PCA (should be G1 only)
        n_components: final SSP dim

    Returns:
        dict with keys: "reference_drug_ids", "pca", "ssp", "drugs_without_smiles"
    """
    reference_drug_ids = [str(d) for d in reference_drug_ids]
    all_drug_ids = [str(d) for d in all_drug_ids]
    train_drug_ids_for_pca_fit = [str(d) for d in train_drug_ids_for_pca_fit]

    fps, missing = compute_morgan_fps(smiles_dict)

    # Tanimoto similarity matrix (all drugs × reference drugs)
    sim_matrix = compute_tanimoto_matrix(fps, all_drug_ids, reference_drug_ids)

    # Fit PCA on the G1 training rows ONLY — never fall back to
    # all_drug_ids, that would leak G2 into the SSP basis and break the
    # cold-start promise. If G1 is too small, we cap n_components down
    # to what G1 can support, raising only if even 2 dims aren't viable.
    train_idx = [i for i, did in enumerate(all_drug_ids) if did in set(train_drug_ids_for_pca_fit)]
    if len(train_idx) < 2:
        raise ValueError(
            f"PCA needs at least 2 G1 drugs, got {len(train_idx)}. "
            "Either expand the G1 set or use a smaller toy."
        )
    train_sub = sim_matrix[train_idx]

    effective_components = min(n_components, train_sub.shape[0], train_sub.shape[1])
    pca = PCA(n_components=effective_components, random_state=random_state)
    pca.fit(train_sub)

    # Transform all drugs
    transformed = pca.transform(sim_matrix).astype(np.float32)
    ssp: Dict[str, np.ndarray] = {
        did: transformed[i] for i, did in enumerate(all_drug_ids)
    }

    return {
        "reference_drug_ids": reference_drug_ids,
        "pca": pca,
        "ssp": ssp,
        "drugs_without_smiles": missing,
        "n_components_effective": effective_components,
    }


def save_ssp_cache(artifacts: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)


def load_ssp_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_ssp_or_zero(artifacts: Dict, drug_id: str) -> np.ndarray:
    """Safe lookup: returns zero vector if the drug is missing (unparsable SMILES)."""
    ssp = artifacts["ssp"]
    did = str(drug_id)
    if did in ssp:
        return ssp[did]
    dim = artifacts["n_components_effective"]
    return np.zeros(dim, dtype=np.float32)
