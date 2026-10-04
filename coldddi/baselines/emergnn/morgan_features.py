"""Compute Morgan fingerprints from SMILES for EmerGNN drug entities.

Produces the (n_drugs, 1024) float array used by upstream
``DB_molecular_feats.pkl``, with optional pickle caching. The baseline
adapter supplies zero feature rows for non-drug entities.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024


def _smiles_to_morgan_np(smiles: str) -> np.ndarray:
    """Return (1024,) float32 bits, or zeros for missing or invalid SMILES."""
    vec = np.zeros(MORGAN_NBITS, dtype=np.float32)
    if not smiles:
        return vec
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return vec
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    onbits = list(fp.GetOnBits())
    vec[onbits] = 1.0
    return vec


def compute_morgan_matrix(
    drug_ids: Sequence[str],
    smiles_dict: Dict[str, str],
) -> Tuple[np.ndarray, List[str]]:
    """Return (n_drugs, 1024) float32 matrix in drug_ids order + list of missing drug_ids."""
    mat = np.zeros((len(drug_ids), MORGAN_NBITS), dtype=np.float32)
    missing = []
    for i, did in enumerate(drug_ids):
        s = smiles_dict.get(str(did), "")
        vec = _smiles_to_morgan_np(s)
        mat[i] = vec
        if vec.sum() == 0:
            missing.append(str(did))
    return mat, missing


def save_morgan_cache(path: str, matrix: np.ndarray, drug_ids: List[str], missing: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"Morgan_Features": matrix, "drug_ids": list(drug_ids), "missing": list(missing)}, f)


def load_morgan_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)
