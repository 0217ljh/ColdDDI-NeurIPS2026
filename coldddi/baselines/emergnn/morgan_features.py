"""
Morgan fingerprint featurizer for EmerGNN drug entities.

EmerGNN's original code expects a (n_drugs, 1024) float array loaded from
`DB_molecular_feats.pkl`. We compute the same on-the-fly from SMILES and
cache by seed under <output_dir>/morgan_cache/<seed>.pkl.

For non-drug entities (targets, enzymes, pathways, ...), we return a zero
row; the model then relies on `nn.Embedding` (args.feat='E') or lets those
rows propagate zeros under `Went` (args.feat='M'). We follow the paper's
default 'M' mode.
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
    """Returns a (1024,) uint8 numpy array (0/1 bits), or zeros if SMILES invalid."""
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
