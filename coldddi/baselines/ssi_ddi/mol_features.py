"""Atom-level featurizer for SSI-DDI's molecular GAT.

Refactored from ``Code-Released/baseline/SSI-DDI/data_preprocessing.py``
to drop the source's module-level globals (``df_drugs_smiles``,
``MOL_EDGE_LIST_FEAT_MTX``, ``ATOM_MAX_NUM`` …) so the same code can
build features from any SMILES dict — e.g. one keyed by DrugBank IDs.

The atom feature layout is exactly the source's ``atom_features`` (with
``explicit_H=True``), giving 70 dims per atom:

* 44 one-hot atom symbol (with 'Unknown' fallback)
*  4 numeric (degree/10, implicit valence, formal charge, num radical)
*  5 one-hot hybridization
*  1 is-aromatic
*  1 explicit total Hs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

if TYPE_CHECKING:
    pass

# Silence rdkit's noisy parse warnings for malformed SMILES.
RDLogger.DisableLog("rdApp.*")


_ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

#: Total dimension produced by :func:`atom_features`. Useful for tests
#: and for the baseline's ``in_features`` config.
ATOM_FEATURE_DIM: int = (
    len(_ATOM_SYMBOLS)  # one-hot symbol
    + 4                 # numeric (degree/10, valence, formal charge, radical)
    + len(_HYBRIDIZATIONS)
    + 1                 # is_aromatic
    + 1                 # explicit total Hs
)


def _one_of_k_unk(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [x == s for s in allowable]


def atom_features(atom) -> torch.Tensor:
    """Map an RDKit atom to a fixed-length float32 feature vector."""
    feats = (
        _one_of_k_unk(atom.GetSymbol(), _ATOM_SYMBOLS)
        + [
            atom.GetDegree() / 10,
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        ]
        + _one_of_k_unk(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [atom.GetIsAromatic()]
        + [atom.GetTotalNumHs()]
    )
    return torch.from_numpy(np.array(feats, dtype=np.float32))


def mol_to_graph(mol) -> Data | None:
    """Convert an RDKit molecule to a PyG :class:`Data` (no labels).

    Returns ``None`` for empty molecules (zero atoms) — the caller is
    responsible for skipping these drugs at training time.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feat_pairs = [(a.GetIdx(), atom_features(a)) for a in mol.GetAtoms()]
    feat_pairs.sort()
    _, feats = zip(*feat_pairs)
    x = torch.stack(feats)

    bonds = mol.GetBonds()
    if len(bonds):
        edge = torch.LongTensor(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds]
        )
        # Make undirected: add the reverse edges.
        edge_index = torch.cat([edge, edge[:, [1, 0]]], dim=0).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def build_drug_graphs(
    smiles_dict: dict[str, str],
) -> tuple[dict[str, Data], list[str]]:
    """Parse SMILES strings → ``{drug_id: pyg.Data}``.

    Drugs whose SMILES fail to parse (or have zero atoms) are skipped
    and reported as the second return value, matching how the upstream
    SSI-DDI ``DrugDataset`` filters its triple list against
    ``MOL_EDGE_LIST_FEAT_MTX``.
    """
    graphs: dict[str, Data] = {}
    missing: list[str] = []
    for drug_id, smi in smiles_dict.items():
        if smi is None or not str(smi).strip():
            missing.append(drug_id)
            continue
        mol = Chem.MolFromSmiles(str(smi).strip())
        data = mol_to_graph(mol)
        if data is None:
            missing.append(drug_id)
            continue
        graphs[drug_id] = data
    return graphs, missing
