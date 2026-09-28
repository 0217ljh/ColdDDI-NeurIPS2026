"""HDN-DDI super-node molecular featurizer (55-dim atom features).

Refactored from the upstream
``Code-Released/baseline/HDN-DDI-NEW/drugbank_test/data_preprocessing.py``
``_mol_to_data`` helper. Produces a PyG :class:`Data` per drug with:

* ``x``       — ``[n_atoms+1, 55]`` float32 (44 element types + 4 scalars
                + 5 hybridisations + 1 aromatic + 1 numHs, plus a zeroed
                row for the super-node)
* ``edge_index`` — ``[2, 2*n_bonds + 2*n_atoms]`` long, super-node
                connected to every atom in both directions
* ``edge_attr``  — ``[2*n_bonds + 2*n_atoms]`` long bond-type ids
* ``y``       — ``[n_atoms+1]`` long, atoms = 1, super-node = 2

The super-node ``y == 2`` marker is required by
:func:`coldddi.baselines.hdn_ddi.models.get_node`, which slices the
molecular-level embedding from each block's output.
"""

from __future__ import annotations

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


_ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In",
    "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]
_HYBRIDS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
_BOND_TYPE = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}

#: Total dimension of the per-atom feature vector. Matches the upstream
#: 44+4+5+1+1 = 55 design.
ATOM_FEATURE_DIM: int = (
    len(_ATOM_SYMBOLS) + 4 + len(_HYBRIDS) + 1 + 1
)


def _one_hot(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [float(x == s) for s in allowable]


def _atom_features(atom) -> torch.Tensor:
    feats = (
        _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS)
        + [
            atom.GetDegree() / 10,
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
        ]
        + _one_hot(atom.GetHybridization(), _HYBRIDS)
        + [float(atom.GetIsAromatic())]
        + [atom.GetTotalNumHs()]
    )
    return torch.tensor(feats, dtype=torch.float32)


def mol_to_data(smiles: str) -> Data | None:
    """Build a PyG ``Data`` with the super-node convention. Returns
    ``None`` if SMILES cannot be parsed or has zero atoms."""
    if smiles is None or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feats = [_atom_features(a) for a in mol.GetAtoms()]
    x = torch.stack(feats)  # [n_atoms, 55]

    bonds = list(mol.GetBonds())
    if bonds:
        src = [b.GetBeginAtomIdx() for b in bonds]
        dst = [b.GetEndAtomIdx() for b in bonds]
        bt = [_BOND_TYPE.get(b.GetBondType(), 0) for b in bonds]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        edge_attr = torch.tensor(bt + bt, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)

    n_atoms = x.shape[0]
    super_x = torch.zeros(1, x.shape[1], dtype=torch.float32)
    x = torch.cat([x, super_x], dim=0)
    super_idx = n_atoms

    atom_idx = torch.arange(n_atoms, dtype=torch.long)
    super_col = torch.full((n_atoms,), super_idx, dtype=torch.long)
    extra_edge = torch.stack(
        [
            torch.cat([atom_idx, super_col]),
            torch.cat([super_col, atom_idx]),
        ],
        dim=0,
    )
    extra_attr = torch.zeros(2 * n_atoms, dtype=torch.long)

    edge_index = torch.cat([edge_index, extra_edge], dim=1)
    edge_attr = torch.cat([edge_attr, extra_attr])

    y = torch.ones(n_atoms + 1, dtype=torch.long)
    y[super_idx] = 2

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_drug_graphs(
    smiles_dict: dict[str, str],
) -> tuple[dict[str, Data], list[str]]:
    """Parse SMILES → ``{drug_id: Data}``; report unparseable as ``missing``."""
    graphs: dict[str, Data] = {}
    missing: list[str] = []
    for drug_id, smi in smiles_dict.items():
        data = mol_to_data(smi)
        if data is None:
            missing.append(drug_id)
            continue
        graphs[drug_id] = data
    return graphs, missing
