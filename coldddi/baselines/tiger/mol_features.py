"""TIGER molecular featurizer with shortest-path encoding.

Each drug's :class:`Data` carries:

* ``x``       — ``[n_atoms, ATOM_FEATURE_DIM]`` (sum-normalized like upstream)
* ``edge_index`` — ``[2, n_directed_bonds]`` long
* ``sp_edge_index`` — ``[2, n_pairs]`` shortest-path edges (over the molecule)
* ``sp_value``      — shortest-path lengths
* ``sp_edge_rel``   — per-pair relation id (bond type for length=1, length+offset otherwise)

The model uses ``sp_*`` for the GraphTransformer's spatial+relational
attention (see :mod:`coldddi.baselines.tiger.graph_transformer`).
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


_ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "X",
]
_NUM_HS = list(range(11))         # 0..10
_IMPL_VAL = list(range(11))       # 0..10

_BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
_BOND_TYPE_IDX = {bt: i for i, bt in enumerate(_BOND_TYPES)}

#: 44 + 11 + 11 + 1 = 67 dims, matching upstream atom_features rather than
#: its legacy num_features_drug=78 default.
ATOM_FEATURE_DIM: int = (
    len(_ATOM_SYMBOLS) + len(_NUM_HS) + len(_IMPL_VAL) + 1
)

#: Offset added to non-direct shortest-path lengths when packing the
#: relation id (matches upstream ``s_rel[np.where(s_value != 1)] += 23``).
SP_REL_OFFSET: int = len(_BOND_TYPES) + 19  # 4 + 19 = 23


def _one_hot_unk(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [float(x == s) for s in allowable]


def _atom_feature_vector(atom) -> np.ndarray:
    feats = (
        _one_hot_unk(atom.GetSymbol(), _ATOM_SYMBOLS)
        + _one_hot_unk(atom.GetTotalNumHs(), _NUM_HS)
        + _one_hot_unk(atom.GetImplicitValence(), _IMPL_VAL)
        + [float(atom.GetIsAromatic())]
    )
    arr = np.array(feats, dtype=np.float32)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr


def _shortest_path_pairs(edge_index_np: np.ndarray) -> np.ndarray:
    """Return ``[[i, j, len_ij], …]`` for nodes present in the edge list.

    Include reachable pairs and length-zero self-pairs, as in upstream
    ``calculate_shortest_path``. Isolated atoms absent from the edges are omitted.
    """
    g = nx.DiGraph()
    g.add_edges_from(edge_index_np.tolist())
    rows = []
    for node_i, node_ij in nx.all_pairs_shortest_path_length(g):
        for node_j, length in node_ij.items():
            rows.append([node_i, node_j, length])
    rows.sort()
    return np.array(rows, dtype=np.int64)


def mol_to_data(smiles: str) -> Data | None:
    if smiles is None or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feats = [_atom_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(feats), dtype=torch.float32)

    bond_pairs = []
    for bond in mol.GetBonds():
        bt = str(bond.GetBondType())
        rel = _BOND_TYPE_IDX.get(bt, 0)
        bond_pairs.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), rel))
        bond_pairs.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), rel))

    if not bond_pairs:
        return None

    bond_arr = np.array(sorted(bond_pairs), dtype=np.int64)
    edge_index = torch.tensor(bond_arr[:, :2].T, dtype=torch.long)
    bond_rel = bond_arr[:, 2]

    # Shortest-path edges (i, j, length) for every reachable pair.
    sp_arr = _shortest_path_pairs(bond_arr[:, :2])
    sp_edge_index = torch.tensor(sp_arr[:, :2].T, dtype=torch.long)
    sp_value = torch.tensor(sp_arr[:, 2], dtype=torch.float32)

    # sp_edge_rel: for length-1 entries use the matching bond rel,
    # otherwise length + SP_REL_OFFSET.
    sp_rel = sp_arr[:, 2].copy()
    direct_mask = sp_arr[:, 2] == 1
    if direct_mask.any():
        # Lookup bond rel for each direct sp_edge by matching (src, dst).
        direct_pairs = sp_arr[direct_mask, :2]
        bond_lookup = {(int(s), int(d)): int(r) for s, d, r in bond_arr}
        for k, (s, d) in enumerate(direct_pairs):
            sp_rel_idx = np.where(direct_mask)[0][k]
            sp_rel[sp_rel_idx] = bond_lookup.get((int(s), int(d)), 0)
    sp_rel[~direct_mask] += SP_REL_OFFSET
    sp_edge_rel = torch.tensor(sp_rel, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        sp_edge_index=sp_edge_index,
        sp_value=sp_value,
        sp_edge_rel=sp_edge_rel,
    )


def build_drug_graphs(
    smiles_dict: dict[str, str],
) -> tuple[dict[str, Data], list[str], int]:
    """Parse SMILES → ``{drug_id: Data}``.

    Returns ``(graphs, missing, max_sp_rel)`` where ``max_sp_rel`` is
    the maximum value of ``sp_edge_rel`` across the corpus, used by the
    adapter to pick a safe ``num_relations_mol``.
    """
    graphs: dict[str, Data] = {}
    missing: list[str] = []
    max_rel = 0
    for drug_id, smi in smiles_dict.items():
        data = mol_to_data(smi)
        if data is None:
            missing.append(drug_id)
            continue
        graphs[drug_id] = data
        max_rel = max(max_rel, int(data.sp_edge_rel.max().item()))
    return graphs, missing, max_rel
