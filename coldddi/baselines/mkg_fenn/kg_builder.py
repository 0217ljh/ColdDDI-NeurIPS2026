"""Build the four MKG-FENN KGs from a :class:`PairDataset`.

Mirrors :file:`coldddi/baselines/mkg_fenn/_legacy/train_custom_bundle.py`
``build_kg1..4`` but operates on the modern release schema:

* KG1 — drug → chemical entity (enzymes / targets / transporters /
  carriers / pathways) from :class:`coldddi.data.kg.KnowledgeGraph`
* KG2 — drug → Morgan-FP bit (radius=2, nBits=512) from SMILES
* KG3 — drug → drug (DDI training positive pairs, bidirectional);
  cold-start drugs without DDI neighbours get a self-reference
* KG4 — drug → molecular property (RDKit descriptors, percentile-binned)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

# RDKit is imported lazily inside KG2/KG4 (some installs are slow to load).


_KG1_RELATIONS = ("enzymes", "targets", "transporters", "carriers", "pathways")
_KG4_PROPS = (
    "MolMR",
    "MolLogP",
    "MolWt",
    "NumRotatableBonds",
    "NumAliphaticRings",
)


def _add_ghost_if_empty(
    kg: dict[int, list], drug_idxs: Iterable[int], ghost_tail: int, ghost_rel: int
) -> None:
    for didx in drug_idxs:
        if not kg.get(didx):
            kg.setdefault(didx, []).append((ghost_tail, ghost_rel))


def build_kg1(kg, dict1: dict[str, int]) -> tuple[dict[int, list], int, int]:
    """Drug → chemical entity. Tail/rel ghost handled by GNN1."""
    table = defaultdict(list)
    entity_to_idx: dict[str, int] = {}
    rel_to_idx: dict[str, int] = {}

    for rel_name in _KG1_RELATIONS:
        df = getattr(kg, rel_name, None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # Find the drug-id and entity columns: drug col is one of
        # {drugbank_id, drug_id, d1}; entity col is the first remaining column.
        drug_col = next(
            (c for c in ("drugbank_id", "drug_id", "d1") if c in df.columns), None
        )
        if drug_col is None:
            continue
        ent_col = next((c for c in df.columns if c != drug_col), None)
        if ent_col is None:
            continue
        if rel_name not in rel_to_idx:
            rel_to_idx[rel_name] = len(rel_to_idx)
        rel_idx = rel_to_idx[rel_name]
        for drug_id, ent_val in zip(df[drug_col].astype(str), df[ent_col]):
            if drug_id not in dict1:
                continue
            if ent_val is None or (isinstance(ent_val, float) and pd.isna(ent_val)):
                continue
            ent = str(ent_val).strip()
            if not ent:
                continue
            if ent not in entity_to_idx:
                entity_to_idx[ent] = len(entity_to_idx)
            table[dict1[drug_id]].append((entity_to_idx[ent], rel_idx))

    tail_len = len(entity_to_idx)
    rel_len = max(len(rel_to_idx), 1)
    return dict(table), tail_len, rel_len


def build_kg2(
    drug_id2smiles: dict[str, str],
    dict1: dict[str, int],
    *,
    radius: int = 2,
    nbits: int = 512,
) -> tuple[dict[int, list], int, int]:
    """Drug → substructure (Morgan FP bit)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    table = defaultdict(list)
    rel_include = 0
    ghost_bit = nbits
    ghost_rel = 1

    for drug_id, smiles in drug_id2smiles.items():
        if drug_id not in dict1 or not smiles:
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        for bit in fp.GetOnBits():
            table[dict1[drug_id]].append((int(bit), rel_include))

    _add_ghost_if_empty(table, dict1.values(), ghost_bit, ghost_rel)
    tail_len = nbits + 1     # 0..nbits-1 + ghost
    rel_len = 2              # 0=include, 1=ghost
    return dict(table), tail_len, rel_len


def build_kg3(
    train_pos_df: pd.DataFrame, dict1: dict[str, int]
) -> tuple[dict[int, list], int, int]:
    """Drug → drug (training-positive DDI pairs, bidirectional). Cold-start
    drugs with zero DDI neighbours get a self-reference so the no-ghost
    GNN3 forward path doesn't trip on empty rows."""
    table = defaultdict(list)
    rel_ddi = 0
    a_col = "drug_a_id" if "drug_a_id" in train_pos_df.columns else "d1"
    b_col = "drug_b_id" if "drug_b_id" in train_pos_df.columns else "d2"
    for _, row in train_pos_df[[a_col, b_col]].iterrows():
        d1 = str(row[a_col])
        d2 = str(row[b_col])
        if d1 not in dict1 or d2 not in dict1:
            continue
        i1, i2 = dict1[d1], dict1[d2]
        table[i1].append((i2, rel_ddi))
        table[i2].append((i1, rel_ddi))
    for didx in dict1.values():
        if not table.get(didx):
            table.setdefault(didx, []).append((didx, rel_ddi))
    tail_len = len(dict1)
    rel_len = 1
    return dict(table), tail_len, rel_len


def build_kg4(
    drug_id2smiles: dict[str, str],
    dict1: dict[str, int],
    *,
    n_bins: int = 10,
) -> tuple[dict[int, list], int, int]:
    """Drug → molecular property (5 RDKit descriptors, percentile-binned)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors as RDDesc

    RDLogger.DisableLog("rdApp.*")

    n_props = len(_KG4_PROPS)
    ghost_prop = n_props
    ghost_rel = n_bins

    raw_vals: dict[str, list[tuple[int, float]]] = {p: [] for p in _KG4_PROPS}
    desc_fn = {p: getattr(RDDesc, p, None) for p in _KG4_PROPS}
    for drug_id, smiles in drug_id2smiles.items():
        if drug_id not in dict1 or not smiles:
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        for p, fn in desc_fn.items():
            if fn is None:
                continue
            try:
                v = float(fn(mol))
                raw_vals[p].append((dict1[drug_id], v))
            except Exception:
                continue

    bin_edges: dict[str, np.ndarray | None] = {}
    for p, entries in raw_vals.items():
        if len(entries) < 2:
            bin_edges[p] = None
        else:
            vals = np.array([e[1] for e in entries], dtype=np.float64)
            edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
            edges[-1] = edges[-1] + 1e-9
            bin_edges[p] = edges

    table = defaultdict(list)
    prop2idx = {p: i for i, p in enumerate(_KG4_PROPS)}
    for p, entries in raw_vals.items():
        prop_idx = prop2idx[p]
        edges = bin_edges[p]
        for drug_idx, val in entries:
            if edges is None:
                bin_id = 0
            else:
                bin_id = int(np.searchsorted(edges[1:], val, side="left"))
                bin_id = min(bin_id, n_bins - 1)
            table[drug_idx].append((prop_idx, bin_id))

    _add_ghost_if_empty(table, dict1.values(), ghost_prop, ghost_rel)
    tail_len = n_props + 1
    rel_len = n_bins + 1
    return dict(table), tail_len, rel_len


def build_all_kgs(
    *,
    kg,
    drug_id2smiles: dict[str, str],
    dict1: dict[str, int],
    train_pos_df: pd.DataFrame,
    fp_radius: int = 2,
    fp_nbits: int = 512,
    n_bins: int = 10,
) -> tuple[dict, dict, dict]:
    """Wrapper that returns ``(kgs, tail_len, relation_len)`` ready for
    :class:`coldddi.baselines.mkg_fenn.model.MKGFENN`."""
    kg1, t1, r1 = build_kg1(kg, dict1)
    kg2, t2, r2 = build_kg2(drug_id2smiles, dict1, radius=fp_radius, nbits=fp_nbits)
    kg3, t3, r3 = build_kg3(train_pos_df, dict1)
    kg4, t4, r4 = build_kg4(drug_id2smiles, dict1, n_bins=n_bins)
    return (
        {"dataset1": kg1, "dataset2": kg2, "dataset3": kg3, "dataset4": kg4},
        {"dataset1": t1, "dataset2": t2, "dataset3": t3, "dataset4": t4},
        {"dataset1": r1, "dataset2": r2, "dataset3": r3, "dataset4": r4},
    )
