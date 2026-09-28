"""
KG builder for EmerGNN — assembles the DrugBank 5-entity KG from a FoldBundle.

Relations (5 base types; reverse edges + self-loop added at graph-load time):
    0: drug--target
    1: drug--enzyme
    2: drug--transporter
    3: drug--carrier
    4: drug--pathway

Entities are indexed by a single unified vocab:
    drug entities come first (block 0 .. n_drugs-1),
    followed by non-drug entities (target/enzyme/transporter/carrier/pathway IDs).

Outputs a dict:
    {
        "entity2id":  {str -> int},       # unified vocab
        "id2entity":  {int -> str},
        "entity_types":  np.ndarray[int], # 0=drug, 1=target, 2=enzyme, 3=transporter, 4=carrier, 5=pathway
        "drug_ids":     List[str],        # drugs in vocab order
        "triplets":  np.ndarray (n_edges, 3),  # (h, t, rel) — 5 base relations
        "n_ent":   int,
        "n_rel":   int,                   # 5 (base); model doubles to 2*n_rel + 1
    }
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REL_DRUG_TARGET = 0
REL_DRUG_ENZYME = 1
REL_DRUG_TRANSPORTER = 2
REL_DRUG_CARRIER = 3
REL_DRUG_PATHWAY = 4
N_BASE_REL = 5

ENT_TYPE_DRUG = 0
ENT_TYPE_TARGET = 1
ENT_TYPE_ENZYME = 2
ENT_TYPE_TRANSPORTER = 3
ENT_TYPE_CARRIER = 4
ENT_TYPE_PATHWAY = 5


_ENTITY_CONFIG = [
    # (kb_key,            id_col,          ent_type,                  rel_id)
    ("my_target_list",       "target_id",       ENT_TYPE_TARGET,      REL_DRUG_TARGET),
    ("my_enzyme_list",       "enzyme_id",       ENT_TYPE_ENZYME,      REL_DRUG_ENZYME),
    ("my_transporter_list",  "transporter_id",  ENT_TYPE_TRANSPORTER, REL_DRUG_TRANSPORTER),
    ("my_carrier_list",      "carrier_id",      ENT_TYPE_CARRIER,     REL_DRUG_CARRIER),
    ("my_pathway_list",      "pathway_id",      ENT_TYPE_PATHWAY,     REL_DRUG_PATHWAY),
]


def build_kg_from_kb(
    kb: Dict[str, pd.DataFrame],
    drug_ids: Sequence[str],
    keep_only_known_drugs: bool = True,
) -> Dict:
    """Assemble the 5-relation DrugBank KG.

    Args:
        kb: bundle.extra["kb"] — dict with my_drugs_list / my_target_list / etc.
        drug_ids: iterable of drug ids that must be first in the entity vocab
            (pass all drugs that appear anywhere in train+val+test splits).
        keep_only_known_drugs: if True, drop rows whose drugbank_id is not in `drug_ids`.

    Returns:
        dict (see module docstring).
    """
    drug_ids = [str(d) for d in drug_ids]
    drug_set = set(drug_ids)

    # Step 1: unified entity vocab. Drugs first.
    entity2id: Dict[str, int] = {d: i for i, d in enumerate(drug_ids)}
    entity_types: List[int] = [ENT_TYPE_DRUG] * len(drug_ids)

    triplets: List[Tuple[int, int, int]] = []
    dropped_per_type: Dict[str, int] = {}
    kept_per_type: Dict[str, int] = {}

    for kb_key, id_col, ent_type, rel_id in _ENTITY_CONFIG:
        df = kb.get(kb_key)
        if df is None or len(df) == 0:
            dropped_per_type[kb_key] = 0
            kept_per_type[kb_key] = 0
            continue
        kept = 0
        dropped = 0
        for row in df[["drugbank_id", id_col]].itertuples(index=False):
            drug_id = str(row[0])
            ent_id = str(row[1])
            if keep_only_known_drugs and drug_id not in drug_set:
                dropped += 1
                continue
            if drug_id not in entity2id:
                # Shouldn't happen if drug_ids covers all, but be safe
                entity2id[drug_id] = len(entity_types)
                entity_types.append(ENT_TYPE_DRUG)
            if ent_id not in entity2id:
                entity2id[ent_id] = len(entity_types)
                entity_types.append(ent_type)
            h = entity2id[drug_id]
            t = entity2id[ent_id]
            triplets.append((h, t, rel_id))
            kept += 1
        kept_per_type[kb_key] = kept
        dropped_per_type[kb_key] = dropped

    # Dedup triplets (a drug may appear multiple times for same target w/ different 'action')
    if triplets:
        arr = np.array(triplets, dtype=np.int64)
        # unique rows
        arr = np.unique(arr, axis=0)
    else:
        arr = np.zeros((0, 3), dtype=np.int64)

    id2entity = {v: k for k, v in entity2id.items()}
    entity_types_arr = np.array(entity_types, dtype=np.int64)

    return {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "entity_types": entity_types_arr,
        "drug_ids": drug_ids,
        "triplets": arr,
        "n_ent": len(entity2id),
        "n_rel": N_BASE_REL,
        "stats": {"kept_per_type": kept_per_type, "dropped_per_type": dropped_per_type},
    }


def build_sparse_adj(
    triplets: np.ndarray,
    n_ent: int,
    n_rel: int,
    device: Optional[str] = None,
):
    """Build a (n_ent, n_ent, 2*n_rel+1) sparse COO tensor used by EmerGNN.

    The tensor encodes:
        * forward edges  (h, t, r)           for every triplet
        * reverse edges  (t, h, r + n_rel)   (i.e., reversed relation shifted by n_rel)
        * self-loops     (e, e, 2*n_rel)     one per entity

    Values are all 1.0. Caller typically lives on GPU; `device=None` keeps on CPU.
    Returns a torch.sparse_coo_tensor.
    """
    import torch

    triplets = np.asarray(triplets, dtype=np.int64)
    heads = triplets[:, 0]
    tails = triplets[:, 1]
    rels = triplets[:, 2]

    # Forward edges (tail,head,rel) — EmerGNN convention: index[0]=dst, index[1]=src? check paper
    # Per original load_data.py double_triple:
    #     new_triples.append([t, h, r])         # reverse with same rel
    #     new_triples.append([h, t, r + n_rel]) # forward with shifted rel
    # So the stored 3-tuples already use (src_index, dst_index, rel_id) where the
    # generalized_rspmm interprets them. We replicate:
    fwd_h = heads
    fwd_t = tails
    fwd_r = rels + n_rel
    rev_h = tails
    rev_t = heads
    rev_r = rels  # "reverse" uses original rel id per original code

    idd_h = np.arange(n_ent, dtype=np.int64)
    idd_t = idd_h
    idd_r = np.full(n_ent, 2 * n_rel, dtype=np.int64)

    all_h = np.concatenate([fwd_h, rev_h, idd_h])
    all_t = np.concatenate([fwd_t, rev_t, idd_t])
    all_r = np.concatenate([fwd_r, rev_r, idd_r])

    indices = torch.from_numpy(np.stack([all_h, all_t, all_r], axis=0)).long()
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    size = torch.Size([n_ent, n_ent, 2 * n_rel + 1])
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=size).coalesce()
    if device is not None:
        adj = adj.to(device)
    return adj


def edges_as_dense_lists(adj):
    """For pure-PyTorch message passing — return (head_idx, tail_idx, rel_idx) LongTensors.

    Operates on a coalesced sparse adjacency. Keeps on the same device.
    """
    import torch

    adj = adj.coalesce()
    idx = adj.indices()  # (3, n_edges)
    return idx[0], idx[1], idx[2]
