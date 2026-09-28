"""BKG (Biomedical Knowledge Graph) construction for dual-channel TIGER.

Extracted from upstream ``Code-Released/baseline/TIGER/train_custom_bundle.py``
(lines ~258-617).  Builds the heterogeneous graph that the KG branch
of :class:`coldddi.baselines.tiger.model.TIGER` walks over:

* Nodes — all drugs (indices ``0..n_drugs-1``) + entities
  (``n_drugs..n_drugs + n_entities - 1``).
* Edges — train-only DDI positives (any pair touching a cold-start
  ``g2`` drug is dropped) + bidirectional drug-entity edges from
  ``bundle.extra['kb']`` + self-loops on isolated drug nodes.
* Relations — ``1`` = DDI / self-loop, ``2..6`` = kb categories
  (targets / enzymes / transporters / carriers / pathways).

Paper KG augmentation (``--kg_source drugbank|kegg|ogbl-biokg`` in
the upstream script) is **not** integrated here because those source
files are 38 MB+ DrugBank-derived and don't ship with this public
release.  See the upstream script if you need them for full
paper-Table-6 reproduction.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# ─── kb DataFrame column resolution helpers ──────────────────────────

def _bkg_drug_col(df: pd.DataFrame) -> str | None:
    for c in ["drug_id", "drug_a_id", "DrugBank ID", "drugbank_id", "d1", "id"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) > 0 else None


def _bkg_entity_col(df: pd.DataFrame, rel_name: str) -> str | None:
    cand = [
        rel_name, "target", "enzyme", "transporter", "carrier", "pathway",
        "name", "gene", "Gene Name", "Uniprot ID", "entity",
    ]
    for c in cand:
        if c in df.columns:
            return c
    if len(df.columns) >= 2:
        return df.columns[1]
    return None


# ─── drug-entity edges from kb ────────────────────────────────────────

def build_drug_entity_edges_from_kb(
    kb_data: dict,
    drug_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    """Extract drug-entity edges from a bundle's ``extra['kb']`` dict.

    Returns ``(edges, entity_to_idx)`` where:

    * ``edges`` is a list of ``(drug_idx, entity_idx, rel_idx)``.
      Relation ids: ``2`` = target, ``3`` = enzyme, ``4`` = transporter,
      ``5`` = carrier, ``6`` = pathway (kept compatible with upstream;
      reserves ``1`` for DDI and self-loops).
    * ``entity_to_idx`` maps entity-name → entity-local index
      (the BKG node id is ``n_drugs + entity_idx``).

    Supports both kb layouts: dict-of-lists keyed on
    ``targets``/``enzymes``/etc., or DataFrame-keyed on
    ``my_target_list``/``my_enzyme_list``/etc.
    """
    entity_to_idx: dict[str, int] = {}
    edges: list[tuple[int, int, int]] = []
    if not isinstance(kb_data, dict):
        kb_data = getattr(kb_data, "__dict__", {}) or {}

    categories_std = ["targets", "enzymes", "transporters", "carriers", "pathways"]
    categories_my = [
        "my_target_list", "my_enzyme_list", "my_transporter_list",
        "my_carrier_list", "my_pathway_list",
    ]
    use_df = False
    for cat in categories_std:
        c = kb_data.get(cat)
        if isinstance(c, dict) and len(c) > 0:
            break
    else:
        use_df = any(kb_data.get(k) is not None for k in categories_my)

    rel_map_std = {c: 2 + i for i, c in enumerate(categories_std)}

    if not use_df:
        for cat in categories_std:
            cat_data = kb_data.get(cat, {})
            if not isinstance(cat_data, dict):
                continue
            rel_idx = rel_map_std.get(cat, 2)
            for drug_id, entities in cat_data.items():
                drug_id = str(drug_id)
                if drug_id not in drug_to_idx:
                    continue
                drug_idx = drug_to_idx[drug_id]
                if isinstance(entities, str):
                    entities = [entities]
                for ent in (entities or []):
                    ent = str(ent).strip() if ent else ""
                    if not ent:
                        continue
                    if ent not in entity_to_idx:
                        entity_to_idx[ent] = len(entity_to_idx)
                    edges.append((drug_idx, entity_to_idx[ent], rel_idx))
    else:
        rel_map_my = {
            "my_target_list": 2,
            "my_enzyme_list": 3,
            "my_transporter_list": 4,
            "my_carrier_list": 5,
            "my_pathway_list": 6,
        }
        for kb_key in categories_my:
            cat_data = kb_data.get(kb_key)
            if cat_data is None or not isinstance(cat_data, pd.DataFrame) or cat_data.empty:
                continue
            df = cat_data
            drug_col = _bkg_drug_col(df)
            if drug_col is None:
                continue
            rel_idx = rel_map_my.get(kb_key, 2)
            entity_col = _bkg_entity_col(
                df, kb_key.replace("my_", "").replace("_list", ""),
            )
            for _, row in df.iterrows():
                try:
                    drug_id = str(row[drug_col])
                except Exception:
                    continue
                if drug_id not in drug_to_idx:
                    continue
                drug_idx = drug_to_idx[drug_id]
                if entity_col is not None and entity_col in df.columns:
                    ent = row[entity_col]
                else:
                    ent = None
                    for c in df.columns:
                        if c != drug_col:
                            ent = row[c]
                            break
                if ent is None or (isinstance(ent, float) and pd.isna(ent)):
                    continue
                ent = str(ent).strip()
                if not ent:
                    continue
                if ent not in entity_to_idx:
                    entity_to_idx[ent] = len(entity_to_idx)
                edges.append((drug_idx, entity_to_idx[ent], rel_idx))

    return edges, entity_to_idx


# ─── BKG assembly ─────────────────────────────────────────────────────

def build_bkg(
    *,
    kb: dict,
    drug_to_idx: dict[str, int],
    train_positive_pairs: Iterable[tuple[str, str]],
    g2_drugs: set[int] | None = None,
) -> tuple[list[list[int]], list[int], int, int]:
    """Build the BKG edge list + relation list for TIGER.

    Parameters
    ----------
    kb
        ``bundle.extra['kb']`` dict (see upstream FoldBundle).
    drug_to_idx
        Mapping ``drugbank_id -> int`` covering every drug that should
        be a BKG node (g1 + g2 combined).
    train_positive_pairs
        Iterable of ``(drug_a_id, drug_b_id)`` train-set DDI positives.
        Edges touching a ``g2_drugs`` entry are dropped (cold-start
        integrity — unseen drugs may not leak via training DDIs).
    g2_drugs
        Set of drug indices that are in the unseen / cold-start
        partition.  Pass ``None`` to disable the cold-start filter
        (every train positive is added).

    Returns
    -------
    ``(network_edge_list, network_rel_list, num_rel, n_drugs)``
    ready to feed into
    :func:`coldddi.baselines.tiger.data_process.generate_node_subgraphs`.
    """
    n_drugs = len(drug_to_idx)
    drug_entity_edges, entity_to_idx = build_drug_entity_edges_from_kb(
        kb, drug_to_idx,
    )
    n_entities = len(entity_to_idx)
    entity_node_offset = n_drugs

    g2_set: set[int] = set(g2_drugs or [])

    network_edge_list: list[list[int]] = []
    network_rel_list: list[int] = []

    # ── DDI edges (train positives only, exclude g2-touching) ──
    ddi_skipped_g2 = 0
    for d1, d2 in train_positive_pairs:
        d1, d2 = str(d1), str(d2)
        if d1 not in drug_to_idx or d2 not in drug_to_idx:
            continue
        i, j = drug_to_idx[d1], drug_to_idx[d2]
        if i in g2_set or j in g2_set:
            ddi_skipped_g2 += 1
            continue
        network_edge_list.append([i, j])
        network_rel_list.append(1)
    if ddi_skipped_g2:
        log.info(
            "BKG: skipped %d DDI edges that involve g2 (unseen) drugs",
            ddi_skipped_g2,
        )

    # ── drug-entity edges (bidirectional) ──
    for drug_idx, entity_idx, rel_idx in drug_entity_edges:
        u, v = drug_idx, entity_node_offset + entity_idx
        network_edge_list.append([u, v])
        network_rel_list.append(rel_idx)
        network_edge_list.append([v, u])
        network_rel_list.append(rel_idx)

    # ── self-loops on isolated drug nodes ──
    nodes_in_edges: set[int] = set()
    for e in network_edge_list:
        nodes_in_edges.add(e[0])
        nodes_in_edges.add(e[1])
    for i in range(n_drugs):
        if i not in nodes_in_edges:
            network_edge_list.append([i, i])
            network_rel_list.append(1)

    num_rel = max(network_rel_list) + 1 if network_rel_list else 7
    log.info(
        "BKG: drugs=%d entities=%d edges=%d num_rel=%d",
        n_drugs, n_entities, len(network_edge_list), num_rel,
    )
    return network_edge_list, network_rel_list, num_rel, n_drugs


__all__ = [
    "build_drug_entity_edges_from_kb",
    "build_bkg",
]
