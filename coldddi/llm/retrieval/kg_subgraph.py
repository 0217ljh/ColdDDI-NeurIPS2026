"""Build the per-drug one-hop subgraph dict consumed by the P3 / P4 / R*
prompt branches.

The output structure matches the legacy
``bundle.extra["drug_subgraph_1hop_map"]`` shape:

    {
        "DB00001": {
            "transporters": ["SLC22A1", "ABCB1", ...],
            "pathways":     ["Coagulation cascade", ...],
            "targets":      ["F2", "F10", ...],
            "enzymes":      ["CYP3A4", ...],
            "carriers":     ["ALB", ...],
            "smiles":       ["<SMILES string>"],   # always a single-element list
        },
        ...
    }

There is no new data source here — :func:`build_subgraph_map` is a
**pure reshape** of :class:`coldddi.data.kg.KnowledgeGraph` (already
loaded by :class:`coldddi.data.dataset.PairDataset`) plus the
``smiles`` column of :attr:`PairDataset.drugs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from coldddi.data.kg import KnowledgeGraph


#: Order matters: the original prompt builder iterates ``list(neighbors.keys())``
#: and renders one line per entity type. Keep this tuple aligned with
#: ``Version_1_1/configs/ddi_finetune_config_k8s.py::selected_entities``.
SUBGRAPH_ENTITY_TYPES: tuple[str, ...] = (
    "transporters",
    "pathways",
    "targets",
    "enzymes",
    "carriers",
    "smiles",
)

#: Mapping from the *plural* entity-type key used in prompts back to the
#: singular ``edge_type`` enum :class:`KnowledgeGraph` exposes.
_PLURAL_TO_EDGE: dict[str, str] = {
    "transporters": "transporter",
    "pathways":     "pathway",
    "targets":      "target",
    "enzymes":      "enzyme",
    "carriers":     "carrier",
}


@dataclass
class SubgraphMap:
    """Per-drug one-hop neighbourhood, keyed by entity type (plural).

    Attributes
    ----------
    data
        ``{drug_id: {entity_type: [neighbor_name, ...]}}``.
        ``entity_type`` follows :data:`SUBGRAPH_ENTITY_TYPES` order;
        ``smiles`` is always a single-element list (the SMILES string).
        Missing entity types are filled with ``["unknown"]`` so the
        prompt builder's singularised "whose X is unknown" branch fires
        consistently.
    topk
        Top-k truncation per entity type; ``None`` = keep all neighbours.
    selected_entities
        The exact list of entity types rendered into the prompt, in the
        rendering order.
    """

    data: dict[str, dict[str, list[str]]]
    topk: int | None
    selected_entities: tuple[str, ...] = field(
        default_factory=lambda: SUBGRAPH_ENTITY_TYPES
    )

    def get_neighbors_block(self, drug_a_id: str, drug_b_id: str) -> dict:
        """Return the ``{"neighbors": {type: {"A": [...], "B": [...]}}}``
        block that :func:`coldddi.llm.prompts.binary_cls._format_pair`
        expects under ``sample["subgraph_1hop"]``."""
        a = self.data.get(str(drug_a_id), {})
        b = self.data.get(str(drug_b_id), {})
        out: dict[str, dict[str, list[str]]] = {}
        for stype in self.selected_entities:
            out[stype] = {
                "A": list(a.get(stype, ["unknown"])),
                "B": list(b.get(stype, ["unknown"])),
            }
        return {"neighbors": out}


def _topk(items: list[str], k: int | None) -> list[str]:
    """Truncate to ``k`` items.

    ``None`` means "no truncation, keep all" (paper "Full KG" condition).
    Any non-None integer is passed to a plain slice — including ``0``
    (which empties the list, matching the upstream ``tolist()[:topk]``
    semantics). Negative values are rejected to avoid the surprising
    Python slice meaning (``items[:-1]`` drops the last element).
    """
    if k is None:
        return items
    if k < 0:
        raise ValueError(f"topk must be None or a non-negative int, got {k}")
    return items[:k]


def build_subgraph_map(
    kg: KnowledgeGraph,
    drugs: pd.DataFrame,
    *,
    topk: int | None = 3,
    selected_entities: tuple[str, ...] = SUBGRAPH_ENTITY_TYPES,
) -> SubgraphMap:
    """Reshape ``KnowledgeGraph`` + drug SMILES into a per-drug subgraph dict.

    Parameters
    ----------
    kg
        :class:`KnowledgeGraph` carrying the five drug-entity tables.
    drugs
        DataFrame with ``drugbank_id`` and ``smiles`` columns (the one
        attached to :attr:`PairDataset.drugs`). Drugs missing from this
        table fall back to ``"unknown"`` in the ``smiles`` slot.
    topk
        Truncate each entity type to the first ``k`` neighbours
        (DataFrame row order is the natural alphabetic/registration
        order from DrugBank XML — same as the original
        ``Preprocessor/subgraph.py``). ``None`` keeps all neighbours
        (paper Table 5/6 "Full KG" condition).
    selected_entities
        Entity types to materialise. Must be a subset of
        :data:`SUBGRAPH_ENTITY_TYPES` (raises ``ValueError`` otherwise).
    """
    bad = set(selected_entities) - set(SUBGRAPH_ENTITY_TYPES)
    if bad:
        raise ValueError(
            f"Unknown entity types in selected_entities: {sorted(bad)}; "
            f"must be a subset of {SUBGRAPH_ENTITY_TYPES}"
        )

    # All drug IDs we want to materialise (union of KG drugs and the
    # passed drugs table).
    drug_ids: set[str] = set(kg.drug_ids)
    if drugs is not None and "drugbank_id" in drugs.columns:
        drug_ids.update(drugs["drugbank_id"].astype(str))

    # Pre-build per-entity-type name dicts (one DataFrame scan per type).
    name_dicts: dict[str, dict[str, list[str]]] = {}
    for stype in selected_entities:
        if stype == "smiles":
            continue
        name_dicts[stype] = kg.name_dict(_PLURAL_TO_EDGE[stype])

    # SMILES lookup from the drugs table.
    smiles_lookup: dict[str, str] = {}
    if drugs is not None and "smiles" in drugs.columns:
        for did, smi in zip(
            drugs["drugbank_id"].astype(str), drugs["smiles"]
        ):
            if pd.isna(smi):
                continue
            smiles_lookup[did] = str(smi)

    data: dict[str, dict[str, list[str]]] = {}
    for did in sorted(drug_ids):
        row: dict[str, list[str]] = {}
        for stype in selected_entities:
            if stype == "smiles":
                smi = smiles_lookup.get(did)
                row[stype] = [smi] if smi else ["unknown"]
                continue
            neighbours = name_dicts[stype].get(did, [])
            if neighbours:
                row[stype] = _topk(neighbours, topk)
            else:
                row[stype] = ["unknown"]
        data[did] = row

    return SubgraphMap(
        data=data,
        topk=topk,
        selected_entities=tuple(selected_entities),
    )


__all__ = [
    "SUBGRAPH_ENTITY_TYPES",
    "SubgraphMap",
    "build_subgraph_map",
]
