"""Build one-hop prompt context from the dataset KG and drug SMILES.

Output maps drug IDs to entity-type lists: transporters, pathways, targets,
enzymes, carriers and smiles. SMILES uses a single-element list; missing
values use ``["unknown"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from coldddi.data.kg import KnowledgeGraph


#: Entity order controls the order of facts in rendered prompts.
SUBGRAPH_ENTITY_TYPES: tuple[str, ...] = (
    "transporters",
    "pathways",
    "targets",
    "enzymes",
    "carriers",
    "smiles",
)

#: Map plural prompt keys to KG edge types.
_PLURAL_TO_EDGE: dict[str, str] = {
    "transporters": "transporter",
    "pathways":     "pathway",
    "targets":      "target",
    "enzymes":      "enzyme",
    "carriers":     "carrier",
}


@dataclass
class SubgraphMap:
    """Per-drug neighbors and prompt rendering order.

    ``data`` maps drug IDs to entity-type lists, with ``["unknown"]`` for
    missing values and one entry for SMILES. ``topk=None`` retains all
    neighbors; ``selected_entities`` defines their rendering order.
    """

    data: dict[str, dict[str, list[str]]]
    topk: int | None
    selected_entities: tuple[str, ...] = field(
        default_factory=lambda: SUBGRAPH_ENTITY_TYPES
    )

    def get_neighbors_block(self, drug_a_id: str, drug_b_id: str) -> dict:
        """Return ``{"neighbors": {type: {"A": [...], "B": [...]}}}`` for a pair."""
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
    """Keep the first k items; None keeps all, zero empties, and negatives raise."""
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
    """Build prompt context from KG tables and a drugbank_id/smiles DataFrame.

    ``topk`` takes the first k neighbors in table order; None retains all.
    Missing drug SMILES become ``"unknown"``. ``selected_entities`` must be
    a subset of SUBGRAPH_ENTITY_TYPES or a ValueError is raised.
    """
    bad = set(selected_entities) - set(SUBGRAPH_ENTITY_TYPES)
    if bad:
        raise ValueError(
            f"Unknown entity types in selected_entities: {sorted(bad)}; "
            f"must be a subset of {SUBGRAPH_ENTITY_TYPES}"
        )

    # Include drug IDs from both the KG and the supplied drug table.
    drug_ids: set[str] = set(kg.drug_ids)
    if drugs is not None and "drugbank_id" in drugs.columns:
        drug_ids.update(drugs["drugbank_id"].astype(str))

    # Scan each KG entity table once.
    name_dicts: dict[str, dict[str, list[str]]] = {}
    for stype in selected_entities:
        if stype == "smiles":
            continue
        name_dicts[stype] = kg.name_dict(_PLURAL_TO_EDGE[stype])


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
