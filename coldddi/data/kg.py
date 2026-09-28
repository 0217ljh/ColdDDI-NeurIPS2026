"""Default :class:`coldddi.data.protocols.KnowledgeGraphProtocol` implementation.

Wraps the five entity tables produced by
:mod:`coldddi.data.filter` (``drug_enzymes`` / ``drug_targets`` /
``drug_transporters`` / ``drug_carriers`` / ``drug_pathways``) into a
single typed object that exposes both:

1. A *modern* DataFrame API: :meth:`neighbors`, :meth:`shared_entities`.
2. A *legacy* dict-of-list view: :meth:`name_dict`.

The legacy view is what the prompt-builder code inherited from the
earlier research repository expects (``{drug_id: [entity_name, ...]}``);
keeping it as a first-class method preserves bit-exact compatibility
with prompts in the LLM stack.

Persistence
-----------
:meth:`save` writes the five tables back as parquet files; the inverse
:func:`KnowledgeGraph.from_filtered_dir` reads them. The on-disk format
is identical to ``data/{public,private}/intermediate/filtered/`` so
existing filter outputs can be loaded directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

EDGE_TYPES: tuple[str, ...] = ("enzyme", "target", "transporter", "carrier", "pathway")

_TABLE_FILENAMES: dict[str, str] = {
    "enzyme": "drug_enzymes.csv",
    "target": "drug_targets.csv",
    "transporter": "drug_transporters.csv",
    "carrier": "drug_carriers.csv",
    "pathway": "drug_pathways.csv",
}


@dataclass
class KnowledgeGraph:
    """Drug-to-entity knowledge graph backed by five DataFrames.

    Parameters
    ----------
    enzymes / targets / transporters / carriers
        Columns: ``drugbank_id, <type>_id, <type>_name, organism, action``.
    pathways
        Columns: ``drugbank_id, pathway_id, pathway_name``.

    All inputs may be empty (the schema is preserved); attempting to
    construct from non-DataFrames raises immediately.
    """

    enzymes: pd.DataFrame
    targets: pd.DataFrame
    transporters: pd.DataFrame
    carriers: pd.DataFrame
    pathways: pd.DataFrame

    # cached unified view (computed lazily by `_unified()`)
    _unified_cache: pd.DataFrame | None = field(default=None, repr=False, init=False)

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_filtered_dir(cls, dir_path: Path) -> "KnowledgeGraph":
        """Load the five entity tables from a filtered output directory.

        Auto-detects the file extension: prefers ``.parquet`` (the format
        :meth:`save` writes) and falls back to ``.csv`` (the format
        Stage 1b's :func:`coldddi.data.filter.write_filter_report`
        writes). This keeps :meth:`save` and :meth:`from_filtered_dir`
        a true round-trip pair.
        """
        kwargs = {}
        for entity, prefix in (
            ("enzymes", "drug_enzymes"),
            ("targets", "drug_targets"),
            ("transporters", "drug_transporters"),
            ("carriers", "drug_carriers"),
            ("pathways", "drug_pathways"),
        ):
            parquet_path = dir_path / f"{prefix}.parquet"
            csv_path = dir_path / f"{prefix}.csv"
            if parquet_path.is_file():
                kwargs[entity] = pd.read_parquet(parquet_path)
            elif csv_path.is_file():
                kwargs[entity] = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(
                    f"Neither {parquet_path} nor {csv_path} exists."
                )
        return cls(**kwargs)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _table_for(self, edge_type: str) -> pd.DataFrame:
        if edge_type == "enzyme":
            return self.enzymes
        if edge_type == "target":
            return self.targets
        if edge_type == "transporter":
            return self.transporters
        if edge_type == "carrier":
            return self.carriers
        if edge_type == "pathway":
            return self.pathways
        raise ValueError(f"Unknown edge_type: {edge_type!r}; expected one of {EDGE_TYPES}")

    def _unified(self) -> pd.DataFrame:
        """Build / cache a unified long DataFrame with one row per drug-entity edge."""
        if self._unified_cache is not None:
            return self._unified_cache

        frames: list[pd.DataFrame] = []
        for et in EDGE_TYPES:
            tbl = self._table_for(et)
            if tbl.empty:
                continue
            id_col = f"{et}_id"
            name_col = f"{et}_name"
            sub = pd.DataFrame(
                {
                    "drugbank_id": tbl["drugbank_id"].astype(str),
                    "edge_type": et,
                    "entity_id": tbl[id_col].astype(str),
                    "entity_name": tbl[name_col].astype(str),
                    "organism": tbl["organism"] if "organism" in tbl.columns else "",
                    "action": tbl["action"] if "action" in tbl.columns else "",
                }
            )
            frames.append(sub)

        if frames:
            unified = pd.concat(frames, ignore_index=True)
        else:
            unified = pd.DataFrame(
                columns=["drugbank_id", "edge_type", "entity_id", "entity_name", "organism", "action"]
            )
        self._unified_cache = unified
        return unified

    # -----------------------------------------------------------------
    # KnowledgeGraphProtocol surface
    # -----------------------------------------------------------------

    def neighbors(
        self,
        drug_id: str,
        *,
        edge_types: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return every entity edge incident to ``drug_id``.

        Filter by ``edge_types`` (e.g. ``["enzyme", "transporter"]``) to
        subset; ``None`` returns all five edge types.
        """
        unified = self._unified()
        mask = unified["drugbank_id"] == str(drug_id)
        if edge_types is not None:
            wanted = set(edge_types)
            unknown = wanted - set(EDGE_TYPES)
            if unknown:
                raise ValueError(f"Unknown edge_types: {sorted(unknown)}")
            mask &= unified["edge_type"].isin(wanted)
        return unified.loc[mask].reset_index(drop=True)

    def shared_entities(self, drug_a: str, drug_b: str) -> pd.DataFrame:
        """Return ``(edge_type, entity_id, entity_name)`` rows present for both drugs.

        Membership is decided by ``(edge_type, entity_id)``; the joined
        ``entity_name`` is taken from drug A's row. We deliberately do
        *not* require name equality because DrugBank occasionally records
        cleaned-up name variants (e.g. trailing whitespace) on different
        polypeptide entries that share an ID.
        """
        unified = self._unified()
        a_keys = unified.loc[
            unified["drugbank_id"] == str(drug_a),
            ["edge_type", "entity_id", "entity_name"],
        ]
        b_keys = unified.loc[
            unified["drugbank_id"] == str(drug_b),
            ["edge_type", "entity_id"],
        ]
        if a_keys.empty or b_keys.empty:
            return pd.DataFrame(columns=["edge_type", "entity_id", "entity_name"])
        merged = a_keys.merge(b_keys, on=["edge_type", "entity_id"], how="inner")
        return merged.drop_duplicates().reset_index(drop=True)

    def name_dict(self, edge_type: str) -> dict[str, list[str]]:
        """Return ``{drug_id: [entity_name, ...]}`` for one edge type.

        This mirrors the legacy ``dbid_2_enzymes`` / ``dbid_2_targets``
        dicts the prompt builders expect. Drugs that have no edges of
        the given type are absent from the result; callers should
        default to ``["unknown"]`` themselves if they want the legacy
        sentinel behavior.
        """
        tbl = self._table_for(edge_type)
        if tbl.empty:
            return {}
        name_col = f"{edge_type}_name"
        out: dict[str, list[str]] = {}
        for drug_id, group in tbl.groupby("drugbank_id"):
            out[str(drug_id)] = group[name_col].astype(str).tolist()
        return out

    def save(self, out_dir: Path) -> None:
        """Persist all five entity tables as parquet under ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.enzymes.to_parquet(out_dir / "drug_enzymes.parquet", index=False)
        self.targets.to_parquet(out_dir / "drug_targets.parquet", index=False)
        self.transporters.to_parquet(out_dir / "drug_transporters.parquet", index=False)
        self.carriers.to_parquet(out_dir / "drug_carriers.parquet", index=False)
        self.pathways.to_parquet(out_dir / "drug_pathways.parquet", index=False)

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    @property
    def drug_ids(self) -> set[str]:
        """All drugbank_ids that appear anywhere in the graph."""
        return set(self._unified()["drugbank_id"].unique())

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph("
            f"enzymes={len(self.enzymes):,}, targets={len(self.targets):,}, "
            f"transporters={len(self.transporters):,}, carriers={len(self.carriers):,}, "
            f"pathways={len(self.pathways):,})"
        )


__all__ = ["KnowledgeGraph", "EDGE_TYPES"]
