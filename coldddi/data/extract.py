"""Stage 1a — DrugBank XML → raw CSV tables.

Stream-parses a DrugBank ``full database.xml`` into the seven raw tables
required by the downstream cleaning and annotation stages:

* ``drugs.csv``
* ``ddi_edges.csv``
* ``drug_enzymes.csv``
* ``drug_targets.csv``
* ``drug_transporters.csv``
* ``drug_carriers.csv``
* ``drug_pathways.csv``

No filtering is applied here: the output mirrors the XML's full content
(17,430 drugs in DrugBank 5.1.13). Filtering and the seven-step pipeline
described in Appendix A.1 live in :mod:`coldddi.data.filter`.

CLI
---
``python -m coldddi.data.extract --xml PATH --out DIR``

Public surface
--------------
- :class:`RawTables` — dataclass holding the seven DataFrames.
- :func:`parse_drugbank_xml` — XML → :class:`RawTables`.
- :func:`load_raw_tables` — read the seven CSV files back into a :class:`RawTables`.
- :func:`write_raw_tables` — dump a :class:`RawTables` as CSVs.
- :func:`main` — CLI entry point.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

DRUGBANK_NS: str = "{http://www.drugbank.ca}"


@dataclass
class RawTables:
    """The seven unfiltered XML-derived tables.

    Each ``drug_*`` table uses ``drugbank_id`` as the foreign key linking
    rows to :attr:`drugs`. The ``edges`` table is symmetric-deduped: each
    undirected DDI pair appears once (with the original orientation
    preserved so the description text can be normalized later).
    """

    drugs: pd.DataFrame
    edges: pd.DataFrame
    enzymes: pd.DataFrame
    targets: pd.DataFrame
    transporters: pd.DataFrame
    carriers: pd.DataFrame
    pathways: pd.DataFrame


# ---------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    if tag.startswith(DRUGBANK_NS):
        return tag[len(DRUGBANK_NS):]
    return tag


def _findtext(elem: ET.Element, name: str) -> str | None:
    child = elem.find(f"{DRUGBANK_NS}{name}")
    if child is None:
        return None
    return child.text


def _findall(elem: ET.Element, name: str) -> list[ET.Element]:
    return elem.findall(f"{DRUGBANK_NS}{name}")


def _primary_drugbank_id(drug: ET.Element) -> str | None:
    for did in _findall(drug, "drugbank-id"):
        if did.attrib.get("primary") == "true":
            return (did.text or "").strip() or None
    return None


def _smiles_of(drug: ET.Element) -> str | None:
    cps = drug.find(f"{DRUGBANK_NS}calculated-properties")
    if cps is None:
        return None
    for prop in cps.findall(f"{DRUGBANK_NS}property"):
        kind = _findtext(prop, "kind")
        if kind == "SMILES":
            return _findtext(prop, "value")
    return None


def _groups_of(drug: ET.Element) -> set[str]:
    groups = drug.find(f"{DRUGBANK_NS}groups")
    if groups is None:
        return set()
    return {(g.text or "").strip() for g in groups.findall(f"{DRUGBANK_NS}group") if g.text}


def _ddi_partners_of(drug: ET.Element) -> Iterator[tuple[str, str]]:
    """Yield (partner_drugbank_id, description) for each interaction."""
    interactions = drug.find(f"{DRUGBANK_NS}drug-interactions")
    if interactions is None:
        return
    for di in interactions.findall(f"{DRUGBANK_NS}drug-interaction"):
        partner_id_elem = di.find(f"{DRUGBANK_NS}drugbank-id")
        desc_elem = di.find(f"{DRUGBANK_NS}description")
        if partner_id_elem is None or desc_elem is None:
            continue
        pid = (partner_id_elem.text or "").strip()
        desc = (desc_elem.text or "").strip()
        if pid and desc:
            yield pid, desc


def _polypeptide_entities(
    drug: ET.Element,
    container_tag: str,
    item_tag: str,
) -> list[dict[str, str]]:
    """Walk a polypeptide-style sub-tree (enzymes/targets/transporters/carriers).

    Each item can have multiple ``<actions>/<action>`` entries; we emit
    one row per action (no actions → one row with ``action=""``).
    Schema: ``drugbank_id, <item>_id, <item>_name, organism, action``.
    """
    rows: list[dict[str, str]] = []
    container = drug.find(f"{DRUGBANK_NS}{container_tag}")
    if container is None:
        return rows
    for item in container.findall(f"{DRUGBANK_NS}{item_tag}"):
        ent_id = (_findtext(item, "id") or "").strip()
        ent_name = (_findtext(item, "name") or "").strip()
        organism = (_findtext(item, "organism") or "").strip()
        if not ent_id and not ent_name:
            continue
        actions_block = item.find(f"{DRUGBANK_NS}actions")
        actions: list[str] = []
        if actions_block is not None:
            for a in actions_block.findall(f"{DRUGBANK_NS}action"):
                txt = (a.text or "").strip()
                if txt:
                    actions.append(txt)
        if not actions:
            actions = [""]
        for action in actions:
            rows.append(
                {
                    f"{item_tag}_id": ent_id,
                    f"{item_tag}_name": ent_name,
                    "organism": organism,
                    "action": action,
                }
            )
    return rows


def _pathway_entities(drug: ET.Element) -> list[dict[str, str]]:
    """Walk ``<pathways>/<pathway>`` (no organism / action columns)."""
    rows: list[dict[str, str]] = []
    container = drug.find(f"{DRUGBANK_NS}pathways")
    if container is None:
        return rows
    for p in container.findall(f"{DRUGBANK_NS}pathway"):
        sid = (_findtext(p, "smpdb-id") or "").strip()
        pname = (_findtext(p, "name") or "").strip()
        if not sid and not pname:
            continue
        rows.append({"pathway_id": sid, "pathway_name": pname})
    return rows


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------


def parse_drugbank_xml(xml_path: Path) -> RawTables:
    """Stream-parse the DrugBank XML into the seven raw tables (no filtering).

    Memory is bounded: each top-level ``<drug>`` is fully consumed and
    cleared before moving on, so a 1.6 GB XML uses on the order of tens
    of megabytes of resident memory.
    """
    drugs: list[dict] = []
    raw_edges: list[dict] = []
    enzymes: list[dict] = []
    targets: list[dict] = []
    transporters: list[dict] = []
    carriers: list[dict] = []
    pathways: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()

    context = ET.iterparse(str(xml_path), events=("end",))
    _, root = next(context)
    for _event, elem in context:
        if _local_tag(elem) != "drug":
            continue
        db_id = _primary_drugbank_id(elem)
        if db_id is None:
            elem.clear()
            continue
        # `type` is a *drug-element attribute* (e.g. <drug type="biotech">),
        # not a child element.
        dtype = elem.attrib.get("type", "")
        name = _findtext(elem, "name") or ""
        smiles = _smiles_of(elem)
        groups = _groups_of(elem)
        drugs.append(
            {
                "drugbank_id": db_id,
                "name": name.strip(),
                "type": dtype.strip(),
                "smiles": smiles,
                "groups": ";".join(sorted(groups)),
            }
        )
        for partner_id, desc in _ddi_partners_of(elem):
            a, b = sorted((db_id, partner_id))
            if (a, b) in seen_edges:
                continue
            seen_edges.add((a, b))
            raw_edges.append(
                {
                    "drug_a_id": db_id,
                    "drug_b_id": partner_id,
                    "drug_a_name": name.strip(),
                    "description": desc,
                }
            )
        for row in _polypeptide_entities(elem, "enzymes", "enzyme"):
            enzymes.append({"drugbank_id": db_id, **row})
        for row in _polypeptide_entities(elem, "targets", "target"):
            targets.append({"drugbank_id": db_id, **row})
        for row in _polypeptide_entities(elem, "transporters", "transporter"):
            transporters.append({"drugbank_id": db_id, **row})
        for row in _polypeptide_entities(elem, "carriers", "carrier"):
            carriers.append({"drugbank_id": db_id, **row})
        for row in _pathway_entities(elem):
            pathways.append({"drugbank_id": db_id, **row})

        elem.clear()
        root.clear()

    drugs_df = pd.DataFrame(
        drugs,
        columns=["drugbank_id", "name", "type", "smiles", "groups"],
    ).drop_duplicates(subset="drugbank_id").reset_index(drop=True)
    edges_df = pd.DataFrame(
        raw_edges,
        columns=["drug_a_id", "drug_b_id", "drug_a_name", "description"],
    ).reset_index(drop=True)
    enzymes_df = pd.DataFrame(
        enzymes,
        columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"],
    )
    targets_df = pd.DataFrame(
        targets,
        columns=["drugbank_id", "target_id", "target_name", "organism", "action"],
    )
    transporters_df = pd.DataFrame(
        transporters,
        columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"],
    )
    carriers_df = pd.DataFrame(
        carriers,
        columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"],
    )
    pathways_df = pd.DataFrame(
        pathways,
        columns=["drugbank_id", "pathway_id", "pathway_name"],
    )
    return RawTables(
        drugs=drugs_df,
        edges=edges_df,
        enzymes=enzymes_df,
        targets=targets_df,
        transporters=transporters_df,
        carriers=carriers_df,
        pathways=pathways_df,
    )


# ---------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------


_TABLE_FILES: tuple[tuple[str, str], ...] = (
    ("drugs", "drugs.csv"),
    ("edges", "ddi_edges.csv"),
    ("enzymes", "drug_enzymes.csv"),
    ("targets", "drug_targets.csv"),
    ("transporters", "drug_transporters.csv"),
    ("carriers", "drug_carriers.csv"),
    ("pathways", "drug_pathways.csv"),
)


def write_raw_tables(raw: RawTables, out_dir: Path) -> None:
    """Dump the seven tables as CSV files into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for attr, fname in _TABLE_FILES:
        df = getattr(raw, attr)
        df.to_csv(out_dir / fname, index=False)


def load_raw_tables(in_dir: Path) -> RawTables:
    """Read the seven CSV files written by :func:`write_raw_tables`."""
    kwargs = {}
    for attr, fname in _TABLE_FILES:
        kwargs[attr] = pd.read_csv(in_dir / fname, low_memory=False)
    return RawTables(**kwargs)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream-parse DrugBank XML into seven raw CSV tables (no filtering).",
    )
    parser.add_argument(
        "--xml",
        required=True,
        type=Path,
        help="Path to DrugBank full database.xml",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory; seven csv files are written here",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    args = parser.parse_args(argv)

    if not args.quiet:
        print(f"[extract] parsing {args.xml}", flush=True)
    raw = parse_drugbank_xml(args.xml)
    if not args.quiet:
        print(
            f"[extract] drugs={len(raw.drugs):,}  edges={len(raw.edges):,}  "
            f"enzymes={len(raw.enzymes):,}  targets={len(raw.targets):,}  "
            f"transporters={len(raw.transporters):,}  carriers={len(raw.carriers):,}  "
            f"pathways={len(raw.pathways):,}",
            flush=True,
        )
    write_raw_tables(raw, args.out)
    if not args.quiet:
        print(f"[extract] wrote 7 csv files under {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
