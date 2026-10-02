"""Stage 1b — seven-step DrugBank filtering pipeline (Appendix A.1).

Consumes the raw CSV tables produced by :mod:`coldddi.data.extract` and
applies the cleaning pipeline described in the paper's Appendix A.1.

Pipeline steps
--------------
The numbers below are the published targets from Table A.1 of the paper.

==== ===========================================  =======  ==========  ======
Step  Operation                                    Drugs    DDI edges   Types
==== ===========================================  =======  ==========  ======
0     DrugBank 5.1.13 raw                           17,430   1,427,655  n/a
1     Retain small molecules                        13,166   1,205,013  n/a
2     Require valid RDKit SMILES                    12,303   1,161,857  n/a
3     Restrict to approved compounds                 2,643     613,514  n/a
4     Extract DDI edges + normalize descriptions     2,643     613,514     451
5     Remove low-frequency types (<10)               2,151     613,055     221
6     Remove inorganic & metal-containing drugs      1,994     565,978     215
7     Remove low-degree drugs (<10 edges)            1,900     565,731     215
==== ===========================================  =======  ==========  ======

Auxiliary tables (``enzymes / targets / transporters / carriers / pathways``)
are pruned in lockstep with the drug list so every row references a
drugbank_id that survives Step 7.

CLI
---
``python -m coldddi.data.filter --raw-dir PATH --out DIR``

Public surface
--------------
- :class:`FilterReport`
- :func:`run_filter_pipeline`
- :func:`main` — module CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from coldddi.data.extract import RawTables, load_raw_tables

LOW_FREQ_TYPE_THRESHOLD: int = 10
LOW_DEGREE_DRUG_THRESHOLD: int = 10

_NORMALIZE_STRIP_CHARS: str = " ,;:."

# Atomic numbers — kept identical to the legacy notebook
# (`Notebooks/Data_Extraction.ipynb` cell 30) so Step 6 reproduces the paper's
# 1,994 drug / 565,978 edge / 215 type counts. Note that `Tc` (43) and `Tl`
# (81) are deliberately omitted: they are chemically transition / post-
# transition metals but the original pipeline never excluded drugs containing
# them, and matching the legacy behaviour is a hard requirement.
METAL_ATOMIC_NUMS: frozenset[int] = frozenset(
    {
        3, 11, 19, 37, 55,                          # Li, Na, K, Rb, Cs
        4, 12, 20, 38, 56,                          # Be, Mg, Ca, Sr, Ba
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,    # Sc - Zn
        39, 40, 41, 42, 44, 45, 46, 47, 48,        # Y - Cd  (no Tc = 43)
        72, 73, 74, 75, 76, 77, 78, 79, 80,        # Hf - Hg
        13, 31, 49, 50, 82, 83,                    # Al, Ga, In, Sn, Pb, Bi  (no Tl = 81)
    }
)


@dataclass
class StepStat:
    """One row of the filtering report (matches paper Table A.1)."""

    step: int
    name: str
    n_drugs: int
    n_edges: int | None
    n_types: int | None


@dataclass
class FilterReport:
    """Cumulative output of :func:`run_filter_pipeline`."""

    drugs: pd.DataFrame  # final drug table after Step 7
    edges: pd.DataFrame  # final DDI edges after Step 7
    enzymes: pd.DataFrame
    targets: pd.DataFrame
    transporters: pd.DataFrame
    carriers: pd.DataFrame
    pathways: pd.DataFrame
    type_to_text: dict[str, str] = field(default_factory=dict)
    step_stats: list[StepStat] = field(default_factory=list)

    def stats_table(self) -> pd.DataFrame:
        rows = [
            {
                "step": s.step,
                "operation": s.name,
                "n_drugs": s.n_drugs,
                "n_edges": s.n_edges if s.n_edges is not None else "n/a",
                "n_types": s.n_types if s.n_types is not None else "n/a",
            }
            for s in self.step_stats
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-step primitives (kept tiny so they compose cleanly)
# ---------------------------------------------------------------------


def _is_valid_smiles(smi: object) -> bool:
    if not isinstance(smi, str) or not smi:
        return False
    try:
        from rdkit import Chem  # local import keeps module importable without RDKit
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Step 2 requires RDKit (paper pins v2023.09). "
            "Install it via `pip install rdkit==2023.9.6`."
        ) from exc
    return Chem.MolFromSmiles(smi) is not None


def _has_carbon_and_no_metal(smi: object) -> tuple[bool, bool]:
    """Return (has_carbon, contains_metal). Mirrors notebook semantics."""
    from rdkit import Chem

    if not isinstance(smi, str) or not smi:
        return False, False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, False
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    contains_metal = any(atom.GetAtomicNum() in METAL_ATOMIC_NUMS for atom in mol.GetAtoms())
    return has_carbon, contains_metal


def _normalize_description(desc: str, name_a: str, name_b: str) -> str:
    """Strip the two drug names and collapse whitespace.

    Matches `Notebooks/Data_Extraction.ipynb` cell 39 verbatim:
    case-insensitive, no word-boundary, strip ``" ,;:."`` then collapse.
    """
    out = desc
    for name in (name_a, name_b):
        if not name:
            continue
        out = re.sub(re.escape(name), "", out, flags=re.IGNORECASE)
    out = out.strip(_NORMALIZE_STRIP_CHARS)
    out = re.sub(r"\s+", " ", out)
    return out


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def _prune_aux_tables(
    drug_set: set[str],
    enzymes: pd.DataFrame,
    targets: pd.DataFrame,
    transporters: pd.DataFrame,
    carriers: pd.DataFrame,
    pathways: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        enzymes[enzymes["drugbank_id"].isin(drug_set)].reset_index(drop=True),
        targets[targets["drugbank_id"].isin(drug_set)].reset_index(drop=True),
        transporters[transporters["drugbank_id"].isin(drug_set)].reset_index(drop=True),
        carriers[carriers["drugbank_id"].isin(drug_set)].reset_index(drop=True),
        pathways[pathways["drugbank_id"].isin(drug_set)].reset_index(drop=True),
    )


def run_filter_pipeline(raw: RawTables, *, verbose: bool = True) -> FilterReport:
    """Apply Steps 1-7 to a :class:`RawTables` and return a :class:`FilterReport`."""

    def log(msg: str) -> None:
        if verbose:
            print(f"[filter] {msg}", flush=True)

    drugs_df = raw.drugs.copy()
    edges = raw.edges.copy()

    step_stats: list[StepStat] = []
    step_stats.append(
        StepStat(step=0, name="DrugBank raw", n_drugs=len(drugs_df), n_edges=len(edges), n_types=None)
    )
    log(f"Step 0: {len(drugs_df):,} drugs, {len(edges):,} raw edges")

    # --- Step 1: small molecules ---
    # Retain edges only when *both* endpoints survive — single-endpoint
    # filtering would leave dangling references that the later steps then
    # have to repair. Apply the same double-end filter to all subsequent
    # drug-level filters (Steps 1, 2, 3, 6).
    drugs_df = drugs_df[drugs_df["type"] == "small molecule"].reset_index(drop=True)
    keep_set = set(drugs_df["drugbank_id"])
    edges = edges[edges["drug_a_id"].isin(keep_set) & edges["drug_b_id"].isin(keep_set)].reset_index(drop=True)
    step_stats.append(
        StepStat(step=1, name="Retain small molecules", n_drugs=len(drugs_df), n_edges=len(edges), n_types=None)
    )
    log(f"Step 1: {len(drugs_df):,} drugs, {len(edges):,} edges")

    # --- Step 2: valid RDKit SMILES ---
    valid_mask = drugs_df["smiles"].map(_is_valid_smiles)
    drugs_df = drugs_df[valid_mask].reset_index(drop=True)
    keep_set = set(drugs_df["drugbank_id"])
    edges = edges[edges["drug_a_id"].isin(keep_set) & edges["drug_b_id"].isin(keep_set)].reset_index(drop=True)
    step_stats.append(
        StepStat(step=2, name="Require valid RDKit SMILES", n_drugs=len(drugs_df), n_edges=len(edges), n_types=None)
    )
    log(f"Step 2: {len(drugs_df):,} drugs, {len(edges):,} edges")

    # --- Step 3: approved ---
    approved_mask = drugs_df["groups"].fillna("").str.contains(r"\bapproved\b")
    drugs_df = drugs_df[approved_mask].reset_index(drop=True)
    keep_set = set(drugs_df["drugbank_id"])
    edges = edges[edges["drug_a_id"].isin(keep_set) & edges["drug_b_id"].isin(keep_set)].reset_index(drop=True)
    step_stats.append(
        StepStat(step=3, name="Restrict to approved", n_drugs=len(drugs_df), n_edges=len(edges), n_types=None)
    )
    log(f"Step 3: {len(drugs_df):,} drugs, {len(edges):,} edges")

    # --- Step 4: extract + normalize descriptions ---
    name_lookup = dict(zip(drugs_df["drugbank_id"], drugs_df["name"]))
    edges["drug_b_name"] = edges["drug_b_id"].map(name_lookup).fillna("")
    edges["ddi_type"] = [
        _normalize_description(d, na, nb)
        for d, na, nb in zip(edges["description"], edges["drug_a_name"], edges["drug_b_name"])
    ]
    step_stats.append(
        StepStat(
            step=4,
            name="Extract DDI + normalize descriptions",
            n_drugs=len(drugs_df),
            n_edges=len(edges),
            n_types=int(edges["ddi_type"].nunique()),
        )
    )
    log(f"Step 4: {len(drugs_df):,} drugs, {len(edges):,} edges, {edges['ddi_type'].nunique()} types")

    # --- Step 5: remove low-frequency types ---
    type_counts = edges["ddi_type"].value_counts()
    keep_types = set(type_counts[type_counts >= LOW_FREQ_TYPE_THRESHOLD].index)
    edges = edges[edges["ddi_type"].isin(keep_types)].reset_index(drop=True)
    drugs_in_edges = set(edges["drug_a_id"]) | set(edges["drug_b_id"])
    drugs_df = drugs_df[drugs_df["drugbank_id"].isin(drugs_in_edges)].reset_index(drop=True)
    step_stats.append(
        StepStat(
            step=5,
            name="Remove low-frequency types (<10)",
            n_drugs=len(drugs_df),
            n_edges=len(edges),
            n_types=int(edges["ddi_type"].nunique()),
        )
    )
    log(f"Step 5: {len(drugs_df):,} drugs, {len(edges):,} edges, {edges['ddi_type'].nunique()} types")

    # --- Step 6: remove inorganic & metal-containing drugs ---
    flags = drugs_df["smiles"].map(_has_carbon_and_no_metal)
    drugs_df["has_carbon"] = flags.map(lambda t: t[0])
    drugs_df["contains_metal"] = flags.map(lambda t: t[1])
    drugs_df = drugs_df[drugs_df["has_carbon"] & ~drugs_df["contains_metal"]].reset_index(drop=True)
    drugs_df = drugs_df.drop(columns=["has_carbon", "contains_metal"])
    keep_set = set(drugs_df["drugbank_id"])
    edges = edges[edges["drug_a_id"].isin(keep_set) & edges["drug_b_id"].isin(keep_set)].reset_index(drop=True)
    # Re-apply low-frequency type filter: removing metal/inorganic drugs
    # drops edges from a few rare types below the 10-occurrence floor.
    # Paper Step 6 reports 215 types (down from 221 at Step 5), which only
    # matches if this prune is repeated here.
    type_counts_after_metals = edges["ddi_type"].value_counts()
    keep_types_step6 = set(
        type_counts_after_metals[type_counts_after_metals >= LOW_FREQ_TYPE_THRESHOLD].index
    )
    edges = edges[edges["ddi_type"].isin(keep_types_step6)].reset_index(drop=True)
    drugs_in_edges = set(edges["drug_a_id"]) | set(edges["drug_b_id"])
    drugs_df = drugs_df[drugs_df["drugbank_id"].isin(drugs_in_edges)].reset_index(drop=True)
    step_stats.append(
        StepStat(
            step=6,
            name="Remove inorganic & metal-containing drugs",
            n_drugs=len(drugs_df),
            n_edges=len(edges),
            n_types=int(edges["ddi_type"].nunique()),
        )
    )
    log(f"Step 6: {len(drugs_df):,} drugs, {len(edges):,} edges, {edges['ddi_type'].nunique()} types")

    # --- Step 7: remove low-degree drugs (degree < 10) ---
    # The original 800-drug sampler consumes this degree-ordered pool.
    # Explicit quicksort preserves pandas 2.x tie ordering under pandas 3.x.
    # Upstream counted canonical (min-ID, max-ID) pairs. Preserve that
    # first-occurrence order without changing the released edge orientation.
    ordered = edges["drug_a_id"] <= edges["drug_b_id"]
    first = edges["drug_a_id"].where(ordered, edges["drug_b_id"])
    second = edges["drug_b_id"].where(ordered, edges["drug_a_id"])
    degree_order = pd.concat([first, second]).value_counts(sort=False)
    degree_order = degree_order.sort_values(ascending=False, kind="quicksort")
    subset_candidates = degree_order[degree_order >= LOW_DEGREE_DRUG_THRESHOLD].index.tolist()
    deg: Counter = Counter()
    deg.update(edges["drug_a_id"].tolist())
    deg.update(edges["drug_b_id"].tolist())
    keep_drugs = {d for d, c in deg.items() if c >= LOW_DEGREE_DRUG_THRESHOLD}
    drugs_df = drugs_df[drugs_df["drugbank_id"].isin(keep_drugs)].reset_index(drop=True)
    keep_set = set(drugs_df["drugbank_id"])
    edges = edges[edges["drug_a_id"].isin(keep_set) & edges["drug_b_id"].isin(keep_set)].reset_index(drop=True)
    step_stats.append(
        StepStat(
            step=7,
            name="Remove low-degree drugs (<10 edges)",
            n_drugs=len(drugs_df),
            n_edges=len(edges),
            n_types=int(edges["ddi_type"].nunique()),
        )
    )
    log(f"Step 7: {len(drugs_df):,} drugs, {len(edges):,} edges, {edges['ddi_type'].nunique()} types")

    # --- Prune auxiliary tables to the final drug set ---
    final_drug_set = set(drugs_df["drugbank_id"])
    enzymes, targets, transporters, carriers, pathways = _prune_aux_tables(
        final_drug_set,
        raw.enzymes,
        raw.targets,
        raw.transporters,
        raw.carriers,
        raw.pathways,
    )
    log(
        f"aux tables pruned to final {len(final_drug_set):,} drugs: "
        f"enzymes={len(enzymes):,}  targets={len(targets):,}  "
        f"transporters={len(transporters):,}  carriers={len(carriers):,}  "
        f"pathways={len(pathways):,}"
    )

    type_to_text = (
        edges.groupby("ddi_type")["description"]
        .first()
        .to_dict()
    )

    drugs_df.attrs["subset_candidates"] = subset_candidates
    return FilterReport(
        drugs=drugs_df,
        edges=edges[["drug_a_id", "drug_b_id", "description", "ddi_type"]].reset_index(drop=True),
        enzymes=enzymes,
        targets=targets,
        transporters=transporters,
        carriers=carriers,
        pathways=pathways,
        type_to_text=type_to_text,
        step_stats=step_stats,
    )


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------


_FILTERED_TABLE_FILES: tuple[tuple[str, str], ...] = (
    ("drugs", "drugs.csv"),
    ("edges", "ddi_edges.csv"),
    ("enzymes", "drug_enzymes.csv"),
    ("targets", "drug_targets.csv"),
    ("transporters", "drug_transporters.csv"),
    ("carriers", "drug_carriers.csv"),
    ("pathways", "drug_pathways.csv"),
)


def write_filter_report(report: FilterReport, out_dir: Path) -> None:
    """Dump the seven filtered tables + ``stats.json`` + ``type_to_text.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for attr, fname in _FILTERED_TABLE_FILES:
        df = getattr(report, attr)
        df.to_csv(out_dir / fname, index=False)
    if "subset_candidates" in report.drugs.attrs:
        (out_dir / "subset_candidates.json").write_text(
            json.dumps(report.drugs.attrs["subset_candidates"], indent=2) + "\n", encoding="utf-8",
        )
    (out_dir / "stats.json").write_text(
        json.dumps([asdict(s) for s in report.step_stats], indent=2)
    )
    (out_dir / "type_to_text.json").write_text(
        json.dumps(report.type_to_text, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the 7-step DrugBank filtering pipeline (Appendix A.1).",
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Directory of raw csvs produced by `python -m coldddi.data.extract`.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the filtered csvs + stats.json.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step prints")
    args = parser.parse_args(argv)

    raw = load_raw_tables(args.raw_dir)
    report = run_filter_pipeline(raw, verbose=not args.quiet)

    print("\n=== Filtering pipeline summary ===")
    print(report.stats_table().to_string(index=False))

    write_filter_report(report, args.out)
    print(f"\nArtifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
