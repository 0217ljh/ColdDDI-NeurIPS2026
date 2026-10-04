"""PK/PD keyword matcher (Appendix A.2).

Label normalized DDI types from :func:`coldddi.data.filter.run_filter_pipeline`
as PK, PD, Mixed, or Unknown. Keywords are frozen for Table A.2's 215-type
distribution; change them only with a corresponding paper update.

``python -m coldddi.annotations.pkpd_keywords --ddi-edges PATH --out PATH``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Frozen keywords; preserve hit order for downstream summary tables.
PK_KEYWORDS: tuple[str, ...] = (
    "metabolism",
    "excretion",
    "absorption",
    "serum concentration",
    "bioavailability",
    "clearance",
    "cyp",
    "transporter",
    "protein binding",
)

PD_KEYWORDS: tuple[str, ...] = (
    "activities",
    "therapeutic efficacy",
    "adverse effect",
    "risk",
    "receptor binding",
    "analgesic",
    "sedative",
    "hypotensive",
    "bleeding",
    "qtc",
    "arrhythmia",
    "bradycardia",
    "tachycardia",
    "cns depressant",
    "hypertension",
    "anticoagulant",
    "hemorrhage",
    "myopathy",
    "sedation",
    "nephrotoxicity",
    "effectiveness",
)

LabelKind = str  # "PK" | "PD" | "Mixed" | "Unknown"


def label_ddi_type(ddi_type: str) -> tuple[LabelKind, list[str], list[str]]:
    """Return ``(label, pk_hits, pd_hits)`` for a single normalized DDI type.

    Match case-insensitive substrings, as in
    ``Label_ddi_for_PK_and_PD/extract_drug_classification.py``.
    """
    text = ddi_type.lower()
    pk_hits = [kw for kw in PK_KEYWORDS if kw in text]
    pd_hits = [kw for kw in PD_KEYWORDS if kw in text]
    if pk_hits and pd_hits:
        return "Mixed", pk_hits, pd_hits
    if pk_hits:
        return "PK", pk_hits, pd_hits
    if pd_hits:
        return "PD", pk_hits, pd_hits
    return "Unknown", pk_hits, pd_hits


_PKPD_OUTPUT_COLUMNS: tuple[str, ...] = (
    "ddi_type",
    "pk_pd_label",
    "matched_pk_keywords",
    "matched_pd_keywords",
)


def label_ddi_types(types: Iterable[str]) -> pd.DataFrame:
    """Label a collection of unique DDI type strings.

    Returns a DataFrame with columns
    ``ddi_type, pk_pd_label, matched_pk_keywords, matched_pd_keywords``,
    sorted by ``ddi_type``. An empty input yields an empty DataFrame
    with the same schema.
    """
    rows: list[dict] = []
    for ddi_type in sorted(set(types)):
        label, pk_hits, pd_hits = label_ddi_type(ddi_type)
        rows.append(
            {
                "ddi_type": ddi_type,
                "pk_pd_label": label,
                "matched_pk_keywords": "; ".join(pk_hits),
                "matched_pd_keywords": "; ".join(pd_hits),
            }
        )
    return pd.DataFrame(rows, columns=list(_PKPD_OUTPUT_COLUMNS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label DDI types with PK/PD/Mixed/Unknown using a keyword matcher.",
    )
    parser.add_argument(
        "--ddi-edges",
        required=True,
        type=Path,
        help="Input ddi_edges.csv (must contain a `ddi_type` column).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output ddi_pk_pd_labels.csv path.",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.ddi_edges, usecols=["ddi_type"])
    types = df["ddi_type"].dropna().astype(str)
    out_df = label_ddi_types(types)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    counts = out_df["pk_pd_label"].value_counts()
    print(f"Wrote {args.out}  ({len(out_df)} types)")
    for label, cnt in counts.items():
        print(f"  {label:>8}: {cnt:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
