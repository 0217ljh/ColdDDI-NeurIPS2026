"""Stage 3 — bundle the four enriched tables as release Parquet artifacts.

Appendix A.6 / Table 16 lists four Parquet files under ``annotations/``:

* ``pkpd.parquet``                  — 215 ddi_type → {PK, PD, Mixed}
* ``ab.parquet``                    — every positive pair → {A, B} + key entity
* ``mediating_entities.parquet``    — Type-A pair → mediating enzyme/transporter/target
* ``action_pairs.parquet``          — Type-A pair → role tuple + mechanism chain

Two release modes exist:

* ``full``    — paths use the bare names above; the resulting Parquet
  files are *not* shipped (they encode the full 565,731-pair graph,
  which redistributes DrugBank). They live under
  ``data/private/outputs_full/annotations/`` and feed downstream
  experiments only.
* ``sample`` — pair-level files gain a ``_sample`` suffix; the resulting
  Parquet files are small enough to ship in the public ``annotations/``
  directory under DrugBank's CC-BY-NC 4.0 redistribution clause for
  reproducibility purposes. ``pkpd.parquet`` is always the full
  215-type table (no pair-level information, license-safe).

CLI
---
::

    # full release artifacts (do not commit)
    python -m coldddi.data.release_parquet \\
      --enriched-dir data/private/intermediate/enriched \\
      --out-dir      data/private/outputs_full/annotations \\
      --mode full

    # sample release artifacts (committed under annotations/)
    python -m coldddi.data.release_parquet \\
      --enriched-dir data/public/intermediate/enriched \\
      --out-dir      annotations \\
      --mode sample \\
      --full-pkpd    data/private/intermediate/enriched/ddi_pk_pd_labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

from coldddi.annotations.derive_type_a_tables import derive_type_a_tables

ReleaseMode = Literal["full", "sample"]


def _coerce_has_key_entity_to_bool(ab: pd.DataFrame) -> pd.DataFrame:
    """Parse CSV ``has_key_entity`` values as booleans; reject unknown values."""
    if "has_key_entity" not in ab.columns:
        return ab
    series = ab["has_key_entity"]
    if series.dtype == bool:
        return ab
    mapping = {"true": True, "false": False, "1": True, "0": False, "": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        unknown = series[parsed.isna()].astype(str).unique().tolist()[:5]
        raise ValueError(
            f"`has_key_entity` contains values that cannot be parsed as bool: {unknown}"
        )
    out = ab.copy()
    out["has_key_entity"] = parsed.fillna(False).astype(bool)
    return out


def dump_release_parquets(
    *,
    enriched_dir: Path,
    out_dir: Path,
    mode: ReleaseMode,
    full_pkpd_csv: Path | None = None,
    also_csv: bool = False,
) -> dict[str, Path]:
    """Write the four release Parquet files into ``out_dir``.

    Parameters
    ----------
    enriched_dir
        Directory containing the Stage-2 outputs:
        ``ddi_pk_pd_labels.csv`` and ``ddi_key_entities.csv``. Derive
        Type-A tables from the latter after boolean normalization.
    out_dir
        Where the Parquet files are written. Created if missing.
    mode
        ``"full"`` keeps bare filenames; ``"sample"``
        appends ``_sample`` to ``ab``, ``mediating_entities`` and
        ``action_pairs`` (``pkpd.parquet`` is always full).
    full_pkpd_csv
        When ``mode="sample"``, override the source for ``pkpd.parquet``.
        The pair-level Stage-2 outputs come from the toy / sample
        ``enriched_dir``, but ``pkpd.parquet`` is meant to be the *full*
        215-type table; pass the licensed full pipeline's
        ``ddi_pk_pd_labels.csv`` here.
    also_csv
        Write ``.csv`` copies alongside each Parquet if True.

    Returns
    -------
    dict mapping logical name ("pkpd", "ab", "mediating_entities",
    "action_pairs") to the written Parquet path.
    """
    if mode not in ("full", "sample"):
        raise ValueError(f"mode must be 'full' or 'sample', got {mode!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if mode == "full" else "_sample"

    # pkpd has no suffix; sample mode can use the licensed full pipeline's table.
    pkpd_csv = full_pkpd_csv if (mode == "sample" and full_pkpd_csv) else enriched_dir / "ddi_pk_pd_labels.csv"
    pkpd_df = pd.read_csv(pkpd_csv)
    pkpd_path = out_dir / "pkpd.parquet"
    pkpd_df.to_parquet(pkpd_path, index=False)

    # All positive pairs, with boolean has_key_entity.
    ab_csv = enriched_dir / "ddi_key_entities.csv"
    ab_df = _coerce_has_key_entity_to_bool(pd.read_csv(ab_csv))
    ab_path = out_dir / f"ab{suffix}.parquet"
    ab_df.to_parquet(ab_path, index=False)

    # Derive both Type-A tables from the normalized A/B table.
    mediating_df, action_df = derive_type_a_tables(ab_df)
    mediating_path = out_dir / f"mediating_entities{suffix}.parquet"
    action_path = out_dir / f"action_pairs{suffix}.parquet"
    mediating_df.to_parquet(mediating_path, index=False)
    action_df.to_parquet(action_path, index=False)

    written = {
        "pkpd": pkpd_path,
        "ab": ab_path,
        "mediating_entities": mediating_path,
        "action_pairs": action_path,
    }
    if also_csv:
        for name, df in (
            ("pkpd", pkpd_df),
            (f"ab{suffix}", ab_df),
            (f"mediating_entities{suffix}", mediating_df),
            (f"action_pairs{suffix}", action_df),
        ):
            df.to_csv(out_dir / f"{name}.csv", index=False)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stage-2 enriched CSVs to release Parquet files "
            "(pkpd / ab / mediating_entities / action_pairs)."
        ),
    )
    parser.add_argument(
        "--enriched-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing ddi_pk_pd_labels.csv and "
            "ddi_key_entities.csv (e.g. data/public/intermediate/enriched)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for the four .parquet files.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full", "sample"],
        help=(
            "'full' keeps bare names (do not commit); "
            "'sample' appends _sample to pair-level files."
        ),
    )
    parser.add_argument(
        "--full-pkpd",
        type=Path,
        default=None,
        help=(
            "Optional path to the licensed full pipeline's ddi_pk_pd_labels.csv. "
            "When --mode=sample, lets pkpd.parquet stay full while pair-level "
            "tables come from the smaller enriched_dir."
        ),
    )
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="Also write .csv copies for human inspection.",
    )
    args = parser.parse_args(argv)

    written = dump_release_parquets(
        enriched_dir=args.enriched_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        full_pkpd_csv=args.full_pkpd,
        also_csv=args.also_csv,
    )
    for name, path in written.items():
        n_rows = pd.read_parquet(path).shape[0]
        print(f"  {name:<20} -> {path}  ({n_rows:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
