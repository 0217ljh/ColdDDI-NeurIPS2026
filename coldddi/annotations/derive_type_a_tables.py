"""Project ``ddi_key_entities.csv`` into the two Type-A release tables.

Paper Table 16 lists two derived artifacts that the release ships as
Parquet alongside ``ab.parquet``:

* ``mediating_entities.parquet`` — for every Type-A pair, the shared
  enzyme / transporter / target (the *bridge* of the interaction). PK-A
  pairs map to enzyme or transporter; PD-A pairs map to a target.
* ``action_pairs.parquet`` — for every Type-A pair, the role tuple
  describing each drug's relationship to the bridge (substrate /
  inhibitor / agonist / antagonist / ...) plus the textual mechanism
  chain.

Both tables are pure projections of ``ddi_key_entities.csv``: rows where
``has_key_entity`` is True, with different column subsets. Type-B pairs
are intentionally excluded because they have no recorded mediating
entity.

CLI
---
``python -m coldddi.annotations.derive_type_a_tables --key-entities PATH --out-dir PATH``

Public surface
--------------
- :func:`derive_type_a_tables`
- :func:`main` — CLI entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MEDIATING_ENTITIES_COLS: tuple[str, ...] = (
    "drug_a_id",
    "drug_b_id",
    "pk_pd_label",
    "entity_id",
    "entity_name",
    "entity_type",
)

ACTION_PAIRS_COLS: tuple[str, ...] = (
    "drug_a_id",
    "drug_b_id",
    "pk_pd_label",
    "action_drug_a",
    "action_drug_b",
    "match_pattern",
    "mechanism_chain",
    "chain_type",
    "confidence",
)


def derive_type_a_tables(
    ddi_key_entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project ``ddi_key_entities`` into the two Type-A release tables.

    Parameters
    ----------
    ddi_key_entities
        The output of :mod:`coldddi.annotations.ab_subdivision`. Must
        contain the columns ``drug_a_id, drug_b_id, pk_pd_label,
        has_key_entity, key_entity_id, key_entity_name, key_entity_type,
        action_drug_a, action_drug_b, match_pattern, mechanism_chain,
        chain_type, confidence``.

    Returns
    -------
    (mediating_entities, action_pairs)
        Two DataFrames whose row count equals the number of Type-A pairs
        in the input (rows where ``has_key_entity`` is True).
    """
    if "has_key_entity" not in ddi_key_entities.columns:
        raise ValueError(
            "Input must contain a `has_key_entity` column "
            "(produced by `coldddi.annotations.ab_subdivision`)."
        )
    # `astype(bool)` on the string "False" returns True (any non-empty string
    # is truthy), which would silently leak Type-B rows into the Type-A
    # projection. Normalize CSV round-tripped values explicitly.
    raw = ddi_key_entities["has_key_entity"]
    if raw.dtype == bool:
        mask = raw
    else:
        mask = raw.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False, "": False}
        )
        if mask.isna().any():
            unknown = raw[mask.isna()].astype(str).unique().tolist()[:5]
            raise ValueError(
                f"`has_key_entity` contains values that cannot be parsed as bool: {unknown}"
            )
    type_a = ddi_key_entities[mask.fillna(False).astype(bool)].copy()

    mediating_entities = (
        type_a.rename(
            columns={
                "key_entity_id": "entity_id",
                "key_entity_name": "entity_name",
                "key_entity_type": "entity_type",
            }
        )
        .loc[:, list(MEDIATING_ENTITIES_COLS)]
        .reset_index(drop=True)
    )

    action_pairs = type_a.loc[:, list(ACTION_PAIRS_COLS)].reset_index(drop=True)

    return mediating_entities, action_pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project ddi_key_entities.csv into mediating_entities + "
            "action_pairs Parquet tables (Type-A pairs only)."
        ),
    )
    parser.add_argument(
        "--key-entities",
        required=True,
        type=Path,
        help="Path to ddi_key_entities.csv from ab_subdivision.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for mediating_entities.parquet + action_pairs.parquet.",
    )
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="In addition to .parquet, write .csv copies of both tables.",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.key_entities)
    mediating_entities, action_pairs = derive_type_a_tables(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mediating_entities.to_parquet(args.out_dir / "mediating_entities.parquet", index=False)
    action_pairs.to_parquet(args.out_dir / "action_pairs.parquet", index=False)
    print(
        f"Wrote {args.out_dir / 'mediating_entities.parquet'}  "
        f"({len(mediating_entities):,} rows)"
    )
    print(
        f"Wrote {args.out_dir / 'action_pairs.parquet'}  "
        f"({len(action_pairs):,} rows)"
    )

    if args.also_csv:
        mediating_entities.to_csv(args.out_dir / "mediating_entities.csv", index=False)
        action_pairs.to_csv(args.out_dir / "action_pairs.csv", index=False)
        print(f"Also wrote .csv copies under {args.out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
