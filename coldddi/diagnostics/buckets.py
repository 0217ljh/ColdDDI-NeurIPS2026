"""Per-pair bucket assignment (PK-A / PK-B / PD-A / PD-B / Other).

Paper §4.3 / Table 6 uses two axes:

* PK vs PD — pharmacokinetic vs pharmacodynamic mechanism class,
  set by :mod:`coldddi.annotations.pkpd_keywords` at Stage 2a.
* A vs B — confirmed mediating entity in the KG (A) vs candidates only
  (B), set by :mod:`coldddi.annotations.ab_subdivision` at Stage 2b.

Labels come from ``annotations/ab_sample.parquet`` (toy) or
``ab.parquet`` (full release). Indicators compute the ``ALL`` aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


#: The four primary buckets plus the aggregate.
BUCKET_NAMES: tuple[str, ...] = ("PK-A", "PK-B", "PD-A", "PD-B", "ALL")

#: Pairs with unknown mechanism or missing AB annotation.
BUCKET_OTHER: str = "Other"


@dataclass
class BucketLookup:
    """Per-pair bucket map ``{(drug_a, drug_b): "PK-A" | ...}``.

    Each pair maps to one primary bucket or :data:`BUCKET_OTHER`.
    Use :meth:`bucket` for one pair or :meth:`buckets_for_pairs` for many.
    The indicator pipeline, not this lookup, computes ``ALL``.
    """

    pair_to_bucket: dict[tuple[str, str], str]

    def bucket(self, drug_a_id: str, drug_b_id: str) -> str:
        """Return the bucket for ``(drug_a, drug_b)``.

        Try the directed pair, then its reverse: annotations use canonical
        ordering, but predictions may use either direction.
        """
        a, b = str(drug_a_id), str(drug_b_id)
        if (a, b) in self.pair_to_bucket:
            return self.pair_to_bucket[(a, b)]
        if (b, a) in self.pair_to_bucket:
            return self.pair_to_bucket[(b, a)]
        return BUCKET_OTHER

    def buckets_for_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
    ) -> list[str]:
        return [self.bucket(a, b) for a, b in pairs]

    def coverage_report(
        self,
        pairs: Iterable[tuple[str, str]],
    ) -> dict[str, int]:
        """Per-bucket counts for a pair iterable.

        ``BUCKET_OTHER`` pairs contribute only to the positive-only ``ALL``
        aggregate. A high ``Other`` count warns of incomplete annotation
        coverage and a potentially misleading A-B gap.
        """
        out: dict[str, int] = {b: 0 for b in BUCKET_NAMES if b != "ALL"}
        out[BUCKET_OTHER] = 0
        for a, b in pairs:
            bk = self.bucket(a, b)
            out[bk] = out.get(bk, 0) + 1
        return out


def build_bucket_lookup(ab_table: pd.DataFrame | str | Path) -> BucketLookup:
    """Build a :class:`BucketLookup` from AB annotations.

    Parameters
    ----------
    ab_table
        DataFrame or parquet/CSV path with columns ``drug_a_id``,
        ``drug_b_id``, ``pk_pd_label`` (``"PK"`` / ``"PD"``), and
        ``has_key_entity`` (bool). Both annotation releases use this schema.
    """
    if isinstance(ab_table, (str, Path)):
        path = Path(ab_table)
        if path.suffix == ".parquet":
            ab_table = pd.read_parquet(path)
        else:
            ab_table = pd.read_csv(path)

    required = {"drug_a_id", "drug_b_id", "pk_pd_label", "has_key_entity"}
    missing = required - set(ab_table.columns)
    if missing:
        raise ValueError(
            f"AB annotation table is missing required columns {sorted(missing)}; "
            f"got {sorted(ab_table.columns)}"
        )

    pair_to_bucket: dict[tuple[str, str], str] = {}
    for row in ab_table.itertuples(index=False):
        a = str(row.drug_a_id)
        b = str(row.drug_b_id)
        pkpd = str(row.pk_pd_label).strip().upper()
        if pkpd not in ("PK", "PD"):
            # Mixed or unknown mechanisms resolve to BUCKET_OTHER.
            continue
        a_b = "A" if bool(row.has_key_entity) else "B"
        pair_to_bucket[(a, b)] = f"{pkpd}-{a_b}"
    return BucketLookup(pair_to_bucket=pair_to_bucket)


__all__ = [
    "BUCKET_NAMES",
    "BUCKET_OTHER",
    "BucketLookup",
    "build_bucket_lookup",
]
