"""Per-pair bucket assignment (PK-A / PK-B / PD-A / PD-B / Other).

The four buckets in paper §4.3 / Table 6 stratify each drug-drug
interaction by two binary axes:

* **PK vs PD** — pharmacokinetic vs pharmacodynamic mechanism class,
  set by :mod:`coldddi.annotations.pkpd_keywords` at Stage 2a.
* **A vs B** — whether a *confirmed* mediating-entity for the DDI
  exists in the knowledge graph (Type-A = present, Type-B = candidates
  only, set by :mod:`coldddi.annotations.ab_subdivision` at Stage 2b).

Both labels are already pre-computed in
``annotations/ab_sample.parquet`` (toy) or ``ab.parquet`` (full
release) — this module just exposes a lookup-by-pair API plus the
aggregate ``"ALL"`` bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


#: The four primary buckets plus the aggregate.
BUCKET_NAMES: tuple[str, ...] = ("PK-A", "PK-B", "PD-A", "PD-B", "ALL")

#: Bucket for pairs whose mechanism class (PK/PD) is unknown or for
#: which the AB annotation file did not produce a label.
BUCKET_OTHER: str = "Other"


@dataclass
class BucketLookup:
    """Per-pair bucket map ``{(drug_a, drug_b): "PK-A" | ...}``.

    A pair always falls into exactly one of the four primary buckets
    (or :data:`BUCKET_OTHER`).  Use :meth:`bucket` to look up a pair;
    use :meth:`buckets_for_pairs` to vectorise.

    Aggregations across all pairs (the ``"ALL"`` bucket) are handled
    by the indicator pipeline directly — not by this lookup.
    """

    pair_to_bucket: dict[tuple[str, str], str]

    def bucket(self, drug_a_id: str, drug_b_id: str) -> str:
        """Return the bucket for ``(drug_a, drug_b)``.

        The directed pair is matched first; ``(drug_b, drug_a)`` is
        tried as a fallback because the annotation table is built
        from canonical ordering and downstream prediction tables may
        re-emit either direction.
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

        Useful before running L6 to verify the annotation table covers
        enough of the prediction set — ``BUCKET_OTHER`` rows pass
        silently into per-bucket aggregates only via the positive-only
        ALL bucket, so a high ``Other`` count indicates the annotation
        is incomplete and the headline A-B gap may be misleading.
        """
        out: dict[str, int] = {b: 0 for b in BUCKET_NAMES if b != "ALL"}
        out[BUCKET_OTHER] = 0
        for a, b in pairs:
            bk = self.bucket(a, b)
            out[bk] = out.get(bk, 0) + 1
        return out


def build_bucket_lookup(ab_table: pd.DataFrame | str | Path) -> BucketLookup:
    """Materialise a :class:`BucketLookup` from the AB annotation table.

    Parameters
    ----------
    ab_table
        Either a loaded DataFrame **or** a path to a parquet/csv
        whose columns include ``drug_a_id``, ``drug_b_id``,
        ``pk_pd_label`` (``"PK"`` / ``"PD"``), and ``has_key_entity``
        (bool).  Both ``annotations/ab_sample.parquet`` and the full
        ``ab.parquet`` use this schema.
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
            # Mixed / unknown — leave to BUCKET_OTHER at lookup time.
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
