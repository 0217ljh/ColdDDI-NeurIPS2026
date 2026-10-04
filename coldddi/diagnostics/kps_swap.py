"""KPS-F (Natural KPS) drug-replacement candidate generator.

Based on upstream ``build_kps_data.py``, with partition preservation
and self-pair exclusion as described below.

For each positive or negative base pair ``(da, db)`` in ``source_split``,
emit both orientations with binary label ``label_uv``::

    (u=da, v=db, u', label_uv)   # find u' that pairs with v=db, label flipped
    (u=db, v=da, u', label_uv)   # find u' that pairs with v=da, label flipped

KPS-F averages over both head-drug positions, not just the canonical
orientation; using one direction under-counts the triples.

Candidates come from ``search_pool_splits`` (upstream default:
``("test_s1", "test_s2")``), subject to per-direction pool selection.

Differences from upstream:

1. When ``drug_pool`` is ``None``, u' comes from u's partition (G1 or
   G2), preserving the cold-start setting. This matches upstream's
   ``u' ∈ g2_drugs`` for ``test_s2`` and also handles ``test_s0/test_s1``.

2. Require ``u' != v`` to exclude degenerate ``(v, v)`` interactions.
   Upstream omits this check; its canonical DrugBank tables have no
   self-loops, so the filter leaves those candidates unchanged.

Specified-KPS, which picks the single u' with minimum KG-entity overlap
with u, is not generated here; callers may apply their own overlap rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

import pandas as pd

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset


@dataclass(frozen=True)
class SwapTriple:
    """One drug-replacement triple ``(u, v, u')``.

    Attributes
    ----------
    qa, qb
        The base test pair ``(u, v)``; its true label is :attr:`label_uv`.
    qa_prime
        Replacement head drug ``u'`` — pairs with the same ``qb`` but
        carries the opposite binary label.
    label_uv
        Binary label (0/1) of ``(qa, qb)``.  ``label_upv = 1 - label_uv``.
    """

    qa: str
    qb: str
    qa_prime: str
    label_uv: int


def _load_pair_label_table(
    ds: "PairDataset", split_name: str,
) -> dict[tuple[str, str], int]:
    """Combine split positives and negatives into ``{(drug_a, drug_b): label}``.

    Matches upstream ``s1_pairs`` / ``s2_pairs``: positives = 1, negatives = 0.
    """
    pairs: dict[tuple[str, str], int] = {}
    splits = dict(ds.splits.items())
    pos = splits.get(split_name)
    if pos is not None:
        for a, b in zip(pos["drug_a_id"].astype(str), pos["drug_b_id"].astype(str)):
            pairs[(a, b)] = 1
    try:
        neg = ds.get_negatives(split_name)
    except Exception:
        neg = None
    if neg is not None:
        for a, b in zip(neg["drug_a_id"].astype(str), neg["drug_b_id"].astype(str)):
            pairs.setdefault((a, b), 0)
    return pairs


def build_swap_candidates(
    ds: "PairDataset",
    *,
    source_split: str = "test_s2",
    search_pool_splits: Iterable[str] | None = None,
    drug_pool: set[str] | None = None,
) -> list[SwapTriple]:
    """Build Natural-KPS swap triples with cold-start and self-pair checks.

    Based on ``build_kps_data.py:147-228``:

    1. Index both orientations of each search-pool pair as
       ``v_to_pool[v] = [(u', label), ...]``.
    2. Iterate both source-pair orientations: ``(u, v) ∈ [(da, db), (db, da)]``.
    3. Emit one triple per unique ``u'`` with ``u' ≠ u``, ``u' != v``,
       ``u' ∈ drug_pool`` and ``label_upv != label_uv``.

    When ``drug_pool`` is ``None``, choose candidates per direction to
    preserve the source split's cold-start setting:

    * ``test_s2``: u and v are in G2, so u' ∈ G2, as upstream.
    * ``test_s1``: u' stays in u's partition. Otherwise a swap such as
      ``(u=G1, v=G2, u'∈G2)`` would change the setting to G2×G2 (S2).
    * ``test_s0``: u and v are in G1, so u' ∈ G1.

    An explicit ``drug_pool`` overrides this restriction in both directions.

    Parameters
    ----------
    ds
        Loaded :class:`PairDataset`.
    source_split
        Base-pair split; defaults to ``"test_s2"`` (paper Table 6).
    search_pool_splits
        Candidate splits; defaults to available ``test_s1`` and ``test_s2``
        splits, following upstream. Uses ``(source_split,)`` if neither exists.
    drug_pool
        Explicit candidate set for ``u'``, or ``None`` for the per-direction
        partition rule above.

    Returns
    -------
    List of :class:`SwapTriple`.  Each unique ``(u, v, u', label_uv)``
    appears exactly once.  Pairs with no qualifying ``u'`` are skipped.
    """
    splits = dict(ds.splits.items())
    if source_split not in splits:
        raise ValueError(
            f"source_split={source_split!r} not in dataset.splits; "
            f"available: {sorted(splits)}"
        )

    # Use available upstream search splits, or source_split if neither exists.
    if search_pool_splits is None:
        search_pool_splits = tuple(
            s for s in ("test_s1", "test_s2") if s in splits
        ) or (source_split,)
    search_pool_splits = tuple(search_pool_splits)

    g1_set = set(map(str, getattr(ds.splits, "g1_drugs", []) or []))
    g2_set = set(map(str, getattr(ds.splits, "g2_drugs", []) or []))

    explicit_pool: set[str] | None = None
    if drug_pool is not None:
        explicit_pool = {str(d) for d in drug_pool}
        # An explicit pool applies to both directions.
        auto_pool_for_partition: Callable[[str], set[str]] | None = None
    else:
        # Match u's partition to preserve the cold-start setting.
        if not (g1_set or g2_set):
            # Without partition metadata, use every drug in source_split.
            pos_df = splits[source_split]
            neg_df = ds.get_negatives(source_split)
            permissive = (
                set(pos_df["drug_a_id"].astype(str))
                | set(pos_df["drug_b_id"].astype(str))
                | set(neg_df["drug_a_id"].astype(str))
                | set(neg_df["drug_b_id"].astype(str))
            )

            def auto_pool_for_partition(drug: str) -> set[str]:
                return permissive
        else:
            def auto_pool_for_partition(drug: str) -> set[str]:
                """Use u's partition (G1 if u∈G1, G2 if u∈G2).

                If u belongs to neither partition, allow their union.
                """
                if drug in g1_set:
                    return g1_set
                if drug in g2_set:
                    return g2_set
                return g1_set | g2_set

    # Index v -> [(u', label), ...]; filter membership per direction below.
    v_to_pool: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split_name in search_pool_splits:
        pair_labels = _load_pair_label_table(ds, split_name)
        for (a, b), lbl in pair_labels.items():
            v_to_pool[b].append((a, lbl))
            v_to_pool[a].append((b, lbl))

    base_pairs = _load_pair_label_table(ds, source_split)

    triples: list[SwapTriple] = []
    seen: set[tuple[str, str, str, int]] = set()
    processed: set[tuple[str, str]] = set()

    for (da, db), label_uv in base_pairs.items():
        # Average over both head-drug directions, as upstream.
        for u, v in ((da, db), (db, da)):
            if (u, v) in processed:
                continue
            processed.add((u, v))

            if explicit_pool is not None:
                pool_here = explicit_pool
            else:
                pool_here = auto_pool_for_partition(u)

            # Deduplicate label-flipping replacements; exclude u and v
            # and enforce the candidate pool's cold-start restriction.
            seen_up: set[str] = set()
            for u_prime, label_upv in v_to_pool.get(v, []):
                if u_prime == u or u_prime == v:
                    continue
                if u_prime not in pool_here:
                    continue
                if label_upv == label_uv:
                    continue
                if u_prime in seen_up:
                    continue
                seen_up.add(u_prime)

                key = (u, v, u_prime, label_uv)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(SwapTriple(
                    qa=u, qb=v, qa_prime=u_prime, label_uv=label_uv,
                ))
    return triples


__all__ = ["SwapTriple", "build_swap_candidates"]
