"""KPS-F (Natural KPS) drug-replacement candidate generator.

Port of upstream ``build_kps_data.py`` with two **deliberate
normative overrides** documented below.

For each base pair ``(da, db)`` in ``source_split`` (positive AND
negative) with binary label ``label_uv``, the upstream code emits
**both orientations** of the triple::

    (u=da, v=db, u', label_uv)   # find u' that pairs with v=db, label flipped
    (u=db, v=da, u', label_uv)   # find u' that pairs with v=da, label flipped

This is the "expectation over both directions" behaviour the paper
relies on — the KPS-F bucket mean averages drug-replacement
sensitivity across **both** possible head-drug positions, not just
the canonical one.  Without it the indicator silently loses half the
triples (~2× under-count vs upstream).

The candidate ``u'`` is drawn from the union of training-validation
splits passed via ``search_pool_splits`` (upstream default:
``("test_s1", "test_s2")``).  See :func:`build_swap_candidates` for
the per-direction pool-selection rule.

Deliberate normative overrides (NOT byte-exact with upstream)
-------------------------------------------------------------
1. **Cold-start setting preservation across source splits**.  When
   ``drug_pool`` is auto-selected (``None``), u' comes from the same
   partition as u (G1 or G2).  For ``test_s2`` this is identical to
   upstream's hard-coded ``u' ∈ g2_drugs``; for ``test_s1`` /
   ``test_s0`` it prevents the swap pair from silently bumping into
   a different cold-start setting.

2. **No self-pair swaps**.  We additionally require ``u' != v`` so
   the resulting swap pair ``(u', v)`` is never a degenerate
   ``(v, v)``.  A drug "interacting with itself" is not a meaningful
   DDI; this filter is biological correctness over upstream-byte-exact
   parity.  In upstream's canonical DrugBank tables self-loops do not
   appear, so the override is silent there.

Specified-KPS (a stricter variant that picks the single u' with
minimum KG-entity overlap with u) is NOT generated here; downstream
callers can filter by their own KG-overlap rule.
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
    """Combine ``splits[split].positive_pairs`` + ``get_negatives(split)``
    into a single ``{(drug_a, drug_b): label}`` dict.

    Mirrors upstream's ``s1_pairs`` / ``s2_pairs`` tables which carry
    BOTH label classes in one map (positives = 1, negatives = 0).
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
    """Build Natural-KPS swap triples — byte-exact port of upstream
    (with a cold-start-setting safety check beyond upstream's S2-only
    main loop).

    Algorithm (mirrors ``build_kps_data.py:147-228``):

    1. Build ``v_to_pool[v] = [(u', label), ...]`` by scanning every
       pair in ``search_pool_splits``, including BOTH orientations
       of each (a, b) so a query partner appearing in either slot
       gets indexed.
    2. For each base pair ``(da, db)`` in ``source_split``, iterate
       BOTH orientations ``(u, v) ∈ [(da, db), (db, da)]``.  This is
       the "expectation over both head-drug directions" step.
    3. For each ``(u, v)``, gather every ``u'`` with ``u' ≠ u``,
       ``u' ∈ drug_pool`` and ``label_upv != label_uv``.  Each unique
       ``u'`` contributes one triple.

    Cold-start setting preservation
    -------------------------------
    When ``drug_pool`` is ``None`` (default) the candidate pool is
    chosen **per direction** so the resulting swap pair preserves
    the source split's cold-start setting:

    * For ``test_s2`` base pairs both u and v are in G2, so u' ∈ G2
      in both directions — identical to upstream's S2-only behaviour.
    * For ``test_s1`` base pairs one drug is in G1 and one in G2;
      u' is restricted to **whichever partition u belongs to** in
      each direction.  Without this, the canonical-direction triple
      ``(u=G1, v=G2, u'∈G2)`` would silently bump the swap pair into
      a G2×G2 (S2) setting, breaking the indicator's semantics.
    * For ``test_s0`` base pairs both drugs are in G1, so u' ∈ G1.

    Pass an explicit ``drug_pool`` to override the auto-restriction
    (e.g. ablations that intentionally cross partitions).

    Parameters
    ----------
    ds
        Loaded :class:`PairDataset`.
    source_split
        Which split's pairs serve as the base ``(u, v)``.  Default
        ``"test_s2"`` (paper Table 6 target).
    search_pool_splits
        Splits to search for candidate ``u'``.  Default = upstream
        convention: combine ``test_s1`` + ``test_s2``.  Falls back to
        ``(source_split,)`` if the upstream splits are missing.
    drug_pool
        Optional explicit candidate set for ``u'``.  When ``None``
        (default) auto-selects per direction per the cold-start
        preservation rule above.

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

    # Default search pool = upstream convention ("test_s1" + "test_s2"
    # combined). Fall back to source_split if either piece is missing.
    if search_pool_splits is None:
        search_pool_splits = tuple(
            s for s in ("test_s1", "test_s2") if s in splits
        ) or (source_split,)
    search_pool_splits = tuple(search_pool_splits)

    # G1 / G2 partitions for the auto-pool path.
    g1_set = set(map(str, getattr(ds.splits, "g1_drugs", []) or []))
    g2_set = set(map(str, getattr(ds.splits, "g2_drugs", []) or []))

    explicit_pool: set[str] | None = None
    if drug_pool is not None:
        explicit_pool = {str(d) for d in drug_pool}
        # When the caller gives an explicit pool, we use it uniformly
        # (no per-direction G1/G2 auto-selection).
        auto_pool_for_partition: Callable[[str], set[str]] | None = None
    else:
        # Auto-mode: pick the candidate pool per (u, v) based on u's
        # partition. This preserves the cold-start setting in all
        # source splits (S0/S1/S2) while staying byte-exact with
        # upstream for the S2 case.
        if not (g1_set or g2_set):
            # Permissive fallback for tiny test fixtures: use every
            # drug seen in source_split as the candidate pool.
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
                """u' must come from the same partition as the
                drug it's replacing (G1 if u∈G1, G2 if u∈G2). When
                the drug is in neither set (shouldn't happen on a
                well-formed PairDataset), allow the union."""
                if drug in g1_set:
                    return g1_set
                if drug in g2_set:
                    return g2_set
                return g1_set | g2_set

    # Build the full search-pool index v -> [(u', label), ...].
    # Candidate-membership filtering happens INSIDE the main loop so
    # the per-direction pool can vary (auto-pool path).
    v_to_pool: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split_name in search_pool_splits:
        pair_labels = _load_pair_label_table(ds, split_name)
        for (a, b), lbl in pair_labels.items():
            v_to_pool[b].append((a, lbl))
            v_to_pool[a].append((b, lbl))

    # Base pairs from source_split (positives + negatives, with labels).
    base_pairs = _load_pair_label_table(ds, source_split)

    triples: list[SwapTriple] = []
    seen: set[tuple[str, str, str, int]] = set()
    processed: set[tuple[str, str]] = set()

    for (da, db), label_uv in base_pairs.items():
        # ── Upstream's "both directions" expectation step ─────────
        for u, v in ((da, db), (db, da)):
            if (u, v) in processed:
                continue
            processed.add((u, v))

            # Pool for THIS direction. explicit_pool overrides
            # auto-selection; otherwise the auto-pool function picks
            # the right partition (G1 vs G2) based on u's membership.
            if explicit_pool is not None:
                pool_here = explicit_pool
            else:
                pool_here = auto_pool_for_partition(u)

            # Find every u' with flipped label, dedup by u'.
            #
            # Filter set:
            #   * u' != u                 (no self-replacement of head)
            #   * u' != v                 (no degenerate (v, v) swap pair —
            #                              a drug interacting with itself
            #                              is not a meaningful DDI;
            #                              biological/normative override
            #                              of upstream which omits this
            #                              check because the canonical
            #                              DrugBank tables happen not to
            #                              contain self-loops in practice)
            #   * u' in pool_here         (cold-start integrity)
            #   * label_upv != label_uv   (label flip)
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
