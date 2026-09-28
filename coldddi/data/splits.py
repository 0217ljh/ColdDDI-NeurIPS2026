"""Drug-wise cold-start S0 / S1 / S2 splits (Appendix B.1).

Two protocols
-------------
* **"fair"** (paper-faithful, default).  Byte-equivalent port of
  upstream ``cold_start_split_fair_step``
  (``Preprocessor/splits.py:207-306`` in the upstream repo),
  the protocol that produced Table B.1 of paper Appendix B.1 and
  the legacy 800-drug pkl bundles shipped under
  ``data/private/outputs_full/splits_legacy/800drug/*.pkl``.

  - ``g1_ratio = 0.8`` → ``|G1| = int(N * 0.8)`` (≈80%/20%)
  - **S0** pool: ``train_size = 0.9`` then val/test = 50/50 of the
    remaining 10% → 90% train / 5% val / 5% test.
  - **S1** pool: val/test = 50%/50%.
  - **S2** pool: val/test = 50%/50%.
  - Random split uses :func:`sklearn.model_selection.train_test_split`
    with ``random_state=seed`` so the partition is reproducible.

* **"legacy"**.  Old upstream protocol
  (``cold_start_split_{0,1,2}_step``), kept for back-compat with the
  earlier release split artefacts under
  ``data/private/intermediate/splits/seed{N}/`` (which were generated
  under this protocol before the audit caught the divergence).

  - ``drug_ratio = 1.5`` → ``|G1| = int(N // 1.5)`` (≈67%/33%)
  - **S0**: 80% train, then val = ``val_ratio`` × 20% rest = 2%,
    test = 18%.
  - **S1**, **S2**: val/test = ``val_ratio`` × pool / rest =
    10%/90% by default.
  - S0 train uses a coverage anchor that guarantees every drug in
    ``G1`` appears in train at least once.

Default is ``"fair"``.  When a caller passes the legacy ``drug_ratio``
kwarg explicitly (without ``g1_ratio``) the dispatcher auto-routes
to ``"legacy"`` for source-compatibility with pre-audit code.

Settings on top of the partition (both protocols)
-------------------------------------------------
* **S0** (transductive): both endpoints in ``G_1``; train + val + test
  all drawn from the ``G_1 × G_1`` pool.
* **S1** (semi-inductive): train edges are ``G_1 × G_1``; val/test
  edges cross between ``G_1`` and ``G_2`` (exactly one endpoint
  unseen).
* **S2** (fully inductive): train edges are ``G_1 × G_1``; val/test
  edges are ``G_2 × G_2`` (both endpoints unseen at training time).

A single :class:`SplitFolds` object stores the train + 6 val/test
DataFrames + the two drug groups.  Persistence: per
:class:`SplitFolds.save`, each split becomes its own parquet under
``<out_dir>/seed{N}/<name>.parquet`` plus a ``manifest.json`` that
records the protocol used so :meth:`SplitFolds.from_dir` can
reload faithfully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Paper-faithful drug-partition ratio (``|G1| = int(N * g1_ratio)``).
DEFAULT_G1_RATIO: float = 0.8

#: Legacy drug-partition ratio (``|G1| = N // drug_ratio``).
#: Kept for back-compat with the pre-audit split artefacts; new
#: callers should rely on the ``"fair"`` protocol default.
DEFAULT_DRUG_RATIO: float = 1.5

#: Legacy val_ratio.  Used only by the ``"legacy"`` protocol.
DEFAULT_VAL_RATIO: float = 0.1

#: ``"fair"`` protocol: S0 train size as a fraction of the G1×G1 pool.
#: Paper-spec (Appendix B.1): 90% train, then val/test 50/50 of the
#: remaining 10%.
FAIR_S0_TRAIN_SIZE: float = 0.9

#: ``"fair"`` protocol: val fraction of the S1 / S2 pool.  Paper-spec:
#: 50% val + 50% test.
FAIR_VAL_FRACTION: float = 0.5

#: Recognised values for the ``protocol`` parameter.
PROTOCOLS: tuple[str, ...] = ("fair", "legacy")

#: Default protocol — paper Table B.1 compatible.
DEFAULT_PROTOCOL: str = "fair"

#: All seven splits, ordered (one train + six val/test buckets).
#: A single train serves all three settings (S0/S1/S2):
#: it is the ``G1 × G1`` pool minus the S0 val/test holdouts, so it is
#: guaranteed disjoint from every val/test bucket.
SPLIT_NAMES: tuple[str, ...] = (
    "train",
    "val_s0",
    "val_s1",
    "val_s2",
    "test_s0",
    "test_s1",
    "test_s2",
)

#: Mapping from a setting label ("s0" / "s1" / "s2") to its (val, test) names.
SETTINGS: dict[str, tuple[str, str]] = {
    "s0": ("val_s0", "test_s0"),
    "s1": ("val_s1", "test_s1"),
    "s2": ("val_s2", "test_s2"),
}


@dataclass
class SplitFolds:
    """Concrete :class:`coldddi.data.protocols.SplitFoldsProtocol` implementation.

    A single canonical ``train`` set serves all three cold-start
    settings (S0 / S1 / S2). It is constructed as
    ``G1 × G1`` **minus** the S0 val/test holdout (and capped at
    ``s0_train_budget`` when that is provided) so that:

    * ``train`` ∩ ``val_s0`` = ``train`` ∩ ``test_s0`` = ∅
      (S0 holdout is explicitly removed from train).
    * ``train`` ⊆ ``G1 × G1`` is automatically disjoint from
      ``val_s1`` / ``test_s1`` (cross edges) and from
      ``val_s2`` / ``test_s2`` (``G2 × G2`` edges).

    This matches the legacy ``cold_start_split_fair_step`` behaviour
    that produced the 800-drug / 1,900-drug bundles shipped in the
    paper, where ``split.train_idx`` is a single index list.
    """

    train: pd.DataFrame
    val_s0: pd.DataFrame
    val_s1: pd.DataFrame
    val_s2: pd.DataFrame
    test_s0: pd.DataFrame
    test_s1: pd.DataFrame
    test_s2: pd.DataFrame
    g1_drugs: list[str]
    g2_drugs: list[str]
    seed: int
    #: Which split protocol produced this bundle.  ``"fair"`` is the
    #: paper-faithful default (Appendix B.1).  ``"legacy"`` is kept
    #: for back-compat with pre-audit artefacts.
    protocol: str = DEFAULT_PROTOCOL
    #: ``"fair"``-only knob: ``|G1| = int(N * g1_ratio)``.
    g1_ratio: float = DEFAULT_G1_RATIO
    #: ``"legacy"``-only knob (kept on the dataclass for round-tripping
    #: legacy manifest.json files; ignored when ``protocol="fair"``).
    drug_ratio: float = DEFAULT_DRUG_RATIO
    #: ``"legacy"``-only knob: val fraction of the per-split pool.
    val_ratio: float = DEFAULT_VAL_RATIO

    # ------------------------------------------------------------------
    # Iteration helpers (used by serialization + tests)
    # ------------------------------------------------------------------

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [(name, getattr(self, name)) for name in SPLIT_NAMES]

    def total_pairs(self) -> int:
        return sum(len(df) for _, df in self.items())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Write each split as its own parquet + a manifest.json.

        Manifest records ``protocol`` so reload via :meth:`from_dir`
        knows which protocol produced the bundle (paper-faithful
        ``"fair"`` vs legacy ``"legacy"``).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        manifest = {
            "seed": self.seed,
            "protocol": self.protocol,
            "g1_ratio": self.g1_ratio,
            "drug_ratio": self.drug_ratio,
            "val_ratio": self.val_ratio,
            "n_pairs": {name: int(len(df)) for name, df in self.items()},
            "g1_drugs": list(self.g1_drugs),
            "g2_drugs": list(self.g2_drugs),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def from_dir(cls, in_dir: Path) -> "SplitFolds":
        """Reload a saved bundle.

        Back-compat: manifests written before the audit had no
        ``protocol`` field — they are interpreted as ``"legacy"``
        (the only protocol that existed at write time).
        """
        manifest = json.loads((in_dir / "manifest.json").read_text())
        kwargs = {
            name: pd.read_parquet(in_dir / f"{name}.parquet") for name in SPLIT_NAMES
        }
        # Pre-audit manifests lack the ``protocol`` field.  We default
        # them to ``"legacy"`` because that's how they were generated.
        protocol = manifest.get("protocol", "legacy")
        return cls(
            **kwargs,
            g1_drugs=list(manifest["g1_drugs"]),
            g2_drugs=list(manifest["g2_drugs"]),
            seed=int(manifest["seed"]),
            protocol=str(protocol),
            g1_ratio=float(manifest.get("g1_ratio", DEFAULT_G1_RATIO)),
            drug_ratio=float(manifest.get("drug_ratio", DEFAULT_DRUG_RATIO)),
            val_ratio=float(manifest.get("val_ratio", DEFAULT_VAL_RATIO)),
        )

    def __repr__(self) -> str:
        sizes = {n: len(df) for n, df in self.items()}
        return (
            f"SplitFolds(seed={self.seed}, |G1|={len(self.g1_drugs)}, |G2|={len(self.g2_drugs)}, "
            f"sizes={sizes})"
        )


# ----------------------------------------------------------------------
# Group construction
# ----------------------------------------------------------------------


def _partition_drugs(
    all_drugs: list[str],
    *,
    seed: int,
    drug_ratio: float,
) -> tuple[list[str], list[str]]:
    """Shuffle and split the drug universe into ``(G1, G2)`` — LEGACY.

    ``len(G1) = len(all_drugs) // drug_ratio`` (matches the legacy
    Preprocessor/splits.py behaviour with ``drug_ratio=1.5``, i.e.
    ~67% of drugs in G1).  Used only by the ``"legacy"`` protocol;
    new callers should rely on :func:`_partition_drugs_fair`.
    """
    drugs = sorted(set(all_drugs))
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    mid = max(1, int(len(drugs) // drug_ratio))
    return drugs[:mid], drugs[mid:]


def _partition_drugs_fair(
    all_drugs: list[str],
    *,
    seed: int,
    g1_ratio: float,
) -> tuple[list[str], list[str]]:
    """Paper-faithful partition — byte-equivalent to upstream
    ``cold_start_split_fair_step`` (Preprocessor/splits.py:227-236).

    ``split_idx = int(len(drugs) * g1_ratio)`` (multiplicative form)
    with ``g1_ratio=0.8`` produces ``|G1|=1520, |G2|=380`` on the
    1,900-drug benchmark — matches paper Table B.1 exactly.

    The same sort + ``np.random.default_rng(seed).shuffle`` ordering
    as upstream is preserved, so the per-seed G1/G2 partition is
    byte-equal to the 800-drug legacy pkl bundles shipped under
    ``data/private/outputs_full/splits_legacy/800drug/``.
    """
    drugs = sorted(set(all_drugs))
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    split_idx = max(1, int(len(drugs) * g1_ratio))
    # Defensive clamp: ensure G2 is non-empty (S1/S2 buckets would be
    # empty otherwise, breaking every downstream cold-start test).
    split_idx = min(split_idx, len(drugs) - 1) if len(drugs) > 1 else split_idx
    return drugs[:split_idx], drugs[split_idx:]


def _coverage_anchor(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
) -> list[int]:
    """S0 anchor: a deterministic index list that touches every drug in ``g1``.

    Mirrors the legacy ``cold_start_split_0_step`` "base_train" loop:
    iterate edges in a shuffled order, keep one whenever it covers a
    not-yet-seen drug.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(edges))
    rng.shuffle(order)

    a_arr = edges["drug_a_id"].astype(str).to_numpy()
    b_arr = edges["drug_b_id"].astype(str).to_numpy()

    unseen = set(g1)
    anchor: list[int] = []
    for i in order:
        if not unseen:
            break
        a, b = a_arr[i], b_arr[i]
        if a in unseen or b in unseen:
            anchor.append(int(i))
            unseen.discard(a)
            unseen.discard(b)
    return anchor


# ----------------------------------------------------------------------
# Per-setting builders (positive-only DataFrames keyed by drug_a_id/drug_b_id)
# ----------------------------------------------------------------------


def _build_s0(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
    val_ratio: float,
    train_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S0 transductive: train + val + test all in G1×G1.

    ``train_budget`` caps the train size to align with S1's train edge
    count (legacy default: 25,690). When ``None``, a 80/10/10 fallback
    is used.
    """
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    mask = a.isin(g1) & b.isin(g1)
    g1g1_idx = np.array(edges.index[mask].tolist())
    if len(g1g1_idx) == 0:
        empty = edges.iloc[0:0].reset_index(drop=True)
        return empty, empty, empty

    anchor = _coverage_anchor(edges.loc[mask].reset_index(drop=True), g1, seed=seed)
    # `anchor` indexes into the *masked* table; map back to global indices.
    g1g1_subset = edges.loc[mask].reset_index(drop=True)
    anchor_global = g1g1_subset.iloc[anchor].index.tolist()

    rng = np.random.default_rng(seed + 17)
    remaining = [int(i) for i in g1g1_subset.index.tolist() if i not in set(anchor_global)]
    rng.shuffle(remaining)

    target_train = train_budget if train_budget is not None else int(0.8 * len(g1g1_subset))
    n_extra = max(0, target_train - len(anchor_global))
    extra = remaining[:n_extra]
    rest = remaining[n_extra:]
    n_val = int(len(rest) * val_ratio)

    train_idx = anchor_global + extra
    val_idx = rest[:n_val]
    test_idx = rest[n_val:]

    return (
        g1g1_subset.iloc[train_idx].reset_index(drop=True),
        g1g1_subset.iloc[val_idx].reset_index(drop=True),
        g1g1_subset.iloc[test_idx].reset_index(drop=True),
    )


def _build_s1(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S1 semi-inductive: train = G1×G1, val/test = cross (G1×G2 ∪ G2×G1)."""
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    train_mask = a.isin(g1) & b.isin(g1)
    cross_mask = (a.isin(g1) & b.isin(g2)) | (a.isin(g2) & b.isin(g1))

    train_df = edges.loc[train_mask].reset_index(drop=True)
    cross_df = edges.loc[cross_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed + 29)
    cross_order = np.arange(len(cross_df))
    rng.shuffle(cross_order)
    n_val = int(len(cross_order) * val_ratio)
    val_idx = cross_order[:n_val]
    test_idx = cross_order[n_val:]
    return (
        train_df,
        cross_df.iloc[val_idx].reset_index(drop=True),
        cross_df.iloc[test_idx].reset_index(drop=True),
    )


def _build_s2(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S2 fully inductive: train = G1×G1, val/test = G2×G2."""
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    train_mask = a.isin(g1) & b.isin(g1)
    g2g2_mask = a.isin(g2) & b.isin(g2)

    train_df = edges.loc[train_mask].reset_index(drop=True)
    g2g2_df = edges.loc[g2g2_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed + 31)
    order = np.arange(len(g2g2_df))
    rng.shuffle(order)
    n_val = int(len(order) * val_ratio)
    val_idx = order[:n_val]
    test_idx = order[n_val:]
    return (
        train_df,
        g2g2_df.iloc[val_idx].reset_index(drop=True),
        g2g2_df.iloc[test_idx].reset_index(drop=True),
    )


# ----------------------------------------------------------------------
# Per-setting builders for the "fair" (paper-faithful) protocol
# ----------------------------------------------------------------------
#
# Byte-equivalent port of upstream ``cold_start_split_fair_step``
# (Preprocessor/splits.py:207-306).  All three builders use
# ``sklearn.model_selection.train_test_split`` with ``random_state=seed``
# so the partition is reproducible and matches the legacy 800-drug
# pkl bundles.


def _build_s0_fair(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
    train_size: float = FAIR_S0_TRAIN_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S0 fair: ``train_size=0.9`` then val/test = 50/50 of the
    remaining 10% → final 90/5/5 (paper App B.1 line 25).

    Returns ``(train, val_s0, test_s0)`` DataFrames keyed by
    ``drug_a_id`` / ``drug_b_id``.
    """
    from sklearn.model_selection import train_test_split

    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    mask = a.isin(g1) & b.isin(g1)
    s0_indices = edges.index[mask].to_numpy()
    if len(s0_indices) == 0:
        empty = edges.iloc[0:0].reset_index(drop=True)
        return empty, empty, empty
    if len(s0_indices) < 2:
        # train_test_split requires ≥ 2 samples; tiny fixtures may
        # have only one S0 edge.  Put that single edge in train and
        # return empty val/test rather than crash.
        empty = edges.iloc[0:0].reset_index(drop=True)
        return edges.loc[s0_indices].reset_index(drop=True), empty, empty

    train_idx, holdout_idx = train_test_split(
        s0_indices, train_size=train_size, random_state=seed,
    )
    if len(holdout_idx) < 2:
        empty = edges.iloc[0:0].reset_index(drop=True)
        return (
            edges.loc[train_idx].reset_index(drop=True),
            empty,
            edges.loc[holdout_idx].reset_index(drop=True),
        )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=FAIR_VAL_FRACTION, random_state=seed,
    )
    return (
        edges.loc[train_idx].reset_index(drop=True),
        edges.loc[val_idx].reset_index(drop=True),
        edges.loc[test_idx].reset_index(drop=True),
    )


def _build_s1_fair(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S1 fair: val/test = 50/50 of the (G1×G2 ∪ G2×G1) pool.

    Train edges are produced by :func:`_build_s0_fair` (which carves
    out the S0 train pool from G1×G1); this returns an empty train
    placeholder for symmetry with the legacy builder signature.
    """
    from sklearn.model_selection import train_test_split

    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    cross_mask = (a.isin(g1) & b.isin(g2)) | (a.isin(g2) & b.isin(g1))
    cross_indices = edges.index[cross_mask].to_numpy()
    empty = edges.iloc[0:0].reset_index(drop=True)
    if len(cross_indices) == 0:
        return empty, empty, empty
    if len(cross_indices) < 2:
        return empty, empty, edges.loc[cross_indices].reset_index(drop=True)
    val_idx, test_idx = train_test_split(
        cross_indices, test_size=FAIR_VAL_FRACTION, random_state=seed,
    )
    return (
        empty,
        edges.loc[val_idx].reset_index(drop=True),
        edges.loc[test_idx].reset_index(drop=True),
    )


def _build_s2_fair(
    edges: pd.DataFrame,
    g1: set[str],
    g2: set[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S2 fair: val/test = 50/50 of the G2×G2 pool."""
    from sklearn.model_selection import train_test_split

    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    g2g2_mask = a.isin(g2) & b.isin(g2)
    g2g2_indices = edges.index[g2g2_mask].to_numpy()
    empty = edges.iloc[0:0].reset_index(drop=True)
    if len(g2g2_indices) == 0:
        return empty, empty, empty
    if len(g2g2_indices) < 2:
        return empty, empty, edges.loc[g2g2_indices].reset_index(drop=True)
    val_idx, test_idx = train_test_split(
        g2g2_indices, test_size=FAIR_VAL_FRACTION, random_state=seed,
    )
    return (
        empty,
        edges.loc[val_idx].reset_index(drop=True),
        edges.loc[test_idx].reset_index(drop=True),
    )


# ----------------------------------------------------------------------
# Top-level builder
# ----------------------------------------------------------------------


_SENTINEL: object = object()  # marks "caller did not pass this kwarg"


def build_splits(
    ddi_edges: pd.DataFrame,
    *,
    seed: int,
    protocol: str | object = _SENTINEL,
    g1_ratio: float | object = _SENTINEL,
    drug_ratio: float | object = _SENTINEL,
    val_ratio: float | object = _SENTINEL,
    s0_train_budget: int | None = None,
) -> SplitFolds:
    """Run S0 + S1 + S2 cold-start splitting in one pass.

    Dispatches on ``protocol``:

    * ``"fair"`` (paper-faithful, default; Appendix B.1 / Table B.1):
      ``g1_ratio=0.8``, S0=90/5/5, S1/S2=50/50.  Byte-equivalent to
      upstream ``cold_start_split_fair_step``.

    * ``"legacy"``: ``drug_ratio=1.5``, S0=80/2/18, S1/S2=10/90.
      Kept for back-compat with the pre-audit release split
      artefacts under ``data/private/intermediate/splits/seed{N}/``.

    Back-compat: a caller passing ``drug_ratio=X`` explicitly without
    naming ``protocol`` is auto-routed to ``"legacy"`` (the only
    protocol that accepts ``drug_ratio``).  This keeps pre-audit
    test code working unchanged.

    Conflict detection: mixing fair-only and legacy-only knobs (e.g.
    ``protocol="fair"`` with ``drug_ratio=...``, or
    ``protocol="legacy"`` with ``g1_ratio=...``) raises
    :class:`ValueError` so a typo doesn't silently take the wrong
    protocol's defaults.

    Parameters
    ----------
    ddi_edges
        Positive-only DataFrame with columns ``drug_a_id, drug_b_id``
        plus any extra columns to carry through.
    seed
        Random seed; the same seed always yields the same splits.
    protocol
        ``"fair"`` (default) or ``"legacy"``.  See module docstring
        for the per-protocol allocation rules.
    g1_ratio
        ``"fair"``-only: ``|G1| = int(N * g1_ratio)``; default
        :data:`DEFAULT_G1_RATIO` (0.8 → 80/20 partition).
    drug_ratio
        ``"legacy"``-only: ``|G1| = N // drug_ratio``.  Passing this
        kwarg without ``protocol`` auto-routes to ``"legacy"``.
    val_ratio
        ``"legacy"``-only: fraction of cold-start edges used for val;
        default 0.1 (gives 10%/90% val/test on S1/S2).
    s0_train_budget
        ``"legacy"``-only: optional cap on the S0 train size.
    """
    if "drug_a_id" not in ddi_edges.columns or "drug_b_id" not in ddi_edges.columns:
        raise ValueError("ddi_edges must contain `drug_a_id` and `drug_b_id` columns")

    # ── Detect which knobs the caller actually passed ──
    explicit_protocol = protocol is not _SENTINEL
    explicit_g1_ratio = g1_ratio is not _SENTINEL
    explicit_drug_ratio = drug_ratio is not _SENTINEL
    explicit_val_ratio = val_ratio is not _SENTINEL
    explicit_s0_train_budget = s0_train_budget is not None

    # ── Resolve protocol with back-compat routing ──
    if explicit_protocol:
        # User-named protocol takes precedence; back-compat does NOT
        # silently override it.
        resolved_protocol = protocol
    elif explicit_drug_ratio:
        # Back-compat: caller passed `drug_ratio` without specifying
        # `protocol` → route to legacy (the only protocol that uses
        # drug_ratio).
        resolved_protocol = "legacy"
    else:
        resolved_protocol = DEFAULT_PROTOCOL

    if resolved_protocol not in PROTOCOLS:
        raise ValueError(
            f"protocol must be one of {PROTOCOLS}; got {resolved_protocol!r}"
        )

    # ── Reject incompatible knob combinations ──
    if resolved_protocol == "fair":
        bad = [
            (explicit_drug_ratio, "drug_ratio"),
            (explicit_val_ratio, "val_ratio"),
            (explicit_s0_train_budget, "s0_train_budget"),
        ]
        offenders = [name for flag, name in bad if flag]
        if offenders:
            raise ValueError(
                f"protocol='fair' does not accept legacy kwargs: "
                f"{offenders!r}.  Use protocol='legacy' or drop these "
                "kwargs.  See coldddi.data.splits module docstring."
            )
    else:  # legacy
        if explicit_g1_ratio:
            raise ValueError(
                "protocol='legacy' does not accept g1_ratio (a fair-"
                "protocol knob).  Use protocol='fair' or drop "
                "g1_ratio."
            )

    # ── Resolve effective kwargs from sentinels to defaults ──
    g1_ratio_val: float = float(g1_ratio) if explicit_g1_ratio else DEFAULT_G1_RATIO  # type: ignore[arg-type]
    drug_ratio_val: float = (
        float(drug_ratio) if explicit_drug_ratio else DEFAULT_DRUG_RATIO  # type: ignore[arg-type]
    )
    val_ratio_val: float = (
        float(val_ratio) if explicit_val_ratio else DEFAULT_VAL_RATIO  # type: ignore[arg-type]
    )
    protocol = resolved_protocol

    edges = ddi_edges.reset_index(drop=True).copy()
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    all_drugs = pd.unique(pd.concat([a, b]))

    if protocol == "fair":
        g1_list, g2_list = _partition_drugs_fair(
            list(all_drugs), seed=seed, g1_ratio=g1_ratio_val,
        )
        g1, g2 = set(g1_list), set(g2_list)
        train, val_s0, test_s0 = _build_s0_fair(edges, g1, seed=seed)
        _train_s1, val_s1, test_s1 = _build_s1_fair(edges, g1, g2, seed=seed)
        _train_s2, val_s2, test_s2 = _build_s2_fair(edges, g1, g2, seed=seed)
    else:  # legacy
        g1_list, g2_list = _partition_drugs(
            list(all_drugs), seed=seed, drug_ratio=drug_ratio_val,
        )
        g1, g2 = set(g1_list), set(g2_list)
        # _build_s0 returns a holdout-safe G1×G1 train. We use this as
        # the canonical train and reuse it across S1 / S2 — it's already
        # disjoint from every val/test bucket the three settings produce.
        train, val_s0, test_s0 = _build_s0(
            edges, g1, seed=seed, val_ratio=val_ratio_val, train_budget=s0_train_budget,
        )
        _train_s1, val_s1, test_s1 = _build_s1(
            edges, g1, g2, seed=seed, val_ratio=val_ratio_val,
        )
        _train_s2, val_s2, test_s2 = _build_s2(
            edges, g1, g2, seed=seed, val_ratio=val_ratio_val,
        )

    return SplitFolds(
        train=train,
        val_s0=val_s0,
        val_s1=val_s1,
        val_s2=val_s2,
        test_s0=test_s0,
        test_s1=test_s1,
        test_s2=test_s2,
        g1_drugs=sorted(g1_list),
        g2_drugs=sorted(g2_list),
        seed=seed,
        protocol=protocol,
        g1_ratio=g1_ratio_val,
        drug_ratio=drug_ratio_val,
        val_ratio=val_ratio_val,
    )


__all__ = [
    "SplitFolds",
    "build_splits",
    "SPLIT_NAMES",
    "PROTOCOLS",
    "DEFAULT_PROTOCOL",
    "DEFAULT_G1_RATIO",
    "DEFAULT_DRUG_RATIO",
    "DEFAULT_VAL_RATIO",
    "FAIR_S0_TRAIN_SIZE",
    "FAIR_VAL_FRACTION",
]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Build cold-start drug-wise S0/S1/S2 splits and (optionally) "
            "the matching static negatives. Writes one parquet per split "
            "plus manifest.json under --out-dir."
        ),
    )
    parser.add_argument(
        "--ddi-edges",
        required=True,
        type=Path,
        help="Input ddi_edges.csv (must contain `drug_a_id`, `drug_b_id`).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory (typically `splits/seed{N}/`).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=DEFAULT_PROTOCOL,
        help=(
            "Split protocol.  'fair' (default, paper App B.1) → "
            "g1_ratio=0.8, S0=90/5/5, S1/S2=50/50.  'legacy' → "
            "old upstream `cold_start_split_{0,1,2}_step` behaviour "
            "(drug_ratio=1.5, S0=80/2/18, S1/S2=10/90); kept for "
            "back-compat with pre-audit split artefacts."
        ),
    )
    parser.add_argument(
        "--g1-ratio",
        type=float,
        default=DEFAULT_G1_RATIO,
        help=(
            f"'fair'-protocol only: |G1| = int(N * g1_ratio); "
            f"default {DEFAULT_G1_RATIO} matches paper Table B.1."
        ),
    )
    parser.add_argument(
        "--drug-ratio",
        type=float,
        default=None,
        help=(
            f"'legacy'-protocol only: |G1| = N // drug_ratio "
            f"(default {DEFAULT_DRUG_RATIO}).  Passing this flag "
            "explicitly auto-switches --protocol to 'legacy'."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=(
            f"'legacy'-protocol only: fraction of cold-start edges "
            f"used for validation (default {DEFAULT_VAL_RATIO}).  "
            "Ignored under --protocol fair."
        ),
    )
    parser.add_argument(
        "--s0-train-budget",
        type=int,
        default=None,
        help=(
            "'legacy'-protocol only: optional cap on the S0 train "
            "size.  Ignored under --protocol fair."
        ),
    )
    parser.add_argument(
        "--build-negatives",
        action="store_true",
        help="Also generate the six static-negative parquet files under <out>/negatives/.",
    )
    parser.add_argument(
        "--n-train-negative-epochs",
        type=int,
        default=0,
        help=(
            "If > 0, pre-bake N epochs of training negatives under "
            "<out>/train_negatives/epoch_{0..N-1}.parquet. Each epoch uses a "
            "distinct deterministic sub-seed."
        ),
    )
    args = parser.parse_args(argv)

    edges = pd.read_csv(args.ddi_edges)
    # Honour the back-compat rule: if --drug-ratio is supplied we
    # route to the legacy protocol (build_splits does the same routing
    # internally, but doing it here too lets us only pass the kwargs
    # the chosen protocol expects, avoiding silent ignored flags).
    build_kwargs: dict = {"seed": args.seed}
    if args.drug_ratio is not None:
        build_kwargs["protocol"] = "legacy"
        build_kwargs["drug_ratio"] = args.drug_ratio
        build_kwargs["val_ratio"] = args.val_ratio
        build_kwargs["s0_train_budget"] = args.s0_train_budget
    else:
        build_kwargs["protocol"] = args.protocol
        if args.protocol == "fair":
            build_kwargs["g1_ratio"] = args.g1_ratio
        else:
            build_kwargs["val_ratio"] = args.val_ratio
            build_kwargs["s0_train_budget"] = args.s0_train_budget
    splits = build_splits(edges, **build_kwargs)
    splits.save(args.out_dir)
    print(splits, file=sys.stderr)

    if args.build_negatives:
        # Local import to avoid a circular dependency at module load time.
        from coldddi.data.negatives import build_static_negatives

        neg_dir = args.out_dir / "negatives"
        neg_dir.mkdir(parents=True, exist_ok=True)
        statics = build_static_negatives(splits, base_seed=args.seed)
        for name, df in statics.items():
            df.to_parquet(neg_dir / f"{name}.parquet", index=False)
            print(f"  negatives/{name}: {len(df):,} rows", file=sys.stderr)

    if args.n_train_negative_epochs > 0:
        # Stream-write each epoch as soon as it's sampled so a long full-
        # scale run shows progress immediately rather than going silent
        # for several minutes.
        from coldddi.data.negatives import build_train_negatives

        train_neg_dir = args.out_dir / "train_negatives"
        train_neg_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_train_negative_epochs):
            df = build_train_negatives(splits, base_seed=args.seed, epoch=i)
            df.to_parquet(train_neg_dir / f"epoch_{i}.parquet", index=False)
            print(
                f"  train_negatives/epoch_{i}: {len(df):,} rows",
                file=sys.stderr,
                flush=True,
            )

    print(f"Wrote splits to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
