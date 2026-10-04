"""Drug-wise cold-start S0 / S1 / S2 splits (Appendix B.1).

``"fair"`` (default) follows upstream ``cold_start_split_fair_step``
(``Preprocessor/splits.py:207-306``), used for Table B.1 and the bundles in
``data/private/outputs_full/splits_legacy/800drug/*.pkl``:

- ``|G1| = int(N * g1_ratio)``, default 0.8 (80/20 drug partition).
- S0: 90% train, then a 50/50 holdout split gives 5% val and 5% test.
- S1/S2: 50% val and 50% test.
- Use ``train_test_split(..., random_state=seed)`` for edge splits.

``"legacy"`` follows ``cold_start_split_{0,1,2}_step`` for older artifacts
under ``data/private/intermediate/splits/seed{N}/``:

- ``|G1| = int(N // drug_ratio)``, default 1.5 (about 67/33).
- S0: target 80% train; val takes ``val_ratio`` of the remainder, giving
  80/2/18 at default 0.1. The coverage anchor can exceed the train target.
- S1/S2: val takes ``val_ratio`` of each pool, giving 10/90 by default.
- S0 anchors cover every G1 drug represented in the G1 x G1 pool.

Passing ``drug_ratio`` without ``protocol`` selects legacy; incompatible
fair/legacy parameters raise ``ValueError``.

Both protocols share one G1 x G1 train set excluding S0 holdouts. S0 val/test
use G1 x G1, S1 uses cross-group pairs (one unseen drug), and S2 uses G2 x G2
(both unseen). All seven DataFrames contain positives only.
``SplitFolds.save(out_dir)`` writes ``<name>.parquet`` and ``manifest.json``
directly under ``out_dir`` (usually ``splits/seed{N}``), recording drug groups
and protocol for reload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Fair drug partition: ``|G1| = int(N * g1_ratio)``.
DEFAULT_G1_RATIO: float = 0.8

#: Legacy drug partition: ``|G1| = int(N // drug_ratio)``.
DEFAULT_DRUG_RATIO: float = 1.5

#: Legacy val_ratio.  Used only by the ``"legacy"`` protocol.
DEFAULT_VAL_RATIO: float = 0.1

#: Fair S0: 90% train, then split the 10% holdout equally (Appendix B.1).
FAIR_S0_TRAIN_SIZE: float = 0.9

#: Fair S1/S2: 50% validation, 50% test.
FAIR_VAL_FRACTION: float = 0.5

#: Recognised values for the ``protocol`` parameter.
PROTOCOLS: tuple[str, ...] = ("fair", "legacy")

#: Default protocol — paper Table B.1 compatible.
DEFAULT_PROTOCOL: str = "fair"

#: Ordered splits: one shared G1 x G1 train excluding S0 holdouts, plus
#: six val/test sets. The train set is disjoint from every holdout.
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
    """Seven positive-only splits with one shared training set.

    Train is a subset of G1 x G1 excluding S0 val/test. S1 cross-group and
    S2 G2 x G2 holdouts are disjoint from that pool. The shared train matches
    ``cold_start_split_fair_step`` and the paper bundles' ``split.train_idx``.
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
    #: Source protocol: fair (Appendix B.1) or legacy.
    protocol: str = DEFAULT_PROTOCOL
    #: ``"fair"``-only knob: ``|G1| = int(N * g1_ratio)``.
    g1_ratio: float = DEFAULT_G1_RATIO
    #: Legacy manifest field; ignored when protocol="fair".
    drug_ratio: float = DEFAULT_DRUG_RATIO
    #: ``"legacy"``-only knob: val fraction of the per-split pool.
    val_ratio: float = DEFAULT_VAL_RATIO

    # Iteration helpers

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [(name, getattr(self, name)) for name in SPLIT_NAMES]

    def total_pairs(self) -> int:
        return sum(len(df) for _, df in self.items())

    # Persistence

    def save(self, out_dir: Path) -> None:
        """Write seven Parquets and a manifest with the source protocol and groups."""
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

        Manifests without ``protocol`` are treated as legacy.
        """
        manifest = json.loads((in_dir / "manifest.json").read_text())
        kwargs = {
            name: pd.read_parquet(in_dir / f"{name}.parquet") for name in SPLIT_NAMES
        }
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


# Group construction


def _partition_drugs(
    all_drugs: list[str],
    *,
    seed: int,
    drug_ratio: float,
) -> tuple[list[str], list[str]]:
    """Sort unique drugs, shuffle, and partition with the legacy ratio.

    Use ``max(1, int(N // drug_ratio))`` for G1, as in
    ``Preprocessor/splits.py``; default 1.5 gives about 67% of drugs.
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
    """Partition as in ``cold_start_split_fair_step`` (Preprocessor/splits.py:227-236).

    Sort unique drugs before ``np.random.default_rng(seed).shuffle``.
    Use ``int(N * g1_ratio)``, clamped to keep both groups nonempty if N > 1.
    The default 0.8 gives 1,520/380 drugs for Table B.1's 1,900-drug graph.
    """
    drugs = sorted(set(all_drugs))
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    split_idx = max(1, int(len(drugs) * g1_ratio))
    # Keep G2 nonempty when there is more than one drug.
    split_idx = min(split_idx, len(drugs) - 1) if len(drugs) > 1 else split_idx
    return drugs[:split_idx], drugs[split_idx:]


def _coverage_anchor(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
) -> list[int]:
    """Choose seeded edge indices covering every G1 drug present in ``edges``.

    Follow ``cold_start_split_0_step``: keep a shuffled edge if it covers
    a not-yet-seen drug.
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


# Per-setting builders (positive-only DataFrames keyed by drug_a_id/drug_b_id)


def _build_s0(
    edges: pd.DataFrame,
    g1: set[str],
    *,
    seed: int,
    val_ratio: float,
    train_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S0 transductive: train + val + test all in G1×G1.

    Target ``train_budget`` edges if supplied, otherwise 80% of the pool.
    The coverage anchor can exceed that target. Validation takes
    ``val_ratio`` of the remainder: default 0.1 gives nominal 80/2/18.
    """
    a = edges["drug_a_id"].astype(str)
    b = edges["drug_b_id"].astype(str)
    mask = a.isin(g1) & b.isin(g1)
    g1g1_idx = np.array(edges.index[mask].tolist())
    if len(g1g1_idx) == 0:
        empty = edges.iloc[0:0].reset_index(drop=True)
        return empty, empty, empty

    anchor = _coverage_anchor(edges.loc[mask].reset_index(drop=True), g1, seed=seed)
    # Anchor indices refer to the reset G1 x G1 subset, not the full edge table.
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


# Fair builders follow cold_start_split_fair_step (Preprocessor/splits.py:207-306).
# Preserve train_test_split random_state=seed for each partition.


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
        # A single edge goes to train; train_test_split needs at least two.
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

    Return an empty train placeholder; :func:`_build_s0_fair` supplies train.
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


# Top-level builder


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
    """Build one train set and six S0/S1/S2 holdouts.

    Fair defaults are G1=80%, S0=90/5/5, and S1/S2=50/50 (Table B.1).
    Legacy defaults are G1~=67%, S0=80/2/18, and S1/S2=10/90.
    Passing ``drug_ratio`` without ``protocol`` selects legacy. Mixed
    fair-only and legacy-only parameters raise :class:`ValueError`.

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
        ``"legacy"``-only: ``|G1| = int(N // drug_ratio)``. Passing this
        kwarg without ``protocol`` auto-routes to ``"legacy"``.
    val_ratio
        ``"legacy"``-only: fraction of cold-start edges used for val;
        default 0.1 (gives 10%/90% val/test on S1/S2).
    s0_train_budget
        ``"legacy"``-only: S0 train target, which the coverage anchor may exceed.
    """
    if "drug_a_id" not in ddi_edges.columns or "drug_b_id" not in ddi_edges.columns:
        raise ValueError("ddi_edges must contain `drug_a_id` and `drug_b_id` columns")

    # Distinguish explicit arguments from defaults.
    explicit_protocol = protocol is not _SENTINEL
    explicit_g1_ratio = g1_ratio is not _SENTINEL
    explicit_drug_ratio = drug_ratio is not _SENTINEL
    explicit_val_ratio = val_ratio is not _SENTINEL
    explicit_s0_train_budget = s0_train_budget is not None

    # Resolve protocol with legacy argument routing.
    if explicit_protocol:
        # An explicit protocol takes precedence.
        resolved_protocol = protocol
    elif explicit_drug_ratio:
        # Only legacy accepts drug_ratio.
        resolved_protocol = "legacy"
    else:
        resolved_protocol = DEFAULT_PROTOCOL

    if resolved_protocol not in PROTOCOLS:
        raise ValueError(
            f"protocol must be one of {PROTOCOLS}; got {resolved_protocol!r}"
        )

    # Reject parameters from the other protocol.
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

    # Fill unspecified parameters with defaults.
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
        # Share S0's train across settings; it excludes all val/test pairs.
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


# CLI


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
    # CLI --drug-ratio selects legacy; pass only that protocol's arguments.
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
        # Write and report each epoch as it is sampled.
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
