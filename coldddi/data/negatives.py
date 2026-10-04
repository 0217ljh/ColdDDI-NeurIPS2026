"""Seeded negative-pair sampling for the paper's 1:1 protocol.

``FairNegativeSampler`` aliases ``UniformNegativeSampler``; "fair" refers
to excluding all observed positives, as in ``Preprocessor/negatives.py``.
Fixed pools, exclusions, counts, and seeds give reproducible pairs.
Static val/test negatives use split-specific pools and seed offsets
``+100/+200/+300/+400/+500/+600`` from the configured base seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from coldddi.data.splits import SplitFolds


@dataclass
class UniformNegativeSampler:
    """Uniform random negative sampler over the configured drug pools.

    The two pools allow asymmetric sampling for cross-split (S1) and
    G2×G2 (S2) settings; for symmetric pools just pass the same list
    twice.
    """

    drug_pool_a: list[str]
    drug_pool_b: list[str]

    def sample(
        self,
        *,
        drug_pool_a: list[str] | None = None,
        drug_pool_b: list[str] | None = None,
        n_pairs: int,
        exclude: set[tuple[str, str]],
        seed: int,
    ) -> pd.DataFrame:
        """Sample ``n_pairs`` undirected negative pairs.

        Optional pools override constructor pools. Canonicalize exclusions
        as ``(min(a, b), max(a, b))``, accepting either orientation.
        Return distinct, sorted, non-self pairs; raise :class:`RuntimeError`
        if capacity is too small or sampling exceeds the attempt limit.
        """
        pool_a = drug_pool_a if drug_pool_a is not None else self.drug_pool_a
        pool_b = drug_pool_b if drug_pool_b is not None else self.drug_pool_b
        if not pool_a or not pool_b:
            if n_pairs == 0:
                return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
            raise RuntimeError(
                f"Empty drug pool(s) — cannot sample {n_pairs} negative pairs "
                f"(|pool_a|={len(pool_a)}, |pool_b|={len(pool_b)})"
            )

        # Exclude both orientations of each pair.
        canonical_exclude: set[tuple[str, str]] = set()
        for a, b in exclude:
            a_s, b_s = str(a), str(b)
            if a_s > b_s:
                a_s, b_s = b_s, a_s
            canonical_exclude.add((a_s, b_s))

        # Capacity upper bound before exclusions; max_attempts handles
        # dense exclusions and overlap between unequal pools.
        pool_a_set = set(map(str, pool_a))
        pool_b_set = set(map(str, pool_b))
        if pool_a_set == pool_b_set:
            reachable = len(pool_a_set) * (len(pool_a_set) - 1) // 2
        else:
            reachable = len(pool_a_set) * len(pool_b_set)
        if reachable < n_pairs:
            raise RuntimeError(
                f"Cannot sample {n_pairs} unique negative pairs from "
                f"|pool_a|={len(pool_a_set)} × |pool_b|={len(pool_b_set)} "
                f"(unique undirected capacity = {reachable})."
            )

        rng = np.random.default_rng(seed)
        sampled: set[tuple[str, str]] = set()
        attempts = 0
        max_attempts = max(n_pairs * 50, 1000)
        while len(sampled) < n_pairs and attempts < max_attempts:
            batch = max((n_pairs - len(sampled)) * 2, 16)
            a_samples = rng.choice(pool_a, size=batch)
            b_samples = rng.choice(pool_b, size=batch)
            for u, v in zip(a_samples, b_samples):
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pair = (str(u), str(v))
                if pair in canonical_exclude or pair in sampled:
                    continue
                sampled.add(pair)
                if len(sampled) >= n_pairs:
                    break
            attempts += 1

        if len(sampled) < n_pairs:
            raise RuntimeError(
                f"Negative-sampling did not converge: produced {len(sampled)}/{n_pairs} "
                f"pairs after {attempts} batches (likely exclude set too dense relative to pool)."
            )

        rows = sorted(sampled)
        return pd.DataFrame(rows, columns=["drug_a_id", "drug_b_id"])


# Legacy Preprocessor/negatives.py::fair_negatives_step uses this distribution;
# "fair" means excluding all observed positives, not changing the sampler.
FairNegativeSampler = UniformNegativeSampler


# Static val/test negatives

#: Legacy seed offsets per static split (matches Preprocessor/negatives.py).
PHASE_OFFSETS: dict[str, int] = {
    "test_s0": 100,
    "val_s0": 200,
    "test_s1": 300,
    "val_s1": 400,
    "test_s2": 500,
    "val_s2": 600,
}


def _canonical_exclude_set(positive_pairs: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for a, b in positive_pairs:
        a, b = str(a), str(b)
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def build_static_negatives(
    splits: SplitFolds,
    *,
    base_seed: int,
    n_per_pos: int = 1,
    exclude_extra: Iterable[tuple[str, str]] = (),
) -> dict[str, pd.DataFrame]:
    """Generate one batch of negatives per val/test split.

    For each of the six val/test splits we pick the appropriate drug
    pools (``G1 × G1`` for S0; cross for S1; ``G2 × G2`` for S2) and
    sample ``n_per_pos`` negatives per positive using a per-split
    sub-seed (``base_seed + PHASE_OFFSETS[name]``).

    The returned dict maps split name → DataFrame of negatives with
    columns ``drug_a_id, drug_b_id``.
    """
    g1, g2 = list(splits.g1_drugs), list(splits.g2_drugs)

    # Build the global positive exclude set across all 7 splits.
    all_pos: list[tuple[str, str]] = []
    for _, df in splits.items():
        all_pos.extend(zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)))
    exclude = _canonical_exclude_set(all_pos)
    exclude.update(_canonical_exclude_set(exclude_extra))

    pool_for: dict[str, tuple[list[str], list[str]]] = {
        "test_s0": (g1, g1),
        "val_s0": (g1, g1),
        "test_s1": (g1, g2),
        "val_s1": (g1, g2),
        "test_s2": (g2, g2),
        "val_s2": (g2, g2),
    }

    sampler = UniformNegativeSampler(drug_pool_a=g1, drug_pool_b=g1)
    out: dict[str, pd.DataFrame] = {}
    for name in ("test_s0", "val_s0", "test_s1", "val_s1", "test_s2", "val_s2"):
        pos_df = getattr(splits, name)
        n_pos = len(pos_df)
        if n_pos == 0:
            out[name] = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
            continue
        pool_a, pool_b = pool_for[name]
        out[name] = sampler.sample(
            drug_pool_a=pool_a,
            drug_pool_b=pool_b,
            n_pairs=n_pos * n_per_pos,
            exclude=exclude,
            seed=base_seed + PHASE_OFFSETS[name],
        )
    return out


#: Training seed offset; nonnegative epochs stay above val/test offsets.
TRAIN_NEGATIVES_SEED_BASE: int = 1000


def build_train_negatives(
    splits: SplitFolds,
    *,
    base_seed: int,
    epoch: int = 0,
    n_per_pos: int = 1,
    exclude_extra: Iterable[tuple[str, str]] = (),
) -> pd.DataFrame:
    """Sample one epoch of training negatives.

    Sample from ``G1 × G1``, excluding positives from all seven splits and
    ``exclude_extra``. Use ``base_seed + TRAIN_NEGATIVES_SEED_BASE + epoch``;
    nonnegative epochs have distinct seeds above val/test offsets 100..600.
    Return ``[drug_a_id, drug_b_id]`` with ``len(splits.train) * n_per_pos`` rows.
    """
    if len(splits.train) == 0:
        return pd.DataFrame(columns=["drug_a_id", "drug_b_id"])

    all_pos: list[tuple[str, str]] = []
    for _, df in splits.items():
        all_pos.extend(zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)))
    exclude = _canonical_exclude_set(all_pos)
    exclude.update(_canonical_exclude_set(exclude_extra))

    sampler = UniformNegativeSampler(
        drug_pool_a=list(splits.g1_drugs), drug_pool_b=list(splits.g1_drugs)
    )
    return sampler.sample(
        n_pairs=len(splits.train) * n_per_pos,
        exclude=exclude,
        seed=base_seed + TRAIN_NEGATIVES_SEED_BASE + epoch,
    )


def build_train_negatives_epochs(
    splits: SplitFolds,
    *,
    base_seed: int,
    n_epochs: int,
    n_per_pos: int = 1,
) -> list[pd.DataFrame]:
    """Return one training-negative DataFrame per epoch, indexed from zero.

    Each epoch uses a distinct seed for reproducible sampling.
    """
    return [
        build_train_negatives(
            splits, base_seed=base_seed, epoch=i, n_per_pos=n_per_pos
        )
        for i in range(n_epochs)
    ]


__all__ = [
    "UniformNegativeSampler",
    "FairNegativeSampler",
    "PHASE_OFFSETS",
    "TRAIN_NEGATIVES_SEED_BASE",
    "build_static_negatives",
    "build_train_negatives",
    "build_train_negatives_epochs",
]
