"""Structural interfaces for graphs, negative samplers, and cold-start splits.

Implementations need the declared methods, not a shared base class.
Defaults live in ``coldddi.data.kg``, ``negatives``, and ``splits``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """A typed drug-to-entity knowledge graph.

    Support DataFrame lookups and the dict-of-lists view used by LLM prompts.
    """

    @property
    def drug_ids(self) -> set[str]:
        """All drugbank IDs that appear anywhere in the graph."""

    def neighbors(
        self,
        drug_id: str,
        *,
        edge_types: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return all (drug_id, edge_type, entity_id, entity_name, ...) rows."""

    def shared_entities(self, drug_a: str, drug_b: str) -> pd.DataFrame:
        """Return entities that *both* drugs are connected to."""

    def name_dict(self, edge_type: str) -> dict[str, list[str]]:
        """Return ``{drug_id: [entity_name, ...]}`` for one edge type.

        Used by legacy ``dataloader/prompts/blocks/subgraph.py``.
        """

    def save(self, out_dir: Path) -> None:
        """Persist all five entity tables to ``out_dir``."""


@runtime_checkable
class NegativeSamplerProtocol(Protocol):
    """Generate negative DDI pairs.

    Identical pools, exclusions, counts, and seeds must return identical pairs.
    """

    def sample(
        self,
        *,
        drug_pool_a: list[str],
        drug_pool_b: list[str],
        n_pairs: int,
        exclude: set[tuple[str, str]],
        seed: int,
    ) -> pd.DataFrame:
        """Return ``n_pairs`` negative pairs as
        ``DataFrame[drug_a_id, drug_b_id]`` (string-typed)."""


@runtime_checkable
class SplitFoldsProtocol(Protocol):
    """Drug-wise S0 / S1 / S2 train/val/test splits.

    Expose positive-only DataFrames and drug groups :math:`G_1` (seen) and
    :math:`G_2` (unseen). One training set serves all three settings and
    must exclude every validation and test pair.
    """

    @property
    def train(self) -> pd.DataFrame:
        """The single canonical training set, shared by all three settings.

        Use ``G1 × G1`` minus S0 val/test holdouts. S1 cross-group and S2
        ``G2 × G2`` val/test pairs are disjoint from this training pool.
        """

    @property
    def val_s0(self) -> pd.DataFrame: ...

    @property
    def val_s1(self) -> pd.DataFrame: ...

    @property
    def val_s2(self) -> pd.DataFrame: ...

    @property
    def test_s0(self) -> pd.DataFrame: ...

    @property
    def test_s1(self) -> pd.DataFrame: ...

    @property
    def test_s2(self) -> pd.DataFrame: ...

    @property
    def g1_drugs(self) -> list[str]: ...

    @property
    def g2_drugs(self) -> list[str]: ...

    @property
    def seed(self) -> int:
        """The random seed that produced this split."""

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        """Iterate over every (split_name, DataFrame) pair."""

    def save(self, out_dir: Path) -> None: ...


__all__ = [
    "KnowledgeGraphProtocol",
    "NegativeSamplerProtocol",
    "SplitFoldsProtocol",
]
