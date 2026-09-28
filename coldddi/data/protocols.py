"""Pluggable interfaces (typing.Protocol) for the data layer.

Defining these as :class:`typing.Protocol` rather than abstract base
classes lets baseline code, LLM prompt builders, and external research
tools plug in their own implementations without inheriting from a
specific class hierarchy. Anything that exposes the listed methods
satisfies the protocol structurally.

Three protocols cover the data-layer plug points:

* :class:`KnowledgeGraphProtocol` — drug-to-entity lookup.
* :class:`NegativeSamplerProtocol` — generate negative DDI pairs.
* :class:`SplitFoldsProtocol` — drug-wise S0/S1/S2 train/val/test access.

The default implementations live in
:mod:`coldddi.data.kg`,
:mod:`coldddi.data.negatives`, and
:mod:`coldddi.data.splits`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class KnowledgeGraphProtocol(Protocol):
    """A typed drug-to-entity knowledge graph.

    Implementations must support both a *modern* DataFrame API and a
    *legacy* dict-of-list view, because the LLM prompt builder
    inherited from earlier code expects the dict shape.
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

        This is the format the legacy prompt code (e.g.
        ``dataloader/prompts/blocks/subgraph.py``) expects.
        """

    def save(self, out_dir: Path) -> None:
        """Persist all five entity tables to ``out_dir``."""


@runtime_checkable
class NegativeSamplerProtocol(Protocol):
    """Generate negative DDI pairs.

    Implementations are pure functions of (drug pool, exclude set, seed)
    so the same call with the same seed always returns the same pairs —
    a hard requirement for reproducing paper results.
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

    Implementations expose every named split as a positive-only
    :class:`pandas.DataFrame` and, separately, the cold-start drug
    groups :math:`G_1` (seen) and :math:`G_2` (unseen). Each of the
    three settings (S0/S1/S2) has its **own** train so that val/test
    can never leak into the corresponding train.
    """

    @property
    def train(self) -> pd.DataFrame:
        """The single canonical training set, shared by all three settings.

        Concretely: ``G1 × G1`` minus the held-out S0 val/test edges.
        Disjoint from every val/test bucket because S1/S2 val/test are
        crosses or ``G2 × G2`` (no overlap with ``G1 × G1`` is possible).
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
