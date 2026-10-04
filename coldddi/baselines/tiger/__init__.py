"""TIGER baseline — Su et al. (AAAI 2024), binary cold-start adaptation.

Uses molecular and KG channels by default, with an optional mol-only mode.

Importing this submodule registers :class:`TIGERBaseline` under
``"tiger"``.
"""

from __future__ import annotations

from coldddi.baselines.tiger.baseline import (
    PAPER_HYPERPARAMS,
    TIGERBaseline,
)

__all__ = ["TIGERBaseline", "PAPER_HYPERPARAMS"]
