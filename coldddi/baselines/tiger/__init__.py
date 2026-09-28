"""TIGER baseline — paper Su et al. (AAAI 2024), binary adaptation
in **mol-only** mode (no KG subgraph branch), for cold-start
compatibility across S0/S1/S2.

Importing this submodule registers :class:`TIGERBaseline` under
``"tiger"``.
"""

from __future__ import annotations

from coldddi.baselines.tiger.baseline import (
    PAPER_HYPERPARAMS,
    TIGERBaseline,
)

__all__ = ["TIGERBaseline", "PAPER_HYPERPARAMS"]
