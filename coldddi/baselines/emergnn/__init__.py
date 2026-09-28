"""EmerGNN baseline (Zhang et al., 2023; ColdDDI pure-PyTorch reimpl).

Importing this submodule registers :class:`EmerGNNBaseline` under
``"emergnn"``.
"""

from __future__ import annotations

from coldddi.baselines.emergnn.baseline import (
    PAPER_HYPERPARAMS,
    EmerGNNBaseline,
)

__all__ = ["EmerGNNBaseline", "PAPER_HYPERPARAMS"]
