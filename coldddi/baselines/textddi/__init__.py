"""TextDDI baseline — paper Zhu et al. (EMNLP 2024 findings), binary
adaptation. Pairs are encoded as a single text sequence (drug name +
SMILES per drug), passed through a Transformer backbone, and a
2-class CLS head predicts the interaction probability.

Default backbone is :data:`DEFAULT_BACKBONE` (``roberta-base`` per
paper Appendix C.1 Table); test fixtures wanting fast / offline
construction pass ``backbone=SMOKE_BACKBONE`` (a randomly-initialised
tiny DistilBert stub) explicitly.

Importing this submodule registers :class:`TextDDIBaseline` under
``"textddi"``.
"""

from __future__ import annotations

from coldddi.baselines.textddi.baseline import (
    DEFAULT_BACKBONE,
    PAPER_HYPERPARAMS,
    SMOKE_BACKBONE,
    TextDDIBaseline,
)

__all__ = [
    "TextDDIBaseline",
    "DEFAULT_BACKBONE",
    "SMOKE_BACKBONE",
    "PAPER_HYPERPARAMS",
]
