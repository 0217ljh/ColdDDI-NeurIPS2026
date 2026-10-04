"""TextDDI baseline — paper Zhu et al. (EMNLP 2024 findings), binary
adaptation. Pairs are encoded as one text sequence (drug names and
descriptions), passed through a Transformer backbone, and a
2-class CLS head predicts the interaction probability.

The paper preset uses :data:`DEFAULT_BACKBONE` (``roberta-base``, Appendix
C.1). Direct construction defaults to :data:`SMOKE_BACKBONE`, a tiny random
DistilBert model for smoke tests.

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
