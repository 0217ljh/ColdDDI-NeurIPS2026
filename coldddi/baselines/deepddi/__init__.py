"""DeepDDI baseline — paper Ryu et al. (PNAS 2018), binary adaptation.

Importing registers :class:`DeepDDIBaseline` as ``"deepddi"``. SSP references
and PCA fitting use only G1 training drugs to prevent G2 leakage.
"""

from __future__ import annotations

from coldddi.baselines.deepddi.baseline import (
    PAPER_HYPERPARAMS,
    DeepDDIBaseline,
)

__all__ = ["DeepDDIBaseline", "PAPER_HYPERPARAMS"]
