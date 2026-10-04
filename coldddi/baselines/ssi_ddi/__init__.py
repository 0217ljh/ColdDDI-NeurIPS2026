"""SSI-DDI baseline — paper Nyamabo et al. (BIB 2022), binary adaptation.

Importing registers :class:`SSIDDIBaseline` as ``"ssi_ddi"``. The model
combines molecular GAT blocks, co-attention, and RESCAL scoring.
"""

from __future__ import annotations

from coldddi.baselines.ssi_ddi.baseline import (
    PAPER_HYPERPARAMS,
    SSIDDIBaseline,
)

__all__ = ["SSIDDIBaseline", "PAPER_HYPERPARAMS"]
