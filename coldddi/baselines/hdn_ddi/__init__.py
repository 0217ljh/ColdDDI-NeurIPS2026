"""HDN-DDI baseline — paper Yu et al., binary adaptation.

HDN-DDI builds a hierarchical drug graph: each molecule gets an extra
"super node" (``y=2``) connected to every atom, and the model extracts
the super-node embedding as the molecular representation. Importing
this submodule registers :class:`HDNDDIBaseline` under ``"hdn_ddi"``.
"""

from __future__ import annotations

from coldddi.baselines.hdn_ddi.baseline import (
    PAPER_HYPERPARAMS,
    HDNDDIBaseline,
)

__all__ = ["HDNDDIBaseline", "PAPER_HYPERPARAMS"]
