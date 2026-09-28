"""MKG-FENN baseline — paper Hou et al. (Inf. Fusion 2024), binary
adaptation. Four parallel KG-GNN channels (drug-entity, drug-substructure,
drug-DDI, drug-property) fused into a 2-class softmax head.

Importing this submodule registers :class:`MKGFENNBaseline` under
``"mkg_fenn"``.
"""

from __future__ import annotations

from coldddi.baselines.mkg_fenn.baseline import (
    PAPER_HYPERPARAMS,
    MKGFENNBaseline,
)

__all__ = ["MKGFENNBaseline", "PAPER_HYPERPARAMS"]
