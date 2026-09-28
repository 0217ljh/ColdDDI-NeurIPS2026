"""DeepDDI baseline — paper Ryu et al. (PNAS 2018), binary adaptation.

Importing this submodule registers :class:`DeepDDIBaseline` under the
``"deepddi"`` name; thereafter ``coldddi.evaluate --method deepddi`` and
``coldddi.baselines.load_baseline(path)`` route to it automatically.

The wrapper is intentionally **thin**: it delegates feature extraction
to :mod:`coldddi.baselines.deepddi.ssp_features` (close to the upstream
implementation, with one principled cold-start guard added — PCA is
fit on the G1 training subset only, never on all drugs, to keep G2
cold-start drugs out of the SSP basis) and the network to
:mod:`coldddi.baselines.deepddi.model`. All training-loop logic lives
in :class:`DeepDDIBaseline.fit` (~80 lines) so adding a new baseline
follows the same recipe: import a model + features module, wire
``fit/predict_proba/save/load`` against the registered ABC.
"""

from __future__ import annotations

from coldddi.baselines.deepddi.baseline import (
    PAPER_HYPERPARAMS,
    DeepDDIBaseline,
)

__all__ = ["DeepDDIBaseline", "PAPER_HYPERPARAMS"]
