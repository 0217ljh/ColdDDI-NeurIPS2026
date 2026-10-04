"""KPS (Knowledge Prediction Sensitivity) entry points.

Re-exports :mod:`coldddi.diagnostics.indicators` and
:mod:`coldddi.diagnostics.kps_swap` under the path listed in
paper Appendix A.6.2, line 503.

Exports:

* :func:`compute_kps_f`         — paper Eq. (1)
* :func:`compute_kps_channel`   — paper Eq. (2)
* :func:`build_swap_candidates` — KPS-F's swap-anchor generator
* :class:`SwapTriple`           — record type emitted by the generator
"""

from __future__ import annotations

from coldddi.diagnostics.indicators import (
    compute_kps_channel,
    compute_kps_f,
)
from coldddi.diagnostics.kps_swap import SwapTriple, build_swap_candidates

__all__ = [
    "compute_kps_f",
    "compute_kps_channel",
    "build_swap_candidates",
    "SwapTriple",
]
