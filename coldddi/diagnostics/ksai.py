"""KSAI (Cross-channel Sensitivity Asymmetry Index) entry points.

Re-exports the KSAI functions from :mod:`coldddi.diagnostics.indicators`.

Exports:

* :func:`compute_ksai`   — paper Eq. (3), LLM-only 2x2 factorial
                            ``|P_R1 - P_R3| - |P_R0 - P_R2|``
* :func:`compute_ab_gap` — A vs B gap for one indicator
"""

from __future__ import annotations

from coldddi.diagnostics.indicators import compute_ab_gap, compute_ksai

__all__ = [
    "compute_ksai",
    "compute_ab_gap",
]
