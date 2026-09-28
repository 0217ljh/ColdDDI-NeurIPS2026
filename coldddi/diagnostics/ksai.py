"""Paper-named shim: KSAI (Cross-channel Sensitivity Asymmetry Index).

Paper Appendix A.6.2 line 503 lists ``coldddi/diagnostics/ksai.py``
as the home of the KSAI entry point.  In the implementation it
lives in :mod:`coldddi.diagnostics.indicators` next to the other
indicator math; this module is a thin re-export so the paper's
file map is accurate.

Exports
-------
* :func:`compute_ksai`   — paper Eq. (3), LLM-only 2x2 factorial
                            ``|P_R1 - P_R3| - |P_R0 - P_R2|``
* :func:`compute_ab_gap` — paper-headline A vs B gap on top of
                            any single indicator
"""

from __future__ import annotations

from coldddi.diagnostics.indicators import compute_ab_gap, compute_ksai

__all__ = [
    "compute_ksai",
    "compute_ab_gap",
]
