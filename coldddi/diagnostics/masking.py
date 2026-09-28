"""Paper-named shim: LLM R0--R7 masking runner.

Paper Appendix A.6.2 line 503 lists ``coldddi/diagnostics/masking.py``
as the home of the R0--R7 masking runner.  In the implementation the
mask conditions are produced by :mod:`coldddi.llm.prompts` /
:mod:`coldddi.llm.inference` (the LLM stack), and the resulting
per-condition prediction dicts are fed through
:func:`coldddi.diagnostics.compute_indicators` for the 7-indicator
panel (KPS-Name / KPS-KG / KPS-KG-Masked / KPS-Name-KGMasked /
KSAI and their aliases).

This module is a thin re-export plus a documented surface for
the LLM-FT masking pipeline.  The actual run-the-LLM step is
``scripts/run_llm.py`` (chains L1-prompt rendering through L6
indicators); call ``compute_indicators(...)`` here once you have
the R0/R1/R2/R3 prediction dicts.

Exports
-------
* :func:`compute_indicators`            — full 7-indicator panel for the
                                          LLM R0/R1/R2/R3 setting
* :data:`LLM_INDICATOR_NAMES`           — tuple of indicator names this
                                          panel emits
* :data:`PRIMARY_BUCKETS`               — PK-A / PK-B / PD-A / PD-B
"""

from __future__ import annotations

from coldddi.diagnostics.indicators import (
    LLM_INDICATOR_NAMES,
    PRIMARY_BUCKETS,
    compute_indicators,
)

__all__ = [
    "compute_indicators",
    "LLM_INDICATOR_NAMES",
    "PRIMARY_BUCKETS",
]
