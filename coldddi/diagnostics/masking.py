"""LLM masking diagnostics (paper Appendix A.6.2, line 503).

The R0--R7 conditions are produced by :mod:`coldddi.llm.prompts` and
:mod:`coldddi.llm.inference`. Run ``scripts/run_llm.py`` for the full
prompt-to-indicator pipeline, or pass R0/R1/R2/R3 prediction dicts to
:func:`compute_indicators` here.

Exports:

* :func:`compute_indicators` — seven-indicator LLM R0/R1/R2/R3 panel
* :data:`LLM_INDICATOR_NAMES` — emitted indicator names
* :data:`PRIMARY_BUCKETS` — PK-A / PK-B / PD-A / PD-B
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
