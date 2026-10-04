"""Diagnostic indicators for paper §4.3 / Table 6.

Based on ``Code-Released/exps/sec5-3/2_indicators/``.

LLM (Llama / Qwen / Gemma) — 4-condition prompt masking:

* ``KPS-F``               — drug replacement, R0 only
* ``KPS-Name``            — ``|R0 - R1|``
* ``KPS-KG``              — ``|R0 - R2|``
* ``KPS-KG-Named``        — alias of ``KPS-KG``
* ``KPS-KG-Masked``       — ``|R1 - R3|``
* ``KPS-Name-KGMasked``   — ``|R2 - R3|``
* ``KSAI``        — per-pair ``|R1-R3| - |R0-R2|``

Baselines (paper Table 6): MKG-FENN and TIGER support KPS-F, KPS-mol,
and KPS-KG. DeepDDI, SSI-DDI, DSN-DDI, HDN-DDI, EmerGNN, and TextDDI
support KPS-F only; inseparable channels return NaN rows per bucket.
KPS-F uses base predictions alone.

Buckets:

* Primary buckets (paper Table 6 columns): ``PK-A``, ``PK-B``,
  ``PD-A``, ``PD-B``, set by :mod:`coldddi.diagnostics.buckets`.
* ``ALL`` aggregates only ``label_uv == 1`` rows (positive base pairs),
  following upstream ``_agg_buckets``.
* Pairs without a primary bucket are omitted from per-bucket reports;
  they contribute to ``ALL`` only when ``label_uv == 1``.
* :func:`compute_ab_gap` returns ``(PK-A + PD-A)/2 - (PK-B + PD-B)/2``.

Submodules:

* :mod:`coldddi.diagnostics.buckets`   — PK/PD × A/B bucket assignment.
* :mod:`coldddi.diagnostics.kps_swap`  — build (u, v, u') swap triples.
* :mod:`coldddi.diagnostics.indicators` — LLM and baseline indicators.
"""

from __future__ import annotations

from coldddi.diagnostics.buckets import (
    BUCKET_NAMES,
    BUCKET_OTHER,
    BucketLookup,
    build_bucket_lookup,
)
from coldddi.diagnostics.indicators import (
    BASELINE_CHANNEL_INDICATOR_NAMES,
    INDICATOR_NAMES,
    LLM_INDICATOR_NAMES,
    PRIMARY_BUCKETS,
    compute_ab_gap,
    compute_baseline_channel_indicators,
    compute_indicators,
    # Paper API aliases (Appendix A.6.2).
    compute_kps_f,
    compute_kps_channel,
    compute_ksai,
)
from coldddi.diagnostics.kps_swap import (
    SwapTriple,
    build_swap_candidates,
)

__all__ = [
    "BUCKET_NAMES",
    "BUCKET_OTHER",
    "BucketLookup",
    "PRIMARY_BUCKETS",
    "LLM_INDICATOR_NAMES",
    "BASELINE_CHANNEL_INDICATOR_NAMES",
    "INDICATOR_NAMES",
    "SwapTriple",
    "build_bucket_lookup",
    "build_swap_candidates",
    "compute_ab_gap",
    "compute_baseline_channel_indicators",
    "compute_indicators",
    "compute_kps_f",
    "compute_kps_channel",
    "compute_ksai",
]
