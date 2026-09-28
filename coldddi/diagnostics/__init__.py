"""Diagnostic indicators for paper §4.3 / Table 6.

This module is a byte-exact port of the upstream
``Code-Released/exps/sec5-3/2_indicators/`` scripts.

Indicator coverage per model
============================

LLM (Llama / Qwen / Gemma) — 4-condition prompt masking:

* ``KPS-F``               — drug replacement, R0 only
* ``KPS-Name``            — ``|R0 - R1|``
* ``KPS-KG``              — ``|R0 - R2|``
* ``KPS-KG-Named``        — alias of ``KPS-KG`` (paper-script parity)
* ``KPS-KG-Masked``       — ``|R1 - R3|``
* ``KPS-Name-KGMasked``   — ``|R2 - R3|``
* ``KSAI``        — per-pair ``|R1-R3| - |R0-R2|``

Baselines — coverage matrix (paper Table 6):

==============  ==========  ============  ===========
Baseline        KPS-F       KPS-mol       KPS-KG
==============  ==========  ============  ===========
DeepDDI         ✓ (R0)      —             —
SSI-DDI         ✓ (R0)      —             —
DSN-DDI         ✓ (R0)      —             —
HDN-DDI         ✓ (R0)      —             —
EmerGNN         ✓ (R0)      —             —
TextDDI         ✓ (R0)      —             —
MKG-FENN        ✓ (base)    ✓             ✓
TIGER           ✓ (base)    ✓             ✓
==============  ==========  ============  ===========

The dash (``—``) means the baseline has only a single fused
modality with no separable channel to mask — channel indicators
return NaN rows per bucket.  ``KPS-F`` is always computable from the
base predictions alone.

Bucket axes
-----------
* Primary buckets (paper Table 6 columns): ``PK-A``, ``PK-B``,
  ``PD-A``, ``PD-B``, set by :mod:`coldddi.diagnostics.buckets`.
* ``ALL`` bucket = aggregate over ``label_uv == 1`` rows only
  (positive base pairs) — matches upstream ``_agg_buckets`` exactly.
* Pairs whose annotation didn't yield a primary bucket are dropped
  from per-bucket reports and excluded from the ALL aggregate by
  the ``label_uv == 1`` filter (since unannotated pairs in upstream
  are also typically negatives or "Mixed" mechanism).
* The paper-headline A-B gap is computed by
  :func:`compute_ab_gap` as ``(PK-A + PD-A)/2 - (PK-B + PD-B)/2``.

Submodules
----------
* :mod:`coldddi.diagnostics.buckets`   — PK/PD × A/B bucket assignment.
* :mod:`coldddi.diagnostics.kps_swap`  — build (u, v, u') swap triples.
* :mod:`coldddi.diagnostics.indicators` — KPS-F / KPS-Name / KPS-KG /
  KPS-KG-Named / KPS-KG-Masked / KPS-Name-KGMasked / KSAI
  for LLMs; KPS-mol / KPS-KG for mol+KG-separable baselines.
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
    # Paper-name aliases (A.6.2 walkthrough, "Diagnostic-Indicator Plugin"):
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
    # Paper-name aliases (A.6.2):
    "compute_kps_f",
    "compute_kps_channel",
    "compute_ksai",
]
