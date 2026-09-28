"""KPS / KSAI diagnostic indicators — byte-exact port of upstream.

Sources (canonical reference):
* ``Code-Released/exps/sec5-3/2_indicators/compute_llm_indicators_3seed.py``
* ``Code-Released/exps/sec5-3/2_indicators/compute_baseline_kpsf_3seed.py``
* ``Code-Released/exps/sec5-3/2_indicators/compute_baseline_kps_channels_3seed.py``

Indicator names (paper §4.3 / Table 6)
--------------------------------------

LLM stack (R0=baseline / R1=mask name / R2=mask entity / R3=mask both):

* ``KPS-F``              — drug-replacement on R0 only;
                           iterates over swap triples (u, v, u').
                           **Always computable** as long as R0 exists.
* ``KPS-Name``           — ``mean(|R0 - R1|)`` over base pairs.
* ``KPS-KG``             — ``mean(|R0 - R2|)`` over base pairs.
* ``KPS-KG-Named``       — alias of ``KPS-KG`` (paper-script keeps both
                           names for cross-reference clarity).
* ``KPS-KG-Masked``      — ``mean(|R1 - R3|)`` (KG channel, name masked).
* ``KPS-Name-KGMasked``  — ``mean(|R2 - R3|)`` (name channel, KG masked).
* ``KSAI``       — per-pair ``(|R1-R3| - |R0-R2|)`` then mean.

Baselines with mol+KG separable channels (MKG-FENN, TIGER):

* ``KPS-F``    — same as LLM.
* ``KPS-mol``  — ``mean(|base - mask_mol|)`` (molecular feature masked).
* ``KPS-KG``   — ``mean(|base - mask_kg|)``  (KG feature masked).

Single-modality baselines (DeepDDI / SSI-DDI / DSN-DDI / HDN-DDI /
EmerGNN / TextDDI):

* ``KPS-F`` only — no separable name/KG channel to mask.

Bucket axes
-----------
* Primary buckets (paper Table 6 columns):  ``PK-A``, ``PK-B``, ``PD-A``,
  ``PD-B`` — strictly from the annotation table.  Pairs whose
  mechanism is unknown / "Mixed" are silently dropped (upstream
  convention, matching how the paper aggregates).
* ``ALL`` bucket = aggregate over ``label_uv == 1`` rows only
  (positive base pairs).  This matches upstream's
  ``_agg_buckets`` exactly and is what the paper reports.
* The earlier ``"Other"`` bucket has been removed for byte-exact
  parity with upstream.

A-B gap
-------
The paper's headline diagnostic on top of the per-bucket mean is the
A-vs-B gap, computed by :func:`compute_ab_gap`::

    A-B gap = (PK-A + PD-A) / 2  -  (PK-B + PD-B) / 2

A positive gap means the model is **more** sensitive on pairs whose
mediating entity IS confirmed in the KG (Type-A); a negative gap
means the model leans the wrong way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from coldddi.diagnostics.kps_swap import SwapTriple


#: Per-bucket primary names emitted in addition to ``ALL``.  The
#: ``"Other"`` bucket used in pre-byte-exact versions is intentionally
#: NOT emitted any more (upstream drops it; see the module docstring).
PRIMARY_BUCKETS: tuple[str, ...] = ("PK-A", "PK-B", "PD-A", "PD-B")


#: All indicator names emitted by :func:`compute_indicators` for the
#: LLM 4-condition setting (R0/R1/R2/R3).
LLM_INDICATOR_NAMES: tuple[str, ...] = (
    "KPS-F",
    "KPS-Name",
    "KPS-KG",
    "KPS-KG-Named",       # alias of KPS-KG, kept for paper parity
    "KPS-KG-Masked",
    "KPS-Name-KGMasked",
    "KSAI",
)


#: All indicator names emitted by :func:`compute_baseline_channel_indicators`
#: for the mol+KG-separable baselines (MKG-FENN, TIGER).
BASELINE_CHANNEL_INDICATOR_NAMES: tuple[str, ...] = (
    "KPS-F",
    "KPS-mol",
    "KPS-KG",
)


#: Kept for backward compatibility with earlier callers.
INDICATOR_NAMES = LLM_INDICATOR_NAMES


# ─── Internal lookup with direction-tolerant fallback ────────────────────────

def _lookup_directed(d: dict, a: str, b: str):
    """``d[(a, b)]`` with reverse-direction fallback.

    Strict superset of the upstream's exact-key lookup: when ``d`` is
    populated in canonical order (the upstream invariant) this
    returns the same value as ``d.get((a, b))``; when ``d`` happens
    to carry ``(b, a)`` instead, we still find it.  Without the
    fallback, the toy smoke test alone drops 32% of swap-candidate
    triples whose ``(qa_prime, qb)`` key lives in reverse direction.
    """
    if (a, b) in d:
        return d[(a, b)]
    if (b, a) in d:
        return d[(b, a)]
    return None


# ─── Aggregation matching upstream _agg_buckets ──────────────────────────────

def _agg_buckets(records: list[dict], value_col: str, indicator: str) -> list[dict]:
    """Replicate ``compute_*_3seed.py::_agg_buckets`` byte-exactly.

    ``records`` must be a list of dicts with keys ``bucket`` (one of
    :data:`PRIMARY_BUCKETS` or ``""``) and ``label_uv`` (0 or 1) and
    ``value_col`` (the per-row delta).
    """
    out: list[dict] = []
    if not records:
        return out
    df = pd.DataFrame(records)

    # Primary buckets: only emit those with at least one record.
    for bk in PRIMARY_BUCKETS:
        sub = df[df["bucket"] == bk]
        if len(sub):
            out.append({
                "indicator": indicator,
                "bucket": bk,
                "value": float(sub[value_col].mean()),
                "std": float(sub[value_col].std(ddof=1)) if len(sub) > 1 else 0.0,
                "n": int(len(sub)),
            })

    # ALL bucket = aggregate over positive base pairs (label_uv == 1),
    # matching upstream's convention exactly.
    pos = df[df["label_uv"] == 1]
    if len(pos):
        out.append({
            "indicator": indicator,
            "bucket": "ALL",
            "value": float(pos[value_col].mean()),
            "std": float(pos[value_col].std(ddof=1)) if len(pos) > 1 else 0.0,
            "n": int(len(pos)),
        })
    return out


def _empty_indicator_rows(indicator: str) -> list[dict]:
    """When a required condition is missing, emit one row per primary
    bucket + ALL with NaN.  This is how single-modality baselines
    surface "no signal here" for channel indicators."""
    rows = []
    for bk in PRIMARY_BUCKETS + ("ALL",):
        rows.append({
            "indicator": indicator,
            "bucket": bk,
            "value": float("nan"),
            "std": float("nan"),
            "n": 0,
        })
    return rows


# ─── Derive base pairs from swap_candidates (one row per (u, v)) ─────────────

def _unique_base_pairs(
    swap_candidates: Iterable[SwapTriple],
    bucket_fn: Callable[[str, str], str],
) -> list[dict]:
    """Dedup swap_candidates by ``(qa, qb)`` and attach bucket + label_uv.

    Mirrors ``swap_df`` in
    ``compute_llm_indicators_3seed.py::compute_indicators_for_seed``
    which carries one row per base pair (not per triple).  Pairs with
    an empty / unknown bucket are KEPT in the returned list (with
    ``bucket=""``) so they can still contribute to the ALL aggregate
    when ``label_uv == 1`` — they only get excluded at
    :func:`_agg_buckets` time from per-primary-bucket rows, exactly
    matching upstream behaviour.
    """
    seen: dict[tuple[str, str], tuple[str, int]] = {}
    for t in swap_candidates:
        key = (t.qa, t.qb)
        if key not in seen:
            seen[key] = (bucket_fn(t.qa, t.qb), int(t.label_uv))
    return [
        {"qa": a, "qb": b, "bucket": bk, "label_uv": lv}
        for (a, b), (bk, lv) in seen.items()
    ]


# ─── LLM 4-condition indicators ──────────────────────────────────────────────

def compute_indicators(
    predictions: dict[str, dict | None],
    swap_candidates: list[SwapTriple],
    *,
    bucket_fn: Callable[[str, str], str] = lambda a, b: "",
    coverage_warnings: bool = False,
) -> pd.DataFrame:
    """Compute the full LLM indicator panel.

    Parameters
    ----------
    predictions
        ``{"R0": {(qa, qb): p_yes}, "R1": ..., "R2": ..., "R3": ...}``.
        ``R0`` is required for any indicator to fire; the other three
        may be ``None`` or missing — channel/KSAI indicators that
        depend on a missing condition come back as NaN rows for every
        bucket (single-modality baseline path).
    swap_candidates
        Output of :func:`coldddi.diagnostics.kps_swap.build_swap_candidates`.
        Used by KPS-F (per-triple) AND used to derive the base-pair
        table for channel/KSAI indicators (per-base-pair).
    bucket_fn
        ``(qa, qb) -> "PK-A" | "PK-B" | "PD-A" | "PD-B" | <anything else>``.
        Pairs that fall outside the four primary buckets are dropped
        from per-bucket rows (and contribute only to the ALL aggregate
        when ``label_uv == 1``).
    coverage_warnings
        When ``True``, print a per-indicator coverage summary to
        stderr.  Defaults to ``False`` so the function stays quiet by
        default — set ``True`` in pipeline runs where silently
        dropping uncovered triples could mask a real bug (e.g. when
        ``predictions`` is built from a smaller pair set than
        ``swap_candidates`` references).

    Returns
    -------
    Long-form DataFrame with columns ``[indicator, bucket, value, std, n]``.
    """
    r0 = predictions.get("R0") or {}
    r1 = predictions.get("R1")
    r2 = predictions.get("R2")
    r3 = predictions.get("R3")

    out_rows: list[dict] = []
    coverage: dict[str, dict[str, int]] = {}

    # ── KPS-F (Natural KPS, drug-replacement) — uses R0 + triples ─────────────
    if r0 and swap_candidates:
        kpsf = []
        kpsf_skipped = 0
        for t in swap_candidates:
            p_uv = _lookup_directed(r0, t.qa, t.qb)
            p_upv = _lookup_directed(r0, t.qa_prime, t.qb)
            if p_uv is None or p_upv is None:
                kpsf_skipped += 1
                continue
            kpsf.append({
                "bucket": bucket_fn(t.qa, t.qb),
                "label_uv": int(t.label_uv),
                "delta": abs(float(p_uv) - float(p_upv)),
            })
        out_rows.extend(_agg_buckets(kpsf, "delta", "KPS-F"))
        coverage["KPS-F"] = {
            "kept": len(kpsf),
            "skipped": kpsf_skipped,
            "total": len(swap_candidates),
        }
    else:
        out_rows.extend(_empty_indicator_rows("KPS-F"))
        coverage["KPS-F"] = {
            "kept": 0,
            "skipped": 0,
            "total": len(swap_candidates) if swap_candidates else 0,
        }

    # ── Channel / KSAI iterate over base pairs (deduped from triples) ─────────
    base_pairs = _unique_base_pairs(swap_candidates, bucket_fn) if swap_candidates else []

    def _delta_indicator(name: str, cond_a, cond_b) -> None:
        if cond_a is None or cond_b is None or not base_pairs:
            out_rows.extend(_empty_indicator_rows(name))
            coverage[name] = {
                "kept": 0,
                "skipped": 0,
                "total": len(base_pairs) if base_pairs else 0,
            }
            return
        recs = []
        skipped = 0
        for p in base_pairs:
            va = _lookup_directed(cond_a, p["qa"], p["qb"])
            vb = _lookup_directed(cond_b, p["qa"], p["qb"])
            if va is None or vb is None:
                skipped += 1
                continue
            recs.append({
                "bucket": p["bucket"],
                "label_uv": p["label_uv"],
                "delta": abs(float(va) - float(vb)),
            })
        if recs:
            out_rows.extend(_agg_buckets(recs, "delta", name))
        else:
            out_rows.extend(_empty_indicator_rows(name))
        coverage[name] = {
            "kept": len(recs),
            "skipped": skipped,
            "total": len(base_pairs),
        }

    _delta_indicator("KPS-Name",          r0, r1)
    _delta_indicator("KPS-KG",            r0, r2)
    _delta_indicator("KPS-KG-Named",      r0, r2)  # alias of KPS-KG
    _delta_indicator("KPS-KG-Masked",     r1, r3)
    _delta_indicator("KPS-Name-KGMasked", r2, r3)

    # ── KSAI = per-pair (|R1-R3| - |R0-R2|), then mean ────────────────
    if r0 and r1 and r2 and r3 and base_pairs:
        recs = []
        ksai_skipped = 0
        for p in base_pairs:
            p_r0 = _lookup_directed(r0, p["qa"], p["qb"])
            p_r1 = _lookup_directed(r1, p["qa"], p["qb"])
            p_r2 = _lookup_directed(r2, p["qa"], p["qb"])
            p_r3 = _lookup_directed(r3, p["qa"], p["qb"])
            if any(v is None for v in (p_r0, p_r1, p_r2, p_r3)):
                ksai_skipped += 1
                continue
            d_kg_named = abs(float(p_r0) - float(p_r2))
            d_kg_masked = abs(float(p_r1) - float(p_r3))
            recs.append({
                "bucket": p["bucket"],
                "label_uv": p["label_uv"],
                "delta": d_kg_masked - d_kg_named,
            })
        if recs:
            out_rows.extend(_agg_buckets(recs, "delta", "KSAI"))
        else:
            out_rows.extend(_empty_indicator_rows("KSAI"))
        coverage["KSAI"] = {
            "kept": len(recs),
            "skipped": ksai_skipped,
            "total": len(base_pairs),
        }
    else:
        out_rows.extend(_empty_indicator_rows("KSAI"))
        coverage["KSAI"] = {
            "kept": 0,
            "skipped": 0,
            "total": len(base_pairs) if base_pairs else 0,
        }

    if coverage_warnings:
        _emit_coverage(coverage)
    return pd.DataFrame(out_rows)


def _emit_coverage(coverage: dict[str, dict[str, int]]) -> None:
    """Print a one-line summary of how many records each indicator
    actually used vs. how many were skipped (missing predictions).
    """
    import sys

    lines = ["[L6 coverage]"]
    for name, c in coverage.items():
        kept = c["kept"]
        skipped = c["skipped"]
        total = c["total"]
        ratio = (kept / total) if total else 0.0
        flag = "" if skipped == 0 else f"  ⚠ skipped {skipped}"
        lines.append(f"  {name:<22} kept {kept}/{total} ({ratio:>5.1%}){flag}")
    print("\n".join(lines), file=sys.stderr)


# ─── Baseline mol+KG channel indicators (MKG-FENN, TIGER) ────────────────────

def compute_baseline_channel_indicators(
    predictions: dict[str, dict | None],
    swap_candidates: list[SwapTriple],
    *,
    bucket_fn: Callable[[str, str], str] = lambda a, b: "",
    coverage_warnings: bool = False,
) -> pd.DataFrame:
    """Compute KPS-F + KPS-mol + KPS-KG for a mol+KG-separable baseline.

    The expected keys in ``predictions`` are::

        "base"      — unmasked predictions (= LLM's R0)
        "mask_mol"  — predictions with the molecular channel masked
        "mask_kg"   — predictions with the KG channel masked

    Missing keys produce a NaN row block for the affected indicator,
    matching the single-modality baseline contract.

    Pass ``coverage_warnings=True`` to print a per-indicator
    kept/skipped/total summary to stderr (same convention as
    :func:`compute_indicators`).
    """
    base = predictions.get("base") or {}
    mol = predictions.get("mask_mol")
    kg = predictions.get("mask_kg")

    out_rows: list[dict] = []
    coverage: dict[str, dict[str, int]] = {}

    # KPS-F (drug replacement) — same logic as the LLM path.
    if base and swap_candidates:
        kpsf = []
        kpsf_skipped = 0
        for t in swap_candidates:
            p_uv = _lookup_directed(base, t.qa, t.qb)
            p_upv = _lookup_directed(base, t.qa_prime, t.qb)
            if p_uv is None or p_upv is None:
                kpsf_skipped += 1
                continue
            kpsf.append({
                "bucket": bucket_fn(t.qa, t.qb),
                "label_uv": int(t.label_uv),
                "delta": abs(float(p_uv) - float(p_upv)),
            })
        out_rows.extend(_agg_buckets(kpsf, "delta", "KPS-F"))
        coverage["KPS-F"] = {
            "kept": len(kpsf),
            "skipped": kpsf_skipped,
            "total": len(swap_candidates),
        }
    else:
        out_rows.extend(_empty_indicator_rows("KPS-F"))
        coverage["KPS-F"] = {
            "kept": 0, "skipped": 0,
            "total": len(swap_candidates) if swap_candidates else 0,
        }

    base_pairs = _unique_base_pairs(swap_candidates, bucket_fn) if swap_candidates else []

    def _delta(name: str, other) -> None:
        if other is None or not base_pairs:
            out_rows.extend(_empty_indicator_rows(name))
            coverage[name] = {
                "kept": 0, "skipped": 0,
                "total": len(base_pairs) if base_pairs else 0,
            }
            return
        recs = []
        skipped = 0
        for p in base_pairs:
            va = _lookup_directed(base, p["qa"], p["qb"])
            vb = _lookup_directed(other, p["qa"], p["qb"])
            if va is None or vb is None:
                skipped += 1
                continue
            recs.append({
                "bucket": p["bucket"],
                "label_uv": p["label_uv"],
                "delta": abs(float(va) - float(vb)),
            })
        if recs:
            out_rows.extend(_agg_buckets(recs, "delta", name))
        else:
            out_rows.extend(_empty_indicator_rows(name))
        coverage[name] = {
            "kept": len(recs),
            "skipped": skipped,
            "total": len(base_pairs),
        }

    _delta("KPS-mol", mol)
    _delta("KPS-KG", kg)
    if coverage_warnings:
        _emit_coverage(coverage)
    return pd.DataFrame(out_rows)


# ─── A-B gap (paper-headline diagnostic) ─────────────────────────────────────

def compute_ab_gap(indicators_df: pd.DataFrame, indicator: str) -> float:
    """Return ``(PK-A + PD-A)/2 - (PK-B + PD-B)/2`` for one indicator.

    A positive value means the model is more sensitive on the
    annotation-confirmed (A) pairs than on the unconfirmed (B) pairs.
    Returns ``float('nan')`` when any of the four primary buckets is
    missing for the given indicator.
    """
    sub = indicators_df[indicators_df["indicator"] == indicator]
    vals = {row["bucket"]: row["value"] for _, row in sub.iterrows()}
    needed = ("PK-A", "PK-B", "PD-A", "PD-B")
    if not all(b in vals and not pd.isna(vals[b]) for b in needed):
        return float("nan")
    return (vals["PK-A"] + vals["PD-A"]) / 2 - (vals["PK-B"] + vals["PD-B"]) / 2


# ─── Paper-name aliases (Appendix A.6.2, "Diagnostic-Indicator Plugin") ──
#
# The paper exposes three per-equation entry points:
#
#   * ``compute_kps_f``        (paper Eq. 1: drug-replacement KPS)
#   * ``compute_kps_channel``  (paper Eq. 2: channel-mask KPS, one channel)
#   * ``compute_ksai``         (paper Eq. 3: 2x2 factorial KSAI, LLM-only)
#
# These names appear in :cref:`app-A.6.2:walkthrough` and are the
# documented Python API surface for downstream users adding a new
# baseline or LLM stack.  Internally they delegate to the dict-based
# :func:`compute_indicators` and :func:`compute_baseline_channel_indicators`
# (the "byte-exact upstream port" path), so the underlying math is
# identical and reusable both ways.
#
# The wrappers accept EITHER a prediction dict ``{(a,b): prob}`` OR a
# callable ``predict_fn(pairs) -> np.ndarray`` matching the paper text
# "takes a method's predicted-probability function".  When given a
# callable the wrapper materialises the union pair set from the swap
# candidates, calls it once per condition, and forwards the resulting
# dict.

from typing import Callable, Union

# Accept either {(a,b): prob} or a callable that takes a 2-col DataFrame.
_PredInput = Union[dict, "Callable[[pd.DataFrame], object]"]


def _union_pairs_from_swap(
    swap_candidates: list[SwapTriple],
) -> pd.DataFrame:
    """Materialise ``DataFrame[drug_a_id, drug_b_id]`` covering every
    pair that any indicator in this module looks up: canonical
    ``(qa, qb)`` for both orientations emitted by
    :func:`coldddi.diagnostics.kps_swap.build_swap_candidates`, plus
    ``(qa_prime, qb)`` for the swap targets used by KPS-F."""
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str]] = []
    for t in swap_candidates:
        for a, b in ((t.qa, t.qb), (t.qa_prime, t.qb)):
            key = (str(a), str(b))
            if key not in seen:
                seen.add(key)
                rows.append(key)
    return pd.DataFrame(rows, columns=["drug_a_id", "drug_b_id"])


def _coerce_predictions(
    predictions: _PredInput,
    swap_candidates: list[SwapTriple],
) -> dict[tuple[str, str], float]:
    """Normalise ``predictions`` to the ``{(a, b): prob}`` dict shape
    expected by :func:`compute_indicators` /
    :func:`compute_baseline_channel_indicators`.

    * dict → returned verbatim.
    * callable → invoked on the union pair set derived from
      ``swap_candidates`` (one pass), result zipped into a dict.
      Raises ``ValueError`` if the callable returns fewer or more
      scores than rows — silent truncation via ``zip`` would
      otherwise produce a per-pair dict that's missing predictions
      and corrupt every downstream indicator.

    This is the bridge between the paper's
    "predicted-probability function" API and the existing
    dict-based engine.
    """
    if isinstance(predictions, dict):
        return predictions
    if callable(predictions):
        pairs = _union_pairs_from_swap(swap_candidates)
        scores = predictions(pairs)
        # Materialise length checks against the input pair set so a
        # bug in the callable surfaces here, not silently downstream
        # via ``zip`` truncation.
        scores_seq = list(scores)
        if len(scores_seq) != len(pairs):
            raise ValueError(
                f"predict_fn returned {len(scores_seq)} scores for "
                f"{len(pairs)} pairs; lengths must match."
            )
        a_vals = pairs["drug_a_id"].to_numpy()
        b_vals = pairs["drug_b_id"].to_numpy()
        return {(a, b): float(s) for a, b, s in zip(a_vals, b_vals, scores_seq)}
    raise TypeError(
        f"predictions must be a dict or a callable; got {type(predictions).__name__}"
    )


def compute_kps_f(
    predictions: _PredInput,
    swap_candidates: list[SwapTriple],
    *,
    bucket_fn: Callable[[str, str], str] = lambda a, b: "",
) -> pd.DataFrame:
    """Paper-name alias for **KPS-F** (Drug-Replacement Sensitivity).

    Paper Eq. (1) — Appendix~E.1.  Computes
    ``mean(|P(u, v) - P(u', v)|)`` over the swap-candidate triples,
    per bucket plus an ``ALL`` aggregate over positive base pairs.

    Parameters
    ----------
    predictions
        Either a ``{(drug_a, drug_b): p_yes}`` dict OR a callable
        matching ``predict_fn(pairs_df) -> np.ndarray`` (the paper's
        "predicted-probability function").  The callable, if given,
        is invoked once on the union of swap-anchor and swap-target
        pairs and the result is zipped back into a dict internally.
    swap_candidates
        Output of :func:`build_swap_candidates`.
    bucket_fn
        ``(a, b) -> bucket-label`` callable (typically
        ``BucketLookup.bucket``).

    Returns
    -------
    Long-form DataFrame with rows for ``indicator == "KPS-F"`` only
    (paper-spec narrowing of the full panel returned by
    :func:`compute_indicators`).
    """
    pred = _coerce_predictions(predictions, swap_candidates)
    full = compute_baseline_channel_indicators(
        {"base": pred}, swap_candidates, bucket_fn=bucket_fn,
    )
    return full[full["indicator"] == "KPS-F"].reset_index(drop=True)


def compute_kps_channel(
    predictions_base: _PredInput,
    predictions_masked: _PredInput,
    swap_candidates: list[SwapTriple],
    *,
    channel: str,
    bucket_fn: Callable[[str, str], str] = lambda a, b: "",
) -> pd.DataFrame:
    """Paper-name alias for **KPS-Channel** (Channel-Mask Sensitivity).

    Paper Eq. (2) — Appendix~E.1.  Computes
    ``mean(|P_base(u, v) - P_mask=c(u, v)|)`` per bucket plus ALL
    aggregate over positive base pairs.

    Parameters
    ----------
    predictions_base, predictions_masked
        Either dicts or callables (see :func:`compute_kps_f`).
    channel
        ``"mol"`` or ``"kg"`` — picks which row block to extract
        from :func:`compute_baseline_channel_indicators`.
    """
    # ``compute_baseline_channel_indicators`` emits "KPS-mol" and
    # "KPS-KG" (KG uppercase, mol lowercase) per the paper-script
    # convention; a naïve ``f"KPS-{channel}"`` would produce
    # ``"KPS-kg"`` and silently return an empty DataFrame.  Map
    # explicitly to defend against that bug class.
    indicator_name = {"mol": "KPS-mol", "kg": "KPS-KG"}.get(channel)
    if indicator_name is None:
        raise ValueError(
            f"channel must be 'mol' or 'kg'; got {channel!r}"
        )
    base = _coerce_predictions(predictions_base, swap_candidates)
    masked = _coerce_predictions(predictions_masked, swap_candidates)
    preds = {"base": base, f"mask_{channel}": masked}
    full = compute_baseline_channel_indicators(
        preds, swap_candidates, bucket_fn=bucket_fn,
    )
    return full[full["indicator"] == indicator_name].reset_index(drop=True)


def compute_ksai(
    predictions_r0: _PredInput,
    predictions_r1: _PredInput,
    predictions_r2: _PredInput,
    predictions_r3: _PredInput,
    swap_candidates: list[SwapTriple],
    *,
    bucket_fn: Callable[[str, str], str] = lambda a, b: "",
) -> pd.DataFrame:
    """Paper-name alias for **KSAI** (Cross-Channel Asymmetry, LLM-only).

    Paper Eq. (3) — Appendix~E.1.  Computes the 2x2 factorial
    interaction ``|P_R1 - P_R3| - |P_R0 - P_R2|`` per pair, then
    averages per bucket plus ALL.

    The four R-conditions correspond to the LLM mask factorial
    (paper Table 8 / :mod:`coldddi.llm.prompts`):

    * ``R0`` — no mask          (name present, KG entities present)
    * ``R1`` — name-masked      ([DRUG_A]/[DRUG_B], KG entities present)
    * ``R2`` — KG-entity-masked (name present, [ENTITY])
    * ``R3`` — both masked      ([DRUG_*], [ENTITY])

    Only meaningful for the LLM stack's R0/R1/R2/R3 mask conditions.
    Single-modality baselines and ``mol+kg`` baselines have no
    separable name channel that can be masked independently of the
    KG channel, so they shouldn't be calling this — the underlying
    :func:`compute_indicators` will produce NaN row blocks when the
    R-conditions don't match the 2x2 factorial.
    """
    coerce = lambda x: _coerce_predictions(x, swap_candidates)
    preds = {
        "R0": coerce(predictions_r0),
        "R1": coerce(predictions_r1),
        "R2": coerce(predictions_r2),
        "R3": coerce(predictions_r3),
    }
    full = compute_indicators(preds, swap_candidates, bucket_fn=bucket_fn)
    return full[full["indicator"] == "KSAI"].reset_index(drop=True)


__all__ = [
    "PRIMARY_BUCKETS",
    "LLM_INDICATOR_NAMES",
    "BASELINE_CHANNEL_INDICATOR_NAMES",
    "INDICATOR_NAMES",
    "compute_indicators",
    "compute_baseline_channel_indicators",
    "compute_ab_gap",
    # Paper-name aliases (A.6.2 walkthrough).
    "compute_kps_f",
    "compute_kps_channel",
    "compute_ksai",
]
