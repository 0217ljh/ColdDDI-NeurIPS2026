"""KPS / KSAI diagnostic indicators (paper §4.3 / Table 6).

Sources:

* ``Code-Released/exps/sec5-3/2_indicators/compute_llm_indicators_3seed.py``
* ``Code-Released/exps/sec5-3/2_indicators/compute_baseline_kpsf_3seed.py``
* ``Code-Released/exps/sec5-3/2_indicators/compute_baseline_kps_channels_3seed.py``

LLM stack (R0=baseline / R1=mask name / R2=mask entity / R3=mask both):

* ``KPS-F``              — drug replacement over R0 swap triples (u, v, u').
* ``KPS-Name``           — ``mean(|R0 - R1|)`` over base pairs.
* ``KPS-KG``             — ``mean(|R0 - R2|)`` over base pairs.
* ``KPS-KG-Named``       — alias of ``KPS-KG``.
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

Buckets follow upstream ``_agg_buckets``:

* Primary buckets ``PK-A``, ``PK-B``, ``PD-A``, ``PD-B`` come from the
  annotation table. Unknown or "Mixed" mechanisms have no primary row;
  no ``"Other"`` bucket is emitted.
* ``ALL`` includes only ``label_uv == 1`` rows (positive base pairs),
  including those without a primary bucket.

:func:`compute_ab_gap` computes::

    A-B gap = (PK-A + PD-A) / 2  -  (PK-B + PD-B) / 2

A positive gap means greater sensitivity on pairs with a confirmed
mediating entity in the KG (Type-A); a negative gap means greater
sensitivity on Type-B pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from coldddi.diagnostics.kps_swap import SwapTriple


#: Primary buckets emitted alongside ``ALL``; upstream omits ``"Other"``.
PRIMARY_BUCKETS: tuple[str, ...] = ("PK-A", "PK-B", "PD-A", "PD-B")


#: Indicators from :func:`compute_indicators` for LLM R0/R1/R2/R3.
LLM_INDICATOR_NAMES: tuple[str, ...] = (
    "KPS-F",
    "KPS-Name",
    "KPS-KG",
    "KPS-KG-Named",       # Paper alias of KPS-KG.
    "KPS-KG-Masked",
    "KPS-Name-KGMasked",
    "KSAI",
)


#: Indicators for mol+KG-separable baselines (MKG-FENN, TIGER).
BASELINE_CHANNEL_INDICATOR_NAMES: tuple[str, ...] = (
    "KPS-F",
    "KPS-mol",
    "KPS-KG",
)


#: Backward-compatible alias.
INDICATOR_NAMES = LLM_INDICATOR_NAMES


def _lookup_directed(d: dict, a: str, b: str):
    """``d[(a, b)]`` with reverse-direction fallback.

    Prefer upstream's exact key, but accept ``(b, a)`` when only the
    reverse prediction exists. Otherwise swap triples can be lost.
    """
    if (a, b) in d:
        return d[(a, b)]
    if (b, a) in d:
        return d[(b, a)]
    return None


def _agg_buckets(records: list[dict], value_col: str, indicator: str) -> list[dict]:
    """Aggregate as in ``compute_*_3seed.py::_agg_buckets``.

    Records have keys ``bucket`` (:data:`PRIMARY_BUCKETS` or ``""``),
    ``label_uv`` (0 or 1), and ``value_col`` (per-row delta).
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

    # ALL includes only positive base pairs (label_uv == 1), as upstream.
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
    """Return NaN rows for each primary bucket and ALL for a missing condition."""
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


def _unique_base_pairs(
    swap_candidates: Iterable[SwapTriple],
    bucket_fn: Callable[[str, str], str],
) -> list[dict]:
    """Deduplicate by ``(qa, qb)`` and attach ``bucket`` and ``label_uv``.

    Like ``swap_df`` in
    ``compute_llm_indicators_3seed.py::compute_indicators_for_seed``,
    use one row per base pair, not per triple. Keep empty/unknown buckets:
    :func:`_agg_buckets` excludes them from primary rows but includes
    them in ALL when ``label_uv == 1``.
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
        ``R0`` supplies base predictions. Other conditions may be ``None``
        or missing; dependent channel/KSAI indicators return NaN rows
        for every bucket.
    swap_candidates
        Output of :func:`coldddi.diagnostics.kps_swap.build_swap_candidates`.
        KPS-F uses one row per triple; channel/KSAI indicators use
        the deduplicated base pairs.
    bucket_fn
        ``(qa, qb) -> "PK-A" | "PK-B" | "PD-A" | "PD-B" | <anything else>``.
        Non-primary pairs contribute only to ALL, when ``label_uv == 1``.
    coverage_warnings
        Print per-indicator coverage to stderr when ``True`` (default:
        ``False``). Missing predictions skip records, so enable this to
        detect incomplete coverage of ``swap_candidates``.

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

    # KPS-F uses R0 predictions and counts each swap triple.
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

    # Channel/KSAI metrics count unique base pairs, not triples.
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

    # KSAI: compute (|R1-R3| - |R0-R2|) per pair, then average.
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
    """Print per-indicator kept/skipped counts; missing predictions skip records."""
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

    Missing keys produce NaN rows for the affected indicator.

    ``coverage_warnings=True`` prints per-indicator kept/skipped/total
    counts to stderr, as in :func:`compute_indicators`.
    """
    base = predictions.get("base") or {}
    mol = predictions.get("mask_mol")
    kg = predictions.get("mask_kg")

    out_rows: list[dict] = []
    coverage: dict[str, dict[str, int]] = {}

    # KPS-F counts swap triples, as in the LLM path.
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


def compute_ab_gap(indicators_df: pd.DataFrame, indicator: str) -> float:
    """Return ``(PK-A + PD-A)/2 - (PK-B + PD-B)/2`` for one indicator.

    Positive means greater sensitivity on confirmed (A) than unconfirmed
    (B) pairs. Returns ``float('nan')`` if any primary bucket is missing
    or NaN for this indicator.
    """
    sub = indicators_df[indicators_df["indicator"] == indicator]
    vals = {row["bucket"]: row["value"] for _, row in sub.iterrows()}
    needed = ("PK-A", "PK-B", "PD-A", "PD-B")
    if not all(b in vals and not pd.isna(vals[b]) for b in needed):
        return float("nan")
    return (vals["PK-A"] + vals["PD-A"]) / 2 - (vals["PK-B"] + vals["PD-B"]) / 2


# Paper API wrappers (Appendix A.6.2, "Diagnostic-Indicator Plugin").
# Each accepts a prediction dict or calls predict_fn once per condition
# on the union of swap-anchor and swap-target pairs.

from typing import Callable, Union

# Accept either {(a,b): prob} or a callable that takes a 2-col DataFrame.
_PredInput = Union[dict, "Callable[[pd.DataFrame], object]"]


def _union_pairs_from_swap(
    swap_candidates: list[SwapTriple],
) -> pd.DataFrame:
    """Return ``DataFrame[drug_a_id, drug_b_id]`` for all indicator lookups.

    Include ``(qa, qb)`` in both orientations emitted by
    :func:`coldddi.diagnostics.kps_swap.build_swap_candidates` and
    the KPS-F swap targets ``(qa_prime, qb)``.
    """
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
    """Return ``{(a, b): prob}`` predictions for the indicator functions.

    Return dicts unchanged. Call a predictor once on the union pair set
    from ``swap_candidates`` and zip its scores into a dict. Raise
    ``ValueError`` unless there is exactly one score per row; otherwise
    ``zip`` truncation would lose predictions and bias the indicators.
    """
    if isinstance(predictions, dict):
        return predictions
    if callable(predictions):
        pairs = _union_pairs_from_swap(swap_candidates)
        scores = predictions(pairs)
        # Reject length mismatches before zip can silently truncate scores.
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
    """KPS-F (Drug-Replacement Sensitivity), paper Eq. (1), Appendix E.1.

    Compute ``mean(|P(u, v) - P(u', v)|)`` over swap triples per bucket,
    plus an ``ALL`` aggregate over positive base pairs.

    Parameters
    ----------
    predictions
        ``{(drug_a, drug_b): p_yes}`` or ``predict_fn(pairs_df) -> np.ndarray``.
        Callables run once on the union of swap-anchor and swap-target
        pairs and must return one score per row.
    swap_candidates
        Output of :func:`build_swap_candidates`.
    bucket_fn
        ``(a, b) -> bucket-label`` callable (typically
        ``BucketLookup.bucket``).

    Returns
    -------
    Long-form DataFrame containing only ``indicator == "KPS-F"`` rows.
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
    """KPS-Channel (Channel-Mask Sensitivity), paper Eq. (2), Appendix E.1.

    Compute ``mean(|P_base(u, v) - P_mask=c(u, v)|)`` per bucket,
    plus an ALL aggregate over positive base pairs.

    Parameters
    ----------
    predictions_base, predictions_masked
        Either dicts or callables (see :func:`compute_kps_f`).
    channel
        ``"mol"`` or ``"kg"`` — picks which row block to extract
        from :func:`compute_baseline_channel_indicators`.
    """
    # Paper names use lowercase "mol" but uppercase "KG"; "KPS-kg"
    # would select no rows from compute_baseline_channel_indicators.
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
    """KSAI (Cross-Channel Asymmetry, LLM-only), paper Eq. (3), Appendix E.1.

    Compute the 2x2 factorial interaction
    ``|P_R1 - P_R3| - |P_R0 - P_R2|`` per pair, then average per bucket
    and over positive base pairs for ALL.

    The four R-conditions correspond to the LLM mask factorial
    (paper Table 8 / :mod:`coldddi.llm.prompts`):

    * ``R0`` — no mask          (name present, KG entities present)
    * ``R1`` — name-masked      ([DRUG_A]/[DRUG_B], KG entities present)
    * ``R2`` — KG-entity-masked (name present, [ENTITY])
    * ``R3`` — both masked      ([DRUG_*], [ENTITY])

    Requires independently masked name and KG channels, so it does not
    apply to single-modality or ``mol+kg`` baselines. Missing required
    R-conditions produce NaN rows in :func:`compute_indicators`.
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
    "compute_kps_f",
    "compute_kps_channel",
    "compute_ksai",
]
