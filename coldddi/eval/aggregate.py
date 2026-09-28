"""Cross-method / cross-seed aggregators for ``runs/`` artefacts.

Backs the four paper-promised reproducibility helpers
(``exps/sec5_{overall,stratified,kps,masking}.sh``) by walking the
``runs/<method>/seed<N>/`` directory tree that ``evaluate.py``
populates and rolling the per-pair predictions / per-bucket
indicators up into the final paper-table shape.

Three aggregation modes, mirroring paper §5:

* :func:`aggregate_overall`     — overall S2 AUC / AUPRC per method,
                                  mean ± std across 3 seeds. Paper
                                  Table for §5.2 (Cold-start results).
* :func:`aggregate_stratified` — per-bucket (PK-A/PK-B/PD-A/PD-B)
                                  AUC per method, mean ± std across
                                  seeds. Paper §5.3 (Stratified
                                  analysis).
* :func:`aggregate_kps`         — KPS-F / KPS-mol / KPS-KG per
                                  (method, bucket), mean ± std across
                                  seeds. Paper §5.4 (KPS table).

All three return a long-form ``pandas.DataFrame`` and also write
a CSV under ``runs/sec5_<mode>.csv`` for downstream tooling.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: ``runs/<method>/seed<N>/`` regex used to walk the artefact tree.
_RUN_DIR_RE = re.compile(r"^seed(\d+)$")


def _iter_method_seed_dirs(
    runs_root: Path,
    methods: Iterable[str] | None = None,
    *,
    flat_method: str | None = None,
) -> Iterable[tuple[str, int, Path]]:
    """Yield ``(method, seed, dir)`` for every seed dir under ``runs_root``.

    Two layouts supported:

    * **nested** (default): ``runs_root/<method>/seed<N>/``.  Used by
      the baseline path (``runs/<method>/seed<N>/``).
    * **flat** (``flat_method`` set): ``runs_root/seed<N>/`` directly.
      Used by the LLM path where the model + prompt cell already
      identifies the method, so ``runs_root`` is the cell directory
      itself (e.g. ``runs/llama-1b/P4``).  The ``flat_method`` arg
      gives the synthetic method label to emit (e.g.
      ``"llama-1b_P4"``) so downstream aggregation keeps cell identity.
    """
    if not runs_root.is_dir():
        return

    if flat_method is not None:
        for seed_dir in sorted(runs_root.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = _RUN_DIR_RE.match(seed_dir.name)
            if m is None:
                continue
            yield flat_method, int(m.group(1)), seed_dir
        return

    for method_dir in sorted(runs_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if methods is not None and method not in methods:
            continue
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = _RUN_DIR_RE.match(seed_dir.name)
            if m is None:
                continue
            yield method, int(m.group(1)), seed_dir


# ─── Overall AUC ────────────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC with graceful fallback when only one class present.

    Single-class folds happen on tiny smoke fixtures — return NaN so
    the downstream aggregation skips them rather than crashing the
    whole table.
    """
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def aggregate_overall(
    runs_root: Path,
    *,
    setting: str = "test_s2",
    methods: Iterable[str] | None = None,
    flat_method: str | None = None,
) -> pd.DataFrame:
    """Roll up per-method overall AUC / AUPRC across seeds.

    Reads ``predictions_<setting>_seed{N}.csv`` from every seed
    directory under each method.  Computes AUC + AUPRC per seed,
    aggregates mean ± std across seeds.  Returns long-form
    DataFrame ``[method, metric, mean, std, n_seeds, seeds]``.
    """
    rows: list[dict] = []
    for method, seed, seed_dir in _iter_method_seed_dirs(
        runs_root, methods, flat_method=flat_method,
    ):
        csv = seed_dir / f"predictions_{setting}_seed{seed}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        y_true = df["true_label"].to_numpy()
        y_score = df["predicted_prob"].to_numpy()
        rows.append({
            "method": method, "seed": seed,
            "metric": "AUROC",
            "value": _safe_auc(y_true, y_score),
        })
        rows.append({
            "method": method, "seed": seed,
            "metric": "AUPRC",
            "value": _safe_auprc(y_true, y_score),
        })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return pd.DataFrame(
            columns=["method", "metric", "mean", "std", "n_seeds", "seeds"]
        )
    out_rows: list[dict] = []
    for (m, met), g in per_seed.groupby(["method", "metric"]):
        non_nan = g.dropna(subset=["value"])
        vals = non_nan["value"].to_numpy()
        if len(vals) == 0:
            continue
        # ``seeds`` lists only the seeds whose AUROC/AUPRC was
        # actually finite — single-class splits drop out of the
        # mean AND the seed list so the audit string matches n_seeds.
        out_rows.append({
            "method": m,
            "metric": met,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": int(len(vals)),
            "seeds": ",".join(str(s) for s in sorted(non_nan["seed"].unique())),
        })
    return pd.DataFrame(out_rows)


# ─── Stratified AUC ─────────────────────────────────────────────────

def aggregate_stratified(
    runs_root: Path,
    *,
    ab_parquet: Path,
    setting: str = "test_s2",
    methods: Iterable[str] | None = None,
    flat_method: str | None = None,
) -> pd.DataFrame:
    """Roll up per-bucket AUC per method, mean ± std across seeds.

    Reads ``predictions_<setting>_seed{N}.csv`` and joins each pair
    against ``ab_parquet`` (via :func:`coldddi.diagnostics.build_bucket_lookup`)
    to assign PK-A / PK-B / PD-A / PD-B labels.  Computes AUC per
    (method, seed, bucket) then aggregates across seeds.
    """
    from coldddi.diagnostics import build_bucket_lookup

    lookup = build_bucket_lookup(ab_parquet)
    rows: list[dict] = []
    for method, seed, seed_dir in _iter_method_seed_dirs(
        runs_root, methods, flat_method=flat_method,
    ):
        csv = seed_dir / f"predictions_{setting}_seed{seed}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        df["bucket"] = [
            lookup.bucket(a, b)
            for a, b in zip(
                df["drug_a_id"].astype(str),
                df["drug_b_id"].astype(str),
            )
        ]
        for bk in ("PK-A", "PK-B", "PD-A", "PD-B"):
            sub = df[df["bucket"] == bk]
            if len(sub) < 2:
                continue
            rows.append({
                "method": method, "seed": seed, "bucket": bk,
                "value": _safe_auc(
                    sub["true_label"].to_numpy(),
                    sub["predicted_prob"].to_numpy(),
                ),
            })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return pd.DataFrame(
            columns=["method", "bucket", "mean", "std", "n_seeds", "seeds"]
        )
    out_rows: list[dict] = []
    for (m, b), g in per_seed.groupby(["method", "bucket"]):
        non_nan = g.dropna(subset=["value"])
        vals = non_nan["value"].to_numpy()
        if len(vals) == 0:
            continue
        out_rows.append({
            "method": m, "bucket": b,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": int(len(vals)),
            "seeds": ",".join(str(s) for s in sorted(non_nan["seed"].unique())),
        })
    return pd.DataFrame(out_rows)


# ─── KPS aggregation ────────────────────────────────────────────────

def aggregate_kps(
    runs_root: Path,
    *,
    methods: Iterable[str] | None = None,
    flat_method: str | None = None,
) -> pd.DataFrame:
    """Roll up per-(method, indicator, bucket) KPS / KSAI values
    across seeds.

    Reads ``indicators_test_s2_seed{N}.csv`` (written by
    :func:`coldddi.evaluate._run_indicators_for_test_s2`).  Aggregates
    mean ± std per (method, indicator, bucket).
    """
    rows: list[dict] = []
    for method, seed, seed_dir in _iter_method_seed_dirs(
        runs_root, methods, flat_method=flat_method,
    ):
        csv = seed_dir / f"indicators_test_s2_seed{seed}.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        df["method"] = method
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=[
                "method", "indicator", "bucket",
                "mean", "std", "n_seeds", "seeds",
            ]
        )
    per_seed = pd.concat(rows, ignore_index=True)
    out_rows: list[dict] = []
    for (m, ind, bk), g in per_seed.groupby(["method", "indicator", "bucket"]):
        non_nan = g.dropna(subset=["value"])
        vals = non_nan["value"].to_numpy()
        if len(vals) == 0:
            # All-NaN block (single-modality baseline KPS-mol/KPS-KG)
            # — propagate one NaN row per (method, indicator, bucket)
            # so the paper table prints "—" cells consistently.  The
            # ``seeds`` field still lists every seed that contributed
            # the NaN row (useful for audit: "all 3 seeds were NaN,
            # not just one") — for non-NaN rows ``seeds`` only lists
            # the contributing seeds.
            out_rows.append({
                "method": m, "indicator": ind, "bucket": bk,
                "mean": float("nan"), "std": float("nan"),
                "n_seeds": 0,
                "seeds": ",".join(str(s) for s in sorted(g["seed"].unique())),
            })
            continue
        out_rows.append({
            "method": m, "indicator": ind, "bucket": bk,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": int(len(vals)),
            "seeds": ",".join(str(s) for s in sorted(non_nan["seed"].unique())),
        })
    return pd.DataFrame(out_rows)


# ─── CLI dispatcher ─────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m coldddi.eval.aggregate {overall,stratified,kps} ...``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="coldddi.eval.aggregate",
        description=(
            "Roll up per-method per-seed evaluate.py artefacts into "
            "the paper-table shape (sec5_{overall,stratified,kps}.sh)."
        ),
    )
    parser.add_argument(
        "mode", choices=("overall", "stratified", "kps"),
        help="Which aggregator to run.",
    )
    parser.add_argument(
        "--runs", type=Path, required=True,
        help="runs/ directory containing <method>/seed<N>/ subdirs.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. Defaults to runs/sec5_<mode>.csv.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Restrict to these methods (default: all under --runs).",
    )
    parser.add_argument(
        "--ab-parquet", type=Path, default=None,
        help="Required for --mode stratified.",
    )
    parser.add_argument(
        "--setting", default="test_s2",
        help="Split to aggregate (overall/stratified only).",
    )
    parser.add_argument(
        "--method", default=None,
        help=(
            "Flat-mode method label.  When set, --runs is treated as "
            "a cell directory whose children are seed<N>/ dirs "
            "directly (no <method>/ level).  Used by the LLM masking "
            "path: sec5_masking.sh passes --runs runs/<model>/<prompt> "
            "--method <model>_<prompt>."
        ),
    )
    args = parser.parse_args(argv)

    # --methods (plural, restrict to a subset of registered methods)
    # is irrelevant in --method (singular, flat-mode) since the latter
    # forces a synthetic single label.  Reject the combination so a
    # user that meant one doesn't silently get the other.
    if args.method is not None and args.methods is not None:
        parser.error(
            "--method and --methods are mutually exclusive: --method "
            "is the flat-mode composite label, --methods restricts "
            "to a subset of nested-mode method dirs."
        )

    if args.mode == "overall":
        df = aggregate_overall(
            args.runs, setting=args.setting,
            methods=args.methods, flat_method=args.method,
        )
    elif args.mode == "stratified":
        if args.ab_parquet is None:
            parser.error("--ab-parquet is required for --mode stratified")
        df = aggregate_stratified(
            args.runs, ab_parquet=args.ab_parquet,
            setting=args.setting,
            methods=args.methods, flat_method=args.method,
        )
    else:
        df = aggregate_kps(
            args.runs, methods=args.methods, flat_method=args.method,
        )

    out = args.out if args.out is not None else args.runs / f"sec5_{args.mode}.csv"
    df.to_csv(out, index=False)
    print(f"[aggregate] {args.mode} → {out} ({len(df)} rows)", file=sys.stderr)
    # Also pretty-print to stdout for quick inspection.
    if not df.empty:
        with pd.option_context("display.max_rows", 200, "display.width", 140):
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
