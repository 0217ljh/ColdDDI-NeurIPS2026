"""Top-level evaluator — paper §A.6.3 entry point.

Dispatches to a registered :class:`coldddi.baselines.BaselineModel`,
fits it on a :class:`PairDataset`, runs evaluation on the requested
setting (S0 / S1 / S2 / all), and writes per-split metrics to ``--out``.

Layer 1 implements the CLI surface and the fit / score / save loop;
the actual metrics are computed by helpers from
:mod:`coldddi.eval.metrics` (added in a later layer).

CLI
---
::

    python evaluate.py \\
      --method {deepddi,ssi-ddi,dsn-ddi,hdn-ddi,emergnn,tiger,mkg-fenn,textddi} \\
      (--data DIR | --subset {800,1900}) \\
      --seed INT \\
      [--setting {S0,S1,S2,all}]   # default: all
      [--device {cuda,cpu,auto}]   # default: cuda
      [--out DIR]                  # default: runs/<method>/seed<N>/
      [--checkpoint PATH | --adapter PATH]
      [--ab-parquet PATH] [--no-indicators]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import inspect

from coldddi.baselines import (
    BaselineModel,
    NAME_TO_MODULE,
    ensure_imported,
    list_baselines,
    load_baseline,
)
from coldddi.baselines.base import _REGISTRY
from coldddi.data.dataset import PairDataset

ALL_SETTINGS: tuple[str, ...] = ("S0", "S1", "S2")


def _setting_splits(setting: str) -> tuple[str, str]:
    """Map a setting label to its (val_split_name, test_split_name)."""
    s = setting.lower()
    return f"val_{s}", f"test_{s}"


#: Canonical column order for per-pair prediction CSVs written by
#: :func:`run_evaluation`.  Matches the schema upstream baseline
#: inference CSVs use (``Code-Released/baseline/Output/<method>/my/
#: seed{seed}/s{1,2}/inference_*.csv``) modulo column names: release
#: canonical ``drug_a_id`` / ``drug_b_id`` instead of upstream's
#: ``d1`` / ``d2``.  A one-line ``df.rename(columns={"drug_a_id":
#: "d1", "drug_b_id": "d2"})`` converts to upstream layout.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "drug_a_id",
    "drug_b_id",
    "true_label",
    "predicted_prob",
    "predicted_label",
)


def _predictions_to_df(
    model: BaselineModel,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
) -> pd.DataFrame:
    """Score positives + negatives once and return a long-form table.

    Used by :func:`_evaluate_split_with_predictions` so the caller
    pays exactly one ``predict_proba`` call per (pos, neg) pair set
    even when both aggregate metrics AND per-pair predictions are
    needed for downstream L6 diagnostics.

    Threshold ``predicted_prob >= 0.5`` mirrors the upstream baseline
    inference CSV convention.
    """
    frames: list[pd.DataFrame] = []
    if len(pos):
        scores_pos = model.predict_proba(pos[["drug_a_id", "drug_b_id"]])
        frames.append(pd.DataFrame({
            "drug_a_id": pos["drug_a_id"].astype(str).to_numpy(),
            "drug_b_id": pos["drug_b_id"].astype(str).to_numpy(),
            "true_label": 1,
            "predicted_prob": scores_pos,
            "predicted_label": (scores_pos >= 0.5).astype(int),
        }))
    if len(neg):
        scores_neg = model.predict_proba(neg[["drug_a_id", "drug_b_id"]])
        frames.append(pd.DataFrame({
            "drug_a_id": neg["drug_a_id"].astype(str).to_numpy(),
            "drug_b_id": neg["drug_b_id"].astype(str).to_numpy(),
            "true_label": 0,
            "predicted_prob": scores_neg,
            "predicted_label": (scores_neg >= 0.5).astype(int),
        }))
    if not frames:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(PREDICTION_COLUMNS)]


def _evaluate_split_with_predictions(
    model: BaselineModel,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute aggregate metrics AND return per-pair predictions.

    Single ``predict_proba`` pass per (pos, neg) — the aggregate stats
    are derived from the same DataFrame written to disk, so the
    metrics JSON and predictions CSV are guaranteed consistent.
    """
    df = _predictions_to_df(model, pos, neg)
    metrics: dict[str, float] = {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
    if len(df):
        pos_rows = df[df["true_label"] == 1]
        neg_rows = df[df["true_label"] == 0]
        if len(pos_rows):
            metrics["mean_pos_score"] = float(pos_rows["predicted_prob"].mean())
        metrics["mean_neg_score"] = (
            float(neg_rows["predicted_prob"].mean()) if len(neg_rows) else None
        )
    return metrics, df


def _evaluate_split(
    model: BaselineModel,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
) -> dict[str, float]:
    """Backward-compat shim — returns only the aggregate metrics dict.

    Kept so any external caller relying on the original signature
    (returns ``dict[str, float]``) keeps working.  New code should
    prefer :func:`_evaluate_split_with_predictions`, which adds a
    per-pair DataFrame at no extra inference cost.
    """
    metrics, _ = _evaluate_split_with_predictions(model, pos, neg)
    return metrics


# ─── L6 indicator helpers (Step 2 of the A.6.2 contract) ─────────────


#: Modality → list of mask channels the L6 dispatch should run.
#:
#: This is the single keyword-based dispatch the user requested when
#: they asked us to "mark modality at registration time so the
#: indicator step can look it up by keyword".  Lives here (not in the
#: ABC) because the values are evaluator-pipeline concerns; the ABC
#: only knows what the model declares.
#:
#: Invariant: ``set(MODALITY_MASK_CHANNELS) == set(MODALITIES)``.
#: A test in ``tests/test_evaluate_indicators_e2e.py`` pins it so a
#: future modality added to :data:`coldddi.baselines.base.MODALITIES`
#: cannot land without a dispatch entry here.
MODALITY_MASK_CHANNELS: dict[str, tuple[str, ...]] = {
    "mol":          (),
    "text":         (),
    "mol+kg-fused": (),         # fused architecture; no separable mask
    "mol+kg":       ("mol", "kg"),
}


def _resolve_ab_parquet(
    explicit: Path | None,
    data_dir: Path,
) -> Path | None:
    """Locate the A/B annotation parquet for L6 bucket assignment.

    Search order:

    1. ``--ab-parquet PATH`` (if user supplied).
    2. ``<data_dir>/annotations/ab.parquet`` (full release).
    3. ``<data_dir>/annotations/ab_sample.parquet`` (toy).
    4. ``<repo_root>/annotations/ab.parquet`` (full).
    5. ``<repo_root>/annotations/ab_sample.parquet`` (toy).

    Returns ``None`` if nothing found — caller skips L6 with a warning
    instead of crashing.  This matches the paper convention that L6 is
    optional in a single ``evaluate.py`` invocation: ``annotations/``
    is shipped with the release artefacts and the auto-discovery hits
    it without any flag.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(
                f"--ab-parquet {p} does not exist"
            )
        return p
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        Path(data_dir) / "annotations" / "ab.parquet",
        Path(data_dir) / "annotations" / "ab_sample.parquet",
        repo_root / "annotations" / "ab.parquet",
        repo_root / "annotations" / "ab_sample.parquet",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _predict_dict(
    model: BaselineModel,
    pairs: pd.DataFrame,
    *,
    mask_channel: str | None = None,
) -> dict[tuple[str, str], float]:
    """Score ``pairs`` and return ``{(drug_a, drug_b): p_yes}``.

    ``mask_channel`` is forwarded to ``predict_proba`` when set — the
    caller is responsible for only doing so when the baseline's
    modality declares the channel is supported.
    """
    if mask_channel is None:
        scores = model.predict_proba(pairs[["drug_a_id", "drug_b_id"]])
    else:
        scores = model.predict_proba(
            pairs[["drug_a_id", "drug_b_id"]], mask_channel=mask_channel,
        )
    out: dict[tuple[str, str], float] = {}
    a_vals = pairs["drug_a_id"].astype(str).to_numpy()
    b_vals = pairs["drug_b_id"].astype(str).to_numpy()
    for a, b, s in zip(a_vals, b_vals, scores):
        out[(a, b)] = float(s)
    return out


def _dict_to_predictions_df(
    pairs: pd.DataFrame,
    pred: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Materialise a per-pair predictions DataFrame from a score dict.

    Used to write the mask-mode CSVs (``predictions_<split>_mask_*_
    seed{N}.csv``) so they share the canonical schema with the base
    predictions CSV — making the L6 inputs reproducible by any user
    who wants to re-run :mod:`coldddi.diagnostics` directly without
    re-training the baseline.
    """
    a_vals = pairs["drug_a_id"].astype(str).to_numpy()
    b_vals = pairs["drug_b_id"].astype(str).to_numpy()
    probs = np.array(
        [pred.get((a, b), float("nan")) for a, b in zip(a_vals, b_vals)],
        dtype=np.float32,
    )
    return pd.DataFrame({
        "drug_a_id": a_vals,
        "drug_b_id": b_vals,
        "true_label": -1,                              # not applicable for mask CSV
        "predicted_prob": probs,
        "predicted_label": (probs >= 0.5).astype(int),
    })[list(PREDICTION_COLUMNS)]


def _run_indicators_for_test_s2(
    *,
    model: BaselineModel,
    ds: "PairDataset",
    ab_parquet: Path,
    out_dir: Path,
    seed: int,
) -> Path:
    """Compute KPS-F (+ KPS-mol / KPS-KG for mol+KG baselines) on test_s2.

    Reads the ``model.modality`` attribute (instance preferred over
    class default; see :class:`coldddi.baselines.base.BaselineModel`)
    to decide how many ``predict_proba`` passes to make:

    * Single-modality baselines (``modality`` in :data:`MODALITIES`
      whose entry in :data:`MODALITY_MASK_CHANNELS` is ``()``):
      one base pass; channel indicators come back as NaN rows.
    * mol+KG-separable (``modality == "mol+kg"``): three passes —
      base, mask_mol, mask_kg — and all three indicators land
      populated.

    Returns the path of the ``indicators_test_s2_seed{N}.csv`` file
    written under ``out_dir``.

    Auxiliary outputs (also written to ``out_dir``):

    * For mol+KG baselines: ``predictions_test_s2_mask_mol_seed{N}.csv``
      and ``predictions_test_s2_mask_kg_seed{N}.csv`` — per-pair mask
      predictions over the union (canonical test_s2 base ∪ swap-anchor
      ``(qa, qb)`` for both orientations ∪ swap-target ``(qa_prime, qb)``
      pairs).  The base predictions over test_s2 are NOT re-written
      here (those already exist as ``predictions_test_s2_seed{N}.csv``
      from the preceding split-evaluation loop).
    """
    # Lazy import — diagnostics pulls in pandas-heavy machinery we don't
    # want loaded on every ``coldddi.evaluate`` import.
    from coldddi.diagnostics import (
        build_bucket_lookup,
        build_swap_candidates,
        compute_baseline_channel_indicators,
    )

    print(f"[evaluate][L6] using ab-parquet {ab_parquet}", file=sys.stderr)
    bucket_lookup = build_bucket_lookup(ab_parquet)
    swap = build_swap_candidates(ds, source_split="test_s2")
    print(
        f"[evaluate][L6] swap-candidate triples: {len(swap)} "
        f"(pos={sum(1 for t in swap if t.label_uv == 1)}, "
        f"neg={sum(1 for t in swap if t.label_uv == 0)})",
        file=sys.stderr,
    )

    # Pair union: canonical test_s2 base ∪ swap-anchor (qa, qb) for
    # BOTH orientations emitted by ``build_swap_candidates`` ∪
    # swap-target (qa_prime, qb).  Including the swap-anchor reverses
    # is critical: ``build_swap_candidates`` emits triples in both
    # orientations ((da, db) AND (db, da)), and not every baseline's
    # ``predict_proba`` is order-invariant.  Without the explicit
    # reverse-anchor coverage, ``compute_baseline_channel_indicators``
    # would fall back via :func:`_lookup_directed` to the canonical-
    # orientation prediction, which silently assumes symmetry —
    # untrue for e.g. TIGER's BKG random-walk subgraph where the
    # head-drug position affects the subgraph context.
    test_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    test_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    base_pairs = pd.concat([test_pos, test_neg], ignore_index=True)
    swap_anchor_pairs = pd.DataFrame(
        [{"drug_a_id": t.qa, "drug_b_id": t.qb} for t in swap],
        columns=["drug_a_id", "drug_b_id"],
    )
    swap_target_pairs = pd.DataFrame(
        [{"drug_a_id": t.qa_prime, "drug_b_id": t.qb} for t in swap],
        columns=["drug_a_id", "drug_b_id"],
    )
    union = (
        pd.concat(
            [base_pairs, swap_anchor_pairs, swap_target_pairs],
            ignore_index=True,
        )
        .drop_duplicates(subset=["drug_a_id", "drug_b_id"])
        .reset_index(drop=True)
    )
    print(
        f"[evaluate][L6] union pairs: {len(union)} "
        f"(test_s2 base {len(base_pairs)}, "
        f"swap anchors {len(swap_anchor_pairs)}, "
        f"swap targets {len(swap_target_pairs)})",
        file=sys.stderr,
    )

    # Keyword-based dispatch (the user-requested style): read modality
    # once, decide what passes to make, no signature introspection.
    # Instance-level ``modality`` overrides the class-level default so
    # mode-dependent baselines (e.g. TIGER with ``mol_only=True``) can
    # downgrade themselves to single-modality at construction time
    # without us hard-coding their internals here.
    modality = getattr(model, "modality", "mol")
    # Strict lookup: if a baseline ever declares an unknown modality
    # that slipped past ``register()`` validation, fail loudly here
    # instead of silently dispatching as single-modality.
    if modality not in MODALITY_MASK_CHANNELS:
        raise KeyError(
            f"baseline modality {modality!r} has no MODALITY_MASK_CHANNELS "
            f"entry; registered dispatch keys are {sorted(MODALITY_MASK_CHANNELS)}"
        )
    mask_channels = MODALITY_MASK_CHANNELS[modality]
    print(
        f"[evaluate][L6] modality={modality!r} → mask_channels={mask_channels}",
        file=sys.stderr,
    )

    predictions: dict[str, dict[tuple[str, str], float] | None] = {
        "base": _predict_dict(model, union),
    }
    # Persist the L6 base-union predictions to a separate CSV.  This is
    # the EXACT input the indicator math saw (test_s2 base ∪ swap
    # anchors ∪ swap targets) and is what downstream hand-recompute
    # tests + users replaying L6 without re-training need.
    # ``predictions_test_s2_seed{N}.csv`` (written by the split-eval
    # loop) covers test_s2 only, a strict subset — re-using it for
    # KPS-F hand-recompute would under-count any triple whose
    # qa_prime is unseen in test_s2.
    union_csv = out_dir / f"predictions_test_s2_union_seed{seed}.csv"
    _dict_to_predictions_df(union, predictions["base"]).to_csv(
        union_csv, index=False,
    )
    print(
        f"[evaluate][L6] base union predictions → {union_csv.name} "
        f"(n={len(predictions['base'])})",
        file=sys.stderr,
    )
    for ch in mask_channels:
        pred = _predict_dict(model, union, mask_channel=ch)
        predictions[f"mask_{ch}"] = pred
        # Persist the mask-mode predictions so L6 can be re-run later
        # without re-training the baseline.
        mask_csv = out_dir / f"predictions_test_s2_mask_{ch}_seed{seed}.csv"
        _dict_to_predictions_df(union, pred).to_csv(mask_csv, index=False)
        print(
            f"[evaluate][L6] mask_{ch} predictions → {mask_csv.name} "
            f"(n={len(pred)})",
            file=sys.stderr,
        )

    indicators_df = compute_baseline_channel_indicators(
        predictions, swap, bucket_fn=bucket_lookup.bucket,
        coverage_warnings=True,
    )
    out_csv = out_dir / f"indicators_test_s2_seed{seed}.csv"
    indicators_df.to_csv(out_csv, index=False)
    print(
        f"[evaluate][L6] indicators → {out_csv.name} "
        f"({len(indicators_df)} rows)",
        file=sys.stderr,
    )
    return out_csv


#: Recognised ``--preset`` values.  ``"paper"`` (default) constructs
#: the baseline with paper-spec hyperparams from Appendix C.1 Table 8
#: (per-baseline ``PAPER_HYPERPARAMS`` constant). ``"smoke"`` skips
#: that materialisation and uses the class ``__init__`` defaults
#: (fast/CI values).
PRESETS: tuple[str, ...] = ("paper", "smoke")
DEFAULT_PRESET: str = "paper"


def run_evaluation(
    *,
    method: str,
    data_dir: Path,
    seed: int,
    settings: Sequence[str],
    out_dir: Path,
    checkpoint: Path | None = None,
    device: str = "cuda",
    ab_parquet: Path | None = None,
    with_indicators: bool = True,
    preset: str = DEFAULT_PRESET,
) -> dict:
    """Train (or load) a baseline and write metrics for each setting.

    Note on ``settings``: when multiple settings are passed, the model
    is **trained once and reused** to score every setting's val/test
    splits. This matches the legacy ``cold_start_split_fair`` design
    where train ⊆ G1×G1 is shared across S0/S1/S2 (no setting-specific
    re-training). Baselines that internally do val-driven early
    stopping should use only the first setting's val for that purpose.

    Parameters
    ----------
    method
        Registered baseline name (``coldddi.baselines.list_baselines()``).
    data_dir
        Release-style directory consumed by
        :meth:`PairDataset.from_release_dir`.
    seed
        Which split seed to load.
    settings
        Iterable of ``"S0" / "S1" / "S2"``.
    out_dir
        Where ``metrics_seed{N}.json`` (aggregate stats) and one
        ``predictions_<split>_seed{N}.csv`` per evaluated split (long-form
        per-pair predictions) are written.  Both flow from a single
        ``predict_proba`` pass per split, so the JSON aggregates
        are always consistent with the CSV.  The per-pair CSV is the
        artefact paper appendix~A.6.2 promises under ``--out`` and is
        the canonical handoff to :mod:`coldddi.diagnostics` for L6
        indicator computation.
    checkpoint
        If given, load instead of training.
    device
        Hint passed via baseline-specific kwargs (most baselines pick up
        ``CUDA_VISIBLE_DEVICES`` themselves).
    ab_parquet
        Optional explicit path to the A/B annotation parquet used by
        :mod:`coldddi.diagnostics` to assign PK-A / PK-B / PD-A / PD-B
        buckets.  When ``None`` (default), the L6 step auto-discovers
        ``annotations/{ab,ab_sample}.parquet`` under ``data_dir`` or
        the repo root.  When neither is found, L6 is skipped with a
        single stderr warning (training + per-pair CSVs are still
        emitted; only the indicators step is dropped).
    with_indicators
        Controls whether the L6 indicator step fires after training.
        Defaults to ``True`` so a plain ``python evaluate.py --method
        <baseline>`` invocation produces the full paper A.6.2
        artefact set (metrics JSON + per-pair predictions CSV +
        indicators CSV).  Set ``False`` to skip the indicator pass
        when you only need predictions — useful in sweeps where L6
        is run separately as a post-processing batch.
    preset
        ``"paper"`` (default) constructs the baseline with the per-
        method :data:`PAPER_HYPERPARAMS` dict (Appendix C.1 Table 8)
        — required for paper-grade reproduction.  ``"smoke"`` uses
        the class ``__init__`` defaults (fast / CI values).  Class
        defaults are intentionally smoke values so unit tests stay
        fast; the paper-grade configuration is materialised at run
        time by this preset switch.
    """
    # Validate preset up front so a bad value fails before paying
    # the cost of dataset loading.
    if preset not in PRESETS:
        raise ValueError(
            f"preset must be one of {PRESETS}; got {preset!r}"
        )
    # Lazy-import the baseline module *before* validating, so a fresh
    # process that only did `import coldddi.evaluate` still finds it.
    ensure_imported(method)
    if method not in list_baselines():
        known = sorted(set(list_baselines()) | set(NAME_TO_MODULE))
        raise ValueError(
            f"Unknown method {method!r}; registered or declared baselines are {known}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] method = {method}", file=sys.stderr)
    print(f"[evaluate] loading PairDataset @ seed={seed} from {data_dir}",
          file=sys.stderr, flush=True)
    # Dispatch based on path extension only (do NOT also require
    # ``is_file()``): we want a missing ``.pkl`` to surface a clean
    # ``FileNotFoundError`` from ``PairDataset.from_pkl`` rather than
    # silently falling through to ``from_release_dir`` (which would
    # produce a less actionable "drugs.csv not found" further down).
    # Test in tests/test_evaluate_cli_alignment.py pins the dispatch
    # to suffix-only.
    data_path = Path(data_dir)
    if data_path.suffix.lower() == ".pkl":
        ds = PairDataset.from_pkl(data_path)
    else:
        ds = PairDataset.from_release_dir(data_path, seed=seed)

    # Train (or load).
    if checkpoint is not None:
        print(f"[evaluate] loading checkpoint {checkpoint}", file=sys.stderr)
        model = load_baseline(checkpoint)
    else:
        # Forward `device` to baselines that accept it.  Accepts when:
        #   * ``device`` is a named parameter on ``__init__``, OR
        #   * ``__init__`` has ``**kwargs`` (VAR_KEYWORD — same policy
        #     the paper-preset block uses so wrappers receive device
        #     consistently with the paper hyperparams).
        cls = _REGISTRY[method]
        kwargs: dict = {}
        sig = inspect.signature(cls)
        has_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if "device" in sig.parameters or has_var_keyword:
            kwargs["device"] = device
        if preset == "paper":
            # Paper-grade construction: layer the per-baseline
            # PAPER_HYPERPARAMS dict (Appendix C.1 Table 8) onto the
            # device kwarg.  Class __init__ defaults are smoke values;
            # this preset materialises the production config from the
            # paper's tables.
            from coldddi.baselines.base import get_paper_hyperparams

            paper_kwargs = get_paper_hyperparams(method)
            # Forwarding policy:
            #   * If the class accepts ``**kwargs`` (a VAR_KEYWORD
            #     parameter — typical for wrapper subclasses), pass
            #     every paper kwarg through unfiltered.  Without this
            #     check, a future real-class wrapper using
            #     ``def __init__(self, **kwargs): super().__init__(**kwargs)``
            #     would silently drop the paper hyperparams.
            #   * Otherwise filter to named ``__init__`` parameters,
            #     which intentionally rejects paper kwargs for test
            #     Tiny classes (they want the hard-coded smoke kwargs
            #     baked into their ``super().__init__(...)`` call).
            sig = inspect.signature(cls)
            has_var_keyword = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_keyword:
                # Real wrapper or test Tiny with **kw — forward all.
                # Test Tiny classes ignore unwanted kwargs by merging
                # over their own hard-coded values; production wrappers
                # receive every paper kwarg explicitly.
                kwargs.update(paper_kwargs)
            else:
                # Strict-signature class (e.g. ``def __init__(self, *,
                # ssp_dim=..., ...)``): only forward kwargs the
                # constructor actually accepts so a paper field a
                # future refactor removed doesn't crash construction.
                sig_params = set(sig.parameters)
                paper_kwargs = {
                    k: v for k, v in paper_kwargs.items() if k in sig_params
                }
                kwargs.update(paper_kwargs)
            print(
                f"[evaluate] preset='paper' → {len(paper_kwargs)} hyperparams "
                f"from PAPER_HYPERPARAMS",
                file=sys.stderr,
            )
        else:
            print("[evaluate] preset='smoke' → class __init__ defaults", file=sys.stderr)
        model = cls(**kwargs)
        print(f"[evaluate] training {method} on seed {seed}", file=sys.stderr)
        model.fit(ds, kg=ds.kg)

    # Evaluate every requested setting.  Each split's predictions are
    # written to ``out_dir/predictions_<split>_seed{N}.csv`` next to the
    # aggregate metrics, mirroring the paper's ``--out DIR`` contract
    # (Appendix A.6.2, "CSV of per-pair predictions + metrics").
    metrics: dict[str, dict] = {}
    for setting in settings:
        if setting not in ALL_SETTINGS:
            raise ValueError(f"Unknown setting {setting!r}; expected one of {ALL_SETTINGS}")
        val_name, test_name = _setting_splits(setting)
        for split_name in (val_name, test_name):
            pos = getattr(ds.splits, split_name)
            neg = ds.get_negatives(split_name)
            split_metrics, pred_df = _evaluate_split_with_predictions(
                model, pos, neg,
            )
            metrics[split_name] = split_metrics
            if len(pred_df):
                pred_csv = out_dir / f"predictions_{split_name}_seed{seed}.csv"
                pred_df.to_csv(pred_csv, index=False)
                print(
                    f"[evaluate] {split_name}: {split_metrics} → {pred_csv.name}",
                    file=sys.stderr,
                )
            else:
                # Empty split (no positives AND no negatives) — skip
                # the CSV.  The metrics JSON still records n_pos=0/n_neg=0
                # so downstream diagnostics can detect the empty case.
                print(
                    f"[evaluate] {split_name}: {split_metrics} (empty split, no CSV)",
                    file=sys.stderr,
                )

    out_path = out_dir / f"metrics_seed{seed}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] wrote {out_path}", file=sys.stderr)

    # ── L6 indicator step (paper A.6.2 end-to-end contract) ─────────
    # Only fire when test_s2 was evaluated — KPS-F / KPS-mol / KPS-KG
    # are defined on the S2 cold-start anchor set per the paper.  Skip
    # silently for runs that target S0 / S1 only.
    if with_indicators and "S2" in {s.upper() for s in settings}:
        resolved_ab = _resolve_ab_parquet(ab_parquet, data_dir)
        if resolved_ab is None:
            print(
                "[evaluate][L6] no AB-annotation parquet found "
                "(searched --ab-parquet, data_dir/annotations/, repo "
                "annotations/) — skipping indicator step.  Pass "
                "--ab-parquet PATH to compute KPS-F / KPS-mol / KPS-KG.",
                file=sys.stderr,
            )
        else:
            _run_indicators_for_test_s2(
                model=model,
                ds=ds,
                ab_parquet=resolved_ab,
                out_dir=out_dir,
                seed=seed,
            )

    return metrics


def _ensure_baseline_imported(method: str) -> None:
    """Backward-compat alias for :func:`coldddi.baselines.ensure_imported`.

    Older tests / code may import this; new code should use
    :func:`coldddi.baselines.ensure_imported` directly. The two share
    the same ``NAME_TO_MODULE`` map.
    """
    if method not in NAME_TO_MODULE:
        raise ImportError(
            f"Could not import baseline module for {method!r}: "
            f"not declared in NAME_TO_MODULE."
        )
    ensure_imported(method)


#: Paper-spec dataset subset shortcuts (Appendix A.6.2 walkthrough).
#:
#: ``800``  : legacy 800-drug bundle (paper's smaller subset, used
#:            in the LLM-FT 3-seed experiments).  Stored as a pkl
#:            per upstream convention; loaded via
#:            :meth:`PairDataset.from_pkl`.
#: ``1900`` : full 1900-drug release directory.  Stored as the
#:            standard release layout; loaded via
#:            :meth:`PairDataset.from_release_dir`.
#:
#: Both resolve to paths under ``data/private/`` that are populated
#: when the user runs ``reconstruct.py``.  Toy fixtures are reached
#: via explicit ``--data data/public/intermediate``.
SUBSET_PATHS: dict[str, Path] = {
    # The "subset" value is the path *template* — ``{seed}`` is
    # substituted before use so a single shortcut covers all seeds.
    "800": Path(
        "data/private/outputs_full/splits_legacy/800drug/"
        "latest_drugbank_ddi-Binary_cls-{seed}+"
        "cold_start_split_fair_step-and-fair_negatives_step.pkl"
    ),
    "1900": Path("data/private/intermediate"),
}


def _resolve_subset(
    subset: str,
    seed: int,
    *,
    repo_root: Path,
) -> Path:
    """Resolve a ``--subset`` shorthand to a concrete path.

    Returns the resolved path verbatim; existence checking happens
    inside :meth:`PairDataset.from_pkl` /
    :meth:`PairDataset.from_release_dir` so the user gets a clear
    error if ``reconstruct.py`` hasn't been run yet.
    """
    if subset == "800":
        release = repo_root / "data/private/subsets/800" / f"seed{seed}" / "intermediate"
        if release.is_dir():
            return release
    template = SUBSET_PATHS[subset]
    # Format the seed into the path string (no-op for the 1900 dir).
    return repo_root / Path(str(template).format(seed=seed))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Run a registered ColdDDI baseline end-to-end (train → "
            "per-pair predictions CSV → L6 KPS / KSAI indicators), "
            "paper Appendix A.6.2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Registered baseline name (will lazy-import "
            "`coldddi.baselines.<method>`)."
        ),
    )
    # --data and --subset are mutually exclusive; exactly one is needed.
    # Implementing as a manual check (not argparse's mutually-exclusive
    # group) so the help text reads cleanly.
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Explicit dataset path: release directory (loaded via "
            "PairDataset.from_release_dir) or .pkl legacy bundle "
            "(via from_pkl).  Mutually exclusive with --subset."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=tuple(SUBSET_PATHS.keys()),
        default=None,
        help=(
            "Paper-spec dataset shorthand: '800' → legacy 800-drug "
            "pkl, '1900' → full 1900-drug release dir.  Mutually "
            "exclusive with --data.  Both resolve under data/private/, "
            "populated by `reconstruct.py`."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    # Paper A.6.2 walkthrough shows --setting default = "all".  Earlier
    # versions defaulted to S2 because the L6 indicator step (added in
    # Step 2 of the release contract roll-out) is only fired when S2 is
    # in the requested settings.  "all" supersets that, so the default
    # is now paper-aligned with no behavioural loss.
    parser.add_argument(
        "--setting",
        choices=["S0", "S1", "S2", "all"],
        default="all",
    )
    # Paper line 555: ``[--device {cuda,cpu}]``.  Constrain to those
    # two values plus ``auto`` (the baselines' own resolver picks
    # between cuda/cpu based on torch.cuda.is_available()).  Free-form
    # device strings (e.g., ``cuda:1``) are accepted by passing
    # CUDA_VISIBLE_DEVICES env var, keeping the CLI surface paper-faithful.
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help=(
            "Device hint passed to baseline __init__ when supported. "
            "For specific GPU indices set CUDA_VISIBLE_DEVICES instead."
        ),
    )
    # Paper line 556 brackets --out as optional.  Default to
    # ``runs/<method>/seed<N>/`` under the repo root so a plain
    # ``python evaluate.py --method <m>`` invocation always lands
    # somewhere predictable.  Resolved lazily in ``main()`` because
    # ``argparse`` defaults can't reference other args (--method, --seed).
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory.  Defaults to "
            "`runs/<method>/seed<seed>/` under the repo root if "
            "omitted (paper A.6.2 line 556 makes --out optional)."
        ),
    )
    # --checkpoint and --adapter are aliases (paper uses --adapter for
    # the LLM-FT path; we accept both so the same CLI works across
    # baselines and the LLM stack).  argparse routes them to the same
    # destination.
    parser.add_argument(
        "--checkpoint",
        "--adapter",
        type=Path,
        default=None,
        dest="checkpoint",
        help=(
            "Optional pre-trained baseline directory (skips fit()). "
            "--adapter is accepted as a paper-name alias."
        ),
    )
    parser.add_argument(
        "--ab-parquet",
        type=Path,
        default=None,
        help=(
            "Optional path to the A/B annotation parquet used by L6 "
            "diagnostics for bucket assignment.  Auto-discovers "
            "annotations/{ab,ab_sample}.parquet under --data or the "
            "repo root when omitted.  Skip L6 if neither found."
        ),
    )
    parser.add_argument(
        "--no-indicators",
        action="store_true",
        help=(
            "Skip the L6 indicator step (KPS-F / KPS-mol / KPS-KG); "
            "only train and write the per-pair predictions CSV plus "
            "the aggregate metrics JSON."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default=DEFAULT_PRESET,
        help=(
            "Hyperparameter preset for the baseline.  'paper' "
            "(default) loads per-method PAPER_HYPERPARAMS (Appendix "
            "C.1 Table 8) so plain `python evaluate.py --method <m>` "
            "reproduces the paper-grade config out of the box.  "
            "'smoke' uses the class __init__ defaults (fast CI "
            "values; pre-audit behaviour)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve dataset path: exactly one of --data / --subset must be set.
    if (args.data is None) == (args.subset is None):
        parser.error(
            "exactly one of --data or --subset must be provided "
            "(got data={!r}, subset={!r})".format(args.data, args.subset)
        )
    repo_root = Path(__file__).resolve().parents[1]
    if args.subset is not None:
        data_dir = _resolve_subset(args.subset, args.seed, repo_root=repo_root)
        print(
            f"[evaluate] --subset {args.subset} → {data_dir}",
            file=sys.stderr,
        )
    else:
        data_dir = args.data

    # Resolve --out default: ``runs/<method>/seed<N>/`` under repo root.
    out_dir = (
        args.out
        if args.out is not None
        else repo_root / "runs" / args.method / f"seed{args.seed}"
    )

    settings = list(ALL_SETTINGS) if args.setting == "all" else [args.setting]
    run_evaluation(
        method=args.method,
        data_dir=data_dir,
        seed=args.seed,
        settings=settings,
        out_dir=out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        ab_parquet=args.ab_parquet,
        with_indicators=not args.no_indicators,
        preset=args.preset,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
