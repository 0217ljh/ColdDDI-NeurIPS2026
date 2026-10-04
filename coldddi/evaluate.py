"""Baseline evaluator (paper §A.6.3).

Fit a registered :class:`coldddi.baselines.BaselineModel` on a
:class:`PairDataset`, evaluate S0/S1/S2, and write metrics and predictions
to ``--out``.

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


#: Prediction CSV column order, matching upstream
#: ``Code-Released/baseline/Output/<method>/my/seed{seed}/s{1,2}/inference_*.csv``.
#: Rename ``drug_a_id`` / ``drug_b_id`` to ``d1`` / ``d2`` for upstream layout.
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
    """Score each nonempty positive/negative set once and return a long-form table.

    Metrics and saved predictions share these scores. The threshold
    ``predicted_prob >= 0.5`` follows upstream inference CSVs.
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
    """Return aggregate metrics and the per-pair predictions they summarize.

    Reuse the scores written to CSV so metrics JSON and predictions agree.
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
    """Backward-compatible wrapper returning only the aggregate metrics dict.

    Use :func:`_evaluate_split_with_predictions` to also get per-pair
    predictions without extra inference.
    """
    metrics, _ = _evaluate_split_with_predictions(model, pos, neg)
    return metrics


#: Modality-to-mask-channel dispatch for L6 (paper Appendix A.6.2).
#: Invariant: ``set(MODALITY_MASK_CHANNELS) == set(MODALITIES)``.
#: Every modality in :mod:`coldddi.baselines.base` needs a dispatch entry.
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

    A missing explicit path raises ``FileNotFoundError``. If discovery
    finds nothing, return ``None``; the caller skips L6 with a warning.
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

    Forward ``mask_channel`` to ``predict_proba`` when set. The caller
    must check that the baseline's modality supports the channel.
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
    """Convert a score dict to the canonical per-pair predictions schema.

    Used for union and ``predictions_<split>_mask_*_seed{N}.csv`` outputs,
    which allow L6 diagnostics to be rerun without training.
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

    ``model.modality`` (instance override preferred over class default;
    see :class:`coldddi.baselines.base.BaselineModel`) selects the passes:

    * Empty :data:`MODALITY_MASK_CHANNELS` entry: base only, with NaN
      channel indicators.
    * ``modality == "mol+kg"``: base, mask_mol, and mask_kg passes.

    Return ``out_dir/indicators_test_s2_seed{N}.csv``.

    Also write ``predictions_test_s2_union_seed{N}.csv`` and, for mol+KG,
    ``predictions_test_s2_mask_mol_seed{N}.csv`` and
    ``predictions_test_s2_mask_kg_seed{N}.csv``. These cover canonical
    test_s2 pairs ∪ swap anchors ``(qa, qb)`` in both orientations ∪
    swap targets ``(qa_prime, qb)``. The split-only
    ``predictions_test_s2_seed{N}.csv`` is not rewritten here.
    """
    # Load diagnostics only when the indicator step runs.
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

    # Score test_s2 ∪ both swap-anchor orientations ∪ swap targets.
    # Reverse-anchor scores are needed for order-sensitive models such as
    # TIGER's head-dependent BKG walks; _lookup_directed's reverse fallback
    # would otherwise assume symmetric predictions.
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

    # Honor instance modality overrides, e.g. TIGER with mol_only=True.
    modality = getattr(model, "modality", "mol")
    # Unknown modalities must not silently receive single-modality dispatch.
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
    # Save the full L6 input union for recomputation. The split-only
    # predictions_test_s2_seed{N}.csv omits swap targets outside test_s2
    # and would under-count their KPS-F triples.
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
        # Save mask predictions for L6 recomputation without training.
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


#: ``paper`` uses PAPER_HYPERPARAMS (Appendix C.1 Table 8);
#: ``smoke`` uses the class __init__ defaults for fast/CI runs.
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

    Train once and reuse the model for all requested val/test splits,
    following ``cold_start_split_fair``: train ⊆ G1×G1 is shared across
    S0/S1/S2. Validation-driven early stopping should use only the first
    setting's validation split.

    Parameters
    ----------
    method
        Registered baseline name (``coldddi.baselines.list_baselines()``).
    data_dir
        Release directory for :meth:`PairDataset.from_release_dir` or
        legacy .pkl bundle for :meth:`PairDataset.from_pkl`.
    seed
        Which split seed to load.
    settings
        Iterable of ``"S0" / "S1" / "S2"``.
    out_dir
        Destination for ``metrics_seed{N}.json`` and
        ``predictions_<split>_seed{N}.csv`` (paper Appendix A.6.2).
        Both use the same scores; CSVs supply per-pair predictions to
        :mod:`coldddi.diagnostics`.
    checkpoint
        If given, load instead of training.
    device
        Hint passed via baseline-specific kwargs (most baselines pick up
        ``CUDA_VISIBLE_DEVICES`` themselves).
    ab_parquet
        A/B annotation parquet for PK-A/PK-B/PD-A/PD-B assignment.
        ``None`` searches ``annotations/{ab,ab_sample}.parquet`` under
        ``data_dir`` and the repo root. If neither exists, warn on stderr
        and skip only L6; training and prediction outputs still run.
    with_indicators
        Run L6 after evaluation when S2 is requested (default: ``True``).
        ``False`` writes metrics and predictions without indicators.
    preset
        ``"paper"`` (default) uses per-method :data:`PAPER_HYPERPARAMS`
        from Appendix C.1 Table 8. ``"smoke"`` uses class ``__init__``
        defaults for fast/CI runs.
    """
    # Reject invalid presets before loading data.
    if preset not in PRESETS:
        raise ValueError(
            f"preset must be one of {PRESETS}; got {preset!r}"
        )
    # Import before validation so the baseline is registered in a fresh process.
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
    # Dispatch by suffix, not is_file(): a missing .pkl must raise from
    # from_pkl, not fall through to a misleading "drugs.csv not found".
    data_path = Path(data_dir)
    if data_path.suffix.lower() == ".pkl":
        ds = PairDataset.from_pkl(data_path)
    else:
        ds = PairDataset.from_release_dir(data_path, seed=seed)

    if checkpoint is not None:
        print(f"[evaluate] loading checkpoint {checkpoint}", file=sys.stderr)
        model = load_baseline(checkpoint)
    else:
        # Forward device to named parameters or **kwargs, as with paper presets.
        cls = _REGISTRY[method]
        kwargs: dict = {}
        sig = inspect.signature(cls)
        has_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if "device" in sig.parameters or has_var_keyword:
            kwargs["device"] = device
        if preset == "paper":
            # Override smoke defaults with Appendix C.1 Table 8 hyperparameters.
            from coldddi.baselines.base import get_paper_hyperparams

            paper_kwargs = get_paper_hyperparams(method)
            # **kwargs wrappers need all paper parameters; filtering by named
            # parameters would drop them. Strict constructors receive only
            # supported names, allowing test classes to retain smoke defaults.
            sig = inspect.signature(cls)
            has_var_keyword = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_keyword:
                kwargs.update(paper_kwargs)
            else:
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
                # Empty splits produce no CSV; metrics still record n_pos=n_neg=0.
                print(
                    f"[evaluate] {split_name}: {split_metrics} (empty split, no CSV)",
                    file=sys.stderr,
                )

    out_path = out_dir / f"metrics_seed{seed}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] wrote {out_path}", file=sys.stderr)

    # Paper A.6.2 defines these indicators on S2 anchors; skip S0/S1-only runs.
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
    """Backward-compatible wrapper for :func:`coldddi.baselines.ensure_imported`.

    New callers should use that function directly; both use ``NAME_TO_MODULE``.
    """
    if method not in NAME_TO_MODULE:
        raise ImportError(
            f"Could not import baseline module for {method!r}: "
            f"not declared in NAME_TO_MODULE."
        )
    ensure_imported(method)


#: Dataset shortcuts (paper Appendix A.6.2), populated by reconstruct.py.
#: ``800``: legacy 800-drug .pkl for LLM-FT 3-seed runs; PairDataset.from_pkl.
#: ``1900``: full release directory; PairDataset.from_release_dir.
#: Both live under data/private/; toy data uses --data data/public/intermediate.
SUBSET_PATHS: dict[str, Path] = {
    # Substitute {seed} in the path template before loading.
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

    :meth:`PairDataset.from_pkl` / :meth:`PairDataset.from_release_dir`
    check that the resolved path exists after ``reconstruct.py``.
    """
    if subset == "800":
        release = repo_root / "data/private/subsets/800" / f"seed{seed}" / "intermediate"
        if release.is_dir():
            return release
    template = SUBSET_PATHS[subset]
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
    # main() enforces exactly one of --data and --subset.
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
    # "all" follows paper Appendix A.6.2 and includes S2 for L6 indicators.
    parser.add_argument(
        "--setting",
        choices=["S0", "S1", "S2", "all"],
        default="all",
    )
    # Paper line 555 uses cuda/cpu; auto delegates to the baseline's resolver.
    # Select GPU indices with CUDA_VISIBLE_DEVICES, not device strings like cuda:1.
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help=(
            "Device hint passed to baseline __init__ when supported. "
            "For specific GPU indices set CUDA_VISIBLE_DEVICES instead."
        ),
    )
    # Optional per paper line 556; main() builds runs/<method>/seed<N>/
    # after method and seed are known.
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
    # --adapter is the paper's LLM-FT alias for --checkpoint.
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
