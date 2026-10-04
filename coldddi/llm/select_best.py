"""Select LoRA checkpoints separately for S0, S1, and S2.

Derived from ``parse_trainer_state.py`` and ``select_best.py`` in
``Version_1_1/exps/sec5-3/0_select_lora/``. For each split, score the
lowest-loss candidates and the latest checkpoint on its validation set,
then select the highest ROC-AUC, breaking ties by earlier step.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd

    from coldddi.data.dataset import PairDataset
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.retrieval.kg_subgraph import SubgraphMap


@dataclass
class CandidateCkpt:
    """A candidate LoRA checkpoint."""

    step: int
    ckpt_path: str
    tag: str
    eval_loss: float | None = None
    eval_metric_key: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    """A :class:`CandidateCkpt` with its validation AUC and sample counts."""

    candidate: CandidateCkpt
    val_auc: float
    n_pos: int
    n_neg: int


# Candidate parsing

def _find_trainer_state(run_dir: Path) -> Path:
    """Find the most recent ``trainer_state.json`` under ``run_dir``.

    Prefer the latest checkpoint's cumulative log history over the
    top-level file, which may be stale after an interrupted resumed run.
    Use the top-level file only if no checkpoint copy exists.
    """
    candidates = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.parent.name).group(1)),
    )
    if candidates:
        return candidates[-1]
    direct = run_dir / "trainer_state.json"
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"No trainer_state.json under {run_dir} or its checkpoint-* subdirs."
    )


def _on_disk_ckpt_steps(run_dir: Path) -> set[int]:
    out: set[int] = set()
    for p in run_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


#: :func:`parse_candidate_ckpts` defaults to cold-start S2, as reported in
#: paper Tables 5, 6, and 8. S0/S1 diagnostics are opt-in.
DEFAULT_VAL_SPLITS: tuple[str, ...] = ("S2",)


def _candidates_for_one_key(
    run_dir: Path,
    on_disk: set[int],
    by_step: dict[int, dict],
    eval_metric_key: str,
    topk: int,
) -> list[CandidateCkpt]:
    """Return top-K by lowest eval_metric_key plus latest, sorted by loss/step."""
    rows: list[CandidateCkpt] = []
    for step, d in by_step.items():
        if step not in on_disk:
            continue
        ev = d.get(eval_metric_key)
        if ev is None:
            continue
        rows.append(
            CandidateCkpt(
                step=step,
                ckpt_path=str(run_dir / f"checkpoint-{step}"),
                tag="",
                eval_loss=float(ev),
                eval_metric_key=eval_metric_key,
            )
        )

    # Sort by ascending (eval_loss, step); ties favor the earlier step.
    rows.sort(key=lambda r: (
        r.eval_loss if r.eval_loss is not None else float("inf"),
        r.step,
    ))
    top = rows[:topk]
    for i, r in enumerate(top):
        r.tag = f"min_{eval_metric_key}_{i + 1}"

    # Always include the latest checkpoint.
    latest_step = max(on_disk)
    latest_match = next((r for r in top if r.step == latest_step), None)
    if latest_match is None:
        latest_match = CandidateCkpt(
            step=latest_step,
            ckpt_path=str(run_dir / f"checkpoint-{latest_step}"),
            tag="latest",
            eval_loss=None,
            eval_metric_key=eval_metric_key,
        )
        top.append(latest_match)
    else:
        latest_match.tag = latest_match.tag + "+latest"

    def _key(r: CandidateCkpt) -> tuple[float, int]:
        return (
            r.eval_loss if r.eval_loss is not None else float("inf"),
            r.step,
        )

    return sorted(top, key=_key)


def parse_candidate_ckpts(
    run_dir: str | Path,
    *,
    splits: tuple[str, ...] = DEFAULT_VAL_SPLITS,
    topk: int = 1,
) -> dict[str, list[CandidateCkpt]]:
    """Return top-K lowest-loss checkpoints plus latest for each validation split.

    Parameters
    ----------
    run_dir
        :class:`coldddi.llm.trainer.LoRATrainer`'s ``output_dir``,
        containing ``checkpoint-<step>/`` subdirectories.
    splits
        Validation splits to rank, defaulting to ``("S2",)`` for the
        cold-start setting in paper Table 8. ``("S0", "S1", "S2")``
        includes all diagnostic rankings. Each split uses its
        ``eval_<split>_loss`` key in the trainer log history.
    topk
        Lowest-loss checkpoints to keep per split. Default ``1`` matches
        upstream ``Code-Released`` (one per setting), validating the loss
        winner by AUC. ``topk > 1`` tests more candidates when loss and AUC
        rankings differ.

    Returns
    -------
    Dict mapping each split to :class:`CandidateCkpt` entries sorted by
    ascending eval loss, then step. A split with no logged eval key maps
    to only the latest checkpoint with ``eval_loss=None``. Checkpoints
    pruned by ``save_total_limit`` are excluded.

    Notes
    -----
    With no on-disk ``checkpoint-*`` directories, each split maps to an
    empty list; callers must handle that result.
    """
    run_dir = Path(run_dir)
    on_disk = _on_disk_ckpt_steps(run_dir)
    if not on_disk:
        return {s: [] for s in splits}

    try:
        state_path = _find_trainer_state(run_dir)
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        log = state.get("log_history", [])
    except FileNotFoundError:
        log = []

    # Group evaluation metrics by step.
    by_step: dict[int, dict] = defaultdict(dict)
    for row in log:
        step = row.get("step")
        if step is None:
            continue
        for k, v in row.items():
            if k.startswith("eval_") and (k.endswith("_loss") or k == "eval_loss"):
                by_step[int(step)][k] = v

    latest_step = max(on_disk)
    out: dict[str, list[CandidateCkpt]] = {}
    for split in splits:
        key = f"eval_{split}_loss"
        if any(key in d for d in by_step.values()):
            out[split] = _candidates_for_one_key(
                run_dir, on_disk, by_step, key, topk
            )
        else:
            # Without this split's eval key, use the latest checkpoint.
            out[split] = [
                CandidateCkpt(
                    step=latest_step,
                    ckpt_path=str(run_dir / f"checkpoint-{latest_step}"),
                    tag="latest_no_eval",
                    eval_loss=None,
                    eval_metric_key=key,
                )
            ]
    return out


# Per-split candidate scoring

def _val_split_name(split: str) -> str:
    """Map eval-log split names to :attr:`PairDataset.splits` names.

    ``S0`` / ``S1`` / ``S2`` become ``val_s0`` / ``val_s1`` / ``val_s2``.
    """
    return f"val_{split.lower()}"


def _build_eval_samples(
    dataset: "PairDataset",
    val_split: str,
    n_neg_per_pos: int,
    subgraph_map,
    fewshot_map,
):
    """Build the per-split eval list once. Returns (samples, y_true, n_pos, n_neg)."""
    import numpy as np
    import pandas as pd

    from coldddi.llm.retrieval.llm_view import to_llm_samples

    pos_df = dict(dataset.splits.items()).get(val_split)
    if pos_df is None or pos_df.empty:
        raise ValueError(
            f"Split {val_split!r} is empty on this PairDataset. "
            f"Available splits: "
            f"{[k for k, df in dataset.splits.items() if not df.empty]}"
        )
    pos_df = pos_df[["drug_a_id", "drug_b_id"]].copy()
    # Cached negatives match the pairs indexed by build_fewshot_smiles_map
    # and build_fewshot_2hop_map. If absent, use the sampler; regenerated
    # negatives lack few-shot entries, so their P2/P5 prompts become zero-shot.
    # This can affect fixtures that skip pre-sampling.
    cached_negs = getattr(dataset, "negatives_by_split", {}) or {}
    cached_neg_df = cached_negs.get(val_split)
    if cached_neg_df is not None and len(cached_neg_df):
        neg_df = cached_neg_df[["drug_a_id", "drug_b_id"]].copy()
    else:
        neg_df = dataset.get_negatives(val_split)[["drug_a_id", "drug_b_id"]].copy()
    target_n_neg = n_neg_per_pos * len(pos_df)
    if target_n_neg > 0 and len(neg_df) > target_n_neg:
        neg_df = neg_df.head(target_n_neg).reset_index(drop=True)
    eval_df = pd.concat([pos_df, neg_df], ignore_index=True)
    y_true = np.concatenate([
        np.ones(len(pos_df), dtype=np.int64),
        np.zeros(len(neg_df), dtype=np.int64),
    ])
    samples = to_llm_samples(
        eval_df, labels=None, ds=dataset,
        subgraph_map=subgraph_map, fewshot_map=fewshot_map,
    )
    return samples, y_true, len(pos_df), len(neg_df)


def _drug_lookup_tables(dataset: "PairDataset"):
    import pandas as pd

    drug_id2name: dict[str, str] = {}
    drug_id2smiles: dict[str, str] = {}
    if dataset.drugs is not None:
        for _, row in dataset.drugs.iterrows():
            did = str(row["drugbank_id"])
            if "name" in dataset.drugs.columns and not pd.isna(row["name"]):
                drug_id2name[did] = str(row["name"])
            if "smiles" in dataset.drugs.columns and not pd.isna(row["smiles"]):
                drug_id2smiles[did] = str(row["smiles"])
    return drug_id2name, drug_id2smiles


def score_candidate_ckpts(
    candidates_by_split: dict[str, list[CandidateCkpt]],
    *,
    base_model_name: str,
    dataset: "PairDataset",
    prompt_cfg: "PromptBuildConfig",
    subgraph_map: "SubgraphMap | None" = None,
    fewshot_map: dict | None = None,
    key_entity_map: dict | None = None,
    n_neg_per_pos: int = 1,
    device: str = "auto",
    dtype: str = "bfloat16",
    batch_size: int = 8,
    max_length: int = 1024,
    cache_dir: str | None = None,
    yes_token: str = " Yes",
    no_token: str = " No",
    shuffle_seed: int = 20260511,
) -> dict[str, list[ScoredCandidate]]:
    """Score each candidate on its own validation split.

    Parameters
    ----------
    candidates_by_split
        Output of :func:`parse_candidate_ckpts` —
        ``{"S0": [...], "S1": [...], "S2": [...]}``.
    base_model_name
        Base causal-LM identifier for the LoRA adapters.
    dataset
        :class:`PairDataset` with ``splits.val_sx`` populated for each
        ``"Sx"`` key in ``candidates_by_split``.
    prompt_cfg, subgraph_map, fewshot_map
        Forwarded to :func:`to_llm_samples` and the inference runner.
        Must match training: use the ``prompt_cfg`` passed to
        :meth:`LoRATrainer.fit` or recover it with :func:`read_fit_info`.
    key_entity_map
        Optional ``{(drug_a_id, drug_b_id): {key_entity_name, ...}}``.
        Required for the ``OHS_Mask_Entity`` / ``OHS_Mask_Name_Entity``
        (R2/R3) validation methods; ignored by the unmasked / name-only
        masking paths.
    n_neg_per_pos
        Negatives per positive in the eval pair list.
    yes_token, no_token
        Must match ``LLMTrainerConfig.yes_token`` / ``no_token`` used in
        training. Recover them from ``fit_info.json`` with :func:`read_fit_info`.
        Defaults match the trainer (`` Yes`` / `` No``).
    shuffle_seed
        Reseed NumPy's global RNG before each candidate to keep few-shot
        order fixed for AUC comparisons. ``None`` disables reseeding;
        non-fewshot methods are unaffected.

    Returns
    -------
    Dict ``{split: [ScoredCandidate, ...]}`` — same outer keys and
    inner ordering as ``candidates_by_split``.  Skipped (missing on
    disk) checkpoints get ``val_auc = NaN``.
    """
    import numpy as np
    import gc

    import torch
    from sklearn.metrics import roc_auc_score

    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig

    drug_id2name, drug_id2smiles = _drug_lookup_tables(dataset)

    # Build each split's samples and labels once for all candidates.
    eval_packs: dict[str, tuple] = {}
    for split in candidates_by_split:
        eval_packs[split] = _build_eval_samples(
            dataset,
            val_split=_val_split_name(split),
            n_neg_per_pos=n_neg_per_pos,
            subgraph_map=subgraph_map,
            fewshot_map=fewshot_map,
        )

    # As in upstream train_t48.py, load the base once, keep adapters resident,
    # and switch with set_adapter(name).
    base_runner = LLMInferenceRunner(LLMRunnerConfig(
        model_name=base_model_name,
        adapter_path=None,            # base model only
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        cache_dir=cache_dir,
        yes_token=yes_token,
        no_token=no_token,
    ))
    base_runner.load()
    base_model = base_runner.model    # no adapter

    from peft import PeftModel

    # Name adapters by source split and step for deterministic selection.
    def _adapter_name(split: str, step: int) -> str:
        return f"{split}__step{step}"

    peft_model: "PeftModel | None" = None
    registered: dict[str, str] = {}  # ckpt_path -> adapter_name
    for split, candidates in candidates_by_split.items():
        for cand in candidates:
            if not os.path.isdir(cand.ckpt_path):
                continue
            # Reuse adapters shared by multiple splits' top-K lists.
            if cand.ckpt_path in registered:
                continue
            name = _adapter_name(split, cand.step)
            try:
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(
                        base_model, cand.ckpt_path,
                        adapter_name=name, is_trainable=False,
                    )
                else:
                    peft_model.load_adapter(
                        cand.ckpt_path, adapter_name=name,
                        is_trainable=False,
                    )
                registered[cand.ckpt_path] = name
            except Exception as e:
                # Record failed adapter loads as NaN AUC below.
                print(
                    f"[L5 score_candidate_ckpts] load_adapter failed for "
                    f"{cand.ckpt_path}: {type(e).__name__}: {e}"
                )

    if peft_model is not None:
        peft_model.eval()

    out: dict[str, list[ScoredCandidate]] = {s: [] for s in candidates_by_split}
    try:
        for split, candidates in candidates_by_split.items():
            samples, y_true, n_pos, n_neg = eval_packs[split]
            for cand in candidates:
                if cand.ckpt_path not in registered:
                    out[split].append(ScoredCandidate(
                        candidate=cand, val_auc=float("nan"),
                        n_pos=n_pos, n_neg=n_neg,
                    ))
                    continue

                df = None
                try:
                    peft_model.set_adapter(registered[cand.ckpt_path])
                    # Fix the P2/P5 few-shot shuffle order across candidates.
                    if shuffle_seed is not None:
                        np.random.seed(int(shuffle_seed))
                    # Reuse the base tokenizer and answer IDs with the active adapter.
                    active_runner = _make_scratch_runner(
                        base_runner, peft_model,
                    )
                    df = active_runner.score_samples(
                        samples, prompt_cfg,
                        drug_id2name=drug_id2name,
                        drug_id2smiles=drug_id2smiles,
                        key_entity_map=key_entity_map,
                    )
                except Exception as e:
                    print(
                        f"[L5 score_candidate_ckpts] scoring failed for "
                        f"{cand.ckpt_path}: {type(e).__name__}: {e}"
                    )

                if df is None:
                    out[split].append(ScoredCandidate(
                        candidate=cand, val_auc=float("nan"),
                        n_pos=n_pos, n_neg=n_neg,
                    ))
                    continue

                y_score = df["p_yes"].to_numpy(dtype=np.float64)
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    auc = float("nan")
                out[split].append(ScoredCandidate(
                    candidate=cand, val_auc=auc,
                    n_pos=n_pos, n_neg=n_neg,
                ))
    finally:
        del peft_model, base_runner, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _make_scratch_runner(base_runner: "LLMInferenceRunner", peft_model):
    """Score via ``peft_model`` with ``base_runner``'s tokenizer and answer IDs.

    The shim keeps ``LLMInferenceRunner.score_samples``'s signature and
    routes forward passes through the LoRA-wrapped model without reloading.
    """
    from coldddi.llm.inference import LLMInferenceRunner

    shim = LLMInferenceRunner.__new__(LLMInferenceRunner)
    shim.cfg = base_runner.cfg
    shim.device = base_runner.device
    shim.model = peft_model
    shim.tokenizer = base_runner.tokenizer
    shim.yes_id = base_runner.yes_id
    shim.no_id = base_runner.no_id
    return shim


# Best-checkpoint selection and manifest

def _select_best_one_split(
    split: str,
    scored: list[ScoredCandidate],
) -> dict:
    valid = [s for s in scored if not _is_nan(s.val_auc)]
    if not valid:
        raise ValueError(
            f"Split {split!r}: no candidate has a valid val AUC — "
            "every entry is NaN or skipped."
        )
    ranked = sorted(valid, key=lambda s: (-s.val_auc, s.candidate.step))
    best = ranked[0]
    return {
        "split": split,
        "best_ckpt": best.candidate.ckpt_path,
        "best_step": best.candidate.step,
        "best_val_auc": best.val_auc,
        "best_tag": best.candidate.tag,
        "all_ranked": [
            {
                "step": s.candidate.step,
                "tag": s.candidate.tag,
                "val_auc": s.val_auc,
                "eval_loss": s.candidate.eval_loss,
                "eval_metric_key": s.candidate.eval_metric_key,
                "ckpt": s.candidate.ckpt_path,
                "n_pos": s.n_pos,
                "n_neg": s.n_neg,
            }
            for s in ranked
        ],
        "selection_criterion": (
            f"max(val_{split}.auroc) with tie-break = earlier step"
        ),
    }


def select_best(
    scored_by_split: dict[str, list[ScoredCandidate]],
) -> dict[str, dict]:
    """Return a per-split manifest dict.

    Parameters
    ----------
    scored_by_split
        Output of :func:`score_candidate_ckpts`.

    Returns
    -------
    Dict ``{split: best_manifest, ...}`` where each ``best_manifest``
    has the keys ``{"split", "best_ckpt", "best_step", "best_val_auc",
    "best_tag", "all_ranked", "selection_criterion"}``.

    Raises
    ------
    ValueError
        If any split has no valid (non-NaN) candidate; splits are never
        silently dropped.
    """
    out: dict[str, dict] = {}
    for split, scored in scored_by_split.items():
        out[split] = _select_best_one_split(split, scored)
    return out


def write_manifest(manifest: dict, out_path: str | Path) -> Path:
    """Pretty-print the manifest as JSON. Returns the resolved path."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and x != x


# Training contract

def read_fit_info(run_dir: str | Path) -> dict:
    """Read ``fit_info.json`` from a :class:`LoRATrainer` output dir.

    Returns the training contract (answer tokens, base model, prompt method,
    etc.) needed to reconstruct the inference configuration.

    Raises
    ------
    FileNotFoundError
        If ``run_dir/fit_info.json`` is missing. :meth:`LoRATrainer.fit`
        writes it on completion; a missing file indicates an unfinished
        or deleted run.
    """
    p = Path(run_dir) / "fit_info.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found — the run either never finished or was wiped. "
            "LoRATrainer.fit() writes fit_info.json on successful completion."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_prompt_cfg_matches_fit_info(
    fit_info: dict,
    prompt_cfg: "PromptBuildConfig",
    *,
    strict: bool = False,
) -> None:
    """Warn or raise when val/test ``prompt_cfg`` differs from training.

    Compare three fields in :func:`LoRATrainer.fit`'s ``fit_info.json``:

    * ``method``: canonicalise with :func:`canon_method` so aliases such as
      ``"One_Hop_Subgraph_Sequence"`` / ``"ohs"`` compare equal.
    * ``task_name``: determines the system/instruction header.
    * ``model_name``: compare chat-template families (Llama / Qwen / Gemma),
      which determine prompt structure even for the same method and task.

    ``strict=True`` raises instead of warning.
    """
    import warnings

    from coldddi.llm.prompts.binary_cls import canon_method, infer_model_family

    fit_prompt = fit_info.get("prompt_cfg") or {}
    if not fit_prompt:
        return  # No recorded prompt_cfg to compare.

    diffs: list[str] = []

    # Canonicalize methods so aliases compare equal.
    fit_method = fit_prompt.get("method")
    cur_method = getattr(prompt_cfg, "method", None)
    if fit_method is not None and cur_method is not None:
        try:
            if canon_method(cur_method) != canon_method(fit_method):
                diffs.append(
                    f"method: train={fit_method!r} vs val={cur_method!r} "
                    f"(canonicalised: {canon_method(fit_method)!r} "
                    f"vs {canon_method(cur_method)!r})"
                )
        except Exception:
            # If canonicalization fails, compare raw values.
            if cur_method != fit_method:
                diffs.append(
                    f"method: train={fit_method!r} vs val={cur_method!r}"
                )

    # Task name determines the system header.
    fit_task = fit_prompt.get("task_name")
    cur_task = getattr(prompt_cfg, "task_name", None)
    if fit_task is not None and cur_task is not None and fit_task != cur_task:
        diffs.append(f"task_name: train={fit_task!r} vs val={cur_task!r}")

    # Compare chat-template families, allowing size variants such as 1B and 3B.
    fit_model = fit_prompt.get("model_name")
    cur_model = getattr(prompt_cfg, "model_name", None)
    if fit_model is not None and cur_model is not None:
        try:
            if infer_model_family(fit_model) != infer_model_family(cur_model):
                diffs.append(
                    f"model_family: train={infer_model_family(fit_model)!r} "
                    f"(from {fit_model!r}) vs "
                    f"val={infer_model_family(cur_model)!r} "
                    f"(from {cur_model!r})"
                )
        except Exception:
            if fit_model != cur_model:
                diffs.append(
                    f"model_name: train={fit_model!r} vs val={cur_model!r}"
                )

    if not diffs:
        return

    msg = (
        "prompt_cfg diverges from the contract the LoRA was trained under "
        "— per-checkpoint AUC may not generalise back to inference time. "
        "Confirm this is intentional. Differences:\n  - "
        + "\n  - ".join(diffs)
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)


__all__ = [
    "DEFAULT_VAL_SPLITS",
    "CandidateCkpt",
    "ScoredCandidate",
    "parse_candidate_ckpts",
    "score_candidate_ckpts",
    "select_best",
    "write_manifest",
]
