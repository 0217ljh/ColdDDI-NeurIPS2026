"""Per-split LoRA checkpoint selection.

Port of ``Version_1_1/exps/sec5-3/0_select_lora/`` (the two scripts
``parse_trainer_state.py`` + ``select_best.py``).  Refactored so the
caller picks the **best checkpoint per validation split** (S0, S1,
S2) rather than only S2 — useful when paper experiments need a
per-setting ranking, or when you want to keep separate "best ckpt for
S0" / "best ckpt for S1" diagnostics alongside the cold-start (S2)
champion.

Why per-split
-------------
HuggingFace Trainer's ``metric_for_best_model`` defaults to
``eval_loss``, which under the multi-eval DataLoader registration
becomes ``eval_S0_loss`` — so the trainer's "best checkpoint"
optimises the wrong split for cold-start. Even with our
``primary_val_split = "S2"`` patch, that only gives ONE best ckpt
per training run.  L5's job is to expose the per-split rankings so
downstream experiments can pick whichever split is appropriate.

Three-step pipeline
-------------------
1. :func:`parse_candidate_ckpts` — read the latest
   ``checkpoint-*/trainer_state.json``, return a
   ``{split_name: [CandidateCkpt, ...]}`` dict.  Each split's
   candidate list = top-K lowest ``eval_{split}_loss`` checkpoints
   plus the "latest" checkpoint.
2. :func:`score_candidate_ckpts` — for each (split, candidate),
   load the LoRA adapter through
   :class:`coldddi.llm.inference.LLMInferenceRunner`, score the
   matching ``val_<split>`` dataset, compute ROC-AUC.
3. :func:`select_best` — per split, pick the highest-AUC candidate;
   tie-break = earlier step (less over-fit).
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
    """One candidate LoRA checkpoint identified by the parser."""

    step: int
    ckpt_path: str
    tag: str
    eval_loss: float | None = None
    eval_metric_key: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    """A :class:`CandidateCkpt` after its val_s2 AUC has been computed."""

    candidate: CandidateCkpt
    val_auc: float
    n_pos: int
    n_neg: int


# ─── Step 1: candidate parser ──────────────────────────────────────────────

def _find_trainer_state(run_dir: Path) -> Path:
    """Find the most recent ``trainer_state.json`` under ``run_dir``.

    HF Trainer writes one inside every ``checkpoint-*`` dir; the latest
    checkpoint's trainer_state has the full log history (it accumulates).
    We **prefer** the latest checkpoint's copy over any top-level
    ``trainer_state.json`` because a stale top-level file (left over
    from an earlier run that resumed and was interrupted) would
    otherwise produce the wrong candidate pool.
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


#: Default validation splits we surface in :func:`parse_candidate_ckpts`.
#: Only ``"S2"`` is materialised by default — paper Table 5/6 / Table 8
#: report cold-start S2; S0 and S1 are diagnostic and opt-in.
DEFAULT_VAL_SPLITS: tuple[str, ...] = ("S2",)


def _candidates_for_one_key(
    run_dir: Path,
    on_disk: set[int],
    by_step: dict[int, dict],
    eval_metric_key: str,
    topk: int,
) -> list[CandidateCkpt]:
    """Top-K (lowest eval_metric_key) + latest, ordered ascending."""
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

    # Stable sort: ascending (eval_loss, step) — ties go to earlier step.
    rows.sort(key=lambda r: (
        r.eval_loss if r.eval_loss is not None else float("inf"),
        r.step,
    ))
    top = rows[:topk]
    for i, r in enumerate(top):
        r.tag = f"min_{eval_metric_key}_{i + 1}"

    # Always include the "latest" checkpoint.
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
    """Identify top-K lowest-eval-loss + latest ckpts **per validation split**.

    Parameters
    ----------
    run_dir
        Training-run output directory (the one passed as
        ``output_dir`` to :class:`coldddi.llm.trainer.LoRATrainer`).
        Must contain ``checkpoint-<step>/`` subdirs.
    splits
        Which validation splits to rank by.  Default ``("S2",)`` —
        the cold-start setting paper Table 8 evaluates on.  Pass
        ``("S0", "S1", "S2")`` to materialise all three rankings
        (diagnostic).  Each entry in this tuple is looked up as
        ``eval_<split>_loss`` in the trainer log history.
    topk
        How many lowest-loss checkpoints to keep per split.  Default
        ``1`` — matches the upstream ``Code-Released`` convention
        (one ckpt per setting) and means L5 effectively does
        AUC-validation of HF Trainer's per-split loss winner.  Pass
        ``topk > 1`` to scan extra candidates as a loss-vs-AUC
        ranking hedge.

    Returns
    -------
    Dict mapping each requested split name to a sorted list of
    :class:`CandidateCkpt` (ascending eval loss then step).  Splits
    whose log key never appears in the history (e.g. you only
    trained with S2 eval but asked for S0) are still in the dict but
    map to a one-element list containing only the latest checkpoint
    (with ``eval_loss=None``).  Splits whose checkpoint dirs were
    pruned by ``save_total_limit`` are filtered out.

    Notes
    -----
    If ``run_dir`` has no on-disk ``checkpoint-*`` directories, every
    requested split maps to an empty list and the caller is expected
    to detect the empty result downstream (this lets a smoke test on a
    pruned run inspect the empty manifest without crashing).
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

    # Pivot log_history into a per-step dict of eval metrics.
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
            # Split's eval key is missing from every log row.  We can
            # still return the latest ckpt so a downstream "best
            # available" path keeps working.
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


# ─── Step 2: per-candidate val_S2 scoring ──────────────────────────────────

def _val_split_name(split: str) -> str:
    """``"S2"`` → ``"val_s2"``.

    :attr:`PairDataset.splits` exposes the validation splits as
    ``val_s0`` / ``val_s1`` / ``val_s2`` (lowercase, snake-cased).
    L5's per-split key (passed in via the ``splits=`` argument) is the
    short form ``S0`` / ``S1`` / ``S2``, matching the eval log key
    convention.  This helper bridges the two.
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
    # Prefer cached static negatives over the regeneration path so the
    # eval pair list lines up exactly with the cached set that
    # ``build_fewshot_smiles_map`` / ``build_fewshot_2hop_map`` indexed.
    # If a dataset has no cached negatives for this split, fall back to
    # the sampler — but then the few-shot map will also have lacked
    # those rows, so the P2/P5 prompt would degenerate to zero-shot for
    # the regenerated negatives; this is at most a contract mismatch
    # for fixtures that explicitly skipped pre-sampling.
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
    """Score every candidate against ITS OWN val split.

    Parameters
    ----------
    candidates_by_split
        Output of :func:`parse_candidate_ckpts` —
        ``{"S0": [...], "S1": [...], "S2": [...]}``.
    base_model_name
        Base causal-LM identifier; the LoRA adapter is layered on top.
    dataset
        Loaded :class:`PairDataset` carrying every needed validation
        split.  For each split key ``"Sx"`` in ``candidates_by_split``,
        the dataset must have ``splits.val_sx`` populated.
    prompt_cfg, subgraph_map, fewshot_map
        Forwarded to :func:`to_llm_samples` and the inference runner.
        Must match what the LoRA was trained against — pass the same
        ``prompt_cfg`` you handed to :meth:`LoRATrainer.fit` (or
        :func:`read_fit_info` it back from the run dir).
    key_entity_map
        Optional ``{(drug_a_id, drug_b_id): {key_entity_name, ...}}``.
        Required for the ``OHS_Mask_Entity`` / ``OHS_Mask_Name_Entity``
        (R2/R3) validation methods; ignored by the unmasked / name-only
        masking paths.
    n_neg_per_pos
        Negatives per positive in the eval pair list.
    yes_token, no_token
        Answer-token strings the LoRA was trained against.  Must match
        ``LLMTrainerConfig.yes_token`` / ``no_token``; the easiest way
        to ensure that is to read them from the run dir's
        ``fit_info.json`` via :func:`read_fit_info`.  Defaults match
        the trainer defaults (`` Yes`` / `` No``).
    shuffle_seed
        Seed reapplied to NumPy's global RNG before scoring **each**
        candidate.  This pins few-shot example ordering across
        candidates so per-checkpoint AUC comparisons stay valid for
        P2 / P5 prompts (their ``_fewshot_*_block`` helpers shuffle
        in-place).  Pass ``None`` to disable the reseed (back-compat
        with non-fewshot methods that are insensitive to this).

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

    # Pre-build per-split eval pair lists (samples + y_true) ONCE so
    # the (potentially expensive) to_llm_samples call doesn't repeat
    # per candidate.
    eval_packs: dict[str, tuple] = {}
    for split in candidates_by_split:
        eval_packs[split] = _build_eval_samples(
            dataset,
            val_split=_val_split_name(split),
            n_neg_per_pos=n_neg_per_pos,
            subgraph_map=subgraph_map,
            fewshot_map=fewshot_map,
        )

    # ── Base + ALL adapters resident; swap via set_adapter ─────────
    # Pattern mirrored from upstream Code-Released ``train_t48.py``:
    #   1. Load base ONCE via LLMInferenceRunner (no adapter).
    #   2. Pre-register EVERY unique candidate ckpt as a named adapter
    #      on the same PeftModel. Each LoRA is <50 MB on disk; 9 of
    #      them ≈ 0.5 GB of GPU memory — negligible next to the
    #      multi-GB base.
    #   3. At scoring time just call ``peft_model.set_adapter(name)``
    #      which is a ~O(num_lora_modules) attribute flip (no disk
    #      IO, no weight copy). This is the key to keeping per-cand
    #      cost near "forward only" rather than "reload-everything".
    base_runner = LLMInferenceRunner(LLMRunnerConfig(
        model_name=base_model_name,
        adapter_path=None,            # base only here
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        cache_dir=cache_dir,
        yes_token=yes_token,
        no_token=no_token,
    ))
    base_runner.load()
    base_model = base_runner.model    # clean causal-LM with no adapter

    from peft import PeftModel

    # Step 2: collect unique (split, cand.ckpt_path) pairs and
    # register all adapters up front. Adapter names encode their
    # source so the per-candidate set_adapter is deterministic.
    def _adapter_name(split: str, step: int) -> str:
        return f"{split}__step{step}"

    peft_model: "PeftModel | None" = None
    registered: dict[str, str] = {}  # ckpt_path -> adapter_name
    for split, candidates in candidates_by_split.items():
        for cand in candidates:
            if not os.path.isdir(cand.ckpt_path):
                continue
            # Reuse an existing adapter_name if the same ckpt_path
            # appears in multiple splits' top-K (rare but possible
            # when training picks the same step as the per-split
            # winner).
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
                # Bad on-disk adapter — surface via NaN AUC below.
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
                    # Reseed NumPy's global RNG so every candidate sees
                    # the same few-shot shuffle order — otherwise
                    # per-ckpt AUC gets confounded by prompt-rendering
                    # noise (P2 / P5).
                    if shuffle_seed is not None:
                        np.random.seed(int(shuffle_seed))
                    # Score via a shim runner that reuses tokenizer +
                    # yes/no ids from base_runner and routes forward
                    # through the LoRA-wrapped model (active adapter
                    # = the one we just selected).
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
    """Build a runner-like object that scores via ``peft_model`` while
    reusing tokenizer / yes-no token ids from ``base_runner``.

    We don't subclass or instantiate ``LLMInferenceRunner`` afresh —
    instead we hand back a shim whose ``score_samples`` matches the
    base runner's signature but routes the forward through the
    LoRA-wrapped model.  Keeps ``LLMInferenceRunner``'s public API
    untouched.
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


# ─── Step 3: pick best + emit manifest ─────────────────────────────────────

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
        If any split has no valid (non-NaN) candidate — surface the
        failure rather than silently dropping a split, so callers
        always get a complete manifest.
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


# ─── fit_info.json read-back ────────────────────────────────────────────────

def read_fit_info(run_dir: str | Path) -> dict:
    """Read ``fit_info.json`` from a :class:`LoRATrainer` output dir.

    Returns the persisted training contract (yes/no token, base
    model name, prompt method etc.) so downstream stages can
    reconstruct the inference configuration without having to
    re-pass it manually.

    Raises
    ------
    FileNotFoundError
        If ``run_dir/fit_info.json`` doesn't exist (the trainer
        always writes it on :meth:`LoRATrainer.fit` completion;
        absence means the run never finished or was wiped).
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
    """Warn (or raise) when a val/test ``prompt_cfg`` diverges from
    the contract the LoRA was actually trained under.

    Compares three fields recorded in :func:`LoRATrainer.fit`'s
    ``fit_info.json``:

    * ``method`` — canonicalised via :func:`canon_method` so aliases
      (``"One_Hop_Subgraph_Sequence"`` / ``"ohs"`` / etc.) don't
      false-alarm.
    * ``task_name`` — drives the system/instruction header text;
      mismatch = different task framing.
    * ``model_name`` — drives the chat-template family
      (Llama / Qwen / Gemma); mismatch = different prompt structure
      even at the same method/task.

    Pass ``strict=True`` to raise instead of warning — useful in CI
    runs where a divergence indicates a config bug.
    """
    import warnings

    from coldddi.llm.prompts.binary_cls import canon_method, infer_model_family

    fit_prompt = fit_info.get("prompt_cfg") or {}
    if not fit_prompt:
        return  # train run didn't record a prompt_cfg — nothing to compare.

    diffs: list[str] = []

    # Method — canonicalise before comparing so alias forms agree.
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
            # canon_method shouldn't raise on a registered alias;
            # fall back to a raw equality check so we never lose the
            # signal silently.
            if cur_method != fit_method:
                diffs.append(
                    f"method: train={fit_method!r} vs val={cur_method!r}"
                )

    # Task name — affects the system header text.
    fit_task = fit_prompt.get("task_name")
    cur_task = getattr(prompt_cfg, "task_name", None)
    if fit_task is not None and cur_task is not None and fit_task != cur_task:
        diffs.append(f"task_name: train={fit_task!r} vs val={cur_task!r}")

    # Model name — affects the chat-template family. Compare family
    # rather than raw model id so "...-1B" vs "...-3B" of the same
    # family doesn't false-alarm.
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
