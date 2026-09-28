"""LoRA fine-tuning runner for the binary DDI task.

Port of ``Version_1_1/LLMs/binary_cls.FTRunner`` +
``build_trainer_binary_cls.TwoCollatorBinaryTrainer``. The runner
exposes a high-level :meth:`fit` that:

1. Loads the base causal-LM (with explicit dtype / device handling
   mirrored from :class:`coldddi.llm.inference.LLMInferenceRunner`).
2. Wraps it with a PEFT LoRA adapter.
3. Builds a :class:`transformers.Trainer` subclass whose
   :meth:`compute_loss` is the "first-non-template-token Yes/No CE"
   loss from the upstream paper.
4. Trains for ``num_epochs`` and saves checkpoints under
   ``output_dir``.

Limitations of this L3 port
---------------------------
* No DDP / SmartBatchSampler — that's an optimisation for later. Single
  GPU or CPU only.
* No dynamic negative sampling callback — the caller must supply the
  full ``train_samples`` list. (The :class:`PairDataset`'s
  ``get_train_negatives(epoch)`` already returns a fresh negatives
  list; the caller can rebuild ``train_samples`` per epoch externally,
  or run multiple ``fit()`` calls with ``resume_from_checkpoint``.)
* No CSV streaming during evaluation. ``predict`` is a thin wrapper
  that delegates to :class:`coldddi.llm.inference.LLMInferenceRunner`
  with the saved LoRA adapter.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from coldddi.llm.collator import BinaryFTCollator
from coldddi.llm.prompts.binary_cls import infer_model_family

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from coldddi.llm.prompts import PromptBuildConfig


@dataclass
class LoRAConfig:
    """LoRA hyperparameters (subset of :class:`peft.LoraConfig`)."""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


#: Paper-spec LoRA + training hyperparameters from Appendix D.2
#: Table 9 (the "every fine-tuned LLM in the paper" config).  Class
#: ``LLMTrainerConfig`` defaults are smoke values for fast CI; the
#: paper-grade configuration is materialised by callers via
#: :func:`paper_llm_trainer_config` (the one-call builder below).
#: Wiring this through ``scripts/run_llm.py`` as a CLI flag is a
#: follow-up workstream tracked separately.
#:
#: Per-model micro-batch tiering (paper Table 9):
#:   * 0.5B / 1B → micro_batch=8, grad_accum=2  (effective batch 16)
#:   * 3B  / 4B → micro_batch=4, grad_accum=4
#:   * 7B        → micro_batch=2, grad_accum=8
#:   * 13B / 14B → micro_batch=1, grad_accum=16
#:
#: This module exposes the per-size choices via
#: :func:`paper_micro_batch_for_size`; the rest of the hyperparams
#: are model-agnostic and live in :data:`PAPER_LORA_HYPERPARAMS`.
PAPER_LORA_HYPERPARAMS: dict[str, object] = {
    # Adapter (Table 9, "Adapter (PEFT/LoRA)" rows):
    "lora": {
        "r":               16,
        "alpha":           16,
        "dropout":         0.10,
        "target_modules":  ("q_proj", "k_proj", "v_proj", "o_proj"),
        "task_type":       "CAUSAL_LM",
        "bias":            "none",
    },
    # Training (Table 9, "Training" rows):
    "num_epochs":          4,
    "learning_rate":       5e-4,
    "warmup_ratio":        0.05,
    "weight_decay":        0.0,
    "max_grad_norm":       1.0,
    "optim":               "adamw_torch",
    "lr_scheduler_type":   "cosine",
    # Top-3 KG max_length per paper; Full KG (R4-R7) uses 4096
    # — toggle with paper_max_length(kg_top3=True/False).
    "max_length":          1250,
    # Effective batch = micro * grad_accum = 16 always; micro_batch
    # is size-tiered (use paper_micro_batch_for_size below).
    "save_total_limit":    20,
    "primary_val_split":   "S2",
}


def paper_micro_batch_for_size(billions: float) -> tuple[int, int]:
    """Return ``(micro_batch, grad_accum)`` per paper Table 9 size
    tier so the effective batch is always 16.

    Tiers (Appendix D.2 Table 9):
        ≤ 1B  → (8, 2)
        ≤ 4B  → (4, 4)
        ≤ 7B  → (2, 8)
        > 7B  → (1, 16)
    """
    if billions <= 1.0:
        return 8, 2
    if billions <= 4.0:
        return 4, 4
    if billions <= 7.0:
        return 2, 8
    return 1, 16


def paper_max_length(kg_top3: bool = True) -> int:
    """Return paper-spec ``max_length`` for the LLM prompt context.

    * Top-3 KG (R0-R3 mask conditions): 1250 (paper default).
    * Full KG (R4-R7 mask conditions): 4096.
    """
    return 1250 if kg_top3 else 4096


def paper_llm_trainer_config(
    *,
    model_name: str,
    output_dir: str,
    billions: float,
    kg_top3: bool = True,
    **overrides: object,
) -> "LLMTrainerConfig":
    """Construct an :class:`LLMTrainerConfig` from paper Table 9.

    Convenience helper for ``scripts/run_llm.py`` and any other code
    that wants the paper-grade config in one call.  ``overrides``
    let callers override individual fields (e.g. ``output_dir``)
    without re-spelling every paper field.
    """
    micro, accum = paper_micro_batch_for_size(billions)
    base = dict(PAPER_LORA_HYPERPARAMS)
    lora_kwargs = base.pop("lora")
    cfg_kwargs = dict(base)
    cfg_kwargs["model_name"] = model_name
    cfg_kwargs["output_dir"] = output_dir
    cfg_kwargs["micro_batch_size"] = micro
    cfg_kwargs["gradient_accumulation_steps"] = accum
    cfg_kwargs["max_length"] = paper_max_length(kg_top3=kg_top3)
    cfg_kwargs["lora"] = LoRAConfig(**lora_kwargs)
    cfg_kwargs.update(overrides)
    return LLMTrainerConfig(**cfg_kwargs)


@dataclass
class LLMTrainerConfig:
    """Training-time configuration for :class:`LoRATrainer`.

    Mirrors the subset of ``Version_1_1/configs/ddi_finetune_config_k8s.py``
    that the upstream FTRunner actually consumes.
    """

    model_name: str
    output_dir: str

    # Tokens
    yes_token: str = " Yes"
    no_token: str = " No"
    max_length: int = 1024

    # Compute
    dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = True
    cache_dir: str | None = None

    # Optimisation
    num_epochs: int = 1
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    eval_strategy: str = "steps"  # "no" / "steps" / "epoch"

    #: Name of the validation split that drives HF Trainer's
    #: ``metric_for_best_model``. Default is ``"S2"`` because the
    #: ColdDDI cold-start setting is what we ultimately ship — S0/S1
    #: are diagnostic-only.  Must match one of the keys passed to
    #: :meth:`LoRATrainer.fit`'s ``val_samples`` dict.
    primary_val_split: str = "S2"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Misc
    seed: int = 42
    report_to: tuple[str, ...] = ()
    disable_tqdm: bool = False


def _resolve_device(d: str) -> str:
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _resolve_dtype(d: str) -> torch.dtype:
    return getattr(torch, d)


class LoRATrainer:
    """LoRA fine-tuning entry point for the binary DDI task.

    Usage
    -----
    >>> cfg = LLMTrainerConfig(model_name="hf-internal-testing/...",
    ...                        output_dir="./ft_out")
    >>> trainer = LoRATrainer(cfg)
    >>> trainer.fit(train_samples=[...], val_samples={"S2": [...]})
    >>> trainer.save_adapter("./ft_out/best")
    """

    def __init__(self, cfg: LLMTrainerConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.family = infer_model_family(cfg.model_name)
        self.model: "PreTrainedModel | None" = None
        self.tokenizer: "PreTrainedTokenizer | None" = None
        self.yes_id: int | None = None
        self.no_id: int | None = None
        self._hf_trainer = None  # set by fit

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.model is not None:
            return

        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            set_seed,
        )

        # Seed before LoRA adapter init so the random matrices are
        # reproducible across runs. TrainingArguments(seed=...) only
        # seeds the optimizer trajectory — it does not retroactively
        # cover get_peft_model's random init.
        set_seed(int(self.cfg.seed))

        tok_kwargs: dict = dict(
            trust_remote_code=self.cfg.trust_remote_code,
            padding_side="left",
        )
        if self.cfg.cache_dir is not None:
            tok_kwargs["cache_dir"] = self.cfg.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, **tok_kwargs
        )
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        yes_ids = self.tokenizer.encode(self.cfg.yes_token, add_special_tokens=False)
        no_ids = self.tokenizer.encode(self.cfg.no_token, add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"YES/NO must be single tokens for tokenizer "
                f"{self.cfg.model_name!r}; got yes={yes_ids} no={no_ids}."
            )
        self.yes_id = int(yes_ids[0])
        self.no_id = int(no_ids[0])

        effective_dtype = self.cfg.dtype
        if self.device == "cpu" and effective_dtype in ("bfloat16", "float16"):
            effective_dtype = "float32"
        model_kwargs = dict(
            torch_dtype=_resolve_dtype(effective_dtype),
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.cfg.cache_dir is not None:
            model_kwargs["cache_dir"] = self.cfg.cache_dir
        if self.cfg.device == "auto" and self.device != "cpu":
            model_kwargs["device_map"] = "auto"
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            base = base.to(torch.device(self.device))

        lora_cfg = LoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=list(self.cfg.lora.target_modules),
            bias=self.cfg.lora.bias,
            task_type=self.cfg.lora.task_type,
        )
        self.model = get_peft_model(base, lora_cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_samples: list[dict],
        val_samples: dict[str, list[dict]] | None = None,
        *,
        prompt_cfg: "PromptBuildConfig | None" = None,
        resume_from_checkpoint: bool | str | None = None,
    ) -> dict:
        """Run LoRA fine-tuning.

        Parameters
        ----------
        train_samples
            List of dicts with ``{"text": <full FT prompt>, "cls_labels": 0/1}``.
            Each prompt should already include the trailing answer token
            (``" Yes"`` / ``" No"``) — i.e. ``build_binary_prompt`` with the
            default ``assistant_content=None``.
        val_samples
            Validation splits as ``{split_name: list[sample_dict]}``.
            The recommended setting is ``{"S2": [...]}`` for cold-start
            evaluation; ``S0`` and ``S1`` are optional diagnostic
            splits.  At least one split is required when
            ``cfg.eval_strategy != "no"`` — that gates HF Trainer's
            ``metric_for_best_model = eval_{primary_val_split}_loss``
            and the L5 :func:`parse_candidate_ckpts` path.  Pass
            ``None`` or an empty dict only when running a no-eval
            smoke / debug session.
        prompt_cfg
            Optional :class:`PromptBuildConfig` that was used to render
            ``train_samples``.  When provided, the prompt method /
            task / model name are persisted to ``fit_info.json`` so
            L5 selection can later verify the val-time prompt matches
            the train-time prompt and surface a warning otherwise.
        resume_from_checkpoint
            Auto-resume control.  ``None`` (default) auto-detects:
            if ``output_dir`` already contains ``checkpoint-*`` dirs,
            HF Trainer is told to resume from the latest one (and a
            log line is printed); otherwise training starts fresh.
            Pass ``False`` to force a fresh run even when checkpoints
            exist (useful when re-running a tuning grid).  Pass a
            specific path string to point at a non-default checkpoint.

        Returns
        -------
        Dict with ``best_ckpt`` (path or None), ``output_dir``,
        ``log_history`` (list of trainer log entries), plus
        ``yes_token`` / ``no_token`` / ``model_name`` /
        ``model_family`` / ``max_length`` / ``prompt_cfg`` so
        downstream stages can reconstruct the inference contract
        without re-passing them.
        """
        self.load()
        from transformers import Trainer, TrainingArguments

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        collator = BinaryFTCollator(
            tokenizer=self.tokenizer,
            model_family=self.family,
            max_length=self.cfg.max_length,
        )

        # Convert val_samples to a single Trainer-compatible dataset or
        # dict-of-datasets so HF Trainer can score each separately.
        train_ds = _SampleList(train_samples)
        eval_ds: dict | _SampleList | None
        if val_samples:
            if not isinstance(val_samples, dict):
                raise TypeError(
                    "val_samples must be a dict {split_name: list[sample_dict]}; "
                    f"got {type(val_samples).__name__}. The recommended primary "
                    f"split is 'S2' (the cold-start eval); add 'S0' and 'S1' as "
                    "optional diagnostics."
                )
            empty = [k for k, v in val_samples.items() if not v]
            if empty:
                raise ValueError(
                    f"val_samples has empty splits {empty}; remove them or "
                    "populate them. At least one non-empty split is required."
                )
            if self.cfg.primary_val_split not in val_samples:
                raise ValueError(
                    f"primary_val_split={self.cfg.primary_val_split!r} is not in "
                    f"val_samples (keys={sorted(val_samples)}). Either change "
                    "cfg.primary_val_split or add the matching split to val_samples."
                )
            eval_ds = {k: _SampleList(v) for k, v in val_samples.items()}
        else:
            if self.cfg.eval_strategy != "no":
                raise ValueError(
                    "val_samples is required when cfg.eval_strategy != 'no'. "
                    "Pass at least one split (e.g. {'S2': [...]}) or set "
                    "cfg.eval_strategy='no' for a no-eval debug run."
                )
            eval_ds = None

        # Align save_strategy with eval_strategy when load_best_model_at_end
        # is True — HF Trainer raises otherwise (e.g. eval=epoch + save=steps).
        effective_eval_strategy = (
            self.cfg.eval_strategy if eval_ds is not None else "no"
        )
        save_strategy = (
            effective_eval_strategy if eval_ds is not None else "steps"
        )

        args_kwargs = dict(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.micro_batch_size,
            per_device_eval_batch_size=self.cfg.micro_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            warmup_ratio=self.cfg.warmup_ratio,
            weight_decay=self.cfg.weight_decay,
            optim=self.cfg.optim,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            max_grad_norm=self.cfg.max_grad_norm,
            num_train_epochs=self.cfg.num_epochs,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            save_strategy=save_strategy,
            save_total_limit=self.cfg.save_total_limit,
            eval_steps=self.cfg.eval_steps if eval_ds is not None else None,
            bf16=(self._effective_dtype() == "bfloat16"),
            fp16=(self._effective_dtype() == "float16"),
            remove_unused_columns=False,
            report_to=list(self.cfg.report_to),
            seed=self.cfg.seed,
            disable_tqdm=self.cfg.disable_tqdm,
            load_best_model_at_end=eval_ds is not None,
        )
        # Drive HF Trainer's "best ckpt" by the primary val split's
        # loss so the saved best model is the right one for cold-start
        # (S2) instead of the trainer default eval_S0_loss.
        if eval_ds is not None:
            args_kwargs["metric_for_best_model"] = (
                f"eval_{self.cfg.primary_val_split}_loss"
            )
            args_kwargs["greater_is_better"] = False

        # transformers 4.x used `evaluation_strategy`; 5.x renamed to
        # `eval_strategy`.  Pick whichever the installed runtime
        # accepts so the trainer works against both pinning regimes.
        import inspect as _inspect

        ta_sig = _inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in ta_sig.parameters:
            args_kwargs["eval_strategy"] = effective_eval_strategy
        else:
            args_kwargs["evaluation_strategy"] = effective_eval_strategy

        args = TrainingArguments(**args_kwargs)

        # transformers 5.x renamed `tokenizer` → `processing_class`. Pass
        # through whichever the runtime supports.
        trainer_kwargs: dict = dict(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        import inspect

        sig = inspect.signature(Trainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = _BinaryClsTrainer(
            yes_token_id=self.yes_id,
            no_token_id=self.no_id,
            **trainer_kwargs,
        )
        self._hf_trainer = trainer

        # Auto-resume: if the caller didn't decide, look for any
        # ``checkpoint-*`` in output_dir and resume from it. Explicit
        # ``False`` skips resume even when checkpoints exist (clean
        # re-run); a string is forwarded as the checkpoint path.
        if resume_from_checkpoint is None:
            ckpt_steps = sorted(
                int(p.name.split("-")[1])
                for p in Path(self.cfg.output_dir).glob("checkpoint-*")
                if p.is_dir() and p.name.startswith("checkpoint-")
                and p.name.split("-", 1)[1].isdigit()
            )
            if ckpt_steps:
                resume_from_checkpoint = True
                print(
                    f"[LoRATrainer] auto-resume: found "
                    f"{len(ckpt_steps)} existing checkpoint(s) in "
                    f"{self.cfg.output_dir} (latest step={ckpt_steps[-1]}); "
                    "passing resume_from_checkpoint=True to HF Trainer. "
                    "Pass resume_from_checkpoint=False to fit() to override."
                )
            else:
                resume_from_checkpoint = False

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Persist enough metadata that L5 / L2 can reconstruct the
        # inference contract (yes/no token, base model, prompt
        # method) without the caller having to re-pass them. Without
        # this, a non-default `yes_token` set on the trainer would
        # silently diverge from L2's runner default (' Yes') and the
        # model would score the wrong logits.
        prompt_cfg_info: dict | None = None
        if prompt_cfg is not None:
            prompt_cfg_info = {
                "task_name": getattr(prompt_cfg, "task_name", None),
                "method": getattr(prompt_cfg, "method", None),
                "model_name": getattr(prompt_cfg, "model_name", None),
            }
        info = {
            "output_dir": self.cfg.output_dir,
            "best_ckpt": getattr(trainer.state, "best_model_checkpoint", None),
            "log_history": list(trainer.state.log_history),
            "model_name": self.cfg.model_name,
            "model_family": self.family,
            "yes_token": self.cfg.yes_token,
            "no_token": self.cfg.no_token,
            "max_length": self.cfg.max_length,
            "primary_val_split": self.cfg.primary_val_split,
            "prompt_cfg": prompt_cfg_info,
        }
        with open(os.path.join(self.cfg.output_dir, "fit_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        return info

    def _effective_dtype(self) -> str:
        d = self.cfg.dtype
        if self.device == "cpu" and d in ("bfloat16", "float16"):
            return "float32"
        return d

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_adapter(self, path: str | Path) -> None:
        """Save the trained LoRA adapter to ``path``.

        Combined with the base ``model_name`` (in ``cfg``), this is
        enough for :class:`coldddi.llm.inference.LLMInferenceRunner` to
        reload the fine-tuned model for inference.
        """
        if self.model is None:
            raise RuntimeError("fit() must be called before save_adapter().")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        # Also dump a small adapter_info.json so the inference runner
        # knows which base model + token tuple this adapter was trained
        # against.
        info = {
            "base_model_name": self.cfg.model_name,
            "yes_token": self.cfg.yes_token,
            "no_token": self.cfg.no_token,
            "model_family": self.family,
            "max_length": self.cfg.max_length,
        }
        with open(Path(path) / "adapter_info.json", "w") as f:
            json.dump(info, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────

class _SampleList(torch.utils.data.Dataset):
    """Trivial wrapper that lets HF Trainer iterate a Python list."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _build_binary_trainer_cls():
    """Lazy-import HF Trainer and build the subclass with our compute_loss."""
    from transformers import Trainer

    class _BinaryClsTrainer(Trainer):
        """HF Trainer with the "first valid token Yes/No CE" loss.

        Mirrors ``TwoCollatorBinaryTrainer.compute_loss`` byte-exactly.
        """

        def __init__(self, *args, yes_token_id, no_token_id, **kwargs):
            super().__init__(*args, **kwargs)
            self.yes_token_id = int(yes_token_id)
            self.no_token_id = int(no_token_id)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]                  # (B, L)
            cls_labels = inputs["cls_labels"].long()   # (B,)
            # Exclude BOTH ``cls_labels`` (our binary target — not a
            # forward arg) AND ``labels`` (the masked-CE label tensor —
            # only used by *this* compute_loss to locate the first
            # non-template token).  Passing ``labels`` into
            # ``model.forward(...)`` triggers HF's auto-attached
            # ``ForCausalLMLoss`` which fp32-casts the full
            # (B, L, vocab_size) logits tensor for numerical stability
            # — typically 60-70% of total training memory at vocab
            # 150k+ — and we throw the resulting loss away anyway. Our
            # binary CE on the {Yes, No} token slot is the only loss
            # that should flow into backward.
            model_inputs = {
                k: v for k, v in inputs.items()
                if k not in ("cls_labels", "labels")
            }
            outputs = model(**model_inputs)
            logits = outputs.logits.float()            # (B, L, V)

            shift_logits = logits[:, :-1, :]           # (B, L-1, V)
            shift_labels = labels[:, 1:]               # (B, L-1)
            valid = shift_labels != -100
            has_valid = valid.any(dim=1)

            if not has_valid.any():
                # No row has a non-masked target — return a 0-loss
                # tensor that still keeps grad-fn intact.
                loss = (logits.sum() * 0.0).requires_grad_()
                return (loss, outputs) if return_outputs else loss

            first_pos = valid.float().argmax(dim=1)
            b_idx = torch.nonzero(has_valid, as_tuple=False).squeeze(1)
            p_idx = first_pos[has_valid]
            first_logits = shift_logits[b_idx, p_idx, :]
            binary_logits = torch.stack(
                [first_logits[:, self.no_token_id], first_logits[:, self.yes_token_id]],
                dim=-1,
            )
            targets = cls_labels[has_valid]
            loss = nn.functional.cross_entropy(binary_logits, targets)
            return (loss, outputs) if return_outputs else loss

    return _BinaryClsTrainer


# Late-bound to avoid importing transformers at module import time.
def _BinaryClsTrainer(*args, yes_token_id, no_token_id, **kwargs):
    cls = _build_binary_trainer_cls()
    return cls(*args, yes_token_id=yes_token_id, no_token_id=no_token_id, **kwargs)


__all__ = [
    "LLMTrainerConfig",
    "LoRAConfig",
    "LoRATrainer",
]
