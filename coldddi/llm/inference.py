"""LLM inference runner — port of
``Version_1_1/LLMs/binary_cls.BinaryClsRunner``.

The runner loads a base causal-LM, optionally attaches a LoRA adapter,
and scores DDI drug-pair samples by extracting the next-token logit at
the position that should emit ``" Yes"`` / ``" No"``. The output is a
:class:`pandas.DataFrame` with per-pair ``p_yes`` / ``p_no`` / ``pred``,
mirroring the column layout the original BinaryClsEvaluator consumes.

Design notes
------------
* **Prompts are built with an empty label** (``sample["label"] = ""``)
  so the assistant content collapses to ``" "`` — the prompt then ends
  right at the position where the model is expected to emit the answer
  token, and ``logits[:, -1, :]`` (without any shift) gives the correct
  next-token distribution. This matches the original
  ``InferenceCollator`` path of ``BinaryClsRunner.predict``.
* **YES/NO tokens are required to be single tokens** for the chosen
  tokenizer. The runner validates this on first load and raises a
  ``ValueError`` otherwise (so callers can pass alternative spellings
  like ``" Interaction"`` / ``" No Interaction"`` for tokenisers where
  ``" Yes"`` would split into multiple subwords).
* **LoRA adapter loading** delegates to :mod:`peft`; we only require
  ``peft`` at adapter-loading time.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from coldddi.llm.prompts.binary_cls import (
    PromptBuildConfig,
    build_binary_prompt,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LLMRunnerConfig:
    """Inference-time configuration for :class:`LLMInferenceRunner`.

    Attributes
    ----------
    model_name
        HuggingFace repo id or local path of the base causal-LM.
    adapter_path
        Optional LoRA adapter directory; passed to
        :class:`peft.PeftModel.from_pretrained` when set.
    yes_token / no_token
        The literal string forms of the answer tokens. Both must
        encode to a single token id by the loaded tokenizer.
    dtype
        ``"bfloat16"`` / ``"float16"`` / ``"float32"``.
    device
        ``"auto"`` (default) or an explicit ``"cuda" / "cuda:0" / "cpu"``.
    batch_size
        Forward batch size for :meth:`score_samples`.
    max_length
        Tokenizer ``max_length`` (and truncation cap).
    """

    model_name: str
    adapter_path: str | None = None
    yes_token: str = " Yes"
    no_token: str = " No"
    dtype: str = "bfloat16"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 1024
    trust_remote_code: bool = True
    cache_dir: str | None = None


def _resolve_device(d: str) -> str:
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _resolve_dtype(d: str) -> torch.dtype:
    return getattr(torch, d)


class LLMInferenceRunner:
    """Inference-only runner for the binary DDI task.

    Usage
    -----
    >>> from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
    >>> runner = LLMInferenceRunner(LLMRunnerConfig(
    ...     model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
    ...     device="cpu",
    ... ))
    >>> runner.load()
    >>> df = runner.score_samples(samples, PromptBuildConfig(method="Zero_Shot_Sequence", ...))
    """

    def __init__(self, cfg: LLMRunnerConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model: "PreTrainedModel | None" = None
        self.tokenizer: "PreTrainedTokenizer | None" = None
        self.yes_id: int | None = None
        self.no_id: int | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer + model and resolve YES/NO token ids.

        Safe to call multiple times — subsequent invocations are no-ops
        once ``self.model`` is populated.
        """
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_kwargs = dict(
            trust_remote_code=self.cfg.trust_remote_code,
            padding_side="left",
        )
        if self.cfg.cache_dir is not None:
            tok_kwargs["cache_dir"] = self.cfg.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, **tok_kwargs
        )
        # Truncate from the LEFT so the assistant header / answer slot
        # at the end of the prompt is preserved when an over-long input
        # gets clipped. Right-truncation would silently strip the very
        # tokens we score on.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Resolve YES/NO ids before model load (cheap and surfaces
        # tokeniser-mismatch errors early).
        yes_ids = self.tokenizer.encode(
            self.cfg.yes_token, add_special_tokens=False
        )
        no_ids = self.tokenizer.encode(
            self.cfg.no_token, add_special_tokens=False
        )
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"YES/NO must be single tokens for tokenizer "
                f"{self.cfg.model_name!r}; got yes={yes_ids} no={no_ids}. "
                "Choose an alternative spelling (e.g. ' Yes' / ' No' "
                "vs ' Interaction' / ' No Interaction') or pass the "
                "yes_token / no_token fields explicitly."
            )
        self.yes_id = int(yes_ids[0])
        self.no_id = int(no_ids[0])

        # CPU has no bfloat16 / float16 kernels for most LM ops; force
        # float32 unless the caller has already passed a CPU-friendly
        # explicit dtype.
        effective_dtype = self.cfg.dtype
        if self.device == "cpu" and effective_dtype in ("bfloat16", "float16"):
            effective_dtype = "float32"
        model_kwargs = dict(
            torch_dtype=_resolve_dtype(effective_dtype),
            trust_remote_code=self.cfg.trust_remote_code,
        )
        # Only delegate device placement to `device_map="auto"` when the
        # caller asked for auto-placement; an explicit "cuda:N" / "cpu"
        # must respect the requested device exactly.
        if self.cfg.device == "auto" and self.device != "cpu":
            model_kwargs["device_map"] = "auto"
        if self.cfg.cache_dir is not None:
            model_kwargs["cache_dir"] = self.cfg.cache_dir
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            self.model = self.model.to(torch.device(self.device))
        self.model.eval()

        if self.cfg.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading LoRA adapter requires `peft` to be installed."
                ) from exc
            peft_kwargs: dict = {"is_trainable": False}
            if self.cfg.cache_dir is not None:
                peft_kwargs["cache_dir"] = self.cfg.cache_dir
            self.model = PeftModel.from_pretrained(
                self.model, self.cfg.adapter_path, **peft_kwargs
            )
            self.model.eval()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _prepare_prompts(
        self,
        samples: list[dict],
        prompt_cfg: PromptBuildConfig,
        *,
        drug_id2name: dict,
        drug_id2smiles: dict,
        key_entity_map: dict | None,
    ) -> list[str]:
        """Render every sample as an open-ended prompt (no answer token).

        We override the assistant message content to the **empty
        string** (not the default ``" "`` with leading space) so the
        rendered prompt ends right at the assistant header / model
        marker.  This is the configuration where ``logits[:, -1, :]``
        directly scores the leading-space ``" Yes"`` / ``" No"`` tokens
        — for Llama-3 / Qwen / Gemma tokenizers those literals encode
        to single token ids that already carry the leading space, so a
        trailing space in the prompt would corrupt the position
        alignment.
        """
        prompts: list[str] = []
        for s in samples:
            prompts.append(
                build_binary_prompt(
                    s,
                    prompt_cfg,
                    drug_id2name=drug_id2name,
                    drug_id2smiles=drug_id2smiles,
                    key_entity_map=key_entity_map,
                    assistant_content="",
                )
            )
        return prompts

    @torch.no_grad()
    def score_samples(
        self,
        samples: list[dict],
        prompt_cfg: PromptBuildConfig,
        *,
        drug_id2name: dict | None = None,
        drug_id2smiles: dict | None = None,
        key_entity_map: dict | None = None,
        progress: bool = False,
        resume_path: str | Path | None = None,
        save_every: int = 50,
    ) -> pd.DataFrame:
        """Score every sample and return a per-row DataFrame.

        Output columns:
        ``drug_a_id, drug_b_id, p_no, p_yes, pred, prompt`` (the rendered
        prompt is included for downstream diagnostics).

        Resume
        ------
        Pass ``resume_path`` to enable crash-safe inference for a long
        run (e.g. test_s1 ≈ 450 k pairs).  The runner writes a
        ``{resume_path}.partial`` parquet every ``save_every`` batches;
        on a re-invocation with the same ``resume_path`` it reads back
        whatever pairs already scored, skips them, and continues.  On
        success the partial file is atomically renamed to the final
        ``resume_path``.  Pass ``None`` to disable (default) — behaves
        identically to pre-resume releases.
        """
        if self.model is None:
            self.load()

        # ── Resume bootstrap ──────────────────────────────────────────
        rows: list[dict] = []
        already: set[tuple[str, str]] = set()
        partial_path: Path | None = None
        final_path: Path | None = None
        if resume_path is not None:
            final_path = Path(resume_path)
            partial_path = final_path.with_suffix(final_path.suffix + ".partial")
            # Final already complete → just read and return it.
            if final_path.is_file():
                return pd.read_parquet(final_path)
            if partial_path.is_file():
                existing = pd.read_parquet(partial_path)
                rows = existing.to_dict("records")
                already = {
                    (str(r["drug_a_id"]), str(r["drug_b_id"])) for r in rows
                }

        # Filter out already-scored samples so we don't re-spend GPU
        # time on duplicates.
        if already:
            todo_samples = [
                s for s in samples
                if (str(s.get("drug_a_id", "")),
                    str(s.get("drug_b_id", ""))) not in already
            ]
        else:
            todo_samples = list(samples)

        prompts = self._prepare_prompts(
            todo_samples,
            prompt_cfg,
            drug_id2name=drug_id2name or {},
            drug_id2smiles=drug_id2smiles or {},
            key_entity_map=key_entity_map,
        )

        bs = self.cfg.batch_size
        iterator = range(0, len(prompts), bs)
        if progress:
            iterator = tqdm(iterator, desc="LLM inference")

        for batch_idx, start in enumerate(iterator):
            batch_prompts = prompts[start : start + bs]
            batch_samples = todo_samples[start : start + bs]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            )
            # `device_map="auto"` may shard the model; use the input
            # embedding's device so we don't accidentally land on a
            # non-input shard.
            dev = self.model.get_input_embeddings().weight.device
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits  # (B, L, V)
            last_logits = logits[:, -1, :]  # next-token distribution
            yn_logits = torch.stack(
                [last_logits[:, self.no_id], last_logits[:, self.yes_id]],
                dim=-1,
            )
            probs = torch.softmax(yn_logits, dim=-1)
            p_no = probs[:, 0].float().cpu().numpy()
            p_yes = probs[:, 1].float().cpu().numpy()
            preds = (p_yes >= 0.5).astype(int)
            for i, sample in enumerate(batch_samples):
                rows.append(
                    {
                        "drug_a_id": sample.get("drug_a_id", ""),
                        "drug_b_id": sample.get("drug_b_id", ""),
                        "p_no": float(p_no[i]),
                        "p_yes": float(p_yes[i]),
                        "pred": int(preds[i]),
                        "prompt": batch_prompts[i],
                    }
                )

            # Incremental save — every save_every batches OR on the
            # last batch. Atomic-via-rename so a crash mid-write can't
            # leave a corrupt partial file.
            if (partial_path is not None
                and ((batch_idx + 1) % max(1, save_every) == 0
                     or start + bs >= len(prompts))):
                tmp_path = partial_path.with_suffix(
                    partial_path.suffix + ".tmp"
                )
                pd.DataFrame(rows).to_parquet(tmp_path)
                tmp_path.replace(partial_path)

        # On success rename partial → final.
        if partial_path is not None and final_path is not None:
            if partial_path.is_file():
                partial_path.replace(final_path)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Convenience: score directly from PairDataset rows
    # ------------------------------------------------------------------

    def score_pairs(
        self,
        pairs: pd.DataFrame,
        *,
        ds,
        prompt_cfg: PromptBuildConfig,
        subgraph_map=None,
        fewshot_map=None,
        labels=None,
        progress: bool = False,
    ) -> pd.DataFrame:
        """Score a DataFrame of ``(drug_a_id, drug_b_id)`` rows.

        Internally builds samples via
        :func:`coldddi.llm.retrieval.llm_view.to_llm_samples` then
        delegates to :meth:`score_samples`.
        """
        from coldddi.llm.retrieval.llm_view import to_llm_samples

        samples = to_llm_samples(
            pairs,
            labels,
            ds=ds,
            subgraph_map=subgraph_map,
            fewshot_map=fewshot_map,
        )
        # Look up smiles/name maps off the dataset once.
        drug_id2name: dict[str, str] = {}
        drug_id2smiles: dict[str, str] = {}
        if ds.drugs is not None:
            for _, row in ds.drugs.iterrows():
                did = str(row["drugbank_id"])
                if "name" in ds.drugs.columns and not pd.isna(row["name"]):
                    drug_id2name[did] = str(row["name"])
                if "smiles" in ds.drugs.columns and not pd.isna(row["smiles"]):
                    drug_id2smiles[did] = str(row["smiles"])
        return self.score_samples(
            samples,
            prompt_cfg,
            drug_id2name=drug_id2name,
            drug_id2smiles=drug_id2smiles,
            key_entity_map=None,
            progress=progress,
        )


__all__ = [
    "LLMRunnerConfig",
    "LLMInferenceRunner",
]
