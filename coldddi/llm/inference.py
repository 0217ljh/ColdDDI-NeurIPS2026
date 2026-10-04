"""Score binary DDI pairs with a causal LM and an optional LoRA adapter.

Derived from ``BinaryClsRunner``. Prompts end before the answer token;
next-token logits for ``" Yes"`` and ``" No"`` produce ``p_yes``, ``p_no``,
and ``pred`` columns. Both answer strings must encode to single tokens.
``peft`` is required only when loading an adapter.
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
        Answer strings; each must encode to one token in the loaded tokenizer.
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
    """Inference-only runner for the binary DDI task."""

    def __init__(self, cfg: LLMRunnerConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model: "PreTrainedModel | None" = None
        self.tokenizer: "PreTrainedTokenizer | None" = None
        self.yes_id: int | None = None
        self.no_id: int | None = None

    # Model loading

    def load(self) -> None:
        """Load the tokenizer and model, and resolve YES/NO token IDs.

        Does nothing if ``self.model`` is already populated.
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
        # Left truncation preserves the assistant header and answer slot.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Check answer-token compatibility before loading the model.
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

        # On CPU, replace bfloat16/float16 with float32 for kernel compatibility.
        effective_dtype = self.cfg.dtype
        if self.device == "cpu" and effective_dtype in ("bfloat16", "float16"):
            effective_dtype = "float32"
        model_kwargs = dict(
            torch_dtype=_resolve_dtype(effective_dtype),
            trust_remote_code=self.cfg.trust_remote_code,
        )
        # Use device_map="auto" only for requested non-CPU auto-placement;
        # explicit "cuda:N" / "cpu" requests must be respected.
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

    # Scoring

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

        Set assistant content to ``""``, not ``" "``, so the prompt ends
        at the assistant header / model marker. ``logits[:, -1, :]`` then
        scores ``" Yes"`` / ``" No"`` directly. For Llama-3 / Qwen / Gemma,
        these single tokens already include the leading space; a trailing
        prompt space would misalign them.
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
        ``drug_a_id, drug_b_id, p_no, p_yes, pred, prompt``; ``prompt``
        contains the rendered prompt for diagnostics.

        Resume
        ------
        With ``resume_path``, write ``{resume_path}.partial`` parquet every
        ``save_every`` batches and on the last batch, using atomic renames.
        Reusing the path skips pairs already scored; an existing final file
        is returned directly. On success, rename the partial file to
        ``resume_path``. ``None`` (default) disables resume support.
        """
        if self.model is None:
            self.load()

        # Restore resume state.
        rows: list[dict] = []
        already: set[tuple[str, str]] = set()
        partial_path: Path | None = None
        final_path: Path | None = None
        if resume_path is not None:
            final_path = Path(resume_path)
            partial_path = final_path.with_suffix(final_path.suffix + ".partial")
            # Return completed results without rescoring.
            if final_path.is_file():
                return pd.read_parquet(final_path)
            if partial_path.is_file():
                existing = pd.read_parquet(partial_path)
                rows = existing.to_dict("records")
                already = {
                    (str(r["drug_a_id"]), str(r["drug_b_id"])) for r in rows
                }

        # Skip pairs already scored.
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
            # For sharded models, send inputs to the input embedding's device.
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

            # Save every save_every batches and on the last batch.
            # Atomic rename protects the partial file from interrupted writes.
            if (partial_path is not None
                and ((batch_idx + 1) % max(1, save_every) == 0
                     or start + bs >= len(prompts))):
                tmp_path = partial_path.with_suffix(
                    partial_path.suffix + ".tmp"
                )
                pd.DataFrame(rows).to_parquet(tmp_path)
                tmp_path.replace(partial_path)

        # Rename partial results to the final path on success.
        if partial_path is not None and final_path is not None:
            if partial_path.is_file():
                partial_path.replace(final_path)
        return pd.DataFrame(rows)

    # Score PairDataset rows.

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

        Build samples with :func:`coldddi.llm.retrieval.llm_view.to_llm_samples`
        and delegate to :meth:`score_samples`.
        """
        from coldddi.llm.retrieval.llm_view import to_llm_samples

        samples = to_llm_samples(
            pairs,
            labels,
            ds=ds,
            subgraph_map=subgraph_map,
            fewshot_map=fewshot_map,
        )
        # Build SMILES and name maps once.
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
