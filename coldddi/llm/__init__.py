"""LLM stack for ColdDDI.

Hosts every module that is specific to large-language-model usage
(prompts, retrieval, training, inference, diagnostics).  Each sibling
is independently importable; this top-level package is a namespace
only.

Submodules
----------
* :mod:`coldddi.llm.prompts`    — chat-template formatters + the
  binary-DDI prompt builder (P1-P5 + R0-R7 masking).
* :mod:`coldddi.llm.retrieval`  — per-drug 1-hop subgraph and (future)
  few-shot example retrieval.
* :mod:`coldddi.llm.inference`  — inference-only runner that wraps a
  HuggingFace causal-LM (optionally with a LoRA adapter) and scores
  drug-pair samples via next-token logit comparison on ``" Yes"`` vs
  ``" No"``.
* :mod:`coldddi.llm.collator`   — FT-time data collator that masks
  every token before the assistant header so the cross-entropy loss
  fires only on the answer slot.
* :mod:`coldddi.llm.trainer`    — :class:`LoRATrainer` wrapper around
  ``transformers.Trainer`` with a custom ``compute_loss`` for binary
  Yes/No CE on the first non-masked token.
* :mod:`coldddi.llm.select_best` — parse-then-rank LoRA checkpoints by
  val-S2 AUC and emit a manifest the masking / KPS pipelines consume.
"""

from __future__ import annotations
