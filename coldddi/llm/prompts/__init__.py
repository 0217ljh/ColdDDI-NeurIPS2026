"""LLM prompt construction for ColdDDI.

Mirrors the layout of the original ``Version_1_1/dataloader/prompts/``
but takes plain dict samples + a :class:`PromptBuildConfig` (no global
SimpleNamespace dependency).

Public surface:

* :func:`format_messages_for_model` — chat-template formatter for
  Llama-3 / Qwen / Gemma / Mistral / DeepSeek / Baichuan / ChatGLM.
* :func:`build_binary_prompt`        — six-section prompt for the
  binary DDI task (P1-P5 + R0-R7 masking).
"""

from __future__ import annotations

from coldddi.llm.prompts.binary_cls import (
    PromptBuildConfig,
    build_binary_prompt,
    canon_method,
    infer_model_family,
)
from coldddi.llm.prompts.chat_formatter import format_messages_for_model

__all__ = [
    "PromptBuildConfig",
    "build_binary_prompt",
    "canon_method",
    "infer_model_family",
    "format_messages_for_model",
]
