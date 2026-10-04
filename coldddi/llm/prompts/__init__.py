"""Chat formatting and binary DDI prompts, including masking variants."""

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
