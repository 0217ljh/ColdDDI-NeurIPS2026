"""Format role/content messages as model-specific chat strings.

Llama, Qwen, Gemma, Mistral and DeepSeek leave the assistant turn open
for answer-token training and inference. ChatGLM appends ``\\n\\n`` and
Baichuan appends ``<reserved_108>`` after the answer; callers must account
for these suffixes when locating the answer boundary.
"""

from __future__ import annotations

from typing import Iterable, Mapping


def format_messages_for_model(
    messages: Iterable[Mapping[str, str]],
    model_name: str,
) -> str:
    """Format role/content messages using a case-insensitive model-name match."""
    m = model_name.lower()
    if any(k in m for k in ("llama", "meta-llama")):
        return _format_llama3_messages(messages)
    if "qwen" in m:
        return _format_qwen_messages(messages)
    if "mistral" in m:
        return _format_mistral_messages(messages)
    if "deepseek" in m:
        return _format_deepseek_messages(messages)
    if "chatglm" in m:
        return _format_chatglm_messages(messages)
    if "baichuan" in m:
        return _format_baichuan_messages(messages)
    if "gemma" in m:
        return _format_gemma_messages(messages)
    # Default: Llama-3 layout.
    return _format_llama3_messages(messages)


def _format_llama3_messages(messages):
    """Llama-3 / Llama-3.x / Llama-3.2 chat template."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{content}<|eot_id|>\n\n"
            )
        elif role == "user":
            text += (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{content}<|eot_id|>\n\n"
            )
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>{content}"
    return text


def _format_qwen_messages(messages):
    """Qwen / Qwen2 / Qwen2.5 ChatML template."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant{content}"
    return text


def _format_mistral_messages(messages):
    """Mistral [INST] template."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<s>[INST]system\n{content}[/INST]\n\n"
        elif role == "user":
            text += f"[INST]user\n{content} [/INST]\n\n"
        elif role == "assistant":
            text += f"[INST]assistant\n{content}"
    return text


def _format_deepseek_messages(messages):
    """Format DeepSeek messages with the Llama-3 layout."""
    return _format_llama3_messages(messages)


def _format_chatglm_messages(messages):
    """ChatGLM round-style template; preserve the Chinese role markers."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"[Round 1]\n\n问：{content}\n\n答："
        elif role == "user":
            text += f"问：{content}\n\n答："
        elif role == "assistant":
            text += f"{content}\n\n"
    return text


def _format_baichuan_messages(messages):
    """Baichuan reserved-token template."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<reserved_106>{content}<reserved_107>"
        elif role == "user":
            text += f"<reserved_106>{content}<reserved_107>"
        elif role == "assistant":
            text += f"<reserved_108>{content}<reserved_108>"
    return text


def _format_gemma_messages(messages):
    """Gemma / Gemma-3 user/model template — ``system`` is merged into
    the first ``user`` block and ``assistant`` is renamed to ``model``."""
    messages = list(messages)
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    system_block = "\n".join(system_parts).strip()
    first_user_seen = False
    text = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if not first_user_seen:
                first_user_seen = True
                if system_block:
                    content = f"{system_block}\n\n{content}"
            text += f"<start_of_turn>user\n{content}<end_of_turn>\n\n"

        elif role == "assistant":
            # Leave the assistant turn open for generation.
            text += f"<start_of_turn>model{content}"

    if not first_user_seen and system_block:
        text = f"<start_of_turn>user\n{system_block}<end_of_turn>\n\n"

    return text


__all__ = ["format_messages_for_model"]
