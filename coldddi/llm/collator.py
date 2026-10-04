"""Collate binary DDI prompts for answer-token fine-tuning.

Inputs contain ``text`` and binary ``cls_labels``. Outputs include token IDs,
attention masks and labels masked with ``-100`` through the assistant header.
The trainer uses ``cls_labels`` to score the Yes/No logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


#: Assistant headers used to locate the answer-token boundary.
ASSISTANT_TEMPLATES: dict[str, str] = {
    "llama":   "<|start_header_id|>assistant<|end_header_id|>",
    "qwen":    "<|im_start|>assistant",
    "gemma":   "<start_of_turn>model",
    "mistral": "[INST]assistant\n",
}


@dataclass
class BinaryFTCollator:
    """Tokenize prompts and mask tokens before the answer.

    ``model_family`` must be a key in :data:`ASSISTANT_TEMPLATES`;
    ``max_length`` sets the tokenizer's truncation limit.
    """

    tokenizer: Any
    model_family: str
    max_length: int = 1024

    def __post_init__(self):
        if self.model_family not in ASSISTANT_TEMPLATES:
            raise ValueError(
                f"Unknown model_family {self.model_family!r}; "
                f"must be one of {sorted(ASSISTANT_TEMPLATES)}"
            )
        self._template_str = ASSISTANT_TEMPLATES[self.model_family]
        self._template_ids = self.tokenizer.encode(
            self._template_str, add_special_tokens=False
        )
        if not self._template_ids:
            raise ValueError(
                f"Tokenizer produced empty ids for assistant template "
                f"{self._template_str!r}. The mask boundary cannot be located."
            )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        cls_labels = [int(f["cls_labels"]) for f in features]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch: dict[str, torch.Tensor] = dict(encodings)
        batch["labels"] = batch["input_ids"].clone()
        batch["cls_labels"] = torch.tensor(cls_labels, dtype=torch.long)

        tlen = len(self._template_ids)
        for i in range(batch["input_ids"].size(0)):
            ids = batch["input_ids"][i].tolist()
            start_pos = self._find_template_end(ids, tlen)
            if start_pos == -1:
                # If truncation or tokenization hides the header, use the last token.
                start_pos = len(ids) - 1
            batch["labels"][i, :start_pos] = -100

        return batch

    def _find_template_end(self, ids: list[int], tlen: int) -> int:
        """Return the position after the last assistant header, or ``-1``.

        Searching from the right skips headers quoted within the user prompt.
        """
        n = len(ids)
        if tlen == 0 or tlen > n:
            return -1
        tids = self._template_ids
        for j in range(n - tlen, -1, -1):
            if ids[j : j + tlen] == tids:
                return j + tlen
        return -1


__all__ = ["BinaryFTCollator", "ASSISTANT_TEMPLATES"]
