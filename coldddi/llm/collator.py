"""Fine-tuning collator for the binary DDI task.

Port of ``Version_1_1/dataloader/collators/finetune.ChatDataCollator``.
Takes a list of dicts ``{"text": str, "cls_labels": int}`` and emits a
batch tensor dict with:

* ``input_ids``      — tokenized prompts (padded on the LEFT)
* ``attention_mask`` — 1/0 mask
* ``labels``         — same as ``input_ids`` but every position up to
                       and including the assistant header (template)
                       is set to ``-100`` so the cross-entropy loss is
                       computed only on the answer token(s)
* ``cls_labels``     — per-row binary label (0/1) used by the custom
                       :meth:`compute_loss` to do BCE on
                       ``[no_id, yes_id]`` logits

The assistant template string is family-specific:

* ``llama``:   ``<|start_header_id|>assistant<|end_header_id|>``
* ``qwen``:    ``<|im_start|>assistant``
* ``gemma``:   ``<start_of_turn>model``
* ``mistral``: ``[INST]assistant\\n``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


#: Family-specific assistant header strings. The collator searches for
#: these token sub-sequences to find the boundary between user prompt
#: and the assistant's first generated token.
ASSISTANT_TEMPLATES: dict[str, str] = {
    "llama":   "<|start_header_id|>assistant<|end_header_id|>",
    "qwen":    "<|im_start|>assistant",
    "gemma":   "<start_of_turn>model",
    "mistral": "[INST]assistant\n",
}


@dataclass
class BinaryFTCollator:
    """Tokenize FT prompts and mask everything before the answer slot.

    Parameters
    ----------
    tokenizer
        A loaded HuggingFace tokenizer.
    model_family
        One of :data:`ASSISTANT_TEMPLATES`. Use the value returned by
        :func:`coldddi.llm.prompts.binary_cls.infer_model_family`.
    max_length
        Tokenizer ``max_length`` cap.
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
        # Cache the encoded template ids once.
        self._template_str = ASSISTANT_TEMPLATES[self.model_family]
        self._template_ids = self.tokenizer.encode(
            self._template_str, add_special_tokens=False
        )
        if not self._template_ids:
            raise ValueError(
                f"Tokenizer produced empty ids for assistant template "
                f"{self._template_str!r}. The mask boundary cannot be located."
            )

    # ------------------------------------------------------------------

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
                # Header not found (truncation killed it or template
                # tokenises differently inside the rendered prompt).
                # Fall back to the very last token, matching upstream.
                start_pos = len(ids) - 1
            batch["labels"][i, :start_pos] = -100

        return batch

    # ------------------------------------------------------------------

    def _find_template_end(self, ids: list[int], tlen: int) -> int:
        """Return ``j + tlen`` where ``ids[j : j+tlen] == self._template_ids``.

        Scans **from the right** so an accidental copy of the assistant
        header inside user-supplied content (for example a few-shot
        example that quoted the chat template) cannot shift the mask
        boundary forward into the user prompt.  The legitimate header
        is always the **last** occurrence — it directly precedes the
        answer token.

        Returns ``-1`` if the template is not found.
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
