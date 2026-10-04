"""Test byte-exact prompt parity, assistant boundaries, and answer tokens.

JSON references come from Version_1_1/dataloader/prompts/binary_cls.build_binary_prompt
via fixtures/prompts/_capture_reference.py. The expected matrix below includes
all paper methods, A/B entity masks, and non-paper formatter families.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX_DIR = REPO_ROOT / "tests" / "fixtures" / "prompts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coldddi.llm.prompts.binary_cls import (  # noqa: E402
    PromptBuildConfig,
    build_binary_prompt,
    infer_model_family,
)


# The builder resolves to qwen/llama/gemma/mistral (default qwen).
# ChatGLM, Baichuan, and DeepSeek use Qwen here; direct formatter tests cover
# their native templates separately.
ASSISTANT_HEADERS = {
    "llama":   "<|start_header_id|>assistant<|end_header_id|>",
    "qwen":    "<|im_start|>assistant",
    "gemma":   "<start_of_turn>model",
    "mistral": "[INST]assistant\n",
}

#: All four families satisfy the FT contract: the answer token sits
#: at the end of the rendered prompt.  (Direct chatglm / baichuan
#: rendering breaks this; that path is separately tested.)
FT_CLEAN_TAIL_FAMILIES = set(ASSISTANT_HEADERS.keys())


def _iter_fixtures():
    """Yield every fixture once. Skip the capture script itself."""
    for path in sorted(FIX_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            yield path.name, json.load(f)


# Materialize once so parametrized consumers share the same fixture list.
_ALL_FIXTURES = list(_iter_fixtures())

#: Expected fixture matrix:
#:   * 12 methods × 3 paper families × 2 labels                    = 72
#:   * 4 entity-mask methods × 3 paper families × 2 labels × B-var = 24
#:   * 4 non-paper families × P1 only × 2 labels                   = 8
EXPECTED_FIXTURE_COUNT = 104


_EXPECTED_KEB_FILENAMES = {
    f"{method}__{family}__label{label}__keB.json"
    for method in (
        "R4_ohs_full_mask_entity",
        "R5_ohs_full_mask_name_entity",
        "R6_ohs_mask_entity",
        "R7_ohs_mask_name_entity",
    )
    for family in ("llama", "qwen", "gemma")
    for label in (0, 1)
}


def test_fixture_matrix_complete():
    """Missing fixtures must fail rather than silently reduce parametrized coverage."""
    assert len(_ALL_FIXTURES) == EXPECTED_FIXTURE_COUNT, (
        f"Expected {EXPECTED_FIXTURE_COUNT} fixtures, found "
        f"{len(_ALL_FIXTURES)}. Re-run _capture_reference.py."
    )
    actual_keb = {n for (n, _) in _ALL_FIXTURES if n.endswith("__keB.json")}
    missing = _EXPECTED_KEB_FILENAMES - actual_keb
    extra = actual_keb - _EXPECTED_KEB_FILENAMES
    assert not missing and not extra, (
        f"B-group fixture set drift detected. "
        f"missing={sorted(missing)}, extra={sorted(extra)}"
    )


def _restore_ke_map(payload):
    return {
        tuple(k.split("|")): v for k, v in payload["key_entity_map"].items()
    }


def _build(payload):
    np.random.seed(payload["np_seed"])
    cfg = PromptBuildConfig(
        task_name="Binary_cls",
        method=payload["method_internal"],
        model_name=payload["model_name"],
    )
    # Deep-copy so the original code's in-place shuffle (P2 / P5) does
    # not contaminate `payload` across parametrized test functions.
    return build_binary_prompt(
        copy.deepcopy(payload["sample"]),
        cfg,
        drug_id2name=payload["kb"]["drug_id2name"],
        drug_id2smiles=payload["kb"]["drug_id2smiles"],
        key_entity_map=_restore_ke_map(payload),
    )


@pytest.mark.parametrize("name,payload", _ALL_FIXTURES)
def test_byte_exact_match_with_reference(name, payload):
    """Rebuilt prompts match the upstream reference byte for byte."""
    actual = _build(payload)
    expected = payload["prompt"]
    if actual != expected:
        for i, (a, b) in enumerate(zip(actual, expected)):
            if a != b:
                start = max(0, i - 40)
                end = min(min(len(actual), len(expected)), i + 40)
                pytest.fail(
                    f"Fixture {name}: first divergence at byte {i}\n"
                    f"  expected[{start}:{end}] = {expected[start:end]!r}\n"
                    f"  actual  [{start}:{end}] = {actual[start:end]!r}"
                )
        pytest.fail(
            f"Fixture {name}: lengths differ "
            f"({len(actual)} vs {len(expected)})"
        )


@pytest.mark.parametrize("name,payload", _ALL_FIXTURES)
def test_assistant_header_boundary(name, payload):
    """Verify the assistant header appears exactly once and that the
    body immediately after it starts with the answer token."""
    actual = _build(payload)
    # Use the EFFECTIVE family (the one build_binary_prompt resolves to),
    # not the family label from the fixture filename.
    effective = infer_model_family(payload["model_name"])
    header = ASSISTANT_HEADERS[effective]
    expected_answer = " Yes" if payload["label"] == 1 else " No"

    assert actual.count(header) == 1, (
        f"{name}: expected 1 occurrence of header {header!r} for "
        f"effective family {effective!r}, got {actual.count(header)}"
    )
    idx = actual.rfind(header)
    body_after_header = actual[idx + len(header):]
    assert body_after_header.startswith(expected_answer), (
        f"{name}: body after assistant header must start with "
        f"{expected_answer!r}, got {body_after_header[:20]!r}"
    )


@pytest.mark.parametrize("name,payload", _ALL_FIXTURES)
def test_answer_token_terminates_prompt(name, payload):
    """Every builder-resolved template ends with the correct leading-space Yes/No token."""
    actual = _build(payload)
    expected_tail = " Yes" if payload["label"] == 1 else " No"
    assert actual.endswith(expected_tail), (
        f"{name}: prompt should end with {expected_tail!r}, "
        f"actually ends with ...{actual[-30:]!r}"
    )


# Direct chat-formatter coverage (bypasses build_binary_prompt)

class TestChatFormatterDirectCalls:
    """Test native ChatGLM, Baichuan, and DeepSeek formatter branches.

    Check byte-exact shape, not FT-readiness: upstream ChatGLM/Baichuan append
    tokens after the assistant content.
    """

    MSGS = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": " Yes"},
    ]

    def test_chatglm_template(self):
        from coldddi.llm.prompts import format_messages_for_model

        out = format_messages_for_model(self.MSGS, "THUDM/chatglm3-6b")
        # ChatGLM uses separate system/user rounds, CJK protocol separators,
        # and a trailing "\n\n" after the assistant.
        assert "[Round 1]" in out
        assert "答：" in out
        assert out.endswith(" Yes\n\n"), (
            f"chatglm template should end with ' Yes\\n\\n', got ...{out[-20:]!r}"
        )

    def test_baichuan_template(self):
        from coldddi.llm.prompts import format_messages_for_model

        out = format_messages_for_model(self.MSGS, "baichuan-inc/Baichuan2-7B")
        # Baichuan opens/closes the assistant with <reserved_108>.
        assert out.count("<reserved_108>") == 2, (
            f"baichuan template should have two <reserved_108> markers, got "
            f"{out.count('<reserved_108>')}"
        )
        assert out.endswith("<reserved_108>"), (
            f"baichuan template ends with <reserved_108>, got ...{out[-20:]!r}"
        )

    def test_mistral_template(self):
        from coldddi.llm.prompts import format_messages_for_model

        out = format_messages_for_model(self.MSGS, "mistralai/Mistral-7B-Instruct")
        # Mistral uses [INST] / [/INST]; assistant header is left open.
        assert "[INST]system" in out
        assert "[INST]assistant" in out
        assert out.endswith(" Yes")

    def test_deepseek_template_alone(self):
        """A *pure* deepseek model name (no 'qwen' or 'llama' in it)
        must take the deepseek branch, which aliases to Llama-3 layout."""
        from coldddi.llm.prompts import format_messages_for_model

        out = format_messages_for_model(
            self.MSGS, "deepseek-ai/deepseek-r1-7b"
        )
        assert out.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>assistant<|end_header_id|>" in out
        assert out.endswith(" Yes")
