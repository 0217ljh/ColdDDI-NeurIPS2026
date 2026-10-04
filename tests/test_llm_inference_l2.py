"""Test CPU inference, P1/P4 prompts, and binary probabilities on Llama-3.2-1B.

This paper model (Appendix B.x) encodes leading-space Yes/No as single tokens;
the tiny random Llama's SentencePiece tokenizer does not. Skip model tests
when the required model is not cached locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TINY_MODEL = "meta-llama/Llama-3.2-1B"


# Skip the entire module if transformers is unavailable.
transformers_available = pytest.importorskip(
    "transformers", reason="transformers is required for the LLM inference runner"
)


def _model_locally_available(hf_id: str) -> bool:
    """Return True iff ``hf_id`` can be loaded offline from HF cache."""
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(hf_id, local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _model_locally_available(TINY_MODEL),
    reason=f"{TINY_MODEL} not in local HF cache; skipping L2 inference tests",
)


@pytest.fixture(scope="module")
def loaded_runner():
    """Load the cached model once for this module."""
    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig

    runner = LLMInferenceRunner(
        LLMRunnerConfig(
            model_name=TINY_MODEL,
            dtype="float32",   # tiny model has float32 weights
            device="cpu",
            batch_size=4,
            max_length=512,
        )
    )
    try:
        runner.load()
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"Could not download tiny model: {exc}")
    return runner


# Sample fixtures (no toy dataset required)

DRUG_A_ID, DRUG_B_ID = "DB001", "DB002"
ID2NAME = {DRUG_A_ID: "Lepirudin", DRUG_B_ID: "Cetuximab"}
ID2SMI  = {DRUG_A_ID: "CC[C@H](C)C", DRUG_B_ID: "CN(C)CC"}

NEIGHBORS = {
    "neighbors": {
        "transporters": {"A": ["SLC22A1", "ABCB1"], "B": ["unknown"]},
        "pathways":     {"A": ["Coagulation cascade"], "B": ["EGFR signaling"]},
        "targets":      {"A": ["F2", "F10"], "B": ["EGFR"]},
        "enzymes":      {"A": ["CYP3A4"], "B": ["CYP2C9"]},
        "carriers":     {"A": ["ALB"], "B": ["unknown"]},
        "smiles":       {"A": [ID2SMI[DRUG_A_ID]], "B": [ID2SMI[DRUG_B_ID]]},
    }
}


def _make_sample(label: int, *, with_kg: bool) -> dict:
    s = {
        "drug_a_id": DRUG_A_ID, "drug_b_id": DRUG_B_ID,
        "drug_a_name": ID2NAME[DRUG_A_ID], "drug_b_name": ID2NAME[DRUG_B_ID],
        "drugA_name":  ID2NAME[DRUG_A_ID], "drugB_name":  ID2NAME[DRUG_B_ID],
        "label": label,
    }
    if with_kg:
        s["subgraph_1hop"] = NEIGHBORS
    return s


class TestLLMInferenceRunner:
    def test_load_resolves_yes_no_ids(self, loaded_runner):
        assert loaded_runner.yes_id is not None
        assert loaded_runner.no_id is not None
        assert loaded_runner.yes_id != loaded_runner.no_id
        assert loaded_runner.model is not None
        assert loaded_runner.tokenizer is not None

    def test_p1_zero_shot_inference(self, loaded_runner):
        from coldddi.llm.prompts import PromptBuildConfig

        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Zero_Shot_Sequence",
            model_name=TINY_MODEL,
        )
        samples = [_make_sample(0, with_kg=False), _make_sample(1, with_kg=False)]
        df = loaded_runner.score_samples(
            samples, cfg, drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
        )
        assert list(df.columns) == [
            "drug_a_id", "drug_b_id", "p_no", "p_yes", "pred", "prompt",
        ]
        assert len(df) == 2
        # p_no + p_yes must sum to ~1 (softmax over the binary projection).
        for _, row in df.iterrows():
            assert 0.0 <= row["p_yes"] <= 1.0
            assert 0.0 <= row["p_no"] <= 1.0
            assert abs(row["p_yes"] + row["p_no"] - 1.0) < 1e-5
            assert row["pred"] in (0, 1)
            # Prompt MUST be the open-ended form (no answer at the end).
            assert not row["prompt"].rstrip().endswith("Yes")
            assert not row["prompt"].rstrip().endswith("No")

    def test_p4_kg_sequence_inference(self, loaded_runner):
        from coldddi.llm.prompts import PromptBuildConfig

        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="One_Hop_Subgraph_Sequence",
            model_name=TINY_MODEL,
        )
        samples = [_make_sample(0, with_kg=True), _make_sample(1, with_kg=True)]
        df = loaded_runner.score_samples(
            samples, cfg, drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
        )
        assert len(df) == 2
        # The KG branch produces longer prompts than zero-shot.
        for _, row in df.iterrows():
            assert "whose targets are F2, F10" in row["prompt"]
            assert "whose enzymes are CYP3A4" in row["prompt"]
            assert 0.0 <= row["p_yes"] <= 1.0
            assert 0.0 <= row["p_no"] <= 1.0

    def test_yes_no_token_validation(self):
        """If yes_token splits into multi-token, the runner must raise."""
        from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig

        bad = LLMInferenceRunner(LLMRunnerConfig(
            model_name=TINY_MODEL,
            yes_token=" Interaction Probable",   # almost certainly multi-token
            no_token=" No Interaction Detected",
            dtype="float32",
            device="cpu",
        ))
        with pytest.raises(ValueError, match="single token"):
            bad.load()

    def test_batching(self, loaded_runner):
        """5 samples through batch_size=4 must produce exactly 5 rows."""
        from coldddi.llm.prompts import PromptBuildConfig

        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Zero_Shot_Sequence",
            model_name=TINY_MODEL,
        )
        samples = [_make_sample(i % 2, with_kg=False) for i in range(5)]
        df = loaded_runner.score_samples(
            samples, cfg, drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
        )
        assert len(df) == 5


class TestInferenceFTPrefixInvariant:
    """Inference prompts equal FT prompts without the answer token.

    This keeps the collator's mask boundary aligned with next-token scoring.
    """

    @pytest.mark.parametrize("model_name,family", [
        ("meta-llama/Llama-3.2-1B",        "llama"),
        ("Qwen/Qwen2.5-3B",                 "qwen"),
        ("google/gemma-3-1b-pt",            "gemma"),
    ])
    @pytest.mark.parametrize("method", [
        "Zero_Shot_Sequence",
        "One_Hop_Subgraph_Sequence",
        "OHS_Mask_Name_Entity",
    ])
    def test_inference_prompt_is_prefix_of_ft_prompt(self, model_name, family, method):
        from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt

        cfg = PromptBuildConfig(
            task_name="Binary_cls", method=method, model_name=model_name,
        )
        s = _make_sample(label=1, with_kg=True)
        s_neg = _make_sample(label=0, with_kg=True)
        inf_prompt = build_binary_prompt(
            s, cfg,
            drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
            assistant_content="",
        )
        ft_pos = build_binary_prompt(s, cfg,
            drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
        )
        ft_neg = build_binary_prompt(s_neg, cfg,
            drug_id2name=ID2NAME, drug_id2smiles=ID2SMI,
        )
        # Inference + leading-space answer must reconstruct the FT
        # prompts byte-exactly (for both labels).
        assert inf_prompt + " Yes" == ft_pos, (
            f"family={family} method={method}: ' Yes' postfix mismatch.\n"
            f"  inf+Yes tail: ...{(inf_prompt + ' Yes')[-40:]!r}\n"
            f"  ft_pos  tail: ...{ft_pos[-40:]!r}"
        )
        assert inf_prompt + " No" == ft_neg, (
            f"family={family} method={method}: ' No' postfix mismatch.\n"
            f"  inf+No tail:  ...{(inf_prompt + ' No')[-40:]!r}\n"
            f"  ft_neg tail:  ...{ft_neg[-40:]!r}"
        )


# Optional: integration with PairDataset via score_pairs

TOY_RELEASE = REPO_ROOT / "data" / "public" / "intermediate"


@pytest.mark.skipif(
    not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
    reason="Toy filtered dir not found — run reconstruct.py --toy first.",
)
def test_score_pairs_with_pair_dataset(loaded_runner):
    """End-to-end: PairDataset rows → score_pairs returns valid DataFrame."""
    pytest.importorskip("rdkit")
    from coldddi.data.dataset import PairDataset
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.retrieval import build_subgraph_map

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
    pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(3)

    cfg = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=TINY_MODEL,
    )
    df = loaded_runner.score_pairs(
        pairs, ds=ds, prompt_cfg=cfg, subgraph_map=sm,
    )
    assert len(df) == 3
    assert (df["p_yes"] >= 0).all() and (df["p_yes"] <= 1).all()
    assert (df["p_no"] >= 0).all() and (df["p_no"] <= 1).all()
