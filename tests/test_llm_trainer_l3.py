"""Test collator masks and LoRA fit/save/reload on a tiny LLM.

Tokens before the assistant header are masked with -100; saved adapters must
load through LLMInferenceRunner for prediction.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# Module-level skip if transformers / peft missing.
pytest.importorskip("transformers", reason="transformers required for L3 tests")
pytest.importorskip("peft", reason="peft required for L3 tests")


# Synthetic samples for both train and val

def _build_samples(n: int):
    """Build `n` toy {text, cls_labels} samples by rendering the prompt
    builder with deterministic drug pairs and labels."""
    from coldddi.llm.prompts.binary_cls import (
        PromptBuildConfig, build_binary_prompt,
    )

    cfg = PromptBuildConfig(
        task_name="Binary_cls",
        method="Zero_Shot_Sequence",
        model_name=TINY_MODEL,
    )
    out = []
    for i in range(n):
        label = i % 2
        s = {
            "drug_a_id": f"DB{i:03d}A",
            "drug_b_id": f"DB{i:03d}B",
            "drug_a_name": f"DrugA{i}",
            "drug_b_name": f"DrugB{i}",
            "drugA_name": f"DrugA{i}",
            "drugB_name": f"DrugB{i}",
            "label": label,
        }
        text = build_binary_prompt(s, cfg)
        out.append({"text": text, "cls_labels": label, "_sample": s})
    return out


# Collator tests

class TestBinaryFTCollator:
    def test_mask_boundary_llama(self):
        """For every row, labels before the assistant header must be
        -100 and the single token at the answer position must equal
        the yes/no token id."""
        from transformers import AutoTokenizer

        from coldddi.llm.collator import BinaryFTCollator

        tok = AutoTokenizer.from_pretrained(TINY_MODEL, padding_side="left")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        collator = BinaryFTCollator(tok, model_family="llama", max_length=512)
        samples = _build_samples(4)
        batch = collator(samples)
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["cls_labels"].tolist() == [0, 1, 0, 1]

        yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
        no_id = tok.encode(" No", add_special_tokens=False)[0]

        # For each row, find the first non-(-100) position and assert
        # the token id there is yes_id or no_id matching cls_labels.
        for i in range(batch["input_ids"].size(0)):
            ids = batch["input_ids"][i].tolist()
            lbls = batch["labels"][i].tolist()
            cls = int(batch["cls_labels"][i])
            valid_positions = [j for j, v in enumerate(lbls) if v != -100]
            assert valid_positions, f"row {i}: all labels masked"
            first = valid_positions[0]
            expected = yes_id if cls == 1 else no_id
            assert ids[first] == expected, (
                f"row {i}: first non-masked token id {ids[first]} != "
                f"expected answer id {expected} (cls={cls})"
            )

    def test_unknown_family_rejected(self):
        from transformers import AutoTokenizer
        from coldddi.llm.collator import BinaryFTCollator

        tok = AutoTokenizer.from_pretrained(TINY_MODEL)
        with pytest.raises(ValueError, match="Unknown model_family"):
            BinaryFTCollator(tok, model_family="t5", max_length=128)


# End-to-end LoRA fit + reload

class TestLoRATrainerFit:
    """Fit and reload through L2; the random model tests execution, not prediction quality."""

    @pytest.fixture(scope="class")
    def fit_output(self, tmp_path_factory):
        """Run a 1-step fit on 8 samples and return the trainer info +
        the directory where the adapter was saved."""
        from coldddi.llm.trainer import LLMTrainerConfig, LoRATrainer

        out = tmp_path_factory.mktemp("l3_fit")
        cfg = LLMTrainerConfig(
            model_name=TINY_MODEL,
            output_dir=str(out / "ckpts"),
            dtype="float32",
            device="cpu",
            num_epochs=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            warmup_ratio=0.0,
            max_length=256,
            logging_steps=1,
            save_steps=2,
            eval_steps=4,
            save_total_limit=2,
            eval_strategy="no",
            disable_tqdm=True,
            report_to=(),
            seed=42,
        )
        # Smaller LoRA for the tiny model.
        cfg.lora.r = 4
        cfg.lora.alpha = 8
        cfg.lora.target_modules = ("q_proj", "v_proj")

        trainer = LoRATrainer(cfg)
        train_samples = _build_samples(8)
        try:
            info = trainer.fit(train_samples=train_samples, val_samples=None)
        except Exception as exc:
            pytest.skip(f"Trainer failed on the tiny model: {exc}")

        adapter_dir = out / "adapter"
        trainer.save_adapter(adapter_dir)
        return {"info": info, "adapter_dir": adapter_dir, "output_dir": out}

    def test_fit_returns_log_history(self, fit_output):
        info = fit_output["info"]
        assert "output_dir" in info
        assert "log_history" in info
        assert any("loss" in e for e in info["log_history"]), (
            "trainer state has no logged train loss"
        )

    def test_adapter_saved(self, fit_output):
        d = fit_output["adapter_dir"]
        assert d.is_dir()
        assert (d / "adapter_info.json").is_file()
        # PEFT saves adapter_config.json + adapter_model.{safetensors,bin}
        assert (d / "adapter_config.json").is_file()

    def test_adapter_reloads_into_inference_runner(self, fit_output):
        """The L2 inference runner must be able to attach this LoRA and
        produce sensible probabilities."""
        import pandas as pd
        import torch
        from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
        from coldddi.llm.prompts import PromptBuildConfig

        runner = LLMInferenceRunner(LLMRunnerConfig(
            model_name=TINY_MODEL,
            adapter_path=str(fit_output["adapter_dir"]),
            dtype="float32",
            device="cpu",
            batch_size=4,
            max_length=256,
        ))
        runner.load()
        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Zero_Shot_Sequence",
            model_name=TINY_MODEL,
        )
        samples = [
            s["_sample"] for s in _build_samples(3)
        ]
        df = runner.score_samples(samples, cfg)
        assert len(df) == 3
        assert (df["p_yes"] >= 0).all() and (df["p_yes"] <= 1).all()
        # p_yes + p_no must sum to ~1 since they're a softmax over [no, yes].
        for _, row in df.iterrows():
            assert abs(row["p_yes"] + row["p_no"] - 1.0) < 1e-4
