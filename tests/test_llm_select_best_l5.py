"""Test per-split LoRA checkpoint parsing, selection, and manifests.

Synthetic scores cover ties and NaNs; a tiny Llama exercises training through
scoring and selection to a usable checkpoint path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _fabricate_run(tmp_path: Path, ckpts: list[tuple[int, dict]]) -> Path:
    """Create checkpoints from (step, {eval_<split>_loss: value}) entries.

    Put trainer_state.json in the latest checkpoint, which L5 prefers.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for step, _ in ckpts:
        (run_dir / f"checkpoint-{step}").mkdir()
    log_history = []
    for step, losses in ckpts:
        entry = {"step": step}
        entry.update(losses)
        log_history.append(entry)
    state = {
        "log_history": log_history,
        "best_model_checkpoint": str(run_dir / f"checkpoint-{ckpts[0][0]}"),
        "best_metric": ckpts[0][1].get("eval_S2_loss"),
    }
    latest_step = max(s for s, _ in ckpts)
    with open(run_dir / f"checkpoint-{latest_step}" / "trainer_state.json", "w") as f:
        json.dump(state, f)
    return run_dir


# Parser tests

class TestParseCandidateCkpts:
    def test_default_returns_only_s2(self, tmp_path):
        """Default ``splits=("S2",)`` materialises just one ranking."""
        from coldddi.llm.select_best import parse_candidate_ckpts

        run = _fabricate_run(tmp_path, [
            (10, {"eval_S0_loss": 0.30, "eval_S1_loss": 0.40, "eval_S2_loss": 0.70}),
            (20, {"eval_S0_loss": 0.40, "eval_S1_loss": 0.50, "eval_S2_loss": 0.30}),
            (30, {"eval_S0_loss": 0.50, "eval_S1_loss": 0.45, "eval_S2_loss": 0.50}),
        ])
        cands = parse_candidate_ckpts(run, topk=2)
        assert set(cands) == {"S2"}
        # top-K=2 lowest eval_S2_loss = [20(0.3), 30(0.5)]; latest = 30 (already in).
        steps = sorted(c.step for c in cands["S2"])
        assert steps == [20, 30]

    def test_multi_split_independent_rankings(self, tmp_path):
        """Each split is ranked by its own ``eval_<split>_loss``."""
        from coldddi.llm.select_best import parse_candidate_ckpts

        # Crafted so the top ckpt differs per split:
        #   S0 prefers step 10 (0.30); S1 prefers step 30 (0.30);
        #   S2 prefers step 20 (0.30). Latest = 30.
        run = _fabricate_run(tmp_path, [
            (10, {"eval_S0_loss": 0.30, "eval_S1_loss": 0.55, "eval_S2_loss": 0.70}),
            (20, {"eval_S0_loss": 0.55, "eval_S1_loss": 0.50, "eval_S2_loss": 0.30}),
            (30, {"eval_S0_loss": 0.65, "eval_S1_loss": 0.30, "eval_S2_loss": 0.40}),
        ])
        cands = parse_candidate_ckpts(run, splits=("S0", "S1", "S2"), topk=1)
        assert set(cands) == {"S0", "S1", "S2"}
        # top-1 + latest (might overlap):
        s0_steps = sorted(c.step for c in cands["S0"])
        s1_steps = sorted(c.step for c in cands["S1"])
        s2_steps = sorted(c.step for c in cands["S2"])
        # S0: top-1 = 10; latest = 30 → [10, 30]
        assert s0_steps == [10, 30]
        # S1: top-1 = 30; latest = 30 → [30]
        assert s1_steps == [30]
        # S2: top-1 = 20; latest = 30 → [20, 30]
        assert s2_steps == [20, 30]

    def test_missing_split_returns_latest_only(self, tmp_path):
        """Requesting a split whose key never appears in log_history
        should still yield a one-element [latest] list (no crash)."""
        from coldddi.llm.select_best import parse_candidate_ckpts

        # Only eval_S2_loss recorded.
        run = _fabricate_run(tmp_path, [
            (10, {"eval_S2_loss": 0.7}),
            (20, {"eval_S2_loss": 0.5}),
        ])
        cands = parse_candidate_ckpts(run, splits=("S0", "S2"))
        assert cands["S2"], "S2 must rank normally"
        assert len(cands["S0"]) == 1
        assert cands["S0"][0].tag == "latest_no_eval"
        assert cands["S0"][0].step == 20

    def test_no_ckpts_returns_empty_per_split(self, tmp_path):
        from coldddi.llm.select_best import parse_candidate_ckpts

        run = tmp_path / "empty_run"
        run.mkdir()
        cands = parse_candidate_ckpts(run, splits=("S2",))
        assert cands == {"S2": []}

    def test_latest_already_in_topk_tag_merges(self, tmp_path):
        from coldddi.llm.select_best import parse_candidate_ckpts

        run = _fabricate_run(tmp_path, [
            (10, {"eval_S2_loss": 0.7}),
            (20, {"eval_S2_loss": 0.5}),
            (30, {"eval_S2_loss": 0.3}),  # latest AND best.
        ])
        cands = parse_candidate_ckpts(run, topk=3)
        tag_30 = next(c.tag for c in cands["S2"] if c.step == 30)
        assert "latest" in tag_30
        assert "min_eval_S2_loss" in tag_30


# select_best with hand-crafted scores

class TestSelectBestSyntheticAUCs:
    def _mk(self, step, auc, tag="t"):
        from coldddi.llm.select_best import CandidateCkpt, ScoredCandidate

        c = CandidateCkpt(step=step, ckpt_path=f"/fake/checkpoint-{step}", tag=tag)
        return ScoredCandidate(candidate=c, val_auc=auc, n_pos=10, n_neg=10)

    def test_per_split_winners(self):
        """Different splits get different best ckpts."""
        from coldddi.llm.select_best import select_best

        scored_by_split = {
            "S0": [self._mk(10, 0.8), self._mk(20, 0.6)],
            "S2": [self._mk(10, 0.6), self._mk(20, 0.85)],
        }
        m = select_best(scored_by_split)
        assert set(m) == {"S0", "S2"}
        assert m["S0"]["best_step"] == 10
        assert m["S0"]["best_val_auc"] == 0.8
        assert m["S2"]["best_step"] == 20
        assert m["S2"]["best_val_auc"] == 0.85

    def test_tie_break_per_split(self):
        from coldddi.llm.select_best import select_best

        m = select_best({
            "S2": [self._mk(20, 0.7), self._mk(10, 0.7), self._mk(30, 0.6)],
        })
        assert m["S2"]["best_step"] == 10

    def test_nan_skipped_within_split(self):
        from coldddi.llm.select_best import select_best

        m = select_best({
            "S2": [self._mk(10, float("nan")), self._mk(20, 0.55)],
        })
        assert m["S2"]["best_step"] == 20

    def test_all_nan_raises_with_split_name(self):
        from coldddi.llm.select_best import select_best

        with pytest.raises(ValueError, match="S2"):
            select_best({
                "S2": [self._mk(10, float("nan"))],
            })

    def test_manifest_round_trip(self, tmp_path):
        from coldddi.llm.select_best import select_best, write_manifest

        m = select_best({
            "S0": [self._mk(10, 0.6), self._mk(20, 0.8)],
            "S2": [self._mk(10, 0.8), self._mk(20, 0.6)],
        })
        out = write_manifest(m, tmp_path / "manifest.json")
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["S0"]["best_step"] == 20
        assert loaded["S2"]["best_step"] == 10


# L3 validation contract.

pytest.importorskip("transformers", reason="transformers required for L5 end-to-end")
pytest.importorskip("peft", reason="peft required for L5 end-to-end")


class TestL3ValSplitContract:
    """Training defaults to S2 validation; evaluation requires at least one split."""

    def _build_train_samples(self, n=4):
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
            s = {
                "drug_a_id": f"DB{i}A", "drug_b_id": f"DB{i}B",
                "drug_a_name": f"A{i}", "drug_b_name": f"B{i}",
                "drugA_name": f"A{i}", "drugB_name": f"B{i}",
                "label": i % 2,
            }
            out.append({"text": build_binary_prompt(s, cfg), "cls_labels": i % 2})
        return out

    def _trainer_cfg(self, tmp_path, **overrides):
        from coldddi.llm.trainer import LLMTrainerConfig

        cfg = LLMTrainerConfig(
            model_name=TINY_MODEL,
            output_dir=str(tmp_path / "ckpts"),
            dtype="float32", device="cpu",
            num_epochs=1, micro_batch_size=2,
            max_length=128, logging_steps=1,
            save_steps=1, save_total_limit=10,
            disable_tqdm=True, report_to=(), seed=42,
        )
        cfg.lora.r = 4
        cfg.lora.alpha = 8
        cfg.lora.target_modules = ("q_proj", "v_proj")
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_fit_rejects_missing_val_split(self, tmp_path):
        """eval_strategy='steps' (default) + no val_samples → raise."""
        from coldddi.llm.trainer import LoRATrainer

        cfg = self._trainer_cfg(tmp_path)
        trainer = LoRATrainer(cfg)
        with pytest.raises(ValueError, match="val_samples is required"):
            trainer.fit(self._build_train_samples(), val_samples=None)

    def test_fit_rejects_primary_val_missing_from_dict(self, tmp_path):
        """If user passes val_samples={'S0': [...]} but primary_val_split='S2',
        it should raise so the wrong split isn't silently used."""
        from coldddi.llm.trainer import LoRATrainer

        cfg = self._trainer_cfg(tmp_path)
        trainer = LoRATrainer(cfg)
        with pytest.raises(ValueError, match="primary_val_split"):
            trainer.fit(
                self._build_train_samples(),
                val_samples={"S0": self._build_train_samples()},
            )

    def test_fit_accepts_s2_only_default(self, tmp_path):
        """Minimum config: cfg.primary_val_split='S2' + val_samples={'S2': ...}."""
        from coldddi.llm.trainer import LoRATrainer

        cfg = self._trainer_cfg(tmp_path, eval_steps=1)
        trainer = LoRATrainer(cfg)
        info = trainer.fit(
            self._build_train_samples(),
            val_samples={"S2": self._build_train_samples(2)},
        )
        assert "log_history" in info
        # Trainer should have logged at least one eval_S2_loss entry.
        s2_losses = [r for r in info["log_history"] if "eval_S2_loss" in r]
        assert s2_losses, "trainer did not produce eval_S2_loss entries"

    def test_fit_allows_no_eval_when_strategy_no(self, tmp_path):
        """eval_strategy='no' bypasses the val_samples requirement."""
        from coldddi.llm.trainer import LoRATrainer

        cfg = self._trainer_cfg(tmp_path, eval_strategy="no")
        trainer = LoRATrainer(cfg)
        trainer.fit(self._build_train_samples(), val_samples=None)


# End-to-end per-split L3 → L5

class TestEndToEnd:
    def test_parse_score_select_per_split(self, tmp_path):
        """Full pipeline: train with S2 eval, parse candidates per
        split (S2 only — only S2 was logged), score, select."""
        import pandas as pd
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds
        from coldddi.llm.prompts.binary_cls import (
            PromptBuildConfig, build_binary_prompt,
        )
        from coldddi.llm.select_best import (
            parse_candidate_ckpts, score_candidate_ckpts, select_best,
            write_manifest,
        )
        from coldddi.llm.trainer import LLMTrainerConfig, LoRATrainer

        # Synthetic 4-drug PairDataset
        drugs = pd.DataFrame({
            "drugbank_id": ["DB001", "DB002", "DB003", "DB004"],
            "name":        ["A", "B", "C", "D"],
            "smiles":      ["CCO", "CCC", "c1ccccc1", "CC(=O)OC"],
            "type":        ["small molecule"] * 4,
            "groups":      ["approved"] * 4,
        })
        train = pd.DataFrame({
            "drug_a_id": ["DB001", "DB001", "DB002", "DB003"],
            "drug_b_id": ["DB002", "DB003", "DB004", "DB004"],
        })
        val_s2 = pd.DataFrame({
            "drug_a_id": ["DB001", "DB002"],
            "drug_b_id": ["DB003", "DB004"],
        })
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        val_s2_neg = pd.DataFrame({
            "drug_a_id": ["DB001", "DB002"],
            "drug_b_id": ["DB004", "DB003"],
        })

        splits = SplitFolds(
            train=train,
            val_s0=empty, val_s1=empty, val_s2=val_s2,
            test_s0=empty, test_s1=empty, test_s2=empty,
            g1_drugs=["DB001", "DB002"],
            g2_drugs=["DB003", "DB004"],
            seed=42,
        )
        empty_e = pd.DataFrame(columns=[
            "drugbank_id", "enzyme_id", "enzyme_name", "organism", "action",
        ])
        empty_t = pd.DataFrame(columns=[
            "drugbank_id", "target_id", "target_name", "organism", "action",
        ])
        empty_tr = pd.DataFrame(columns=[
            "drugbank_id", "transporter_id", "transporter_name", "organism", "action",
        ])
        empty_c = pd.DataFrame(columns=[
            "drugbank_id", "carrier_id", "carrier_name", "organism", "action",
        ])
        empty_p = pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"])
        kg = KnowledgeGraph(
            enzymes=empty_e, targets=empty_t,
            transporters=empty_tr, carriers=empty_c, pathways=empty_p,
        )
        ds = PairDataset(edges=train.copy(), splits=splits, kg=kg, drugs=drugs)

        def _get_negatives(split: str):
            return val_s2_neg if split == "val_s2" else empty
        ds.get_negatives = _get_negatives  # type: ignore[method-assign]

        cfg = LLMTrainerConfig(
            model_name=TINY_MODEL,
            output_dir=str(tmp_path / "ckpts"),
            dtype="float32", device="cpu",
            num_epochs=1, micro_batch_size=2,
            learning_rate=1e-4, max_length=128,
            logging_steps=1, save_steps=1, eval_steps=1,
            save_total_limit=10, eval_strategy="steps",
            disable_tqdm=True, report_to=(), seed=42,
        )
        cfg.lora.r = 4
        cfg.lora.alpha = 8
        cfg.lora.target_modules = ("q_proj", "v_proj")

        trainer = LoRATrainer(cfg)
        prompt_cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Zero_Shot_Sequence",
            model_name=TINY_MODEL,
        )
        train_samples = []
        for i, (a, b) in enumerate(zip(train["drug_a_id"], train["drug_b_id"])):
            s = {"drug_a_id": a, "drug_b_id": b,
                 "drug_a_name": "n", "drug_b_name": "n",
                 "drugA_name": "n", "drugB_name": "n",
                 "label": i % 2}
            train_samples.append({
                "text": build_binary_prompt(s, prompt_cfg),
                "cls_labels": i % 2,
            })
        # Build val samples (same format as train: text + cls_labels).
        val_samples = []
        for i, (a, b) in enumerate(zip(val_s2["drug_a_id"], val_s2["drug_b_id"])):
            s = {"drug_a_id": a, "drug_b_id": b,
                 "drug_a_name": "n", "drug_b_name": "n",
                 "drugA_name": "n", "drugB_name": "n",
                 "label": i % 2}
            val_samples.append({
                "text": build_binary_prompt(s, prompt_cfg),
                "cls_labels": i % 2,
            })
        trainer.fit(
            train_samples=train_samples,
            val_samples={"S2": val_samples},
        )

        cands = parse_candidate_ckpts(cfg.output_dir, splits=("S2",), topk=3)
        assert "S2" in cands and cands["S2"]
        scored = score_candidate_ckpts(
            cands, base_model_name=TINY_MODEL, dataset=ds,
            prompt_cfg=prompt_cfg, device="cpu", dtype="float32",
            batch_size=2, max_length=128,
        )
        manifest = select_best(scored)
        assert "S2" in manifest
        assert Path(manifest["S2"]["best_ckpt"]).is_dir()
        out = write_manifest(manifest, tmp_path / "manifest.json")
        assert out.is_file()
