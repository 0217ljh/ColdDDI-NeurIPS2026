"""Opt-in real CLI tests: COLDDDI_RUN_INTEGRATION=1 python -m pytest ... .

Downloads a tiny random Qwen model on first use. Failures are never converted
to skips after the integration test has been explicitly enabled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from coldddi.benchmark_data import file_sha256
from coldddi.diagnostics import build_bucket_lookup


pytestmark = pytest.mark.skipif(os.environ.get("COLDDDI_RUN_INTEGRATION") != "1",
                                reason="opt-in real model download/training: set COLDDDI_RUN_INTEGRATION=1")


@pytest.fixture(scope="module", params=["smoke", "uncapped"])
def completed_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp(f"benchmark-{request.param}")
    args = [sys.executable, "scripts/run_benchmark.py", "--data", "data/public/intermediate",
            "--ab-parquet", "annotations/ab_sample.parquet", "--model", "tiny-random-qwen",
            "--device", "cpu", "--epochs", "1", "--max-length", "512", "--batch-size", "2",
            "--output", str(output)]
    if request.param == "smoke":
        args.append("--smoke")
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    result = subprocess.run(args, cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return repo, output, args, env


def test_complete_run_and_actual_counts(completed_run: tuple) -> None:
    _, output, _, _ = completed_run
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "complete"
    for name, digest in state["artifacts"].items():
        assert file_sha256(output / name) == digest
    config = json.loads((output / "config.json").read_text())
    counts = json.loads((output / "effective_counts.json").read_text())
    inputs = json.loads((output / "input_report.json").read_text())
    if config["smoke"]:
        assert counts["train"] == 16
        assert set(counts["test"].values()) == {16}
    else:
        assert set(config["caps"].values()) == {None}
        assert counts["train"] == sum(inputs["splits"]["train"].values())
        for split in ("S0", "S1", "S2"):
            assert counts["test"][split] == sum(inputs["splits"][f"test_{split.lower()}"].values())
        trainer = config["trainer"]
        assert trainer["micro_batch_size"] * trainer["gradient_accumulation_steps"] == 16
    metrics = json.loads((output / "metrics.json").read_text())
    for split in ("S0", "S1", "S2"):
        table = pd.read_csv(output / f"test_{split}.csv")
        assert len(table) == counts["test"][split] == metrics[split]["pairs"]
        assert np.isfinite(table.p_yes).all()
        assert 0 <= metrics[split]["auroc"] <= 1
    assert json.loads((output / "coverage.json").read_text())["missing_predictions"] == 0


def test_all_indicator_cells_by_independent_recomputation(completed_run: tuple) -> None:
    repo, output, _, _ = completed_run
    buckets = build_bucket_lookup(repo / "annotations/ab_sample.parquet")
    swaps = list(pd.read_parquet(output / "swap_candidates.parquet").itertuples(index=False))
    probabilities = []
    for condition in range(4):
        frame = pd.read_parquet(output / f"predictions_S2_R{condition}.parquet")
        probabilities.append({(r.drug_a_id, r.drug_b_id): r.p_yes for r in frame.itertuples(index=False)})
    channels = {"KPS-Name": (0, 1), "KPS-KG": (0, 2), "KPS-KG-Named": (0, 2),
                "KPS-KG-Masked": (1, 3), "KPS-Name-KGMasked": (2, 3)}
    for cell in pd.read_csv(output / "indicators.csv").itertuples(index=False):
        selected = [t for t in swaps if (t.label_uv == 1 if cell.bucket == "ALL" else buckets.bucket(t.qa, t.qb) == cell.bucket)]
        if cell.indicator == "KPS-F":
            values = [abs(probabilities[0][t.qa, t.qb] - probabilities[0][t.qa_prime, t.qb]) for t in selected]
        else:
            anchors = {(t.qa, t.qb) for t in selected}
            if cell.indicator == "KSAI":
                values = [abs(probabilities[1][p] - probabilities[3][p]) - abs(probabilities[0][p] - probabilities[2][p]) for p in anchors]
            else:
                a, b = channels[cell.indicator]
                values = [abs(probabilities[a][p] - probabilities[b][p]) for p in anchors]
        assert cell.n == len(values)
        assert cell.value == pytest.approx(np.mean(values), abs=1e-12)
        assert cell.std == pytest.approx(np.std(values, ddof=1) if len(values) > 1 else 0, abs=1e-12)


def test_completed_resume_and_refusal_to_overwrite(completed_run: tuple) -> None:
    repo, output, args, env = completed_run
    state = (output / "status.json").read_bytes()
    for extra, expected in (([], 1), (["--resume"], 0), (["--resume", "--epochs", "2"], 1)):
        result = subprocess.run(args + extra, cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        assert result.returncode == expected, result.stdout + result.stderr
        assert (output / "status.json").read_bytes() == state
        if expected == 0:
            assert "Verified completed run" in result.stdout


def test_completed_artifact_tampering_is_rejected(completed_run: tuple) -> None:
    repo, output, args, env = completed_run
    path = output / "metrics.json"
    original = path.read_bytes()
    try:
        path.write_text("{}", encoding="utf-8")
        result = subprocess.run(args + ["--resume"], cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        assert result.returncode == 1
        assert "Completed artifact missing or changed: metrics.json" in result.stderr
    finally:
        path.write_bytes(original)


def test_resume_partial_inference_without_retraining(completed_run: tuple) -> None:
    repo, output, args, env = completed_run
    state_path = output / "status.json"
    state = json.loads(state_path.read_text())
    trained = {name: digest for name, digest in state["artifacts"].items() if name.startswith("ckpts/")}
    target = output / "predictions_S2_R3.parquet"
    expected = pd.read_parquet(target)
    expected.iloc[:len(expected) // 2].to_parquet(target.with_suffix(".parquet.partial"), index=False)
    target.unlink()  # only this test's temporary output, to model interrupted inference
    for name in (target.name, "coverage.json", "metrics.json", "indicators.csv", "ab_gaps.json"):
        state["artifacts"].pop(name)
    state.update(status="failed", stage="inference", error="simulated interruption")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    result = subprocess.run(args + ["--resume"], cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = pd.read_parquet(target)
    columns = ["drug_a_id", "drug_b_id"]
    pd.testing.assert_frame_equal(actual.sort_values(columns).reset_index(drop=True),
                                  expected.sort_values(columns).reset_index(drop=True))
    assert json.loads(state_path.read_text())["status"] == "complete"
    for name, digest in trained.items():
        assert file_sha256(output / name) == digest
