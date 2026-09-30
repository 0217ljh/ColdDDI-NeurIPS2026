"""Offline contract tests for the one-command runner; no model downloads."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from coldddi import benchmark
from coldddi.benchmark_data import pair_keys, validate_dataset
from coldddi.diagnostics import build_bucket_lookup, compute_indicators
from coldddi.diagnostics.kps_swap import SwapTriple


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "release"
    shutil.copytree(repo / "data/public/intermediate/filtered", root / "filtered")
    shutil.copytree(repo / "data/public/intermediate/splits", root / "splits")
    ab = tmp_path / "ab.parquet"
    shutil.copyfile(repo / "annotations/ab_sample.parquet", ab)
    return root, ab


def test_valid_dataset_and_fingerprint(dataset: tuple[Path, Path]) -> None:
    root, ab = dataset
    _, _, first = validate_dataset(root, ab, 42)
    assert first["drugs"] == 86
    assert first["splits"]["train"] == {"positive": 437, "negative": 437}
    assert first["train_negative_overlap"]["test_s0"] == 40
    assert first["warnings"]
    frame = pd.read_parquet(ab)
    frame.loc[0, "pk_pd_label"] = "Unknown"
    frame.to_parquet(ab, index=False)
    assert validate_dataset(root, ab, 42)[2]["fingerprint"] != first["fingerprint"]


@pytest.mark.parametrize("change, message", [
    ("missing", "Missing benchmark input"),
    ("ab_subset", "cover exactly"),
    ("ab_bool", "must contain booleans"),
    ("ab_label", "pk_pd_label must"),
    ("ab_name", "Type-A pairs require"),
    ("positive_overlap", "positive pairs overlap"),
    ("negative_positive", "conflict with known positive"),
    ("wrong_partition", "violates the s0"),
    ("unknown_drug", "unknown drug IDs"),
    ("manifest_seed", "seed does not match"),
    ("manifest_count", "n_pairs does not match"),
    ("overlapping_drug_groups", "G1/G2 must be disjoint"),
    ("kg_unknown", "unknown drug IDs"),
])
def test_bad_data_fails(dataset: tuple[Path, Path], change: str, message: str) -> None:
    root, ab = dataset
    split = root / "splits/seed42"
    if change == "missing":
        (split / "negatives/test_s2.parquet").unlink()
    elif change.startswith("ab_"):
        frame = pd.read_parquet(ab)
        if change == "ab_subset":
            frame = frame.iloc[1:]
        elif change == "ab_bool":
            frame["has_key_entity"] = frame.has_key_entity.astype(str)
        elif change == "ab_label":
            frame.loc[0, "pk_pd_label"] = "typo"
        else:
            frame.loc[frame.has_key_entity, "key_entity_name"] = ""
        frame.to_parquet(ab, index=False)
    elif change in ("manifest_seed", "manifest_count", "overlapping_drug_groups"):
        path = split / "manifest.json"
        value = json.loads(path.read_text())
        if change == "manifest_seed":
            value["seed"] = 0
        elif change == "manifest_count":
            value["n_pairs"]["train"] += 1
        else:
            value["g2_drugs"].append(value["g1_drugs"][0])
        benchmark.write_json(path, value)
    elif change == "kg_unknown":
        path = root / "filtered/drug_enzymes.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "drugbank_id"] = "not-a-drug"
        frame.to_csv(path, index=False)
    else:
        path = split / ("negatives/val_s0.parquet" if change == "negative_positive" else "val_s0.parquet")
        frame = pd.read_parquet(path)
        source = pd.read_parquet(split / ("test_s2.parquet" if change == "wrong_partition" else "train.parquet"))
        frame.loc[0, ["drug_a_id", "drug_b_id"]] = source.iloc[0][["drug_a_id", "drug_b_id"]].values
        if change == "unknown_drug":
            frame.loc[0, "drug_a_id"] = "not-a-drug"
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        validate_dataset(root, ab, 42)


@pytest.mark.parametrize("pairs", [
    [("a", "b"), ("b", "a")], [("a", "a")], [(None, "b")], [(" a", "b")], [("", "b")],
])
def test_invalid_pair_ids(pairs: list[tuple]) -> None:
    with pytest.raises(ValueError):
        pair_keys(pd.DataFrame(pairs, columns=["drug_a_id", "drug_b_id"]), "test")


def test_sample_caps_and_directed_union() -> None:
    pos = pd.DataFrame({"drug_a_id": [f"a{i}" for i in range(20)], "drug_b_id": ["b"] * 20})
    neg = pd.DataFrame({"drug_a_id": [f"c{i}" for i in range(20)], "drug_b_id": ["b"] * 20})
    assert len(benchmark.balanced_pairs(pos, neg, None)) == 40
    assert benchmark.balanced_pairs(pos, neg, 16).label.value_counts().to_dict() == {1: 8, 0: 8}
    swaps = [SwapTriple("b", "a0", "c0", 1), SwapTriple("a0", "b", "c1", 1)]
    union = benchmark.diagnostic_pairs(pos.head(1), swaps)
    assert set(map(tuple, union.values)) == {("a0", "b"), ("b", "a0"), ("c0", "a0"), ("c1", "b")}


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "nan", "infinity", "negative", "above_one"])
def test_invalid_predictions(case: str) -> None:
    pairs = pd.DataFrame({"drug_a_id": ["a", "c"], "drug_b_id": ["b", "d"]})
    predictions = pairs.assign(p_yes=[0.1, 0.9])
    if case == "missing":
        predictions = predictions.head(1)
    elif case == "extra":
        predictions.loc[0, "drug_a_id"] = "x"
    elif case == "duplicate":
        predictions = pd.concat([predictions, predictions.head(1)])
    else:
        predictions.loc[0, "p_yes"] = {"nan": np.nan, "infinity": np.inf, "negative": -0.1, "above_one": 1.1}[case]
    with pytest.raises(ValueError):
        benchmark.validate_predictions(predictions, pairs)


def test_unordered_and_partial_predictions() -> None:
    pairs = pd.DataFrame({"drug_a_id": ["a", "c"], "drug_b_id": ["b", "d"]})
    predictions = pairs.assign(p_yes=[0.1, 0.9]).iloc[::-1]
    benchmark.validate_predictions(predictions, pairs)
    benchmark.validate_predictions(predictions.head(1), pairs, complete=False)


def test_diagnostics_match_existing_and_hand_computation(tmp_path: Path) -> None:
    ab = pd.DataFrame({"drug_a_id": ["a"], "drug_b_id": ["b"], "pk_pd_label": ["PK"], "has_key_entity": [True]})
    buckets = build_bucket_lookup(ab)
    swaps = [SwapTriple("a", "b", "c", 1), SwapTriple("a", "b", "d", 1)]
    predictions = {condition: {("a", "b"): value, ("c", "b"): 0.3, ("d", "b"): 0.1}
                   for condition, value in zip(("R0", "R1", "R2", "R3"), (0.8, 0.7, 0.4, 0.2))}
    coverage = benchmark.save_diagnostics(predictions, swaps, buckets, tmp_path)
    actual = pd.read_csv(tmp_path / "indicators.csv")
    expected = compute_indicators(predictions, swaps, bucket_fn=buckets.bucket)
    pd.testing.assert_frame_equal(actual, expected)
    aggregate = actual.set_index(["indicator", "bucket"])
    assert aggregate.loc[("KPS-F", "ALL"), "value"] == pytest.approx(0.6)
    assert aggregate.loc[("KSAI", "ALL"), "value"] == pytest.approx(0.1)
    assert aggregate.loc[("KPS-F", "ALL"), "n"] == 2
    assert aggregate.loc[("KSAI", "ALL"), "n"] == 1
    assert coverage["missing_predictions"] == 0
    assert "KSAI" in coverage["undefined_ab_gaps"]
    assert json.loads((tmp_path / "ab_gaps.json").read_text())["KSAI"]["value"] is None
    del predictions["R1"][("c", "b")]
    with pytest.raises(ValueError, match="missing swap anchor/target"):
        benchmark.save_diagnostics(predictions, swaps, buckets, tmp_path)


@pytest.mark.parametrize("flags", [["--epochs", "0"], ["--seed", "-1"], ["--batch-size", "3"], ["--prompt", "P1"]])
def test_invalid_cli_arguments(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--data", "unused", "--ab-parquet", "unused"] + flags)
    assert exc.value.code == 2


def test_check_only_has_no_model_or_output(dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ab = dataset
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("check-only must not resolve/download a model")
    monkeypatch.setattr(benchmark, "resolve_model", forbidden)
    output = tmp_path / "run"
    assert benchmark.main(["--data", str(root), "--ab-parquet", str(ab), "--output", str(output), "--check-only"]) == 0
    assert not output.exists()


def test_failed_run_resume_contract(dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from coldddi.llm.trainer import LoRATrainer

    root, ab = dataset
    output = tmp_path / "run"
    monkeypatch.setattr(benchmark, "resolve_model", lambda *args: ("qwen-test", "/qwen-test", "modelhash"))
    def fail_fit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deliberate test interruption")
    monkeypatch.setattr(LoRATrainer, "fit", fail_fit)
    flags = ["--data", str(root), "--ab-parquet", str(ab), "--output", str(output), "--smoke", "--device", "cpu"]
    assert benchmark.main(flags) == 1
    state_path = output / "status.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == "failed"
    assert "deliberate test interruption" in state["error"]
    # An unchanged resume reaches fit again, rather than a tuple/list config mismatch.
    assert benchmark.main(flags + ["--resume"]) == 1
    assert "deliberate test interruption" in json.loads(state_path.read_text())["error"]
    before = state_path.read_bytes()
    assert benchmark.main(flags + ["--resume", "--epochs", "2"]) == 1
    assert state_path.read_bytes() == before
    assert benchmark.main(flags) == 1  # no overwrite without --resume
    (output / "effective_counts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed artifact missing or changed"):
        benchmark.run(benchmark.build_parser().parse_args(flags + ["--resume"]))
