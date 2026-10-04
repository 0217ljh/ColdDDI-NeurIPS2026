"""Test prediction CSV schemas and split layout from Appendix A.6.2, lines 549-557.

These files supply per-pair probabilities to the L6 diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_RELEASE = REPO_ROOT / "data" / "public" / "intermediate"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
    reason="Toy filtered dir not found — run reconstruct.py --toy first.",
)


# PREDICTION_COLUMNS contract

class TestPredictionColumnsSchema:
    def test_canonical_column_order(self):
        from coldddi.evaluate import PREDICTION_COLUMNS

        assert PREDICTION_COLUMNS == (
            "drug_a_id",
            "drug_b_id",
            "true_label",
            "predicted_prob",
            "predicted_label",
        )


# _predictions_to_df helper math

class _ConstantBaseline:
    """Tiny stand-in: every pair gets a fixed predicted_prob."""

    def __init__(self, prob: float):
        self._prob = prob

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.full(len(pairs), self._prob, dtype=np.float32)


class TestPredictionsToDf:
    def test_columns_match_canonical(self):
        from coldddi.evaluate import PREDICTION_COLUMNS, _predictions_to_df

        pos = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["B"]})
        neg = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["C"]})
        df = _predictions_to_df(_ConstantBaseline(0.7), pos, neg)
        assert list(df.columns) == list(PREDICTION_COLUMNS)

    def test_true_label_per_class(self):
        from coldddi.evaluate import _predictions_to_df

        pos = pd.DataFrame({"drug_a_id": ["A", "X"], "drug_b_id": ["B", "Y"]})
        neg = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["C"]})
        df = _predictions_to_df(_ConstantBaseline(0.42), pos, neg)
        # 2 positives + 1 negative.
        assert int((df["true_label"] == 1).sum()) == 2
        assert int((df["true_label"] == 0).sum()) == 1

    def test_predicted_label_threshold(self):
        """``predicted_prob >= 0.5`` → predicted_label == 1.
        Mirrors upstream baseline inference CSV convention."""
        from coldddi.evaluate import _predictions_to_df

        pos = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["B"]})
        neg = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        # Above threshold.
        df_hi = _predictions_to_df(_ConstantBaseline(0.5), pos, neg)
        assert int(df_hi["predicted_label"].iloc[0]) == 1
        df_hi2 = _predictions_to_df(_ConstantBaseline(0.51), pos, neg)
        assert int(df_hi2["predicted_label"].iloc[0]) == 1
        # Below threshold.
        df_lo = _predictions_to_df(_ConstantBaseline(0.499), pos, neg)
        assert int(df_lo["predicted_label"].iloc[0]) == 0

    def test_empty_split_returns_empty_frame_with_schema(self):
        """Empty splits retain the canonical prediction columns."""
        from coldddi.evaluate import PREDICTION_COLUMNS, _predictions_to_df

        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        df = _predictions_to_df(_ConstantBaseline(0.5), empty, empty)
        assert df.empty
        assert list(df.columns) == list(PREDICTION_COLUMNS)


# _evaluate_split_with_predictions: metrics-vs-CSV consistency

class TestEvaluateSplitWithPredictions:
    def test_aggregate_metrics_match_df_means(self):
        """Metrics means match CSV rows, avoiding drift from separate prediction calls."""
        from coldddi.evaluate import _evaluate_split_with_predictions

        pos = pd.DataFrame({"drug_a_id": ["A", "X"], "drug_b_id": ["B", "Y"]})
        neg = pd.DataFrame({"drug_a_id": ["A", "X"], "drug_b_id": ["C", "Z"]})
        metrics, df = _evaluate_split_with_predictions(
            _ConstantBaseline(0.7), pos, neg,
        )
        # Aggregates derive from df rows, byte-equivalent.
        assert metrics["n_pos"] == 2
        assert metrics["n_neg"] == 2
        assert metrics["mean_pos_score"] == pytest.approx(
            float(df.loc[df["true_label"] == 1, "predicted_prob"].mean())
        )
        assert metrics["mean_neg_score"] == pytest.approx(
            float(df.loc[df["true_label"] == 0, "predicted_prob"].mean())
        )

    def test_legacy_evaluate_split_returns_only_metrics(self):
        """The backward-compatible _evaluate_split returns a plain dict."""
        from coldddi.evaluate import _evaluate_split

        pos = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["B"]})
        neg = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        out = _evaluate_split(_ConstantBaseline(0.6), pos, neg)
        assert isinstance(out, dict)
        assert out["n_pos"] == 1
        assert out["n_neg"] == 0

    def test_zero_positives_still_reports_neg_mean(self):
        """Negative-only splits retain mean_neg_score despite having no positives."""
        from coldddi.evaluate import _evaluate_split_with_predictions

        empty_pos = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        neg = pd.DataFrame({"drug_a_id": ["A", "X"], "drug_b_id": ["C", "Z"]})
        metrics, df = _evaluate_split_with_predictions(
            _ConstantBaseline(0.3), empty_pos, neg,
        )
        assert metrics["n_pos"] == 0
        assert metrics["n_neg"] == 2
        assert metrics["mean_neg_score"] == pytest.approx(0.3)
        # ``mean_pos_score`` correctly absent when there are no positives.
        assert "mean_pos_score" not in metrics
        # CSV has only the 2 negative rows, schema preserved.
        assert len(df) == 2
        assert int((df["true_label"] == 0).sum()) == 2

    def test_fully_empty_split_returns_minimal_metrics(self):
        """pos=0 AND neg=0 → only ``n_pos``/``n_neg`` keys, empty df."""
        from coldddi.evaluate import (
            PREDICTION_COLUMNS,
            _evaluate_split_with_predictions,
        )

        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        metrics, df = _evaluate_split_with_predictions(
            _ConstantBaseline(0.5), empty, empty,
        )
        assert metrics == {"n_pos": 0, "n_neg": 0}
        assert df.empty
        assert list(df.columns) == list(PREDICTION_COLUMNS)


# NaN-output behavior.

class _NaNBaseline:
    """Return NaN predictions to verify the pipeline preserves them without crashing."""

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.full(len(pairs), np.nan, dtype=np.float32)


class TestNanPredictionsSurviveHelpers:
    def test_nan_predictions_survive_to_df(self):
        from coldddi.evaluate import _predictions_to_df

        pos = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["B"]})
        neg = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["C"]})
        df = _predictions_to_df(_NaNBaseline(), pos, neg)
        # NaN predicted_prob → predicted_label is 0 under (NaN >= 0.5) == False.
        assert df["predicted_prob"].isna().all()
        assert (df["predicted_label"] == 0).all()

    def test_nan_predictions_propagate_to_metrics(self):
        from coldddi.evaluate import _evaluate_split_with_predictions

        pos = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["B"]})
        neg = pd.DataFrame({"drug_a_id": ["A"], "drug_b_id": ["C"]})
        metrics, df = _evaluate_split_with_predictions(_NaNBaseline(), pos, neg)
        # Preserve NaN means for downstream debugging.
        import math

        assert math.isnan(metrics["mean_pos_score"])
        assert math.isnan(metrics["mean_neg_score"])


# Multi-setting end-to-end: --setting all writes 6 CSVs

class TestRunEvaluationMultiSettingWritesAllCsvs:
    """Pin paper A.6.2 promise that ``--setting all`` writes a CSV per
    evaluated split — 3 settings × (val + test) = 6 CSVs."""

    @pytest.fixture(scope="class")
    def multi_out(self, tmp_path_factory):
        pytest.importorskip("torch")
        from coldddi.evaluate import run_evaluation

        out_dir = tmp_path_factory.mktemp("eval_multi")
        run_evaluation(
            method="deepddi",
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S0", "S1", "S2"],
            out_dir=out_dir,
            device="cpu",
            preset="smoke",   # CI: skip paper-spec 100-epoch DeepDDI training
        )
        return out_dir

    def test_all_six_split_csvs_present(self, multi_out):
        expected = {
            "predictions_val_s0_seed42.csv",
            "predictions_test_s0_seed42.csv",
            "predictions_val_s1_seed42.csv",
            "predictions_test_s1_seed42.csv",
            "predictions_val_s2_seed42.csv",
            "predictions_test_s2_seed42.csv",
        }
        actual_csvs = {p.name for p in multi_out.glob("predictions_*.csv")}
        # All six toy splits are non-empty and must produce files.
        missing = expected - actual_csvs
        assert not missing, f"missing per-split CSV: {sorted(missing)}"

    def test_metrics_json_has_all_split_keys(self, multi_out):
        import json

        metrics = json.loads(
            (multi_out / "metrics_seed42.json").read_text()
        )
        for k in (
            "val_s0", "test_s0",
            "val_s1", "test_s1",
            "val_s2", "test_s2",
        ):
            assert k in metrics, f"missing metrics key {k}"


# run_evaluation end-to-end: CSV files exist with right schema

class TestRunEvaluationWritesPredictionCsv:
    """End-to-end through DeepDDI (smallest baseline) on the toy fixture.
    Verifies the paper-promised per-pair CSV is written under ``--out``."""

    @pytest.fixture(scope="class")
    def run_out(self, tmp_path_factory):
        pytest.importorskip("torch")
        from coldddi.evaluate import run_evaluation

        out_dir = tmp_path_factory.mktemp("eval_out")
        run_evaluation(
            method="deepddi",
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S2"],
            out_dir=out_dir,
            device="cpu",
            preset="smoke",   # CI: skip paper-spec 100-epoch DeepDDI training
        )
        return out_dir

    def test_metrics_json_still_written(self, run_out):
        assert (run_out / "metrics_seed42.json").is_file()

    def test_predictions_csv_per_split_written(self, run_out):
        """One CSV per evaluated split (val + test for each setting).
        For ``settings=['S2']`` that's val_s2 + test_s2 = two CSVs."""
        assert (run_out / "predictions_val_s2_seed42.csv").is_file()
        assert (run_out / "predictions_test_s2_seed42.csv").is_file()

    def test_predictions_csv_schema(self, run_out):
        from coldddi.evaluate import PREDICTION_COLUMNS

        df = pd.read_csv(run_out / "predictions_test_s2_seed42.csv")
        assert list(df.columns) == list(PREDICTION_COLUMNS)
        assert (df["predicted_prob"] >= 0).all()
        assert (df["predicted_prob"] <= 1).all()
        assert df["true_label"].isin([0, 1]).all()
        assert df["predicted_label"].isin([0, 1]).all()

    def test_predictions_csv_row_count_matches_metrics(self, run_out):
        """``n_pos + n_neg`` from metrics JSON == row count in CSV."""
        import json

        metrics = json.loads(
            (run_out / "metrics_seed42.json").read_text()
        )
        df = pd.read_csv(run_out / "predictions_test_s2_seed42.csv")
        n_expected = metrics["test_s2"]["n_pos"] + metrics["test_s2"]["n_neg"]
        assert len(df) == n_expected

    def test_predictions_csv_label_breakdown_matches_metrics(self, run_out):
        """``true_label == 1`` count in CSV == ``n_pos`` in metrics."""
        import json

        metrics = json.loads(
            (run_out / "metrics_seed42.json").read_text()
        )
        df = pd.read_csv(run_out / "predictions_test_s2_seed42.csv")
        assert int((df["true_label"] == 1).sum()) == metrics["test_s2"]["n_pos"]
        assert int((df["true_label"] == 0).sum()) == metrics["test_s2"]["n_neg"]
