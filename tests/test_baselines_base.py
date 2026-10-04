"""Test the BaselineModel ABC and registry with a mock, without PyTorch or RDKit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from coldddi.data.dataset import PairDataset


# A private registry name avoids collisions with real baselines.


def _make_mock_class():
    """Build the mock lazily to avoid registration at module import."""
    from coldddi.baselines.base import BaselineModel, register, write_manifest

    @register("__mock_baseline__")
    class MockBaseline(BaselineModel):
        VERSION = "1.0"

        def __init__(self) -> None:
            self.fitted = False
            self.score_const = 0.5

        def fit(self, train, val=None, *, kg=None):
            self.fitted = True
            self.score_const = 0.7  # mark "trained"

        def predict_proba(self, pairs, *, kg=None):
            return np.full(len(pairs), self.score_const, dtype=float)

        def save(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "weights.json").write_text(
                json.dumps({"score_const": self.score_const, "fitted": self.fitted})
            )
            write_manifest(
                path,
                baseline_name=self.name,
                extra={"version": self.VERSION},
            )

        @classmethod
        def load(cls, path):
            path = Path(path)
            inst = cls()
            payload = json.loads((path / "weights.json").read_text())
            inst.score_const = payload["score_const"]
            inst.fitted = payload["fitted"]
            return inst

    return MockBaseline


@pytest.fixture(scope="module")
def MockBaseline():
    cls = _make_mock_class()
    yield cls
    # Remove the mock to keep registry assertions independent of test order.
    from coldddi.baselines.base import _REGISTRY

    _REGISTRY.pop("__mock_baseline__", None)


# Registry


class TestRegistry:
    def test_register_decorator_sets_name(self, MockBaseline):
        assert MockBaseline.name == "__mock_baseline__"

    def test_list_baselines_includes_mock(self, MockBaseline):
        from coldddi.baselines import list_baselines

        assert "__mock_baseline__" in list_baselines()

    def test_double_register_same_name_raises(self, MockBaseline):
        from coldddi.baselines.base import BaselineModel, register

        with pytest.raises(ValueError, match="already registered"):
            @register("__mock_baseline__")
            class _Other(BaselineModel):
                def fit(self, train, val=None, *, kg=None): ...
                def predict_proba(self, pairs, *, kg=None):
                    return np.zeros(len(pairs))
                def save(self, path): ...
                @classmethod
                def load(cls, path): ...


# Save / load round-trip


class TestSaveLoadDispatch:
    def test_save_writes_manifest(self, MockBaseline, tmp_path):
        m = MockBaseline()
        m.fit(train=None)
        m.save(tmp_path / "ckpt")
        manifest = tmp_path / "ckpt" / "manifest.json"
        assert manifest.is_file()
        payload = json.loads(manifest.read_text())
        assert payload["baseline_name"] == "__mock_baseline__"
        assert payload["version"] == "1.0"

    def test_load_baseline_dispatches_via_manifest(self, MockBaseline, tmp_path):
        from coldddi.baselines import load_baseline

        m = MockBaseline()
        m.fit(train=None)
        m.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, MockBaseline)
        assert loaded.fitted is True
        assert loaded.score_const == 0.7

    def test_load_baseline_round_trip_preserves_predict(self, MockBaseline, tmp_path):
        from coldddi.baselines import load_baseline

        m = MockBaseline()
        m.fit(train=None)
        m.save(tmp_path / "ckpt")
        reloaded = load_baseline(tmp_path / "ckpt")
        pairs = pd.DataFrame(
            {"drug_a_id": ["DB1", "DB2"], "drug_b_id": ["DB3", "DB4"]}
        )
        np.testing.assert_array_equal(
            m.predict_proba(pairs), reloaded.predict_proba(pairs)
        )

    def test_missing_manifest_raises(self, tmp_path):
        from coldddi.baselines import load_baseline

        empty = tmp_path / "nope"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_baseline(empty)

    def test_unknown_baseline_name_in_manifest_raises(self, tmp_path):
        from coldddi.baselines import load_baseline

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({"baseline_name": "__no_such_baseline__"})
        )
        with pytest.raises(ValueError, match="Unknown baseline"):
            load_baseline(bad)


# predict_proba contract


class TestPredictContract:
    def test_predict_returns_1d_ndarray_with_correct_length(self, MockBaseline):
        m = MockBaseline()
        pairs = pd.DataFrame(
            {"drug_a_id": ["DB1", "DB2", "DB3"], "drug_b_id": ["DB4", "DB5", "DB6"]}
        )
        out = m.predict_proba(pairs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)


# evaluate.py CLI surface


class TestEvaluateCLI:
    def test_unknown_method_raises(self, tmp_path):
        from coldddi.evaluate import run_evaluation

        with pytest.raises(ValueError, match="Unknown method"):
            run_evaluation(
                method="__never_registered__",
                data_dir=tmp_path,
                seed=42,
                settings=["S2"],
                out_dir=tmp_path / "out",
            )

    def test_lazy_import_fails_when_module_absent(self):
        from coldddi.evaluate import _ensure_baseline_imported

        with pytest.raises(ImportError, match="Could not import"):
            _ensure_baseline_imported("__no_such_module__")
