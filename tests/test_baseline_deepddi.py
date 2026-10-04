"""Test DeepDDI fitting, S2 predictions, and save/load parity on the toy fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_FILTERED = REPO_ROOT / "data" / "public" / "intermediate" / "filtered"
TOY_SPLITS = REPO_ROOT / "data" / "public" / "intermediate" / "splits" / "seed42"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = [
    pytest.mark.skipif(
        not TOY_FILTERED.is_dir(),
        reason="Toy filtered dir not found — run reconstruct.py --toy first.",
    ),
    pytest.mark.skipif(
        not TOY_SPLITS.is_dir(),
        reason="Toy splits not present — run reconstruct.py --toy first.",
    ),
]


@pytest.fixture(scope="module")
def toy_release_root():
    return REPO_ROOT / "data" / "public" / "intermediate"


@pytest.fixture(scope="module")
def trained_deepddi(toy_release_root):
    """Fit a tiny DeepDDI network on the toy subset."""
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")
    from coldddi.baselines.deepddi import DeepDDIBaseline
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(toy_release_root, seed=42)
    model = DeepDDIBaseline(
        ssp_dim=8,
        hidden_dim=32,
        n_layers=2,
        n_epochs=1,
        batch_size=64,
        device="cpu",
    )
    model.fit(ds, kg=ds.kg)
    return model, ds


class TestDeepDDIFit:
    def test_fit_populates_model_and_ssp(self, trained_deepddi):
        model, _ = trained_deepddi
        assert model._model is not None
        assert model._ssp_artifacts is not None
        # SSP keyed by drug id
        assert "ssp" in model._ssp_artifacts
        assert len(model._ssp_artifacts["ssp"]) > 0

    def test_predict_proba_returns_1d_in_unit_interval(self, trained_deepddi):
        model, ds = trained_deepddi
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(40)
        neg = ds.get_negatives("test_s2").head(40)
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        scores = model.predict_proba(all_pairs)
        assert scores.shape == (len(all_pairs),)
        assert scores.dtype == np.float32 or scores.dtype == np.float64
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_before_fit_raises(self):
        from coldddi.baselines.deepddi import DeepDDIBaseline

        m = DeepDDIBaseline(ssp_dim=4, hidden_dim=8, n_layers=2, n_epochs=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict_proba(
                pd.DataFrame({"drug_a_id": ["DB1"], "drug_b_id": ["DB2"]})
            )

    def test_fit_with_empty_g1_raises(self, toy_release_root):
        """Reject empty G1 to prevent leaking G2 into the SSP basis."""
        from coldddi.baselines.deepddi import DeepDDIBaseline
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_release_dir(toy_release_root, seed=42)
        # Manually empty out g1 to simulate a missing cold-start partition.
        ds.splits.g1_drugs = []

        m = DeepDDIBaseline(ssp_dim=4, hidden_dim=8, n_layers=2, n_epochs=1)
        with pytest.raises(ValueError, match="g1_drugs is empty"):
            m.fit(ds)

    def test_fit_without_drugs_table_raises(self):
        """When ds.drugs is None DeepDDI cannot build SSP — must fail fast."""
        from coldddi.baselines.deepddi import DeepDDIBaseline
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds

        empty_kg = KnowledgeGraph(
            enzymes=pd.DataFrame(),
            targets=pd.DataFrame(),
            transporters=pd.DataFrame(),
            carriers=pd.DataFrame(),
            pathways=pd.DataFrame(),
        )
        empty_splits = SplitFolds(
            train=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            val_s0=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            val_s1=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            val_s2=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            test_s0=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            test_s1=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            test_s2=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            g1_drugs=[],
            g2_drugs=[],
            seed=42,
        )
        ds = PairDataset(
            edges=pd.DataFrame(columns=["drug_a_id", "drug_b_id"]),
            splits=empty_splits,
            kg=empty_kg,
            drugs=None,
        )
        m = DeepDDIBaseline(ssp_dim=4, hidden_dim=8, n_layers=2, n_epochs=1)
        with pytest.raises(ValueError, match="`drugs` table"):
            m.fit(ds)


class TestDeepDDISaveLoad:
    def test_save_writes_required_files(self, trained_deepddi, tmp_path):
        model, _ = trained_deepddi
        model.save(tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "model.pt").is_file()
        assert (tmp_path / "ckpt" / "ssp.pkl").is_file()
        assert (tmp_path / "ckpt" / "manifest.json").is_file()

    def test_load_baseline_dispatcher_routes_to_deepddi(self, trained_deepddi, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.deepddi import DeepDDIBaseline

        model, _ = trained_deepddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, DeepDDIBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_deepddi, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_deepddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")

        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        np.testing.assert_allclose(
            model.predict_proba(pairs),
            loaded.predict_proba(pairs),
            rtol=1e-5,
            atol=1e-5,
        )


class TestDeepDDIRegistry:
    def test_deepddi_is_registered_after_import(self):
        # Import side-effect: registers under "deepddi"
        import coldddi.baselines.deepddi  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "deepddi" in list_baselines()


class TestCleanProcessAutoLoad:
    """Load DeepDDI in a fresh process via NAME_TO_MODULE's lazy import."""

    def test_load_baseline_in_fresh_subprocess(self, trained_deepddi, tmp_path):
        import subprocess

        model, _ = trained_deepddi
        ckpt = tmp_path / "ckpt"
        model.save(ckpt)

        # Subprocess: never touches `coldddi.baselines.deepddi` directly.
        code = (
            "from coldddi.baselines import load_baseline\n"
            f"m = load_baseline(r'{ckpt}')\n"
            "print(type(m).__name__)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"clean-process load failed:\n--- stderr ---\n{result.stderr}"
        )
        assert "DeepDDIBaseline" in result.stdout

    def test_run_evaluation_in_fresh_subprocess(self, tmp_path):
        """Evaluate through lazy registration in a fresh process.

        Apply Tiny after ``ensure_imported`` to avoid registering DeepDDI
        through a direct import.
        """
        import subprocess

        code = f"""
from pathlib import Path
from coldddi.baselines.base import ensure_imported, _REGISTRY

# This is the call C1 must make on its own — emulating run_evaluation()'s
# first line. Without it, _REGISTRY['deepddi'] would not exist.
ensure_imported('deepddi')
assert 'deepddi' in _REGISTRY, "lazy-import did not register deepddi"

orig_cls = _REGISTRY['deepddi']
class Tiny(orig_cls):
    def __init__(self):
        super().__init__(ssp_dim=4, hidden_dim=8, n_layers=2, n_epochs=1, batch_size=64, device='cpu')
_REGISTRY['deepddi'] = Tiny

from coldddi.evaluate import run_evaluation
run_evaluation(
    method='deepddi',
    data_dir=Path(r'{REPO_ROOT / "data" / "public" / "intermediate"}'),
    seed=42,
    settings=['S2'],
    out_dir=Path(r'{tmp_path / "out"}'),
)
print('OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"clean-process run_evaluation failed:\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        assert "OK" in result.stdout
        assert (tmp_path / "out" / "metrics_seed42.json").is_file()
        # Paper A.6.2 contract: --out also writes per-pair predictions
        # CSVs (one per evaluated split).  See test_evaluate_predictions_csv.
        assert (tmp_path / "out" / "predictions_test_s2_seed42.csv").is_file()
        assert (tmp_path / "out" / "predictions_val_s2_seed42.csv").is_file()
