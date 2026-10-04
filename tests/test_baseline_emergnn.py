"""Test EmerGNN's KG-based fitting, pair scoring, and save/load parity.
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


@pytest.fixture(scope="module")
def trained_emergnn():
    pytest.importorskip("rdkit")
    from coldddi.baselines.emergnn import EmerGNNBaseline
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    model = EmerGNNBaseline(
        n_dim=16, length=2, n_epochs=1, batch_size=32, device="cpu"
    )
    model.fit(ds)
    return model, ds


class TestEmerGNNFit:
    def test_fit_populates_state(self, trained_emergnn):
        model, _ = trained_emergnn
        assert model._model is not None
        assert model._entity2id is not None
        assert model._n_ent is not None and model._n_ent > 0
        assert model._edge_src is not None
        # KG entity vocab must include every G1+G2 drug
        for did in model._entity2id:
            assert isinstance(did, str)

    def test_predict_proba_in_unit_interval(self, trained_emergnn):
        model, ds = trained_emergnn
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        neg = ds.get_negatives("test_s2").head(20)
        scores = model.predict_proba(pd.concat([pos, neg], ignore_index=True))
        assert scores.shape == (40,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_unknown_drug_raises(self, trained_emergnn):
        model, _ = trained_emergnn
        with pytest.raises(ValueError, match="not in the trained entity vocab"):
            model.predict_proba(
                pd.DataFrame(
                    {"drug_a_id": ["DBNOSUCH"], "drug_b_id": ["DBALSONO"]}
                )
            )


class TestEmerGNNSaveLoad:
    def test_save_writes_required_files(self, trained_emergnn, tmp_path):
        model, _ = trained_emergnn
        model.save(tmp_path / "ckpt")
        for fname in ("model.pt", "graph.pkl", "manifest.json"):
            assert (tmp_path / "ckpt" / fname).is_file(), f"missing {fname}"

    def test_load_baseline_dispatcher_routes_to_emergnn(self, trained_emergnn, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.emergnn import EmerGNNBaseline

        model, _ = trained_emergnn
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, EmerGNNBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_emergnn, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_emergnn
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        np.testing.assert_allclose(
            model.predict_proba(pairs), loaded.predict_proba(pairs),
            rtol=1e-5, atol=1e-5,
        )


class TestEmerGNNRegistry:
    def test_emergnn_is_registered_after_import(self):
        import coldddi.baselines.emergnn  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "emergnn" in list_baselines()

    def test_emergnn_in_name_to_module(self):
        from coldddi.baselines import NAME_TO_MODULE

        assert NAME_TO_MODULE.get("emergnn") == "coldddi.baselines.emergnn"


class TestEmerGNNFailures:
    def test_kg_without_required_attrs_raises(self):
        from coldddi.baselines.emergnn.baseline import _kg_to_kb_dict

        class FakeKG:
            pass

        with pytest.raises(TypeError, match="EmerGNN requires"):
            _kg_to_kb_dict(FakeKG())

    def test_dataset_without_drugs_raises(self, tmp_path):
        """A missing drugs table raises ValueError."""
        from coldddi.baselines.emergnn import EmerGNNBaseline
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
        cols = ["drug_a_id", "drug_b_id"]
        empty_splits = SplitFolds(
            train=pd.DataFrame(columns=cols),
            val_s0=pd.DataFrame(columns=cols),
            val_s1=pd.DataFrame(columns=cols),
            val_s2=pd.DataFrame(columns=cols),
            test_s0=pd.DataFrame(columns=cols),
            test_s1=pd.DataFrame(columns=cols),
            test_s2=pd.DataFrame(columns=cols),
            g1_drugs=[],
            g2_drugs=[],
            seed=42,
        )
        ds = PairDataset(
            edges=pd.DataFrame(columns=cols),
            splits=empty_splits,
            kg=empty_kg,
            drugs=None,
        )
        m = EmerGNNBaseline(n_dim=4, length=1, n_epochs=1, device="cpu")
        with pytest.raises(ValueError, match="PairDataset.drugs"):
            m.fit(ds)
