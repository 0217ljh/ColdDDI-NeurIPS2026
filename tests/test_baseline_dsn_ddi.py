"""End-to-end smoke test: DSN-DDI on the toy fixture."""

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
def trained_dsn_ddi():
    pytest.importorskip("rdkit")
    pytest.importorskip("torch_geometric")
    from coldddi.baselines.dsn_ddi import DSNDDIBaseline
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    # Intra/InterGraphAttention each output 64 dims; their concat requires
    # SAGPooling's n_heads * head_out_feats == 128.
    model = DSNDDIBaseline(
        hidd_dim=64,
        kge_dim=128,
        heads_out_feat_params=(64, 64),
        blocks_params=(2, 2),
        n_epochs=1,
        batch_size=16,
        device="cpu",
    )
    model.fit(ds)
    return model, ds


class TestDSNDDIFit:
    def test_fit_populates_state(self, trained_dsn_ddi):
        model, _ = trained_dsn_ddi
        assert model._model is not None
        assert model._graphs is not None and len(model._graphs) > 0

    def test_predict_proba_in_unit_interval(self, trained_dsn_ddi):
        model, ds = trained_dsn_ddi
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        neg = ds.get_negatives("test_s2").head(20)
        scores = model.predict_proba(pd.concat([pos, neg], ignore_index=True))
        assert scores.shape == (40,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_before_fit_raises(self):
        from coldddi.baselines.dsn_ddi import DSNDDIBaseline

        m = DSNDDIBaseline(
            hidd_dim=64, kge_dim=128,
            heads_out_feat_params=(64,), blocks_params=(2,),
            n_epochs=1,
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict_proba(
                pd.DataFrame({"drug_a_id": ["DB1"], "drug_b_id": ["DB2"]})
            )

    def test_unknown_drug_returns_default_score(self, trained_dsn_ddi):
        model, _ = trained_dsn_ddi
        scores = model.predict_proba(
            pd.DataFrame(
                {"drug_a_id": ["DBNOSUCH"], "drug_b_id": ["DBALSONO"]}
            )
        )
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.5)


class TestDSNDDISaveLoad:
    def test_save_writes_required_files(self, trained_dsn_ddi, tmp_path):
        model, _ = trained_dsn_ddi
        model.save(tmp_path / "ckpt")
        for fname in ("model.pt", "graphs.pkl", "manifest.json"):
            assert (tmp_path / "ckpt" / fname).is_file(), f"missing {fname}"

    def test_load_baseline_dispatcher_routes_to_dsn_ddi(self, trained_dsn_ddi, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.dsn_ddi import DSNDDIBaseline

        model, _ = trained_dsn_ddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, DSNDDIBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_dsn_ddi, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_dsn_ddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        np.testing.assert_allclose(
            model.predict_proba(pairs),
            loaded.predict_proba(pairs),
            rtol=1e-5,
            atol=1e-5,
        )


class TestDSNDDIRegistry:
    def test_dsn_ddi_is_registered_after_import(self):
        import coldddi.baselines.dsn_ddi  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "dsn_ddi" in list_baselines()

    def test_dsn_ddi_in_name_to_module(self):
        from coldddi.baselines import NAME_TO_MODULE

        assert NAME_TO_MODULE.get("dsn_ddi") == "coldddi.baselines.dsn_ddi"
