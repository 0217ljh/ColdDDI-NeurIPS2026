"""End-to-end smoke test: TIGER (dual-channel default + mol-only opt-in) on the toy fixture."""

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
def trained_tiger():
    pytest.importorskip("rdkit")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("networkx")
    from coldddi.baselines.tiger import TIGERBaseline
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    model = TIGERBaseline(
        max_layer=2,
        output_dim=32,
        n_epochs=1,
        batch_size=16,
        device="cpu",
    )
    model.fit(ds)
    return model, ds


class TestTIGERFit:
    def test_fit_populates_state(self, trained_tiger):
        model, _ = trained_tiger
        assert model._model is not None
        assert model._mol_graphs is not None and len(model._mol_graphs) > 0

    def test_default_is_dual_channel(self, trained_tiger):
        """The paper configuration enables both mol and KG channels by default."""
        model, _ = trained_tiger
        assert model.mol_only is False
        assert model._model.mol_only is False
        # KG-channel modules exist on the model.
        assert hasattr(model._model, "drug_node_feature")
        assert hasattr(model._model, "node_representation_learning")
        assert hasattr(model._model, "cold_start_proj")
        # BKG random-walk subgraphs were generated for every drug.
        assert model._subgraphs is not None
        assert len(model._subgraphs) == len(model._drug_to_idx)
        # Cold-start partition populated (toy fixture has g2 drugs).
        assert len(model._g2_idx) > 0

    def test_mol_only_opt_in(self):
        """The inductive mol_only variant still fits and predicts."""
        pytest.importorskip("rdkit")
        pytest.importorskip("torch_geometric")
        from coldddi.baselines.tiger import TIGERBaseline
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        m = TIGERBaseline(
            mol_only=True,
            max_layer=2, output_dim=16, n_epochs=1, batch_size=16,
            device="cpu",
        )
        m.fit(ds)
        assert m._model.mol_only is True
        assert m._subgraphs is None  # no BKG / subgraphs built in mol_only
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(8)
        scores = m.predict_proba(pos)
        assert scores.shape == (8,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_proba_in_unit_interval(self, trained_tiger):
        model, ds = trained_tiger
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        neg = ds.get_negatives("test_s2").head(20)
        scores = model.predict_proba(pd.concat([pos, neg], ignore_index=True))
        assert scores.shape == (40,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_before_fit_raises(self):
        from coldddi.baselines.tiger import TIGERBaseline

        m = TIGERBaseline(max_layer=2, output_dim=8, n_epochs=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict_proba(
                pd.DataFrame({"drug_a_id": ["DB1"], "drug_b_id": ["DB2"]})
            )

    def test_unknown_drug_returns_default_score(self, trained_tiger):
        model, _ = trained_tiger
        scores = model.predict_proba(
            pd.DataFrame(
                {"drug_a_id": ["DBNOSUCH"], "drug_b_id": ["DBALSONO"]}
            )
        )
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.5)

    def test_channel_mask_outputs_differ_from_base(self, trained_tiger):
        """Zeroing either channel must change predictions.

        Matches upstream
        ``Code-Released/exps/sec5-3/2_indicators/baseline_mask_predictors/_tiger_runner_mask.py``
        which zeros ``mol{1,2}_emb`` (mol mask) or
        ``drug{1,2}_node_emb`` (kg mask) before the ``fc1`` fusion.
        """
        model, ds = trained_tiger
        pairs = pd.concat(
            [
                ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20),
                ds.get_negatives("test_s2").head(20),
            ],
            ignore_index=True,
        )
        base = model.predict_proba(pairs)
        mask_kg = model.predict_proba(pairs, mask_channel="kg")
        mask_mol = model.predict_proba(pairs, mask_channel="mol")
        for arr in (base, mask_kg, mask_mol):
            assert arr.shape == base.shape
            assert (arr >= 0).all() and (arr <= 1).all()
        assert not np.allclose(base, mask_kg), (
            "mask_channel='kg' produced identical scores to base — "
            "KG GraphTransformer output not contributing or mask not wired."
        )
        assert not np.allclose(base, mask_mol), (
            "mask_channel='mol' produced identical scores to base — "
            "mol GraphTransformer output not contributing or mask not wired."
        )
        assert not np.allclose(mask_kg, mask_mol)

    def test_invalid_mask_channel_raises(self, trained_tiger):
        model, ds = trained_tiger
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(4)
        with pytest.raises(ValueError, match="mask_channel must be"):
            model.predict_proba(pos, mask_channel="kgmol")

    def test_mask_channel_disallowed_in_mol_only(self):
        """Reject channel masks in mol_only mode, which has no KG branch."""
        pytest.importorskip("rdkit")
        pytest.importorskip("torch_geometric")
        from coldddi.baselines.tiger import TIGERBaseline
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        m = TIGERBaseline(
            mol_only=True,
            max_layer=2, output_dim=16, n_epochs=1, batch_size=16,
            device="cpu",
        )
        m.fit(ds)
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(4)
        with pytest.raises(ValueError, match="mol_only=True"):
            m.predict_proba(pos, mask_channel="kg")
        with pytest.raises(ValueError, match="mol_only=True"):
            m.predict_proba(pos, mask_channel="mol")


class TestTIGERSaveLoad:
    def test_save_writes_required_files(self, trained_tiger, tmp_path):
        model, _ = trained_tiger
        model.save(tmp_path / "ckpt")
        # artefacts.pkl stores molecular graphs, drug indices, and KG subgraphs.
        for fname in ("model.pt", "artefacts.pkl", "manifest.json"):
            assert (tmp_path / "ckpt" / fname).is_file(), f"missing {fname}"

    def test_load_baseline_dispatcher_routes_to_tiger(self, trained_tiger, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.tiger import TIGERBaseline

        model, _ = trained_tiger
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, TIGERBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_tiger, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_tiger
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        np.testing.assert_allclose(
            model.predict_proba(pairs),
            loaded.predict_proba(pairs),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_load_round_trip_preserves_mask_channel_predictions(
        self, trained_tiger, tmp_path,
    ):
        """Save/load preserves channel-mask predictions used by indicator jobs."""
        from coldddi.baselines import load_baseline

        model, ds = trained_tiger
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        for ch in ("mol", "kg"):
            src = model.predict_proba(pairs, mask_channel=ch)
            dst = loaded.predict_proba(pairs, mask_channel=ch)
            assert (dst >= 0).all() and (dst <= 1).all(), (
                f"mask_channel={ch!r} produced out-of-[0,1] scores after load"
            )
            np.testing.assert_allclose(src, dst, rtol=1e-5, atol=1e-5)


class TestTIGERRegistry:
    def test_tiger_is_registered_after_import(self):
        import coldddi.baselines.tiger  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "tiger" in list_baselines()

    def test_tiger_in_name_to_module(self):
        from coldddi.baselines import NAME_TO_MODULE

        assert NAME_TO_MODULE.get("tiger") == "coldddi.baselines.tiger"
