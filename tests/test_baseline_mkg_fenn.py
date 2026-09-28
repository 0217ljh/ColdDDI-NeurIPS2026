"""End-to-end smoke test: MKG-FENN on the toy fixture."""

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
def trained_mkg_fenn():
    pytest.importorskip("rdkit")
    from coldddi.baselines.mkg_fenn import MKGFENNBaseline
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    model = MKGFENNBaseline(
        embedding_num=16,
        neighbor_sample_size=4,
        n_epochs=1,
        batch_size=64,
        fp_nbits=64,
        n_bins=4,
        device="cpu",
    )
    model.fit(ds)
    return model, ds


class TestMKGFENNFit:
    def test_fit_populates_state(self, trained_mkg_fenn):
        model, _ = trained_mkg_fenn
        assert model._model is not None
        assert model._dict1 is not None and len(model._dict1) > 0

    def test_predict_proba_in_unit_interval(self, trained_mkg_fenn):
        model, ds = trained_mkg_fenn
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        neg = ds.get_negatives("test_s2").head(20)
        scores = model.predict_proba(pd.concat([pos, neg], ignore_index=True))
        assert scores.shape == (40,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_before_fit_raises(self):
        from coldddi.baselines.mkg_fenn import MKGFENNBaseline

        m = MKGFENNBaseline(embedding_num=8, n_epochs=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict_proba(
                pd.DataFrame({"drug_a_id": ["DB1"], "drug_b_id": ["DB2"]})
            )

    def test_unknown_drug_returns_default_score(self, trained_mkg_fenn):
        model, _ = trained_mkg_fenn
        scores = model.predict_proba(
            pd.DataFrame(
                {"drug_a_id": ["DBNOSUCH"], "drug_b_id": ["DBALSONO"]}
            )
        )
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.5)

    def test_channel_mask_outputs_differ_from_base(self, trained_mkg_fenn):
        """Paper-spec KPS-mol / KPS-KG ablation: zero a channel and
        verify the prediction changes.  Equal scores would mean the
        mask either doesn't take effect or the model never learned to
        use that channel — both red flags for the indicator."""
        model, ds = trained_mkg_fenn
        # Use enough pairs that random equality is implausible.
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
        # All three are valid probabilities.
        for arr in (base, mask_kg, mask_mol):
            assert arr.shape == base.shape
            assert (arr >= 0).all() and (arr <= 1).all()
        # The masks must actually change predictions (≥ 1 pair differs).
        assert not np.allclose(base, mask_kg), (
            "mask_channel='kg' produced identical scores to base — "
            "GNN1+GNN3 channels not contributing or mask not wired."
        )
        assert not np.allclose(base, mask_mol), (
            "mask_channel='mol' produced identical scores to base — "
            "GNN2+GNN4 channels not contributing or mask not wired."
        )
        # The two masks should produce DIFFERENT outputs (otherwise
        # we'd be ablating the same thing twice).
        assert not np.allclose(mask_kg, mask_mol)

    def test_invalid_mask_channel_raises(self, trained_mkg_fenn):
        model, ds = trained_mkg_fenn
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(4)
        with pytest.raises(ValueError, match="mask_channel must be"):
            model.predict_proba(pos, mask_channel="kgmol")

    def test_fusion_mask_byte_exact_with_manual_zero(self):
        """Deterministic guard against the per-pair mask implementation
        diverging from "manually zero the rows then concatenate"."""
        pytest.importorskip("torch")
        import torch

        from coldddi.baselines.mkg_fenn.model import FusionLayer

        torch.manual_seed(0)
        emb_dim = 8
        n_drugs = 5
        # Random per-drug embedding tables, one per GNN channel.
        g1 = torch.randn(n_drugs, emb_dim)
        g2 = torch.randn(n_drugs, emb_dim)
        g3 = torch.randn(n_drugs, emb_dim)
        g4 = torch.randn(n_drugs, emb_dim)
        idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        fusion = FusionLayer(emb_dim, dropout=0.0)
        fusion.eval()

        with torch.no_grad():
            base = fusion((g4, g3, g2, g1, idx))
            masked_kg = fusion((g4, g3, g2, g1, idx), mask_channel="kg")
            masked_mol = fusion((g4, g3, g2, g1, idx), mask_channel="mol")

            # Reference: manually zero the rows of the masked pair
            # in the input tables before fusion.
            g1z = g1.clone(); g3z = g3.clone()
            for p in idx.flatten().tolist():
                g1z[p] = 0
                g3z[p] = 0
            ref_kg = fusion((g4, g3z, g2, g1z, idx))

            g2z = g2.clone(); g4z = g4.clone()
            for p in idx.flatten().tolist():
                g2z[p] = 0
                g4z[p] = 0
            ref_mol = fusion((g4z, g3, g2z, g1, idx))

        assert torch.allclose(masked_kg, ref_kg, atol=1e-6), (
            "mask_channel='kg' did not match manual zero-row reference"
        )
        assert torch.allclose(masked_mol, ref_mol, atol=1e-6), (
            "mask_channel='mol' did not match manual zero-row reference"
        )
        # Base must differ from masks (random embeddings → non-trivial).
        assert not torch.allclose(base, masked_kg)
        assert not torch.allclose(base, masked_mol)


class TestMKGFENNSaveLoad:
    def test_save_writes_required_files(self, trained_mkg_fenn, tmp_path):
        model, _ = trained_mkg_fenn
        model.save(tmp_path / "ckpt")
        for fname in ("model.pt", "kg_state.pkl", "manifest.json"):
            assert (tmp_path / "ckpt" / fname).is_file(), f"missing {fname}"

    def test_load_baseline_dispatcher_routes_to_mkg_fenn(self, trained_mkg_fenn, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.mkg_fenn import MKGFENNBaseline

        model, _ = trained_mkg_fenn
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, MKGFENNBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_mkg_fenn, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_mkg_fenn
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        # MKG-FENN samples neighbours stochastically per `precompute_adj`;
        # to make the round-trip deterministic, sync the loaded model's
        # adjacency buffers with the source model's. After that, predictions
        # should be bit-exact (the rest of the model is pure forward pass).
        loaded._model.gnn1.adj_tail.copy_(model._model.gnn1.adj_tail)
        loaded._model.gnn1.adj_relation.copy_(model._model.gnn1.adj_relation)
        loaded._model.gnn2.adj_tail.copy_(model._model.gnn2.adj_tail)
        loaded._model.gnn2.adj_relation.copy_(model._model.gnn2.adj_relation)
        loaded._model.gnn3.adj_tail.copy_(model._model.gnn3.adj_tail)
        loaded._model.gnn3.adj_relation.copy_(model._model.gnn3.adj_relation)
        loaded._model.gnn4.adj_tail.copy_(model._model.gnn4.adj_tail)
        loaded._model.gnn4.adj_relation.copy_(model._model.gnn4.adj_relation)
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(20)
        np.testing.assert_allclose(
            model.predict_proba(pairs),
            loaded.predict_proba(pairs),
            rtol=1e-5,
            atol=1e-5,
        )


class TestMKGFENNRegistry:
    def test_mkg_fenn_is_registered_after_import(self):
        import coldddi.baselines.mkg_fenn  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "mkg_fenn" in list_baselines()

    def test_mkg_fenn_in_name_to_module(self):
        from coldddi.baselines import NAME_TO_MODULE

        assert NAME_TO_MODULE.get("mkg_fenn") == "coldddi.baselines.mkg_fenn"
