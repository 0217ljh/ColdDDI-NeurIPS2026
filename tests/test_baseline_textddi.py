"""Test TextDDI on toy data with the tiny random SMOKE_BACKBONE.

Check the paper preset's ``roberta-base`` configuration separately.
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
def trained_textddi():
    pytest.importorskip("transformers")
    from coldddi.baselines.textddi import TextDDIBaseline
    from coldddi.baselines.textddi.baseline import SMOKE_BACKBONE
    from coldddi.data.dataset import PairDataset

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    model = TextDDIBaseline(
        backbone=SMOKE_BACKBONE,   # avoid downloading the paper backbone
        max_length=64,
        n_epochs=1,
        batch_size=8,
        device="cpu",
    )
    # Skip download/cache failures only; implementation errors must still fail.
    try:
        model._ensure_backbone()
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"Could not download text backbone: {exc}")
    model.fit(ds)
    return model, ds


class TestTextDDIFit:
    def test_fit_populates_state(self, trained_textddi):
        model, _ = trained_textddi
        assert model._model is not None
        assert model._tokenizer is not None
        # The cache stores (name, description) per drug.
        assert model._desc_cache is not None and len(model._desc_cache) > 0
        sample_value = next(iter(model._desc_cache.values()))
        assert isinstance(sample_value, tuple) and len(sample_value) == 2

    def test_predict_proba_in_unit_interval(self, trained_textddi):
        model, ds = trained_textddi
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(10)
        neg = ds.get_negatives("test_s2").head(10)
        scores = model.predict_proba(pd.concat([pos, neg], ignore_index=True))
        assert scores.shape == (20,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_before_fit_raises(self):
        from coldddi.baselines.textddi import TextDDIBaseline

        m = TextDDIBaseline(n_epochs=1)
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict_proba(
                pd.DataFrame({"drug_a_id": ["DB1"], "drug_b_id": ["DB2"]})
            )

    def test_unknown_drug_returns_default_score(self, trained_textddi):
        model, _ = trained_textddi
        scores = model.predict_proba(
            pd.DataFrame(
                {"drug_a_id": ["DBNOSUCH"], "drug_b_id": ["DBALSONO"]}
            )
        )
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.5)

    def test_prompt_uses_paper_query_template(self, trained_textddi):
        """Preserve the upstream query template from train_custom_bundle.py:417-419."""
        model, ds = trained_textddi
        pair = ds.splits.test_s2.iloc[0]
        a, b = str(pair["drug_a_id"]), str(pair["drug_b_id"])
        prompt = model._build_prompt(a, b)
        assert prompt is not None
        assert "In the above context, we can predict that the drug-drug interactions between" in prompt
        assert "is that:" in prompt

    def test_modern_kb_fallback_uses_kg_name_dict(self, trained_textddi):
        """Release-dir descriptions use kg.name_dict's singular keys, not names alone."""
        model, ds = trained_textddi
        # Toy fixture has a populated KG. At least some drugs in the
        # cache should have non-empty descriptions (from kg.name_dict).
        assert model._desc_cache is not None
        n_with_desc = sum(
            1 for _, desc in model._desc_cache.values() if desc
        )
        assert n_with_desc > 0, (
            "kb fallback produced 0 non-empty descriptions; the "
            "singular/plural mismatch on kg.name_dict() has regressed."
        )


class TestTextDDISaveLoad:
    def test_save_writes_required_files(self, trained_textddi, tmp_path):
        model, _ = trained_textddi
        model.save(tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "manifest.json").is_file()
        # Persist the (name, description) cache.
        assert (tmp_path / "ckpt" / "desc_cache.json").is_file()
        assert (tmp_path / "ckpt" / "model").is_dir()
        assert (tmp_path / "ckpt" / "tokenizer").is_dir()

    def test_load_baseline_dispatcher_routes_to_textddi(self, trained_textddi, tmp_path):
        from coldddi.baselines import load_baseline
        from coldddi.baselines.textddi import TextDDIBaseline

        model, _ = trained_textddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        assert isinstance(loaded, TextDDIBaseline)

    def test_load_round_trip_preserves_predictions(self, trained_textddi, tmp_path):
        from coldddi.baselines import load_baseline

        model, ds = trained_textddi
        model.save(tmp_path / "ckpt")
        loaded = load_baseline(tmp_path / "ckpt")
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].head(10)
        np.testing.assert_allclose(
            model.predict_proba(pairs),
            loaded.predict_proba(pairs),
            rtol=1e-4,
            atol=1e-4,
        )


class TestTextDDIRegistry:
    def test_textddi_is_registered_after_import(self):
        import coldddi.baselines.textddi  # noqa: F401
        from coldddi.baselines import list_baselines

        assert "textddi" in list_baselines()

    def test_textddi_in_name_to_module(self):
        from coldddi.baselines import NAME_TO_MODULE

        assert NAME_TO_MODULE.get("textddi") == "coldddi.baselines.textddi"


class TestDefaultBackboneIsPaperSpec:
    """Distinguish the paper CLI backbone from lightweight constructor defaults."""

    def test_default_backbone_constant_is_roberta_base(self):
        from coldddi.baselines.textddi.baseline import DEFAULT_BACKBONE

        assert DEFAULT_BACKBONE == "roberta-base", (
            f"DEFAULT_BACKBONE drifted to {DEFAULT_BACKBONE!r}; paper "
            "Appendix C.1 Table requires 'roberta-base' as the "
            "released script default."
        )

    def test_smoke_backbone_constant_exposed(self):
        """Expose the smoke backbone without duplicating its model ID."""
        from coldddi.baselines.textddi.baseline import SMOKE_BACKBONE

        assert SMOKE_BACKBONE == (
            "hf-internal-testing/tiny-random-DistilBertModel"
        )
        assert SMOKE_BACKBONE != "roberta-base", (
            "SMOKE_BACKBONE must be different from the paper default; "
            "otherwise tests opting into smoke mode would silently "
            "download the real RoBERTa-base."
        )

    def test_instance_default_backbone_is_smoke_stub(self):
        """Direct construction uses the smoke stub; the paper CLI preset overrides it."""
        from coldddi.baselines.textddi import TextDDIBaseline
        from coldddi.baselines.textddi.baseline import SMOKE_BACKBONE

        m = TextDDIBaseline()
        assert m.backbone == SMOKE_BACKBONE

    def test_paper_hyperparams_backbone_is_roberta_base(self):
        """The paper preset forwards roberta-base, as specified in Appendix C.1."""
        from coldddi.baselines.textddi import PAPER_HYPERPARAMS

        assert PAPER_HYPERPARAMS["backbone"] == "roberta-base"

    def test_constants_are_re_exported_from_package_root(self):
        """Both backbone constants are exported from coldddi.baselines.textddi."""
        from coldddi.baselines.textddi import DEFAULT_BACKBONE, SMOKE_BACKBONE

        assert DEFAULT_BACKBONE == "roberta-base"
        assert SMOKE_BACKBONE.endswith("tiny-random-DistilBertModel")
