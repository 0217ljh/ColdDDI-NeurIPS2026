"""Test Appendix A.6.2's end-to-end indicator outputs by modality.

KPS-F is universal; channel indicators are populated only for separable mol+KG
baselines. Representative models cover dispatch; dedicated tests cover math.
"""

from __future__ import annotations

import math
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


# Modality registration sanity

class TestModalityRegistration:
    """Each baseline declares its modality so channel dispatch cannot default incorrectly."""

    EXPECTED_MODALITY = {
        "deepddi":  "mol",
        "ssi_ddi":  "mol",
        "dsn_ddi":  "mol",
        "hdn_ddi":  "mol",
        "emergnn":  "mol+kg-fused",
        "textddi":  "text",
        "mkg_fenn": "mol+kg",
        "tiger":    "mol+kg",
    }

    @pytest.mark.parametrize("name,expected", list(EXPECTED_MODALITY.items()))
    def test_modality_attr_matches_table(self, name, expected):
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY

        ensure_imported(name)
        cls = _REGISTRY[name]
        assert getattr(cls, "modality", None) == expected, (
            f"baseline {name!r} declares modality="
            f"{getattr(cls, 'modality', None)!r}, expected {expected!r}"
        )

    def test_register_rejects_unknown_modality(self):
        """Registration rejects unknown modalities, including case mismatches."""
        from coldddi.baselines.base import BaselineModel, register

        with pytest.raises(ValueError, match="modality="):
            @register("__test_bad_modality__")
            class _Bad(BaselineModel):
                modality = "mol+KG"   # case mismatch — not in MODALITIES

                def fit(self, *a, **k): ...
                def predict_proba(self, *a, **k): ...
                def save(self, *a, **k): ...

                @classmethod
                def load(cls, *a, **k): ...


# Modality → mask-channel dispatch table

class TestModalityMaskChannelDispatch:
    def test_only_mol_kg_separable_runs_masks(self):
        from coldddi.evaluate import MODALITY_MASK_CHANNELS

        # Single-modality entries → empty tuple (no mask passes).
        for m in ("mol", "text", "mol+kg-fused"):
            assert MODALITY_MASK_CHANNELS[m] == ()
        # Only the separable label triggers mol + kg mask passes.
        assert MODALITY_MASK_CHANNELS["mol+kg"] == ("mol", "kg")

    def test_dispatch_table_covers_every_modality(self):
        """The mask-channel dispatch table covers every registered modality."""
        from coldddi.baselines.base import MODALITIES
        from coldddi.evaluate import MODALITY_MASK_CHANNELS

        assert set(MODALITY_MASK_CHANNELS) == set(MODALITIES), (
            "MODALITY_MASK_CHANNELS keys drifted from base.MODALITIES; "
            f"extra={set(MODALITY_MASK_CHANNELS) - set(MODALITIES)}, "
            f"missing={set(MODALITIES) - set(MODALITY_MASK_CHANNELS)}"
        )


# Instance-level modality override

class TestInstanceModalityOverride:
    """TIGER's mol_only instance reports mol to prevent unsupported channel-mask calls."""

    def test_tiger_mol_only_instance_downgrades_modality(self):
        pytest.importorskip("rdkit")
        pytest.importorskip("torch_geometric")
        from coldddi.baselines.tiger import TIGERBaseline

        dual = TIGERBaseline()  # default dual-channel
        assert dual.modality == "mol+kg"

        mol_only = TIGERBaseline(mol_only=True)
        assert mol_only.modality == "mol", (
            "mol_only=True instance must report modality='mol' so the "
            "L6 dispatch skips the mask passes that would crash inside "
            "TIGER.predict_proba"
        )

    def test_class_attribute_unchanged_by_instance_override(self):
        """Setting ``self.modality`` on one instance must not leak
        to the class attribute (sanity for the instance/class shadow)."""
        pytest.importorskip("rdkit")
        pytest.importorskip("torch_geometric")
        from coldddi.baselines.tiger import TIGERBaseline

        _ = TIGERBaseline(mol_only=True)
        assert TIGERBaseline.modality == "mol+kg"


# AB-parquet discovery

class TestResolveAbParquet:
    def test_auto_discovery_finds_toy_sample(self):
        """Discover the toy annotations/ab_sample.parquet without --ab-parquet."""
        from coldddi.evaluate import _resolve_ab_parquet

        path = _resolve_ab_parquet(None, TOY_RELEASE)
        assert path is not None
        assert path.name in ("ab.parquet", "ab_sample.parquet")
        assert path.is_file()

    def test_explicit_path_takes_precedence(self, tmp_path):
        from coldddi.evaluate import _resolve_ab_parquet

        # Use the real toy sample as the explicit-path target.
        sample = REPO_ROOT / "annotations" / "ab_sample.parquet"
        if not sample.is_file():
            pytest.skip("toy sample not present")
        assert _resolve_ab_parquet(sample, TOY_RELEASE) == sample

    def test_missing_explicit_raises(self, tmp_path):
        from coldddi.evaluate import _resolve_ab_parquet

        with pytest.raises(FileNotFoundError, match="ab-parquet"):
            _resolve_ab_parquet(tmp_path / "nope.parquet", TOY_RELEASE)

    def test_none_returned_when_nothing_found(self, tmp_path, monkeypatch):
        """Missing AB parquet returns None so training and prediction output can continue."""
        from coldddi import evaluate

        # Monkeypatch __file__ so the repo-root candidate path also
        # lives under tmp_path (where there's no annotations/).
        orig_file = evaluate.__file__
        fake_pkg = tmp_path / "fake_pkg"
        fake_pkg.mkdir()
        (fake_pkg / "evaluate.py").write_text("# stub\n")
        monkeypatch.setattr(evaluate, "__file__", str(fake_pkg / "evaluate.py"))
        try:
            path = evaluate._resolve_ab_parquet(None, tmp_path)
            assert path is None
        finally:
            monkeypatch.setattr(evaluate, "__file__", orig_file)


# E2E: single-modality baseline (DeepDDI = "mol")

class TestRunEvaluationDeepDDIWritesIndicators:
    """Single-modality smoke: KPS-F populated, KPS-mol / KPS-KG NaN."""

    @pytest.fixture(scope="class")
    def out_dir(self, tmp_path_factory):
        pytest.importorskip("torch")
        from coldddi.evaluate import run_evaluation

        out = tmp_path_factory.mktemp("eval_deepddi_l6")
        run_evaluation(
            method="deepddi",
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S2"],
            out_dir=out,
            device="cpu",
            preset="smoke",   # CI: skip paper-spec 100-epoch DeepDDI
        )
        return out

    def test_indicators_csv_written(self, out_dir):
        assert (out_dir / "indicators_test_s2_seed42.csv").is_file()

    def test_no_mask_csvs_for_single_modality(self, out_dir):
        """DeepDDI is mol-only — mask CSVs must NOT be written."""
        assert not (out_dir / "predictions_test_s2_mask_mol_seed42.csv").is_file()
        assert not (out_dir / "predictions_test_s2_mask_kg_seed42.csv").is_file()

    def test_kps_f_rows_populated(self, out_dir):
        df = pd.read_csv(out_dir / "indicators_test_s2_seed42.csv")
        kpsf = df.query("indicator == 'KPS-F'")
        # At least the ALL row should exist and be a finite probability.
        all_row = kpsf.query("bucket == 'ALL'")
        if len(all_row):
            v = float(all_row.iloc[0]["value"])
            assert not math.isnan(v)
            assert 0 <= v <= 1

    def test_channel_indicators_all_nan(self, out_dir):
        """Without channel masks, single-modality KPS-mol and KPS-KG rows are NaN."""
        df = pd.read_csv(out_dir / "indicators_test_s2_seed42.csv")
        for ind in ("KPS-mol", "KPS-KG"):
            sub = df.query(f"indicator == '{ind}'")
            assert len(sub), f"{ind} rows missing"
            assert sub["value"].isna().all(), (
                f"{ind} unexpectedly populated for mol-only baseline"
            )
            assert (sub["n"] == 0).all()


# E2E: mol+KG baseline (MKG-FENN) — three indicators populated

class TestRunEvaluationMKGFENNWritesIndicators:
    @pytest.fixture(scope="class")
    def out_dir(self, tmp_path_factory):
        pytest.importorskip("torch")
        pytest.importorskip("rdkit")
        from coldddi.evaluate import run_evaluation

        out = tmp_path_factory.mktemp("eval_mkgfenn_l6")
        # Test finite output and artifacts with a tiny model, not prediction quality.
        from coldddi.baselines.base import _REGISTRY
        from coldddi.baselines import ensure_imported

        ensure_imported("mkg_fenn")
        OrigCls = _REGISTRY["mkg_fenn"]

        class Tiny(OrigCls):
            def __init__(self, **kw):
                super().__init__(
                    embedding_num=8,
                    neighbor_sample_size=4,
                    n_epochs=1,
                    batch_size=64,
                    fp_nbits=64,
                    n_bins=4,
                    **kw,
                )

        _REGISTRY["mkg_fenn"] = Tiny
        try:
            run_evaluation(
                method="mkg_fenn",
                data_dir=TOY_RELEASE,
                seed=42,
                settings=["S2"],
                out_dir=out,
                device="cpu",
                preset="smoke",
            )
        finally:
            _REGISTRY["mkg_fenn"] = OrigCls
        return out

    def test_indicators_csv_written(self, out_dir):
        assert (out_dir / "indicators_test_s2_seed42.csv").is_file()

    def test_mask_csvs_written_for_mol_and_kg(self, out_dir):
        """mol+KG modality → mask predictions CSVs must exist so a user
        can re-run L6 directly without re-training."""
        assert (out_dir / "predictions_test_s2_mask_mol_seed42.csv").is_file()
        assert (out_dir / "predictions_test_s2_mask_kg_seed42.csv").is_file()

    def test_mask_csvs_share_canonical_schema(self, out_dir):
        from coldddi.evaluate import PREDICTION_COLUMNS

        for fname in (
            "predictions_test_s2_mask_mol_seed42.csv",
            "predictions_test_s2_mask_kg_seed42.csv",
        ):
            df = pd.read_csv(out_dir / fname)
            assert list(df.columns) == list(PREDICTION_COLUMNS), (
                f"{fname} column order drifted from PREDICTION_COLUMNS"
            )

    def test_all_three_indicators_populated(self, out_dir):
        """KPS-F + KPS-mol + KPS-KG must each have at least one
        non-NaN value (the ``ALL`` bucket, populated from positives,
        is essentially guaranteed on the toy fixture)."""
        df = pd.read_csv(out_dir / "indicators_test_s2_seed42.csv")
        for ind in ("KPS-F", "KPS-mol", "KPS-KG"):
            sub = df.query(f"indicator == '{ind}'")
            assert len(sub), f"{ind} missing from indicators table"
            non_nan = sub["value"].dropna()
            assert len(non_nan) > 0, (
                f"{ind} produced zero non-NaN rows on mol+KG baseline "
                "— mask_channel path is broken or mask preds didn't match swap"
            )


# E2E: text baseline (TextDDI) — KPS-F populated, channels NaN
# Optional text-modality coverage; skip if the backbone is unavailable.

class TestRunEvaluationTextDDIWritesIndicators:
    @pytest.fixture(scope="class")
    def out_dir(self, tmp_path_factory):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from coldddi.evaluate import run_evaluation
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY

        ensure_imported("textddi")
        from coldddi.baselines.textddi.baseline import SMOKE_BACKBONE

        OrigCls = _REGISTRY["textddi"]

        class Tiny(OrigCls):
            def __init__(self, **kw):
                # Explicit smoke stub — paper default 'roberta-base'
                # would download ~500MB on every test session.
                super().__init__(
                    backbone=SMOKE_BACKBONE,
                    max_length=32, n_epochs=1, batch_size=4, **kw,
                )

        _REGISTRY["textddi"] = Tiny
        out = tmp_path_factory.mktemp("eval_textddi_l6")
        try:
            try:
                run_evaluation(
                    method="textddi",
                    data_dir=TOY_RELEASE,
                    seed=42,
                    settings=["S2"],
                    out_dir=out,
                    device="cpu",
                    preset="smoke",
                )
            except (OSError, ConnectionError) as exc:
                pytest.skip(f"could not download text backbone: {exc}")
        finally:
            _REGISTRY["textddi"] = OrigCls
        return out

    def test_indicators_csv_written(self, out_dir):
        assert (out_dir / "indicators_test_s2_seed42.csv").is_file()

    def test_no_mask_csvs_for_text_modality(self, out_dir):
        assert not (out_dir / "predictions_test_s2_mask_mol_seed42.csv").is_file()
        assert not (out_dir / "predictions_test_s2_mask_kg_seed42.csv").is_file()

    def test_channel_indicators_all_nan(self, out_dir):
        df = pd.read_csv(out_dir / "indicators_test_s2_seed42.csv")
        for ind in ("KPS-mol", "KPS-KG"):
            sub = df.query(f"indicator == '{ind}'")
            assert sub["value"].isna().all()


# --no-indicators flag respected

class TestNoIndicatorsFlagSkipsL6:
    def test_with_indicators_false_skips_csv(self, tmp_path):
        pytest.importorskip("torch")
        from coldddi.evaluate import run_evaluation

        run_evaluation(
            method="deepddi",
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S2"],
            out_dir=tmp_path,
            device="cpu",
            with_indicators=False,
            preset="smoke",   # CI: skip paper-spec DeepDDI training
        )
        # Disabling indicators must not suppress predictions.
        assert (tmp_path / "predictions_test_s2_seed42.csv").is_file()
        # L6 CSV intentionally absent.
        assert not (tmp_path / "indicators_test_s2_seed42.csv").is_file()


# L6 silently skipped for S0/S1-only runs

class TestSkipIndicatorsWhenNoS2Setting:
    def test_s0_s1_only_skips_indicators(self, tmp_path):
        """Skip L6 without S2, since these indicators require the S2 anchor set."""
        pytest.importorskip("torch")
        from coldddi.evaluate import run_evaluation

        run_evaluation(
            method="deepddi",
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S0", "S1"],
            out_dir=tmp_path,
            device="cpu",
            preset="smoke",   # CI: skip paper-spec DeepDDI training
        )
        # S0/S1 prediction CSVs land.
        assert (tmp_path / "predictions_test_s0_seed42.csv").is_file()
        assert (tmp_path / "predictions_test_s1_seed42.csv").is_file()
        # No L6 output for an S0/S1-only run.
        assert not (tmp_path / "indicators_test_s2_seed42.csv").is_file()
