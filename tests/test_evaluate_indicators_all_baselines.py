"""Verify evaluate.py indicators and CSVs across seven torch baselines.

Check modality-specific outputs, shared swap counts, exact union key sets,
and KPS values recomputed from CSVs within 1e-6. ALL uses positives only;
primary buckets include both labels. Channel indicators deduplicate anchors.

TextDDI downloads are covered separately in test_evaluate_indicators_e2e.py.
Per-baseline MKG-FENN/TIGER tests check that masks change predictions: tiny
end-to-end fixtures can yield constant scores, especially TIGER's sparse KG.
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


# Modality table (single source of truth for parametrize)

ALL_BASELINES: list[tuple[str, str]] = [
    ("deepddi",  "mol"),
    ("ssi_ddi",  "mol"),
    ("dsn_ddi",  "mol"),
    ("hdn_ddi",  "mol"),
    ("emergnn",  "mol+kg-fused"),
    ("textddi",  "text"),
    ("mkg_fenn", "mol+kg"),
    ("tiger",    "mol+kg"),
]

#: TextDDI's download-dependent single-modality path is covered in the e2e file.
PARAMETRIZED_BASELINES = [
    (m, mod) for m, mod in ALL_BASELINES if m != "textddi"
]

#: All baselines whose ``modality`` triggers the mol+kg mask passes.
MOL_KG_BASELINES = [
    (m, mod) for m, mod in PARAMETRIZED_BASELINES if mod == "mol+kg"
]


# Tiny-hyperparam kwargs per baseline


def _make_tiny_kwargs(method: str) -> dict:
    """Use small configs that respect each architecture's dimensions.

    SSI-DDI needs head_out_feats * n_heads == kge_dim per block.
    DSN/HDN concatenate two fixed 64-dim attention outputs into 128 dims.
    """
    if method == "deepddi":
        return dict(ssp_dim=4, hidden_dim=8, n_layers=2, n_epochs=1, batch_size=64)
    if method == "ssi_ddi":
        return dict(
            hidd_dim=16, kge_dim=16,
            heads_out_feat_params=(8, 8), blocks_params=(2, 2),
            n_epochs=1, batch_size=32,
        )
    if method == "dsn_ddi":
        return dict(
            hidd_dim=64, kge_dim=128,
            heads_out_feat_params=(64, 64), blocks_params=(2, 2),
            n_epochs=1, batch_size=32,
        )
    if method == "hdn_ddi":
        return dict(
            hidd_dim=64, kge_dim=128,
            heads_out_feat_params=(64, 64), blocks_params=(2, 2),
            n_epochs=1, batch_size=16,
        )
    if method == "emergnn":
        return dict(n_dim=16, length=2, n_epochs=1, batch_size=32)
    if method == "mkg_fenn":
        return dict(
            embedding_num=8, neighbor_sample_size=4, n_epochs=1,
            batch_size=64, fp_nbits=64, n_bins=4,
        )
    if method == "tiger":
        return dict(
            max_layer=2, output_dim=16, n_epochs=1, batch_size=16,
        )
    return {}


def _run_evaluate(method: str, out_dir: Path) -> None:
    """Run evaluate.py end-to-end for one baseline with tiny kwargs."""
    from coldddi.baselines import ensure_imported
    from coldddi.baselines.base import _REGISTRY
    from coldddi.evaluate import run_evaluation

    ensure_imported(method)
    OrigCls = _REGISTRY[method]
    tiny_kwargs = _make_tiny_kwargs(method)

    class Tiny(OrigCls):
        def __init__(self, **kw):
            merged = {**tiny_kwargs, **kw}
            super().__init__(**merged)

    _REGISTRY[method] = Tiny
    try:
        run_evaluation(
            method=method,
            data_dir=TOY_RELEASE,
            seed=42,
            settings=["S2"],
            out_dir=out_dir,
            device="cpu",
            preset="smoke",   # prevent paper kwargs from overriding the tiny config
        )
    finally:
        _REGISTRY[method] = OrigCls


@pytest.fixture(scope="module")
def trained_outputs(tmp_path_factory):
    """Run each baseline once and return {method: out_dir} for shared assertions."""
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    out: dict[str, Path] = {}
    for method, _modality in PARAMETRIZED_BASELINES:
        d = tmp_path_factory.mktemp(f"e2e_{method}")
        _run_evaluate(method, d)
        out[method] = d
    return out


def _load_pred_dict(csv_path: Path) -> dict[tuple[str, str], float]:
    """Load a per-pair predictions CSV → ``{(a, b): prob}`` dict.
    NaN predictions are silently skipped (mask CSVs may carry the
    ``true_label=-1`` sentinel but never NaN probs by construction)."""
    df = pd.read_csv(csv_path)
    return {
        (str(r["drug_a_id"]), str(r["drug_b_id"])): float(r["predicted_prob"])
        for _, r in df.iterrows()
        if not math.isnan(float(r["predicted_prob"]))
    }


# 1. Required output files per modality


class TestOutputFilesExistPerBaseline:
    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_indicators_csv_written(self, trained_outputs, method, modality):
        assert (trained_outputs[method] / "indicators_test_s2_seed42.csv").is_file()

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_base_predictions_csv_written(self, trained_outputs, method, modality):
        """Base predictions cover test_s2 only."""
        assert (trained_outputs[method] / "predictions_test_s2_seed42.csv").is_file()

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_union_predictions_csv_written(self, trained_outputs, method, modality):
        """L6 artefact: the exact base-union pair set the indicators
        saw — needed for downstream hand-recompute without re-training."""
        assert (trained_outputs[method] / "predictions_test_s2_union_seed42.csv").is_file()

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_mask_csvs_dispatch_per_modality(
        self, trained_outputs, method, modality,
    ):
        d = trained_outputs[method]
        mol_csv = d / "predictions_test_s2_mask_mol_seed42.csv"
        kg_csv = d / "predictions_test_s2_mask_kg_seed42.csv"
        if modality == "mol+kg":
            assert mol_csv.is_file(), f"{method} (mol+kg) missing mask_mol CSV"
            assert kg_csv.is_file(), f"{method} (mol+kg) missing mask_kg CSV"
        else:
            assert not mol_csv.is_file(), (
                f"{method} ({modality}) wrote a mask_mol CSV — modality "
                "dispatch leaked a mask pass to a single-modality baseline"
            )
            assert not kg_csv.is_file()


# 2. Unified swap_candidates set across baselines


class TestUnifiedSwapCandidatesAcrossBaselines:
    """Swap triples and bucket counts depend on the dataset and seed, not the baseline."""

    def test_full_bucket_n_map_identical_across_baselines(self, trained_outputs):
        """Compare complete bucket/count maps to catch extra or missing rows."""
        ref_method, _ = PARAMETRIZED_BASELINES[0]
        ref_df = pd.read_csv(
            trained_outputs[ref_method] / "indicators_test_s2_seed42.csv"
        )
        ref_map = dict(
            zip(
                ref_df.query("indicator == 'KPS-F'")["bucket"],
                ref_df.query("indicator == 'KPS-F'")["n"].astype(int),
            )
        )
        assert ref_map, "reference baseline emitted zero KPS-F rows"

        for method, _mod in PARAMETRIZED_BASELINES[1:]:
            df = pd.read_csv(
                trained_outputs[method] / "indicators_test_s2_seed42.csv"
            )
            m = dict(
                zip(
                    df.query("indicator == 'KPS-F'")["bucket"],
                    df.query("indicator == 'KPS-F'")["n"].astype(int),
                )
            )
            assert m == ref_map, (
                f"{method} KPS-F bucket map {m} diverges from "
                f"reference {ref_method} {ref_map}"
            )

    def test_union_csv_keys_match_reconstructed_union(self, trained_outputs):
        """The union CSV contains exactly the deduplicated S2 base, anchor, and target pairs."""
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import build_swap_candidates

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")

        # Reconstruct the union the way evaluate.py does.
        expected_keys: set[tuple[str, str]] = set()
        for _, r in ds.splits.test_s2.iterrows():
            expected_keys.add((str(r["drug_a_id"]), str(r["drug_b_id"])))
        for _, r in ds.get_negatives("test_s2").iterrows():
            expected_keys.add((str(r["drug_a_id"]), str(r["drug_b_id"])))
        for t in swap:
            expected_keys.add((str(t.qa), str(t.qb)))
            expected_keys.add((str(t.qa_prime), str(t.qb)))

        # Check the reference baseline; shared swap counts are verified separately.
        ref_method, _ = PARAMETRIZED_BASELINES[0]
        union_df = pd.read_csv(
            trained_outputs[ref_method] / "predictions_test_s2_union_seed42.csv"
        )
        union_keys = set(
            zip(
                union_df["drug_a_id"].astype(str),
                union_df["drug_b_id"].astype(str),
            )
        )
        missing = expected_keys - union_keys
        extra = union_keys - expected_keys
        assert not missing, (
            f"L6 union CSV missing {len(missing)} expected pairs "
            f"(first 5: {sorted(missing)[:5]})"
        )
        assert not extra, (
            f"L6 union CSV has {len(extra)} unexpected pairs "
            f"(first 5: {sorted(extra)[:5]})"
        )
        # Set equality alone would miss duplicate rows.
        assert len(union_df) == len(union_keys), (
            f"L6 union CSV has {len(union_df) - len(union_keys)} "
            "duplicate (drug_a_id, drug_b_id) rows"
        )

    def test_n_matches_reconstructed_swap_candidates(self, trained_outputs):
        """KPS-F's ALL count matches positive triples from build_swap_candidates."""
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import build_swap_candidates

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")
        expected_positive_triples = sum(1 for t in swap if t.label_uv == 1)

        ref_method, _ = PARAMETRIZED_BASELINES[0]
        df = pd.read_csv(
            trained_outputs[ref_method] / "indicators_test_s2_seed42.csv"
        )
        all_row = df.query("indicator == 'KPS-F' and bucket == 'ALL'").iloc[0]
        assert int(all_row["n"]) == expected_positive_triples, (
            f"reported KPS-F ALL n={int(all_row['n'])} doesn't match "
            f"build_swap_candidates positive-triple count "
            f"{expected_positive_triples}"
        )


# 3. Modality dispatch correctness


class TestModalityDispatchOnE2E:
    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_kps_f_always_populated(self, trained_outputs, method, modality):
        """KPS-F is the universal indicator — every baseline must
        emit a finite value in at least the ALL bucket."""
        df = pd.read_csv(
            trained_outputs[method] / "indicators_test_s2_seed42.csv"
        )
        all_row = df.query("indicator == 'KPS-F' and bucket == 'ALL'")
        assert len(all_row), f"{method} missing KPS-F ALL row"
        v = float(all_row.iloc[0]["value"])
        assert not math.isnan(v)
        assert 0 <= v <= 1

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_channel_indicators_match_modality(
        self, trained_outputs, method, modality,
    ):
        df = pd.read_csv(
            trained_outputs[method] / "indicators_test_s2_seed42.csv"
        )
        for ind in ("KPS-mol", "KPS-KG"):
            sub = df.query(f"indicator == '{ind}'")
            assert len(sub), f"{method} missing {ind} rows"
            if modality == "mol+kg":
                all_row = sub.query("bucket == 'ALL'").iloc[0]
                assert int(all_row["n"]) > 0, (
                    f"{method} ({modality}) {ind} has n=0 — mask pass "
                    "didn't produce any matched predictions"
                )
                assert not math.isnan(float(all_row["value"]))
            else:
                assert sub["value"].isna().all(), (
                    f"{method} ({modality}) unexpectedly populated {ind}"
                )
                assert (sub["n"] == 0).all()


# Recompute KPS-F from persisted predictions (1e-6 tolerance).


class TestKpsFMathMatchesUnionRecomputation:
    """Recompute mean(|P(u,v) - P(u',v)|) and counts from union CSVs.

    Values match within 1e-6 for every baseline. Following upstream
    _agg_buckets, ALL uses positives only; primary buckets use both labels.
    """

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    def test_kps_f_all_bucket_exact(self, trained_outputs, method, modality):
        """ALL bucket = positive base pairs only (label_uv == 1)."""
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import build_swap_candidates
        from coldddi.diagnostics.indicators import _lookup_directed

        d = trained_outputs[method]
        preds = _load_pred_dict(d / "predictions_test_s2_union_seed42.csv")
        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")

        deltas: list[float] = []
        for t in swap:
            if t.label_uv != 1:
                continue
            p_uv = _lookup_directed(preds, t.qa, t.qb)
            p_upv = _lookup_directed(preds, t.qa_prime, t.qb)
            assert p_uv is not None and p_upv is not None, (
                "union CSV missing predictions — L6 union persistence "
                f"incomplete for {method}"
            )
            deltas.append(abs(p_uv - p_upv))
        hand = float(np.mean(deltas))

        df = pd.read_csv(d / "indicators_test_s2_seed42.csv")
        all_row = df.query("indicator == 'KPS-F' and bucket == 'ALL'").iloc[0]
        assert float(all_row["value"]) == pytest.approx(hand, abs=1e-6), (
            f"{method}: reported KPS-F ALL={float(all_row['value']):.6f} "
            f"!= hand-recomputed {hand:.6f}"
        )
        assert int(all_row["n"]) == len(deltas)

    @pytest.mark.parametrize("method,modality", PARAMETRIZED_BASELINES)
    @pytest.mark.parametrize("bucket", ["PK-A", "PK-B", "PD-A", "PD-B"])
    def test_kps_f_primary_bucket_exact(
        self, trained_outputs, method, modality, bucket,
    ):
        """Primary buckets aggregate both labels, unlike positives-only ALL."""
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import (
            build_bucket_lookup,
            build_swap_candidates,
        )
        from coldddi.diagnostics.indicators import _lookup_directed
        from coldddi.evaluate import _resolve_ab_parquet

        d = trained_outputs[method]
        preds = _load_pred_dict(d / "predictions_test_s2_union_seed42.csv")
        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")
        ab = _resolve_ab_parquet(None, TOY_RELEASE)
        assert ab is not None
        lookup = build_bucket_lookup(ab)

        deltas: list[float] = []
        for t in swap:
            if lookup.bucket(t.qa, t.qb) != bucket:
                continue
            p_uv = _lookup_directed(preds, t.qa, t.qb)
            p_upv = _lookup_directed(preds, t.qa_prime, t.qb)
            if p_uv is None or p_upv is None:
                continue
            deltas.append(abs(p_uv - p_upv))

        df = pd.read_csv(d / "indicators_test_s2_seed42.csv")
        rows = df.query(f"indicator == 'KPS-F' and bucket == '{bucket}'")
        if not deltas:
            # No triples in this bucket on toy → no row emitted.
            assert len(rows) == 0, (
                f"{method} {bucket}: no swap triples cover this bucket "
                f"but indicator row exists ({rows.iloc[0].to_dict()})"
            )
            return
        assert len(rows), (
            f"{method} {bucket}: {len(deltas)} swap triples found but "
            "no indicator row emitted"
        )
        reported = float(rows.iloc[0]["value"])
        hand = float(np.mean(deltas))
        assert reported == pytest.approx(hand, abs=1e-6), (
            f"{method} KPS-F[{bucket}]: reported {reported:.6f} != "
            f"hand-recomputed {hand:.6f}"
        )
        assert int(rows.iloc[0]["n"]) == len(deltas)


# Recompute KPS-mol / KPS-KG (1e-6 tolerance).


def _channel_indicator_name(channel: str) -> str:
    """Reverse of the dispatch in compute_baseline_channel_indicators:
    ``"mol"`` → ``"KPS-mol"``; ``"kg"`` → ``"KPS-KG"`` (case-sensitive)."""
    return {"mol": "KPS-mol", "kg": "KPS-KG"}[channel]


class TestChannelMathMatchesUnionRecomputation:
    """For each mol+kg baseline × {mol, kg} channel × bucket cell,
    recompute ``mean(|P_base(u, v) - P_mask=c(u, v)|)`` per
    deduplicated swap-anchor pair, assert equal (abs=1e-6) to the
    reported KPS-mol / KPS-KG value AND ``n``.  Covers ALL +
    each primary bucket."""

    @pytest.mark.parametrize("method,modality", MOL_KG_BASELINES)
    @pytest.mark.parametrize("channel", ["mol", "kg"])
    def test_kps_channel_all_bucket_exact(
        self, trained_outputs, method, modality, channel,
    ):
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import build_swap_candidates
        from coldddi.diagnostics.indicators import (
            _lookup_directed, _unique_base_pairs,
        )

        d = trained_outputs[method]
        base = _load_pred_dict(d / "predictions_test_s2_union_seed42.csv")
        mask = _load_pred_dict(
            d / f"predictions_test_s2_mask_{channel}_seed42.csv"
        )
        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")
        base_pairs = _unique_base_pairs(swap, lambda a, b: "")

        deltas: list[float] = []
        for p in base_pairs:
            if p["label_uv"] != 1:
                continue
            va = _lookup_directed(base, p["qa"], p["qb"])
            vb = _lookup_directed(mask, p["qa"], p["qb"])
            if va is None or vb is None:
                continue
            deltas.append(abs(va - vb))
        if not deltas:
            pytest.skip(f"no positive base pairs covered for {method}/{channel}")
        hand = float(np.mean(deltas))

        df = pd.read_csv(d / "indicators_test_s2_seed42.csv")
        ind = _channel_indicator_name(channel)
        all_row = df.query(f"indicator == '{ind}' and bucket == 'ALL'").iloc[0]
        assert float(all_row["value"]) == pytest.approx(hand, abs=1e-6), (
            f"{method}: reported {ind} ALL={float(all_row['value']):.6f} "
            f"!= hand-recomputed {hand:.6f}"
        )
        assert int(all_row["n"]) == len(deltas)

    @pytest.mark.parametrize("method,modality", MOL_KG_BASELINES)
    @pytest.mark.parametrize("channel", ["mol", "kg"])
    @pytest.mark.parametrize("bucket", ["PK-A", "PK-B", "PD-A", "PD-B"])
    def test_kps_channel_primary_bucket_exact(
        self, trained_outputs, method, modality, channel, bucket,
    ):
        from coldddi.data.dataset import PairDataset
        from coldddi.diagnostics import (
            build_bucket_lookup,
            build_swap_candidates,
        )
        from coldddi.diagnostics.indicators import (
            _lookup_directed, _unique_base_pairs,
        )
        from coldddi.evaluate import _resolve_ab_parquet

        d = trained_outputs[method]
        base = _load_pred_dict(d / "predictions_test_s2_union_seed42.csv")
        mask = _load_pred_dict(
            d / f"predictions_test_s2_mask_{channel}_seed42.csv"
        )
        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        swap = build_swap_candidates(ds, source_split="test_s2")
        ab = _resolve_ab_parquet(None, TOY_RELEASE)
        assert ab is not None
        lookup = build_bucket_lookup(ab)
        base_pairs = _unique_base_pairs(swap, lookup.bucket)

        deltas: list[float] = []
        for p in base_pairs:
            if p["bucket"] != bucket:
                continue
            va = _lookup_directed(base, p["qa"], p["qb"])
            vb = _lookup_directed(mask, p["qa"], p["qb"])
            if va is None or vb is None:
                continue
            deltas.append(abs(va - vb))

        df = pd.read_csv(d / "indicators_test_s2_seed42.csv")
        ind = _channel_indicator_name(channel)
        rows = df.query(f"indicator == '{ind}' and bucket == '{bucket}'")
        if not deltas:
            assert len(rows) == 0, (
                f"{method} {ind}[{bucket}]: no pairs cover this bucket "
                f"but row exists"
            )
            return
        assert len(rows), (
            f"{method} {ind}[{bucket}]: {len(deltas)} pairs found but "
            "no indicator row emitted"
        )
        reported = float(rows.iloc[0]["value"])
        hand = float(np.mean(deltas))
        assert reported == pytest.approx(hand, abs=1e-6), (
            f"{method} {ind}[{bucket}]: reported {reported:.6f} != "
            f"hand-recomputed {hand:.6f}"
        )
        assert int(rows.iloc[0]["n"]) == len(deltas)


# 6. Channel mask CSV schema sanity


class TestChannelMaskCsvSchemaSanity:
    """Mask CSVs use the canonical schema and probabilities in [0, 1].

    Per-baseline tests check mask effects; toy scores can be constant.
    """

    @pytest.mark.parametrize("method,modality", MOL_KG_BASELINES)
    @pytest.mark.parametrize("channel", ["mol", "kg"])
    def test_mask_csv_schema_and_range(
        self, trained_outputs, method, modality, channel,
    ):
        from coldddi.evaluate import PREDICTION_COLUMNS

        csv = (
            trained_outputs[method]
            / f"predictions_test_s2_mask_{channel}_seed42.csv"
        )
        df = pd.read_csv(csv)
        assert list(df.columns) == list(PREDICTION_COLUMNS)
        probs = df["predicted_prob"].dropna()
        assert len(probs)
        assert (probs >= 0).all() and (probs <= 1).all()

    @pytest.mark.parametrize("method,modality", MOL_KG_BASELINES)
    @pytest.mark.parametrize("channel", ["mol", "kg"])
    def test_mask_csv_covers_full_union_keyset(
        self, trained_outputs, method, modality, channel,
    ):
        """Mask and base CSV key sets match, preventing reduced-set recomputation."""
        d = trained_outputs[method]
        union = pd.read_csv(d / "predictions_test_s2_union_seed42.csv")
        mask = pd.read_csv(
            d / f"predictions_test_s2_mask_{channel}_seed42.csv"
        )
        u_keys = set(zip(
            union["drug_a_id"].astype(str),
            union["drug_b_id"].astype(str),
        ))
        m_keys = set(zip(
            mask["drug_a_id"].astype(str),
            mask["drug_b_id"].astype(str),
        ))
        missing = u_keys - m_keys
        extra = m_keys - u_keys
        assert not missing, (
            f"{method} mask_{channel} CSV missing {len(missing)} "
            f"union pairs"
        )
        assert not extra, (
            f"{method} mask_{channel} CSV has {len(extra)} "
            f"unexpected pairs not in the base union"
        )
        # No duplicate (a, b) rows.
        assert len(mask) == len(m_keys), (
            f"{method} mask_{channel} CSV has duplicate pairs"
        )
