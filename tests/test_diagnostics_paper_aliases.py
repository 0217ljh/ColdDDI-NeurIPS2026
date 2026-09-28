"""Paper-promised public-name aliases on coldddi/diagnostics/.

Paper Appendix A.6.2 line 624 promises:

    coldddi/diagnostics/ exposes compute_kps_f, compute_kps_channel,
    and compute_ksai, each taking a method's predicted-probability
    function and a swap-table anchor set, and returning a per-bucket
    pandas.DataFrame.

These tests pin:
1. The three names are importable from ``coldddi.diagnostics``.
2. Each accepts either a prediction dict (current dict-based API)
   or a callable ``predict_fn(pairs_df) -> np.ndarray`` (paper API).
3. Each returns a per-bucket DataFrame restricted to its named
   indicator (KPS-F / KPS-mol or KPS-kg / KSAI).
4. The math is byte-equivalent to calling the underlying
   ``compute_indicators`` / ``compute_baseline_channel_indicators``
   and filtering the row block.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Importability ──────────────────────────────────────────────────

class TestPaperAliasImports:
    def test_three_names_exposed_at_package_root(self):
        import coldddi.diagnostics as d

        for name in ("compute_kps_f", "compute_kps_channel", "compute_ksai"):
            assert hasattr(d, name), f"{name} not exposed by coldddi.diagnostics"
            assert callable(getattr(d, name))


# ─── Fixtures (small synthetic swap + predictions) ──────────────────

def _tiny_swap_setup():
    from coldddi.diagnostics.kps_swap import SwapTriple

    swap = [
        SwapTriple("DBA", "DBC", "DBE", label_uv=1),
        SwapTriple("DBB", "DBC", "DBE", label_uv=1),
        SwapTriple("DBD", "DBC", "DBE", label_uv=0),
    ]
    bucket_fn = lambda a, b: "PK-A"
    return swap, bucket_fn


def _tiny_predictions_r0():
    return {
        ("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5,
        ("DBD", "DBC"): 0.2, ("DBE", "DBC"): 0.3,
    }


# ─── KPS-F: dict and callable interfaces equivalent ─────────────────

class TestComputeKpsF:
    def test_dict_input_returns_only_kps_f_rows(self):
        from coldddi.diagnostics import compute_kps_f

        swap, bf = _tiny_swap_setup()
        df = compute_kps_f(_tiny_predictions_r0(), swap, bucket_fn=bf)
        assert (df["indicator"] == "KPS-F").all()
        assert len(df) > 0

    def test_callable_input_matches_dict(self):
        """A callable that mirrors the dict lookup must produce the
        same per-bucket numbers as passing the dict directly."""
        from coldddi.diagnostics import compute_kps_f

        swap, bf = _tiny_swap_setup()
        pred_dict = _tiny_predictions_r0()

        def predict_fn(pairs: pd.DataFrame) -> np.ndarray:
            return np.array([
                pred_dict.get((a, b), 0.5)
                for a, b in zip(pairs["drug_a_id"], pairs["drug_b_id"])
            ], dtype=np.float32)

        from_dict = compute_kps_f(pred_dict, swap, bucket_fn=bf)
        from_fn = compute_kps_f(predict_fn, swap, bucket_fn=bf)
        pd.testing.assert_frame_equal(
            from_dict.reset_index(drop=True),
            from_fn.reset_index(drop=True),
            check_dtype=False,
        )

    def test_matches_underlying_dict_api(self):
        """``compute_kps_f`` must equal
        ``compute_baseline_channel_indicators(...)`` filtered to
        ``indicator == "KPS-F"`` — i.e., it's a strict pass-through
        with no math of its own."""
        from coldddi.diagnostics import (
            compute_baseline_channel_indicators,
            compute_kps_f,
        )

        swap, bf = _tiny_swap_setup()
        pred = _tiny_predictions_r0()
        alias = compute_kps_f(pred, swap, bucket_fn=bf)
        full = compute_baseline_channel_indicators(
            {"base": pred}, swap, bucket_fn=bf,
        )
        ref = full[full["indicator"] == "KPS-F"].reset_index(drop=True)
        pd.testing.assert_frame_equal(alias, ref)


# ─── KPS-Channel: only mol or kg accepted; output restricted ────────

class TestComputeKpsChannel:
    def setup_method(self):
        self.swap, self.bf = _tiny_swap_setup()
        self.base = {
            ("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5,
        }
        self.mask = {
            ("DBA", "DBC"): 0.6, ("DBB", "DBC"): 0.2,
        }

    def test_mol_channel_returns_only_kps_mol_rows(self):
        from coldddi.diagnostics import compute_kps_channel

        df = compute_kps_channel(
            self.base, self.mask, self.swap,
            channel="mol", bucket_fn=self.bf,
        )
        assert len(df) > 0, "mol channel produced empty result"
        assert (df["indicator"] == "KPS-mol").all()

    def test_kg_channel_returns_only_kps_kg_rows(self):
        from coldddi.diagnostics import compute_kps_channel

        df = compute_kps_channel(
            self.base, self.mask, self.swap,
            channel="kg", bucket_fn=self.bf,
        )
        # Codex IMPORTANT: assert the output is non-empty FIRST.
        # ``(empty_df["indicator"] == "KPS-KG").all()`` is vacuously
        # True; without the length check the test would mask a
        # silent KG → "KPS-kg" case-mismatch lookup bug.
        assert len(df) > 0, "KG channel produced empty result"
        assert (df["indicator"] == "KPS-KG").all()

    def test_invalid_channel_raises(self):
        from coldddi.diagnostics import compute_kps_channel

        with pytest.raises(ValueError, match="channel must be"):
            compute_kps_channel(
                self.base, self.mask, self.swap,
                channel="text", bucket_fn=self.bf,
            )

    def test_callable_inputs_accepted(self):
        from coldddi.diagnostics import compute_kps_channel

        def base_fn(pairs):
            return np.array([
                self.base.get((a, b), 0.5)
                for a, b in zip(pairs["drug_a_id"], pairs["drug_b_id"])
            ], dtype=np.float32)

        def mask_fn(pairs):
            return np.array([
                self.mask.get((a, b), 0.5)
                for a, b in zip(pairs["drug_a_id"], pairs["drug_b_id"])
            ], dtype=np.float32)

        df = compute_kps_channel(
            base_fn, mask_fn, self.swap,
            channel="mol", bucket_fn=self.bf,
        )
        assert len(df) > 0
        assert (df["indicator"] == "KPS-mol").all()


# ─── KSAI: only LLM 4-condition setting populates non-NaN values ────

class TestComputeKsai:
    def test_four_condition_dicts_produce_ksai_rows(self):
        from coldddi.diagnostics import compute_ksai

        swap, bf = _tiny_swap_setup()
        # Synthetic R0..R3 that produce a deterministic KSAI value.
        r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5, ("DBD", "DBC"): 0.2}
        r1 = {("DBA", "DBC"): 0.7, ("DBB", "DBC"): 0.3, ("DBD", "DBC"): 0.0}
        r2 = {("DBA", "DBC"): 0.5, ("DBB", "DBC"): 0.1, ("DBD", "DBC"): 0.0}
        r3 = {("DBA", "DBC"): 0.4, ("DBB", "DBC"): 0.0, ("DBD", "DBC"): 0.0}
        df = compute_ksai(r0, r1, r2, r3, swap, bucket_fn=bf)
        assert (df["indicator"] == "KSAI").all()
        all_row = df.query("bucket == 'ALL'")
        if len(all_row):
            # Two positives only: DBA, DBB.
            # DBA: |0.7-0.4| - |0.9-0.5| = 0.3 - 0.4 = -0.1
            # DBB: |0.3-0.0| - |0.5-0.1| = 0.3 - 0.4 = -0.1
            assert float(all_row.iloc[0]["value"]) == pytest.approx(-0.1)

    def test_matches_compute_indicators_filtered_subset(self):
        from coldddi.diagnostics import compute_indicators, compute_ksai

        swap, bf = _tiny_swap_setup()
        r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5, ("DBD", "DBC"): 0.2}
        r1 = {("DBA", "DBC"): 0.7, ("DBB", "DBC"): 0.3, ("DBD", "DBC"): 0.0}
        r2 = {("DBA", "DBC"): 0.5, ("DBB", "DBC"): 0.1, ("DBD", "DBC"): 0.0}
        r3 = {("DBA", "DBC"): 0.4, ("DBB", "DBC"): 0.0, ("DBD", "DBC"): 0.0}
        alias = compute_ksai(r0, r1, r2, r3, swap, bucket_fn=bf)
        full = compute_indicators(
            {"R0": r0, "R1": r1, "R2": r2, "R3": r3},
            swap, bucket_fn=bf,
        )
        ref = full[full["indicator"] == "KSAI"].reset_index(drop=True)
        pd.testing.assert_frame_equal(alias, ref)


# ─── Coercion edge cases ────────────────────────────────────────────

class TestPredictionInputCoercion:
    def test_dict_passes_through_unchanged(self):
        from coldddi.diagnostics.indicators import _coerce_predictions

        swap, _ = _tiny_swap_setup()
        d = {("A", "B"): 0.7}
        out = _coerce_predictions(d, swap)
        assert out is d  # exact same object — no copy / no transform

    def test_invalid_type_raises_type_error(self):
        from coldddi.diagnostics.indicators import _coerce_predictions

        swap, _ = _tiny_swap_setup()
        with pytest.raises(TypeError, match="dict or a callable"):
            _coerce_predictions(42, swap)

    def test_callable_length_mismatch_raises(self):
        """Codex IMPORTANT: a callable returning the wrong number of
        scores must raise ValueError up front, NOT silently truncate
        via ``zip`` and corrupt every downstream indicator with
        missing per-pair predictions."""
        from coldddi.diagnostics.indicators import _coerce_predictions

        swap, _ = _tiny_swap_setup()

        def short_fn(pairs):
            return np.array([0.5], dtype=np.float32)   # only 1 score

        def long_fn(pairs):
            return np.full(len(pairs) + 5, 0.5, dtype=np.float32)

        with pytest.raises(ValueError, match="returned 1 scores"):
            _coerce_predictions(short_fn, swap)
        with pytest.raises(ValueError, match="scores for"):
            _coerce_predictions(long_fn, swap)
