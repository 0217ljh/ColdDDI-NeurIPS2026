"""Test kps.py, ksai.py, and masking.py shims from Appendix A.6.2, line 503.

The documented paths must re-export the implementation's entry points.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestPaperFileNamesImport:
    """Paper-named modules expose the documented entry points."""

    def test_kps_module_exposes_kps_entry_points(self):
        mod = importlib.import_module("coldddi.diagnostics.kps")
        for name in (
            "compute_kps_f",
            "compute_kps_channel",
            "build_swap_candidates",
            "SwapTriple",
        ):
            assert hasattr(mod, name), (
                f"coldddi.diagnostics.kps is missing paper-promised {name!r}"
            )

    def test_ksai_module_exposes_ksai_entry_points(self):
        mod = importlib.import_module("coldddi.diagnostics.ksai")
        for name in ("compute_ksai", "compute_ab_gap"):
            assert hasattr(mod, name), (
                f"coldddi.diagnostics.ksai is missing paper-promised {name!r}"
            )

    def test_masking_module_exposes_runner_entry_points(self):
        mod = importlib.import_module("coldddi.diagnostics.masking")
        for name in ("compute_indicators", "LLM_INDICATOR_NAMES", "PRIMARY_BUCKETS"):
            assert hasattr(mod, name), (
                f"coldddi.diagnostics.masking is missing paper-promised {name!r}"
            )


class TestShimsReExportSameObjects:
    """Shims re-export the original function objects, not separate implementations."""

    def test_kps_compute_kps_f_is_indicators_function(self):
        from coldddi.diagnostics import indicators as impl
        from coldddi.diagnostics import kps as shim

        assert shim.compute_kps_f is impl.compute_kps_f

    def test_kps_swap_helper_is_kps_swap_function(self):
        from coldddi.diagnostics import kps as shim
        from coldddi.diagnostics import kps_swap as impl

        assert shim.build_swap_candidates is impl.build_swap_candidates
        assert shim.SwapTriple is impl.SwapTriple

    def test_ksai_compute_ksai_is_indicators_function(self):
        from coldddi.diagnostics import indicators as impl
        from coldddi.diagnostics import ksai as shim

        assert shim.compute_ksai is impl.compute_ksai
        assert shim.compute_ab_gap is impl.compute_ab_gap

    def test_masking_compute_indicators_is_indicators_function(self):
        from coldddi.diagnostics import indicators as impl
        from coldddi.diagnostics import masking as shim

        assert shim.compute_indicators is impl.compute_indicators

    def test_kps_compute_kps_channel_is_indicators_function(self):
        """Every re-exported symbol retains identity with its implementation source."""
        from coldddi.diagnostics import indicators as impl
        from coldddi.diagnostics import kps as shim

        assert shim.compute_kps_channel is impl.compute_kps_channel

    def test_masking_constants_are_indicators_constants(self):
        from coldddi.diagnostics import indicators as impl
        from coldddi.diagnostics import masking as shim

        assert shim.LLM_INDICATOR_NAMES is impl.LLM_INDICATOR_NAMES
        assert shim.PRIMARY_BUCKETS is impl.PRIMARY_BUCKETS


class TestShimsFunctionalEndToEnd:
    """KPS-F results match through the shim and implementation modules."""

    def test_kps_shim_matches_indicators(self):
        import pandas as pd

        from coldddi.diagnostics import kps as shim_kps
        from coldddi.diagnostics.indicators import compute_kps_f
        from coldddi.diagnostics.kps_swap import SwapTriple

        swap = [
            SwapTriple("DBA", "DBC", "DBE", label_uv=1),
            SwapTriple("DBB", "DBC", "DBE", label_uv=1),
        ]
        bf = lambda a, b: "PK-A"
        preds = {("DBA", "DBC"): 0.8, ("DBB", "DBC"): 0.4,
                 ("DBE", "DBC"): 0.3}
        via_shim = shim_kps.compute_kps_f(preds, swap, bucket_fn=bf)
        via_impl = compute_kps_f(preds, swap, bucket_fn=bf)
        pd.testing.assert_frame_equal(via_shim, via_impl)
