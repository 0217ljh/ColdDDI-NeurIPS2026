"""Aggregator and exps/ shell-script tests for Step 5.

Backs the paper-promised ``exps/sec5_{overall,stratified,kps,
masking}.sh`` one-command reproduction scripts.  Tests:

1. ``coldddi.eval.aggregate`` rolls up synthetic ``runs/`` trees
   correctly (per-method mean ± std across seeds).
2. The CLI dispatcher writes the right CSV at the right path.
3. Each shipped ``exps/sec5_*.sh`` exists, is executable, and
   passes ``bash -n`` syntax validation.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Synthetic runs/ tree helpers ────────────────────────────────────

def _make_predictions_csv(
    out_dir: Path,
    split: str,
    seed: int,
    rows: list[tuple[str, str, int, float]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=["drug_a_id", "drug_b_id", "true_label", "predicted_prob"],
    )
    df["predicted_label"] = (df["predicted_prob"] >= 0.5).astype(int)
    path = out_dir / f"predictions_{split}_seed{seed}.csv"
    df.to_csv(path, index=False)
    return path


def _make_indicators_csv(
    out_dir: Path,
    seed: int,
    rows: list[tuple[str, str, float, float, int]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=["indicator", "bucket", "value", "std", "n"],
    )
    path = out_dir / f"indicators_test_s2_seed{seed}.csv"
    df.to_csv(path, index=False)
    return path


# ─── aggregate_overall ──────────────────────────────────────────────

class TestAggregateOverall:
    def test_three_seeds_yield_mean_std(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_overall

        # 1 method, 3 seeds — fully predictable AUC values per seed
        # (perfect classifier ⇒ AUROC=1.0 for all 3 seeds).
        for seed in (42, 43, 44):
            _make_predictions_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                split="test_s2", seed=seed,
                rows=[
                    ("A", "B", 1, 0.9),
                    ("A", "C", 0, 0.1),
                    ("X", "Y", 1, 0.8),
                    ("X", "Z", 0, 0.2),
                ],
            )
        df = aggregate_overall(tmp_path)
        assert set(df["metric"].unique()) == {"AUROC", "AUPRC"}
        assert (df["method"] == "deepddi").all()
        auroc = df.query("metric == 'AUROC'").iloc[0]
        assert auroc["mean"] == pytest.approx(1.0)
        assert auroc["std"] == pytest.approx(0.0, abs=1e-9)
        assert auroc["n_seeds"] == 3
        assert auroc["seeds"] == "42,43,44"

    def test_method_filter(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_overall

        for m in ("deepddi", "tiger"):
            _make_predictions_csv(
                tmp_path / m / "seed42",
                split="test_s2", seed=42,
                rows=[("A", "B", 1, 0.9), ("A", "C", 0, 0.1)],
            )
        df = aggregate_overall(tmp_path, methods=["tiger"])
        assert set(df["method"].unique()) == {"tiger"}

    def test_empty_runs_root_returns_empty_frame_with_schema(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_overall

        df = aggregate_overall(tmp_path)
        assert df.empty
        for col in ("method", "metric", "mean", "std", "n_seeds", "seeds"):
            assert col in df.columns

    def test_single_class_split_yields_nan_safely(self, tmp_path):
        """Tiny smoke fixtures sometimes produce a fully-positive or
        fully-negative split.  AUROC is undefined there; aggregator
        should silently drop those seeds from the mean instead of
        crashing the whole table."""
        from coldddi.eval.aggregate import aggregate_overall

        # 2 seeds with 2 classes (AUROC valid) + 1 seed with only positives.
        for seed in (42, 43):
            _make_predictions_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                split="test_s2", seed=seed,
                rows=[("A", "B", 1, 0.9), ("A", "C", 0, 0.1)],
            )
        _make_predictions_csv(
            tmp_path / "deepddi" / "seed44",
            split="test_s2", seed=44,
            rows=[("A", "B", 1, 0.9)],   # only positives
        )
        df = aggregate_overall(tmp_path)
        auroc = df.query("metric == 'AUROC'").iloc[0]
        assert auroc["n_seeds"] == 2     # seed44 dropped due to single class
        assert auroc["seeds"] == "42,43"


# ─── aggregate_stratified ────────────────────────────────────────────

class TestAggregateStratified:
    def test_per_bucket_means(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_stratified

        # Build a tiny AB parquet covering all 4 buckets.
        ab = pd.DataFrame({
            "drug_a_id":      ["A", "B", "C", "D"],
            "drug_b_id":      ["X", "Y", "Z", "W"],
            "pk_pd_label":    ["PK", "PK", "PD", "PD"],
            "has_key_entity": [True, False, True, False],
        })
        ab_path = tmp_path / "ab.parquet"
        ab.to_parquet(ab_path)

        for seed in (42, 43, 44):
            _make_predictions_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                split="test_s2", seed=seed,
                rows=[
                    ("A", "X", 1, 0.9),   # PK-A
                    ("A", "X", 0, 0.1),   # bucket lookup is direction-tolerant
                    ("B", "Y", 1, 0.7),   # PK-B
                    ("B", "Y", 0, 0.3),
                    ("C", "Z", 1, 0.95),  # PD-A
                    ("C", "Z", 0, 0.05),
                    ("D", "W", 1, 0.85),  # PD-B
                    ("D", "W", 0, 0.15),
                ],
            )
        df = aggregate_stratified(tmp_path, ab_parquet=ab_path)
        buckets = set(df["bucket"].unique())
        assert buckets == {"PK-A", "PK-B", "PD-A", "PD-B"}, (
            f"missing bucket rows; got {sorted(buckets)}"
        )
        # All 4 are perfect classifiers (AUROC=1) on this fixture.
        for _, row in df.iterrows():
            assert row["mean"] == pytest.approx(1.0)


# ─── aggregate_kps ───────────────────────────────────────────────────

class TestAggregateKps:
    def test_mean_std_per_indicator_bucket(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_kps

        for seed in (42, 43, 44):
            _make_indicators_csv(
                tmp_path / "mkg_fenn" / f"seed{seed}",
                seed=seed,
                rows=[
                    ("KPS-F",   "PK-A", 0.10, 0.01, 50),
                    ("KPS-F",   "ALL",  0.12, 0.02, 200),
                    ("KPS-mol", "PK-A", 0.30, 0.03, 50),
                    ("KPS-mol", "ALL",  0.32, 0.04, 200),
                    ("KPS-KG",  "PK-A", 0.40, 0.05, 50),
                    ("KPS-KG",  "ALL",  0.42, 0.06, 200),
                ],
            )
        df = aggregate_kps(tmp_path)
        assert (df["method"] == "mkg_fenn").all()
        # 3 indicators × 2 buckets = 6 (method, indicator, bucket) cells.
        assert len(df) == 6
        all_kpsf = df.query("indicator == 'KPS-F' and bucket == 'ALL'").iloc[0]
        assert all_kpsf["mean"] == pytest.approx(0.12)
        assert all_kpsf["n_seeds"] == 3

    def test_all_nan_indicator_block_propagated(self, tmp_path):
        """Single-modality baseline KPS-mol/KPS-KG rows are all NaN
        per the diagnostics contract.  Aggregator must propagate one
        NaN row per (method, indicator, bucket) so the final paper
        table prints "—" cells uniformly."""
        from coldddi.eval.aggregate import aggregate_kps

        for seed in (42, 43, 44):
            _make_indicators_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                seed=seed,
                rows=[
                    ("KPS-F",   "ALL", 0.12 + 0.01 * (seed - 42), 0.0, 50),
                    ("KPS-mol", "ALL", float("nan"), float("nan"), 0),
                    ("KPS-KG",  "ALL", float("nan"), float("nan"), 0),
                ],
            )
        df = aggregate_kps(tmp_path)
        # KPS-F populated.
        f_row = df.query("indicator == 'KPS-F' and bucket == 'ALL'").iloc[0]
        assert not np.isnan(f_row["mean"])
        # Channel indicators NaN preserved.
        for ind in ("KPS-mol", "KPS-KG"):
            row = df.query(f"indicator == '{ind}' and bucket == 'ALL'").iloc[0]
            assert np.isnan(row["mean"])
            assert row["n_seeds"] == 0


# ─── CLI dispatcher ─────────────────────────────────────────────────

class TestAggregateCli:
    def test_kps_cli_writes_csv(self, tmp_path):
        for seed in (42, 43):
            _make_indicators_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                seed=seed,
                rows=[("KPS-F", "ALL", 0.10, 0.0, 50)],
            )
        from coldddi.eval.aggregate import main

        rc = main([
            "kps",
            "--runs", str(tmp_path),
            "--out", str(tmp_path / "out.csv"),
        ])
        assert rc == 0
        out = pd.read_csv(tmp_path / "out.csv")
        assert len(out) >= 1
        assert (out["method"] == "deepddi").all()

    def test_overall_cli_writes_csv(self, tmp_path):
        for seed in (42, 43):
            _make_predictions_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                split="test_s2", seed=seed,
                rows=[("A", "B", 1, 0.9), ("A", "C", 0, 0.1)],
            )
        from coldddi.eval.aggregate import main

        rc = main([
            "overall",
            "--runs", str(tmp_path),
            "--out", str(tmp_path / "ov.csv"),
        ])
        assert rc == 0
        out = pd.read_csv(tmp_path / "ov.csv")
        assert set(out["metric"].unique()) == {"AUROC", "AUPRC"}

    def test_stratified_cli_requires_ab_parquet(self, tmp_path):
        from coldddi.eval.aggregate import main

        with pytest.raises(SystemExit):
            main([
                "stratified",
                "--runs", str(tmp_path),
                # --ab-parquet intentionally omitted
            ])

    def test_default_out_path_is_runs_subdir(self, tmp_path):
        for seed in (42, 43):
            _make_indicators_csv(
                tmp_path / "deepddi" / f"seed{seed}",
                seed=seed,
                rows=[("KPS-F", "ALL", 0.10, 0.0, 50)],
            )
        from coldddi.eval.aggregate import main

        rc = main(["kps", "--runs", str(tmp_path)])
        assert rc == 0
        # Default writes runs/sec5_<mode>.csv.
        assert (tmp_path / "sec5_kps.csv").is_file()


# ─── Shell-script sanity ────────────────────────────────────────────

# ─── Flat-mode aggregator (LLM masking path) ────────────────────────

class TestFlatModeAggregator:
    """The LLM masking output tree is ``runs/<model>/<prompt>/seed<N>/``
    — one level deeper than the baseline ``runs/<method>/seed<N>/``
    layout.  The ``flat_method`` kwarg lets the aggregator treat the
    cell directory as the seed-dir parent and tag everything with a
    synthetic composite method label."""

    def test_kps_flat_method_picks_up_seed_dirs_directly(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_kps

        # Cell dir layout: <tmp>/seed42/, <tmp>/seed43/ (no method level).
        for seed in (42, 43):
            _make_indicators_csv(
                tmp_path / f"seed{seed}",
                seed=seed,
                rows=[("KPS-F", "ALL", 0.10, 0.0, 50)],
            )
        # Nested-mode dispatch finds nothing (no <method>/ dir).
        df_nested = aggregate_kps(tmp_path)
        assert df_nested.empty, "nested mode should not see flat layout"
        # Flat-mode dispatch with explicit method label sees both seeds.
        df_flat = aggregate_kps(tmp_path, flat_method="llama-1b_P4")
        assert len(df_flat) == 1
        assert df_flat["method"].iloc[0] == "llama-1b_P4"
        assert df_flat["n_seeds"].iloc[0] == 2

    def test_overall_flat_method_works(self, tmp_path):
        from coldddi.eval.aggregate import aggregate_overall

        for seed in (42, 43):
            _make_predictions_csv(
                tmp_path / f"seed{seed}",
                split="test_s2", seed=seed,
                rows=[("A", "B", 1, 0.9), ("A", "C", 0, 0.1)],
            )
        df = aggregate_overall(tmp_path, flat_method="qwen-0.5b_P4")
        assert (df["method"] == "qwen-0.5b_P4").all()

    def test_cli_method_flag_routes_to_flat_mode(self, tmp_path):
        for seed in (42, 43):
            _make_indicators_csv(
                tmp_path / f"seed{seed}",
                seed=seed,
                rows=[("KPS-F", "ALL", 0.10, 0.0, 50)],
            )
        from coldddi.eval.aggregate import main

        rc = main([
            "kps",
            "--runs", str(tmp_path),
            "--method", "llama-1b_P4",
            "--out", str(tmp_path / "out.csv"),
        ])
        assert rc == 0
        out = pd.read_csv(tmp_path / "out.csv")
        assert (out["method"] == "llama-1b_P4").all()

    def test_method_and_methods_are_mutually_exclusive(self, tmp_path):
        """--method (singular, flat-mode label) and --methods (plural,
        nested-mode subset filter) are different concepts.  Mixing
        them is almost certainly a CLI typo, so reject up front."""
        from coldddi.eval.aggregate import main

        with pytest.raises(SystemExit):
            main([
                "kps",
                "--runs", str(tmp_path),
                "--method", "llama-1b_P4",
                "--methods", "deepddi", "tiger",
            ])


EXPS_SCRIPTS = (
    "sec5_overall.sh",
    "sec5_stratified.sh",
    "sec5_kps.sh",
    "sec5_masking.sh",
)


class TestExpsShellScriptsExist:
    @pytest.mark.parametrize("name", EXPS_SCRIPTS)
    def test_script_present_and_nonempty(self, name):
        p = REPO_ROOT / "exps" / name
        assert p.is_file(), f"missing {p}"
        assert p.stat().st_size > 100, f"{p} suspiciously empty"

    @pytest.mark.skipif(
        not shutil.which("bash"),
        reason="bash unavailable on this host",
    )
    @pytest.mark.parametrize("name", EXPS_SCRIPTS)
    def test_bash_syntax_valid(self, name):
        p = REPO_ROOT / "exps" / name
        result = subprocess.run(
            ["bash", "-n", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{name} has bash syntax errors:\n{result.stderr}"
        )

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows filesystem doesn't expose exec bit reliably",
    )
    @pytest.mark.parametrize("name", EXPS_SCRIPTS)
    def test_executable_bit_set(self, name):
        p = REPO_ROOT / "exps" / name
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} not executable (chmod +x missing)"
