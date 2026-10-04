"""Test evaluate.py against the Appendix A.6.2 CLI, lines 549-557.

Cover defaults, subset/data exclusivity, the adapter alias, and PKL/directory
dispatch through PairDataset.from_pkl/from_release_dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_RELEASE = REPO_ROOT / "data" / "public" / "intermediate"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# argparse default surfaces

class TestArgparseDefaults:
    def test_setting_default_is_all(self):
        """Appendix A.6.2 specifies --setting all by default."""
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--method", "deepddi",
            "--data", "/tmp/x",
            "--out", "/tmp/y",
        ])
        assert args.setting == "all"

    def test_subset_choices_match_paper(self):
        """Paper line 554: ``[--subset 800]`` — release ships the 800
        and 1900 subset shortcuts.  Toy fixtures use explicit ``--data``."""
        from coldddi.evaluate import SUBSET_PATHS, _build_parser

        # 800 + 1900 are the paper-spec shortcuts.
        assert set(SUBSET_PATHS) == {"800", "1900"}

        parser = _build_parser()
        # Valid subset values parse cleanly.
        for v in ("800", "1900"):
            args = parser.parse_args([
                "--method", "deepddi",
                "--subset", v,
                "--out", "/tmp/y",
            ])
            assert args.subset == v
        # Invalid subset rejected by argparse choices.
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--method", "deepddi",
                "--subset", "42",       # not a recognised shortcut
                "--out", "/tmp/y",
            ])

    def test_data_and_subset_both_optional_at_argparse_level(self):
        """main(), not argparse, validates --data/--subset for clearer errors."""
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        # Neither passed — parses successfully at argparse level.
        args = parser.parse_args([
            "--method", "deepddi",
            "--out", "/tmp/y",
        ])
        assert args.data is None and args.subset is None


# --adapter accepted as alias for --checkpoint

class TestAdapterAlias:
    def test_adapter_routes_to_checkpoint_dest(self):
        """The paper's --adapter alias stores its value in args.checkpoint."""
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        a1 = parser.parse_args([
            "--method", "deepddi", "--data", "/tmp/x", "--out", "/tmp/y",
            "--checkpoint", "/tmp/ckpt",
        ])
        a2 = parser.parse_args([
            "--method", "deepddi", "--data", "/tmp/x", "--out", "/tmp/y",
            "--adapter", "/tmp/ckpt",
        ])
        assert a1.checkpoint == Path("/tmp/ckpt")
        assert a2.checkpoint == Path("/tmp/ckpt")
        # Both flags share one destination.
        assert not hasattr(a1, "adapter")
        assert not hasattr(a2, "adapter")


# --subset resolver

class TestSubsetResolver:
    def test_subset_800_resolves_to_legacy_pkl_with_seed(self):
        from coldddi.evaluate import _resolve_subset

        path = _resolve_subset("800", seed=42, repo_root=Path("/repo"))
        s = str(path).replace("\\", "/")
        assert s.endswith(".pkl"), f"800-drug shortcut should resolve to a pkl, got {s}"
        assert "Binary_cls-42+" in s, "seed must be substituted into the pkl name"
        # Test another seed substitutes correctly.
        path43 = _resolve_subset("800", seed=43, repo_root=Path("/repo"))
        assert "Binary_cls-43+" in str(path43).replace("\\", "/")

    def test_subset_1900_resolves_to_release_dir(self):
        from coldddi.evaluate import _resolve_subset

        path = _resolve_subset("1900", seed=42, repo_root=Path("/repo"))
        s = str(path).replace("\\", "/")
        assert s.endswith("data/private/intermediate"), (
            f"1900-drug shortcut should resolve to release directory, got {s}"
        )
        # 1900 doesn't include seed in path — same dir, seed selects file inside.
        path43 = _resolve_subset("1900", seed=43, repo_root=Path("/repo"))
        assert path == path43


# main() mutual-exclusion + dispatch

class TestMainMutualExclusion:
    def test_both_data_and_subset_errors(self, tmp_path):
        from coldddi.evaluate import main

        with pytest.raises(SystemExit):
            main([
                "--method", "deepddi",
                "--data", str(tmp_path),
                "--subset", "800",
                "--out", str(tmp_path / "out"),
            ])

    def test_neither_data_nor_subset_errors(self, tmp_path):
        from coldddi.evaluate import main

        with pytest.raises(SystemExit):
            main([
                "--method", "deepddi",
                "--out", str(tmp_path / "out"),
            ])


# from_pkl path dispatch on .pkl extension

class TestPklDispatchInRunEvaluation:
    def test_pkl_path_routes_through_from_pkl(self, tmp_path, monkeypatch):
        """When ``data_dir`` has a ``.pkl`` extension, ``run_evaluation``
        must call ``PairDataset.from_pkl`` instead of ``from_release_dir``.
        This is what makes ``--subset 800`` work end-to-end."""
        from coldddi import evaluate
        from coldddi.data import dataset as dataset_mod

        called: dict[str, object] = {}

        def fake_from_pkl(p, **kw):
            called["from_pkl"] = Path(p)
            raise RuntimeError("dispatch verified — abort before fit()")

        def fake_from_release_dir(*a, **kw):
            called["from_release_dir"] = a
            raise RuntimeError("should not be called")

        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_pkl",
            classmethod(lambda cls, p, **kw: fake_from_pkl(p, **kw)),
        )
        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_release_dir",
            classmethod(lambda cls, *a, **kw: fake_from_release_dir(*a, **kw)),
        )

        fake_pkl = tmp_path / "bundle.pkl"
        fake_pkl.write_bytes(b"")     # not a real pkl, but path-shape is what we test
        with pytest.raises(RuntimeError, match="dispatch verified"):
            evaluate.run_evaluation(
                method="deepddi",
                data_dir=fake_pkl,
                seed=42,
                settings=["S2"],
                out_dir=tmp_path / "out",
                device="cpu",
                with_indicators=False,
            )
        assert called.get("from_pkl") == fake_pkl
        assert "from_release_dir" not in called

    def test_directory_path_routes_through_from_release_dir(
        self, tmp_path, monkeypatch,
    ):
        from coldddi import evaluate
        from coldddi.data import dataset as dataset_mod

        called: dict[str, object] = {}

        def fake_from_release_dir(p, **kw):
            called["from_release_dir"] = (Path(p), kw)
            raise RuntimeError("dispatch verified — abort before fit()")

        def fake_from_pkl(*a, **kw):
            called["from_pkl"] = a
            raise RuntimeError("should not be called")

        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_release_dir",
            classmethod(lambda cls, p, **kw: fake_from_release_dir(p, **kw)),
        )
        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_pkl",
            classmethod(lambda cls, *a, **kw: fake_from_pkl(*a, **kw)),
        )

        with pytest.raises(RuntimeError, match="dispatch verified"):
            evaluate.run_evaluation(
                method="deepddi",
                data_dir=tmp_path,            # a directory
                seed=42,
                settings=["S2"],
                out_dir=tmp_path / "out",
                device="cpu",
                with_indicators=False,
            )
        assert called["from_release_dir"][0] == tmp_path
        assert "from_pkl" not in called

    def test_missing_pkl_routes_to_from_pkl_not_silent_fallthrough(
        self, tmp_path, monkeypatch,
    ):
        """Missing PKLs still use from_pkl, preserving the relevant file-not-found error."""
        from coldddi import evaluate
        from coldddi.data import dataset as dataset_mod

        called: dict[str, object] = {}

        def fake_from_pkl(p, **kw):
            called["from_pkl"] = Path(p)
            raise FileNotFoundError(f"no such pkl: {p}")

        def fake_from_release_dir(*a, **kw):
            called["from_release_dir"] = a
            raise RuntimeError("must not be called for .pkl input")

        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_pkl",
            classmethod(lambda cls, p, **kw: fake_from_pkl(p, **kw)),
        )
        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_release_dir",
            classmethod(lambda cls, *a, **kw: fake_from_release_dir(*a, **kw)),
        )

        nonexistent = tmp_path / "missing_bundle.pkl"
        # ``nonexistent`` is NOT created — pure path-shape dispatch.
        with pytest.raises(FileNotFoundError):
            evaluate.run_evaluation(
                method="deepddi",
                data_dir=nonexistent,
                seed=42,
                settings=["S2"],
                out_dir=tmp_path / "out",
                device="cpu",
                with_indicators=False,
            )
        assert called["from_pkl"] == nonexistent
        assert "from_release_dir" not in called


# --device choices.

class TestDeviceChoices:
    def test_device_accepts_cuda_cpu_auto(self):
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        for v in ("cuda", "cpu", "auto"):
            args = parser.parse_args([
                "--method", "deepddi", "--data", "/tmp/x",
                "--out", "/tmp/y", "--device", v,
            ])
            assert args.device == v

    def test_device_rejects_freeform_string(self):
        """Accept only cuda/cpu per paper line 555; GPU indices use CUDA_VISIBLE_DEVICES."""
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--method", "deepddi", "--data", "/tmp/x",
                "--out", "/tmp/y", "--device", "cuda:0",
            ])
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--method", "deepddi", "--data", "/tmp/x",
                "--out", "/tmp/y", "--device", "CPU",
            ])


# Optional --out default.

class TestOutOptional:
    def test_out_is_not_required_by_argparse(self):
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        # No --out flag — parses successfully; default is None at
        # argparse level (main() fills in repo-rooted default).
        args = parser.parse_args([
            "--method", "deepddi", "--data", "/tmp/x",
        ])
        assert args.out is None

    def test_main_defaults_out_to_runs_subdir(self, tmp_path, monkeypatch):
        """When --out is omitted, ``main()`` writes under
        ``<repo_root>/runs/<method>/seed<N>/`` per paper A.6.2."""
        from coldddi import evaluate
        from coldddi.data import dataset as dataset_mod

        captured: dict[str, object] = {}

        def fake_run_evaluation(**kwargs):
            captured["out_dir"] = kwargs["out_dir"]
            return {}

        monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
        # Avoid touching the real dataset loaders.
        monkeypatch.setattr(
            dataset_mod.PairDataset, "from_release_dir",
            classmethod(lambda cls, *a, **kw: None),
        )

        rc = evaluate.main([
            "--method", "deepddi",
            "--data", str(tmp_path),
            "--seed", "42",
        ])
        assert rc == 0
        out_dir = captured["out_dir"]
        s = str(out_dir).replace("\\", "/")
        assert s.endswith("runs/deepddi/seed42"), (
            f"--out default should be runs/<method>/seed<N>/, got {s}"
        )


# Smoke: default --setting all on the toy fixture writes all 6 CSVs

class TestDefaultSettingAllSmoke:
    """Default --setting all writes six CSVs: val/test for S0, S1, and S2."""

    @pytest.mark.skipif(
        not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
        reason="toy fixture missing",
    )
    def test_main_default_setting_emits_six_csvs(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from coldddi.evaluate import main

        monkeypatch.setattr(sys, "argv", [
            "evaluate.py",
            "--method", "deepddi",
            "--data", str(TOY_RELEASE),
            "--seed", "42",
            "--out", str(tmp_path),
            "--device", "cpu",
            "--no-indicators",      # keep test fast; L6 is covered in e2e tests
            "--preset", "smoke",    # avoid paper-spec 100-epoch DeepDDI training
        ])
        rc = main([
            "--method", "deepddi",
            "--data", str(TOY_RELEASE),
            "--seed", "42",
            "--out", str(tmp_path),
            "--device", "cpu",
            "--no-indicators",
            "--preset", "smoke",
        ])
        assert rc == 0
        # No explicit --setting flag → default "all" → 6 CSVs.
        expected = {
            "predictions_val_s0_seed42.csv",
            "predictions_test_s0_seed42.csv",
            "predictions_val_s1_seed42.csv",
            "predictions_test_s1_seed42.csv",
            "predictions_val_s2_seed42.csv",
            "predictions_test_s2_seed42.csv",
        }
        present = {p.name for p in tmp_path.glob("predictions_*.csv")}
        missing = expected - present
        assert not missing, (
            f"--setting default 'all' did not emit all 6 per-split CSVs; "
            f"missing: {sorted(missing)}"
        )
