"""End-to-end smoke tests for `coldddi.reconstruct`.

Runs the full driver against the public toy XML and asserts every stage
produces the expected on-disk artifacts with the locked-in toy numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_XML = REPO_ROOT / "data" / "public" / "drugbank_toy.xml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not TOY_XML.exists(),
    reason=f"Toy XML not found at {TOY_XML}",
)


@pytest.fixture(scope="module")
def toy_recon_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run `run_reconstruction` once on the toy XML, share output across tests."""
    from coldddi.reconstruct import run_reconstruction

    pytest.importorskip("rdkit")  # filter step needs rdkit

    out = tmp_path_factory.mktemp("reconstruct")
    run_reconstruction(
        drugbank=TOY_XML,
        output=out,
        seeds=(42, 43),
        release_mode="sample",
        n_train_negative_epochs=2,
        skip_stages=(),
        full_pkpd_csv=None,
        quiet=True,
    )
    return out


class TestReconstructEndToEnd:
    def test_stage1a_outputs(self, toy_recon_root):
        raw = toy_recon_root / "intermediate" / "raw"
        for name in (
            "drugs.csv", "ddi_edges.csv",
            "drug_enzymes.csv", "drug_targets.csv",
            "drug_transporters.csv", "drug_carriers.csv", "drug_pathways.csv",
        ):
            assert (raw / name).is_file(), f"missing {name}"

    def test_stage1b_outputs_filtered_csvs_and_stats(self, toy_recon_root):
        filtered = toy_recon_root / "intermediate" / "filtered"
        assert (filtered / "drugs.csv").is_file()
        assert (filtered / "stats.json").is_file()
        drugs = pd.read_csv(filtered / "drugs.csv")
        edges = pd.read_csv(filtered / "ddi_edges.csv")
        assert len(drugs) == 86
        assert len(edges) == 1383

    def test_stage2_outputs_enriched(self, toy_recon_root):
        enriched = toy_recon_root / "intermediate" / "enriched"
        assert (enriched / "ddi_pk_pd_labels.csv").is_file()
        assert (enriched / "ddi_key_entities.csv").is_file()
        assert (enriched / "ddi_key_entities_type_summary.csv").is_file()
        assert (enriched / "mediating_entities.parquet").is_file()
        assert (enriched / "action_pairs.parquet").is_file()
        labels = pd.read_csv(enriched / "ddi_pk_pd_labels.csv")
        ke = pd.read_csv(enriched / "ddi_key_entities.csv")
        assert len(labels) == 24
        assert len(ke) == 1383
        assert int(ke["has_key_entity"].sum()) == 523

    def test_stage3_outputs_sample_parquets(self, toy_recon_root):
        # sample mode → annotations_sample/
        out = toy_recon_root / "annotations_sample"
        assert (out / "pkpd.parquet").is_file()
        assert (out / "ab_sample.parquet").is_file()
        assert (out / "mediating_entities_sample.parquet").is_file()
        assert (out / "action_pairs_sample.parquet").is_file()
        ab = pd.read_parquet(out / "ab_sample.parquet")
        assert len(ab) == 1383
        assert ab["has_key_entity"].dtype == bool

    def test_stage4_outputs_each_seed(self, toy_recon_root):
        for seed in (42, 43):
            seed_dir = toy_recon_root / "intermediate" / "splits" / f"seed{seed}"
            assert (seed_dir / "manifest.json").is_file()
            for name in (
                "train", "val_s0", "val_s1", "val_s2",
                "test_s0", "test_s1", "test_s2",
            ):
                assert (seed_dir / f"{name}.parquet").is_file()
            for name in ("test_s0", "val_s0", "test_s1", "val_s1", "test_s2", "val_s2"):
                assert (seed_dir / "negatives" / f"{name}.parquet").is_file()
            # 2 train_negative epochs were requested
            for i in range(2):
                assert (seed_dir / "train_negatives" / f"epoch_{i}.parquet").is_file()

    def test_stage4_no_data_leakage(self, toy_recon_root):
        """Training pairs are disjoint from every val/test split for every seed."""
        from coldddi.data.splits import SplitFolds

        for seed in (42, 43):
            seed_dir = toy_recon_root / "intermediate" / "splits" / f"seed{seed}"
            sf = SplitFolds.from_dir(seed_dir)

            def canon(df):
                out = set()
                for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
                    if a > b:
                        a, b = b, a
                    out.add((a, b))
                return out

            train = canon(sf.train)
            for name in ("val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"):
                other = canon(getattr(sf, name))
                assert train.isdisjoint(other), f"seed{seed}: train overlaps {name}"

    def test_stage4_manifest_records_seed(self, toy_recon_root):
        manifest = json.loads(
            (toy_recon_root / "intermediate" / "splits" / "seed42" / "manifest.json").read_text()
        )
        assert manifest["seed"] == 42
        assert "g1_drugs" in manifest and "g2_drugs" in manifest


class TestReconstructSkipStages:
    def test_skip_stage4(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        run_reconstruction(
            drugbank=TOY_XML,
            output=tmp_path,
            seeds=(42,),
            release_mode="sample",
            n_train_negative_epochs=1,
            skip_stages=("stage4",),
            full_pkpd_csv=None,
            quiet=True,
        )
        # Stage 1-3 outputs are present
        assert (tmp_path / "intermediate" / "raw" / "drugs.csv").is_file()
        assert (tmp_path / "intermediate" / "filtered" / "drugs.csv").is_file()
        assert (tmp_path / "intermediate" / "enriched" / "ddi_pk_pd_labels.csv").is_file()
        # Stage 4 splits dir does NOT exist
        assert not (tmp_path / "intermediate" / "splits").exists()

    def test_unknown_skip_stage_raises(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        with pytest.raises(ValueError, match="Unknown stages"):
            run_reconstruction(
                drugbank=TOY_XML,
                output=tmp_path,
                seeds=(42,),
                skip_stages=("typo_stage",),
                quiet=True,
            )


class TestReconstructCLI:
    """Verify the argparse layer accepts the documented signatures."""

    def test_toy_shortcut_accepted(self, tmp_path, monkeypatch):
        from coldddi.reconstruct import main

        monkeypatch.chdir(REPO_ROOT)
        rc = main(
            [
                "--toy",
                "--output", str(tmp_path),
                "--seeds", "42",
                "--release-mode", "sample",
                "--n-train-negative-epochs", "0",
                "--quiet",
            ]
        )
        assert rc == 0
        assert (tmp_path / "intermediate" / "filtered" / "drugs.csv").is_file()

    def test_drugbank_and_toy_mutually_exclusive(self, tmp_path):
        from coldddi.reconstruct import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--toy", "--drugbank", "x.xml", "--output", str(tmp_path)])

    def test_neither_drugbank_nor_toy_fails(self, tmp_path):
        """argparse rejects runs without a data source."""
        from coldddi.reconstruct import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--output", str(tmp_path)])

    def test_toy_path_is_repo_relative(self, tmp_path, monkeypatch):
        """--toy works from any working directory."""
        from coldddi.reconstruct import main

        # Run from an arbitrary unrelated cwd
        monkeypatch.chdir(tmp_path)
        rc = main([
            "--toy",
            "--output", str(tmp_path / "out"),
            "--seeds", "42",
            "--n-train-negative-epochs", "0",
            "--quiet",
        ])
        assert rc == 0
        assert (tmp_path / "out" / "intermediate" / "filtered" / "drugs.csv").is_file()

    def test_toy_default_release_mode_is_sample(self, tmp_path, monkeypatch):
        """--toy defaults to sample mode, keeping toy artifacts out of outputs_full/."""
        from coldddi.reconstruct import main

        monkeypatch.chdir(REPO_ROOT)
        rc = main([
            "--toy",
            "--output", str(tmp_path),
            "--seeds", "42",
            "--n-train-negative-epochs", "0",
            "--quiet",
        ])
        assert rc == 0
        # Sample mode → annotations_sample/, NOT outputs_full/annotations/
        assert (tmp_path / "annotations_sample" / "ab_sample.parquet").is_file()
        assert not (tmp_path / "outputs_full" / "annotations").exists()

    def test_negative_train_epochs_rejected(self, tmp_path):
        """--n-train-negative-epochs cannot be negative."""
        from coldddi.reconstruct import run_reconstruction

        with pytest.raises(ValueError, match="must be >= 0"):
            run_reconstruction(
                drugbank=TOY_XML,
                output=tmp_path,
                seeds=(42,),
                n_train_negative_epochs=-1,
                quiet=True,
            )


class TestReconstructPreflight:
    """Reject skipped stages whose required outputs are missing before doing work."""

    def test_skip_producer_with_missing_output_raises(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        # Skip stage1b (the producer of intermediate/filtered/) but try
        # to run stage2a, which needs that directory. Output dir is empty.
        with pytest.raises(FileNotFoundError, match="stage1b"):
            run_reconstruction(
                drugbank=TOY_XML,
                output=tmp_path,
                seeds=(42,),
                n_train_negative_epochs=0,
                skip_stages=("stage1a", "stage1b"),
                quiet=True,
            )


class TestReconstructFullMode:
    """Verify the full-mode output layout."""

    def test_full_mode_writes_outputs_full_annotations(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        run_reconstruction(
            drugbank=TOY_XML,
            output=tmp_path,
            seeds=(42,),
            release_mode="full",
            n_train_negative_epochs=0,
            quiet=True,
        )
        annotations = tmp_path / "outputs_full" / "annotations"
        for fname in (
            "pkpd.parquet",
            "ab.parquet",
            "mediating_entities.parquet",
            "action_pairs.parquet",
        ):
            assert (annotations / fname).is_file(), f"missing {fname}"
        # No `_sample` suffix in full mode
        assert not (annotations / "ab_sample.parquet").exists()


class TestReconstructIdempotent:
    """Reruns preserve outputs and remove stale per-epoch parquets."""

    def test_rerun_with_fewer_epochs_cleans_old(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        # First run: 3 epochs
        run_reconstruction(
            drugbank=TOY_XML, output=tmp_path,
            seeds=(42,), release_mode="sample",
            n_train_negative_epochs=3, quiet=True,
        )
        train_neg = tmp_path / "intermediate" / "splits" / "seed42" / "train_negatives"
        assert {p.name for p in train_neg.glob("epoch_*.parquet")} == {
            "epoch_0.parquet", "epoch_1.parquet", "epoch_2.parquet"
        }

        # Second run: 1 epoch — old epoch_1 / epoch_2 must be cleaned
        run_reconstruction(
            drugbank=TOY_XML, output=tmp_path,
            seeds=(42,), release_mode="sample",
            n_train_negative_epochs=1, quiet=True,
        )
        assert {p.name for p in train_neg.glob("epoch_*.parquet")} == {"epoch_0.parquet"}

    def test_rerun_same_seed_yields_identical_splits(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        for out in (out_a, out_b):
            run_reconstruction(
                drugbank=TOY_XML, output=out,
                seeds=(42,), release_mode="sample",
                n_train_negative_epochs=1, quiet=True,
            )
        a = pd.read_parquet(out_a / "intermediate" / "splits" / "seed42" / "test_s2.parquet")
        b = pd.read_parquet(out_b / "intermediate" / "splits" / "seed42" / "test_s2.parquet")
        pd.testing.assert_frame_equal(a, b)


class TestReconstructLoggerStream:
    """Reconstruction progress goes to stderr, leaving stdout pipe-safe."""

    def test_logger_writes_to_stderr(self, tmp_path, capfd):
        from coldddi.reconstruct import run_reconstruction

        run_reconstruction(
            drugbank=TOY_XML, output=tmp_path,
            seeds=(42,), release_mode="sample",
            n_train_negative_epochs=0, quiet=False,
        )
        out, err = capfd.readouterr()
        assert "[reconstruct]" in err
        # Neither driver nor stage progress may leak to stdout.
        assert out == "", f"stdout should be empty for piping, got: {out!r}"


class TestReconstructFullPkpdOverride:
    """--full-pkpd selects the PK/PD source in sample mode."""

    def test_full_pkpd_csv_takes_priority(self, tmp_path):
        from coldddi.reconstruct import run_reconstruction

        # Construct a "full" pkpd CSV with a sentinel row count.
        full_pkpd = tmp_path / "synth_pkpd.csv"
        pd.DataFrame(
            {
                "ddi_type": [f"t{i}" for i in range(7)],
                "pk_pd_label": ["PD"] * 7,
                "matched_pk_keywords": [""] * 7,
                "matched_pd_keywords": [""] * 7,
            }
        ).to_csv(full_pkpd, index=False)

        run_reconstruction(
            drugbank=TOY_XML, output=tmp_path / "out",
            seeds=(42,), release_mode="sample",
            n_train_negative_epochs=0,
            full_pkpd_csv=full_pkpd, quiet=True,
        )
        out = pd.read_parquet(tmp_path / "out" / "annotations_sample" / "pkpd.parquet")
        assert len(out) == 7  # sentinel from --full-pkpd, not toy's 24
