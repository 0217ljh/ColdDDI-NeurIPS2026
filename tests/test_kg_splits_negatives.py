"""Test KG, splits, negatives, and PairDataset composition.

Expected counts use the filtered 100-drug toy XML fixture with seed 42.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_FILTERED = REPO_ROOT / "data" / "public" / "intermediate" / "filtered"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not TOY_FILTERED.is_dir(),
    reason=f"Toy filtered dir not found at {TOY_FILTERED}",
)


# Fixtures


@pytest.fixture(scope="module")
def toy_kg():
    from coldddi.data.kg import KnowledgeGraph

    return KnowledgeGraph.from_filtered_dir(TOY_FILTERED)


@pytest.fixture(scope="module")
def toy_edges() -> pd.DataFrame:
    return pd.read_csv(TOY_FILTERED / "ddi_edges.csv")


@pytest.fixture(scope="module")
def toy_splits(toy_edges):
    from coldddi.data.splits import build_splits

    return build_splits(toy_edges, seed=42, drug_ratio=1.5, val_ratio=0.1)


# KnowledgeGraph


class TestKnowledgeGraph:
    def test_loaded_table_sizes_nonzero(self, toy_kg):
        # Filtered toy KG counts: 322 / 289 / 162 / 38 / 29.
        assert len(toy_kg.enzymes) == 322
        assert len(toy_kg.targets) == 289
        assert len(toy_kg.transporters) == 162
        assert len(toy_kg.carriers) == 38
        assert len(toy_kg.pathways) == 29

    def test_neighbors_returns_known_drug(self, toy_kg, toy_edges):
        # The post-filter toy pool includes entity neighbors for this drug.
        a_first = str(toy_edges["drug_a_id"].iloc[0])
        result = toy_kg.neighbors(a_first)
        assert isinstance(result, pd.DataFrame)
        # The first toy drug is approved + small molecule, so at least
        # one of the five tables should mention it.
        if not result.empty:
            assert (result["drugbank_id"] == a_first).all()

    def test_neighbors_filtered_by_edge_type(self, toy_kg):
        # Every row whose edge_type is set must match the requested set.
        for drug in toy_kg.drug_ids:
            r = toy_kg.neighbors(drug, edge_types=["enzyme"])
            if not r.empty:
                assert (r["edge_type"] == "enzyme").all()
                break

    def test_neighbors_unknown_edge_type_raises(self, toy_kg):
        from coldddi.data.kg import KnowledgeGraph

        with pytest.raises(ValueError, match="Unknown edge_types"):
            toy_kg.neighbors("DB00001", edge_types=["bogus_type"])

    def test_name_dict_legacy_shape(self, toy_kg):
        d = toy_kg.name_dict("enzyme")
        # {drug_id: [name, ...]} structure
        assert isinstance(d, dict)
        if d:
            sample_key = next(iter(d))
            assert isinstance(d[sample_key], list)
            assert all(isinstance(x, str) for x in d[sample_key])

    def test_save_and_round_trip(self, toy_kg, tmp_path):
        from coldddi.data.kg import KnowledgeGraph

        out = tmp_path / "kg_roundtrip"
        toy_kg.save(out)
        en = pd.read_parquet(out / "drug_enzymes.parquet")
        assert len(en) == len(toy_kg.enzymes)
        assert set(en.columns) == set(toy_kg.enzymes.columns)


# Splits


class TestSplits:
    def test_drug_partition_split_by_ratio(self, toy_splits):
        # drug_ratio=1.5 → roughly 67% / 33%.
        total = len(toy_splits.g1_drugs) + len(toy_splits.g2_drugs)
        assert total >= 80  # toy retains 86 drugs after Step 7
        # Allow a wide tolerance because of the integer division.
        ratio = len(toy_splits.g1_drugs) / total
        assert 0.55 < ratio < 0.75

    def test_seven_splits_present(self, toy_splits):
        for name in (
            "train", "val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2",
        ):
            df = getattr(toy_splits, name)
            assert isinstance(df, pd.DataFrame)
            assert {"drug_a_id", "drug_b_id"}.issubset(df.columns)

    def test_train_drugs_are_g1g1(self, toy_splits):
        g1 = set(toy_splits.g1_drugs)
        a = toy_splits.train["drug_a_id"].astype(str)
        b = toy_splits.train["drug_b_id"].astype(str)
        assert a.isin(g1).all()
        assert b.isin(g1).all()

    def test_single_train_disjoint_from_every_split(self, toy_splits):
        """Canonical training pairs are disjoint from all S0/S1/S2 val/test splits."""

        def canonical_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
            out: set[tuple[str, str]] = set()
            for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
                if a > b:
                    a, b = b, a
                out.add((a, b))
            return out

        train = canonical_pairs(toy_splits.train)
        for name in ("val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"):
            other = canonical_pairs(getattr(toy_splits, name))
            assert train.isdisjoint(other), f"train overlaps {name}"

    def test_test_s2_drugs_are_g2g2(self, toy_splits):
        g2 = set(toy_splits.g2_drugs)
        if len(toy_splits.test_s2) == 0:
            pytest.skip("toy fixture has no G2×G2 edges (sample too small)")
        a = toy_splits.test_s2["drug_a_id"].astype(str)
        b = toy_splits.test_s2["drug_b_id"].astype(str)
        assert a.isin(g2).all()
        assert b.isin(g2).all()

    def test_total_pair_count_does_not_exceed_input(self, toy_splits, toy_edges):
        # Count distinct positive pairs across all splits against the input edge set.
        seen: set[tuple[str, str]] = set()
        for _, df in toy_splits.items():
            for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
                if a > b:
                    a, b = b, a
                seen.add((a, b))
        all_input: set[tuple[str, str]] = set()
        for a, b in zip(toy_edges["drug_a_id"].astype(str), toy_edges["drug_b_id"].astype(str)):
            if a > b:
                a, b = b, a
            all_input.add((a, b))
        assert seen <= all_input

    def test_save_and_load_round_trip(self, toy_splits, tmp_path):
        from coldddi.data.splits import SplitFolds

        out = tmp_path / "splits_roundtrip"
        toy_splits.save(out)
        loaded = SplitFolds.from_dir(out)
        assert loaded.seed == toy_splits.seed
        assert loaded.g1_drugs == toy_splits.g1_drugs
        for name, df in toy_splits.items():
            assert len(getattr(loaded, name)) == len(df)


# Negatives


class TestNegatives:
    def test_uniform_sampler_is_deterministic(self, toy_splits):
        from coldddi.data.negatives import UniformNegativeSampler

        sampler = UniformNegativeSampler(
            drug_pool_a=toy_splits.g1_drugs, drug_pool_b=toy_splits.g1_drugs
        )
        a = sampler.sample(n_pairs=20, exclude=set(), seed=123)
        b = sampler.sample(n_pairs=20, exclude=set(), seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_uniform_sampler_excludes_known_pairs(self, toy_splits):
        from coldddi.data.negatives import UniformNegativeSampler

        sampler = UniformNegativeSampler(
            drug_pool_a=toy_splits.g1_drugs, drug_pool_b=toy_splits.g1_drugs
        )
        # Build an exclude set from the train positive edges
        exclude = set()
        for a, b in zip(
            toy_splits.train["drug_a_id"].astype(str),
            toy_splits.train["drug_b_id"].astype(str),
        ):
            if a > b:
                a, b = b, a
            exclude.add((a, b))

        out = sampler.sample(n_pairs=50, exclude=exclude, seed=7)
        for a, b in zip(out["drug_a_id"], out["drug_b_id"]):
            if a > b:
                a, b = b, a
            assert (a, b) not in exclude

    def test_static_negatives_keys(self, toy_splits):
        from coldddi.data.negatives import build_static_negatives

        out = build_static_negatives(toy_splits, base_seed=42)
        assert set(out.keys()) == {
            "test_s0", "val_s0", "test_s1", "val_s1", "test_s2", "val_s2",
        }
        # For non-empty splits the negatives should come back 1:1.
        for name, neg in out.items():
            pos = getattr(toy_splits, name)
            if len(pos) > 0:
                assert len(neg) == len(pos)


# PairDataset (legacy + modern paths)


class TestPairDatasetLegacy:
    """Legacy 800-drug pkl must still load through `from_pkl`."""

    LEGACY_PKL = (
        REPO_ROOT
        / "data"
        / "private"
        / "outputs_full"
        / "splits_legacy"
        / "800drug"
        / "latest_drugbank_ddi-Binary_cls-42+cold_start_split_fair_step-and-fair_negatives_step.pkl"
    )

    @pytest.mark.skipif(
        not (
            REPO_ROOT
            / "data"
            / "private"
            / "outputs_full"
            / "splits_legacy"
            / "800drug"
            / "latest_drugbank_ddi-Binary_cls-42+cold_start_split_fair_step-and-fair_negatives_step.pkl"
        ).is_file(),
        reason="Legacy 800drug pkl not present",
    )
    def test_legacy_pkl_loads(self):
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(self.LEGACY_PKL)
        assert ds.legacy_bundle is not None
        # 800-drug bundle has 7 splits; train must be non-empty.
        assert len(ds.splits.train) > 0

    @pytest.mark.skipif(
        not (
            REPO_ROOT
            / "data"
            / "private"
            / "outputs_full"
            / "splits_legacy"
            / "800drug"
            / "latest_drugbank_ddi-Binary_cls-42+cold_start_split_fair_step-and-fair_negatives_step.pkl"
        ).is_file(),
        reason="Legacy 800drug pkl not present",
    )
    def test_legacy_train_negatives_accessible(self):
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(self.LEGACY_PKL)
        # Legacy 800-drug bundle has at least one epoch of train negatives.
        epoch0 = ds.get_legacy_train_negatives(0)
        assert isinstance(epoch0, pd.DataFrame)
        assert len(epoch0) > 0


class TestPairDatasetModern:
    """`from_release_dir` end-to-end on the toy fixture."""

    def test_round_trip_from_release_dir(self, toy_splits, tmp_path):
        from coldddi.data.dataset import PairDataset

        # Build a minimal release-style directory layout in tmp_path.
        root = tmp_path / "release_root"
        (root / "filtered").mkdir(parents=True)
        for fname in (
            "drugs.csv",
            "ddi_edges.csv",
            "drug_enzymes.csv",
            "drug_targets.csv",
            "drug_transporters.csv",
            "drug_carriers.csv",
            "drug_pathways.csv",
        ):
            (root / "filtered" / fname).write_bytes((TOY_FILTERED / fname).read_bytes())

        toy_splits.save(root / "splits" / "seed42")

        ds = PairDataset.from_release_dir(root, seed=42)
        # The release directory's edges are the full toy filtered ddi_edges.csv;
        # the canonical train must be ≤ that count.
        assert len(ds.edges) > 0
        assert len(ds.splits.train) <= len(ds.edges)
        assert isinstance(ds.kg.name_dict("enzyme"), dict)
        # Static negatives should be cached on first build, and split-aware.
        g1, g2 = set(ds.splits.g1_drugs), set(ds.splits.g2_drugs)
        pool_check = {
            "test_s0": (g1, g1),
            "val_s0": (g1, g1),
            "test_s1": ({*g1, *g2}, {*g1, *g2}),  # cross — both pools are valid
            "val_s1": ({*g1, *g2}, {*g1, *g2}),
            "test_s2": (g2, g2),
            "val_s2": (g2, g2),
        }
        for name in ("test_s0", "val_s0", "test_s1", "val_s1", "test_s2", "val_s2"):
            neg = ds.get_negatives(name)
            pos = getattr(ds.splits, name)
            if len(pos) > 0:
                assert len(neg) == len(pos)
                allowed_a, allowed_b = pool_check[name]
                a = neg["drug_a_id"].astype(str)
                b = neg["drug_b_id"].astype(str)
                # Both endpoints must be in the union of the allowed pools
                # (canonicalization may have flipped a/b).
                allowed = allowed_a | allowed_b
                assert a.isin(allowed).all()
                assert b.isin(allowed).all()

    def test_regenerate_negatives_path(self, toy_splits, tmp_path):
        from coldddi.data.dataset import PairDataset

        root = tmp_path / "release_root"
        (root / "filtered").mkdir(parents=True)
        for fname in (
            "drugs.csv",
            "ddi_edges.csv",
            "drug_enzymes.csv",
            "drug_targets.csv",
            "drug_transporters.csv",
            "drug_carriers.csv",
            "drug_pathways.csv",
        ):
            (root / "filtered" / fname).write_bytes((TOY_FILTERED / fname).read_bytes())
        toy_splits.save(root / "splits" / "seed42")

        ds_a = PairDataset.from_release_dir(root, seed=42, regenerate_negatives=False)
        ds_b = PairDataset.from_release_dir(root, seed=42, regenerate_negatives=True)
        # Both paths must yield the *same* deterministic negatives because
        # `build_static_negatives` is seeded by base_seed + PHASE_OFFSETS.
        for name in PHASE_OFFSETS_KEYS:
            a = ds_a.negatives_by_split.get(name)
            b = ds_b.negatives_by_split.get(name)
            if a is None or b is None:
                continue
            pd.testing.assert_frame_equal(
                a.sort_values(["drug_a_id", "drug_b_id"]).reset_index(drop=True),
                b.sort_values(["drug_a_id", "drug_b_id"]).reset_index(drop=True),
            )


# Module-level constant used in the parametrized test above.
PHASE_OFFSETS_KEYS = ("test_s0", "val_s0", "test_s1", "val_s1", "test_s2", "val_s2")


# Train negatives, separate from val/test negatives.


class TestTrainNegatives:
    def test_build_train_negatives_size_matches_train(self, toy_splits):
        from coldddi.data.negatives import build_train_negatives

        out = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        # 1:1 negative-to-positive ratio against the canonical train set
        assert len(out) == len(toy_splits.train)
        assert list(out.columns) == ["drug_a_id", "drug_b_id"]

    def test_train_negatives_pool_is_g1_only(self, toy_splits):
        """Train negatives sample from G1×G1 — never from G2."""
        from coldddi.data.negatives import build_train_negatives

        out = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        g1 = set(toy_splits.g1_drugs)
        a = out["drug_a_id"].astype(str)
        b = out["drug_b_id"].astype(str)
        assert a.isin(g1).all()
        assert b.isin(g1).all()

    def test_train_negatives_excludes_positives(self, toy_splits):
        """No sampled train negative may collide with any positive in any split."""
        from coldddi.data.negatives import build_train_negatives

        positives: set[tuple[str, str]] = set()
        for _, df in toy_splits.items():
            for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
                if a > b:
                    a, b = b, a
                positives.add((a, b))

        out = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        for a, b in zip(out["drug_a_id"], out["drug_b_id"]):
            if a > b:
                a, b = b, a
            assert (a, b) not in positives

    def test_train_negatives_seed_determinism(self, toy_splits):
        """Same (base_seed, epoch) must yield identical draws."""
        from coldddi.data.negatives import build_train_negatives

        a = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        b = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        pd.testing.assert_frame_equal(a, b)

    def test_train_negatives_different_epochs_differ(self, toy_splits):
        """Different epochs from the same base_seed must yield different draws."""
        from coldddi.data.negatives import build_train_negatives

        e0 = build_train_negatives(toy_splits, base_seed=42, epoch=0)
        e1 = build_train_negatives(toy_splits, base_seed=42, epoch=1)
        assert not e0.equals(e1)

    def test_build_epochs_returns_n_dataframes(self, toy_splits):
        from coldddi.data.negatives import build_train_negatives_epochs

        epochs = build_train_negatives_epochs(toy_splits, base_seed=42, n_epochs=3)
        assert len(epochs) == 3
        for df in epochs:
            assert len(df) == len(toy_splits.train)

    def test_train_seed_disjoint_from_phase_offsets(self):
        """Training sub-seed base must not collide with val/test PHASE_OFFSETS."""
        from coldddi.data.negatives import PHASE_OFFSETS, TRAIN_NEGATIVES_SEED_BASE

        # PHASE_OFFSETS values are 100..600; TRAIN_NEGATIVES_SEED_BASE is 1000+
        # so train_neg sub-seeds (base + 1000+epoch) cannot collide.
        assert TRAIN_NEGATIVES_SEED_BASE > max(PHASE_OFFSETS.values())

    def test_train_subseeds_disjoint_from_val_test_subseeds_real(self):
        """Concrete (base_seed × epoch) sub-seed sets must not intersect
        the val/test sub-seeds for the seeds shipped in the paper."""
        from coldddi.data.negatives import PHASE_OFFSETS, TRAIN_NEGATIVES_SEED_BASE

        for base_seed in (42, 43, 44):
            train_subseeds = {
                base_seed + TRAIN_NEGATIVES_SEED_BASE + e for e in range(64)
            }
            valtest_subseeds = {base_seed + off for off in PHASE_OFFSETS.values()}
            assert train_subseeds.isdisjoint(valtest_subseeds), (
                f"sub-seed collision at base_seed={base_seed}: "
                f"{train_subseeds & valtest_subseeds}"
            )

    def test_pair_dataset_get_train_negatives_pre_baked(self, toy_splits, tmp_path):
        """PairDataset prefers pre-baked train negatives when present."""
        from coldddi.data.dataset import PairDataset
        from coldddi.data.negatives import build_train_negatives_epochs

        root = tmp_path / "release_root"
        (root / "filtered").mkdir(parents=True)
        for fname in (
            "drugs.csv", "ddi_edges.csv",
            "drug_enzymes.csv", "drug_targets.csv",
            "drug_transporters.csv", "drug_carriers.csv", "drug_pathways.csv",
        ):
            (root / "filtered" / fname).write_bytes((TOY_FILTERED / fname).read_bytes())
        toy_splits.save(root / "splits" / "seed42")

        # Pre-bake 2 epochs of train negatives
        epochs = build_train_negatives_epochs(toy_splits, base_seed=42, n_epochs=2)
        train_neg_dir = root / "splits" / "seed42" / "train_negatives"
        train_neg_dir.mkdir()
        for i, df in enumerate(epochs):
            df.to_parquet(train_neg_dir / f"epoch_{i}.parquet", index=False)

        ds = PairDataset.from_release_dir(root, seed=42)
        assert len(ds.train_negatives_epochs) == 2
        # Loaded epoch 0 must match the in-memory one
        pd.testing.assert_frame_equal(
            ds.get_train_negatives(0).reset_index(drop=True),
            epochs[0].reset_index(drop=True),
        )


# Legacy bundle KG: must build a real KnowledgeGraph from `my_X_list`


_LEGACY_PKL = (
    REPO_ROOT
    / "data" / "private" / "outputs_full" / "splits_legacy" / "800drug"
    / "latest_drugbank_ddi-Binary_cls-42+cold_start_split_fair_step-and-fair_negatives_step.pkl"
)


@pytest.mark.skipif(
    not _LEGACY_PKL.is_file(), reason="Legacy 800drug pkl not present"
)
class TestLegacyKG:
    def test_legacy_kb_my_x_list_becomes_real_kg(self):
        """Promote legacy my_X_list tables to a populated KnowledgeGraph."""
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        # Concrete type, not adapter (the regression we are guarding).
        assert isinstance(ds.kg, KnowledgeGraph)
        # All five entity tables must have rows.
        assert len(ds.kg.enzymes) > 0
        assert len(ds.kg.targets) > 0
        assert len(ds.kg.transporters) > 0
        assert len(ds.kg.carriers) > 0
        assert len(ds.kg.pathways) > 0

    def test_legacy_kg_drug_ids_nonempty(self):
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        assert len(ds.kg.drug_ids) > 0

    def test_legacy_kg_name_dict_returns_lists(self):
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        d = ds.kg.name_dict("enzyme")
        assert isinstance(d, dict)
        assert len(d) > 0
        sample_value = next(iter(d.values()))
        assert isinstance(sample_value, list)

    def test_legacy_get_train_negatives_round_trip(self):
        """Reuse legacy train_neg_epochs without resampling.

        The modern API keeps only pair columns; their rows must match the
        legacy API, which preserves all original columns.
        """
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        modern = ds.get_train_negatives(0)
        legacy = ds.get_legacy_train_negatives(0)
        assert len(modern) == len(legacy) > 0
        assert list(modern.columns) == ["drug_a_id", "drug_b_id"]
        # The modern API's content must equal the legacy bundle's pair
        # coordinates row-for-row (no silent re-sampling).
        pd.testing.assert_frame_equal(
            modern.reset_index(drop=True),
            legacy.loc[:, ["drug_a_id", "drug_b_id"]].reset_index(drop=True),
        )


# Edge-case path: legacy KB with the older `dbid_2_X` schema


def test_legacy_kg_dbid_format_falls_back_to_adapter():
    """Older bundles stored kb as flat `dbid_2_X` dicts; the loader must
    still produce a usable KG object (the adapter)."""
    from coldddi.data.dataset import _build_kg_from_legacy_kb, _LegacyKGAdapter

    legacy_kb = {
        "dbid_2_enzymes": {"DB00001": ["CYP3A4"], "DB00002": ["CYP2D6"]},
        "dbid_2_targets": {"DB00001": ["Thrombin"]},
    }
    kg = _build_kg_from_legacy_kb(legacy_kb)
    assert isinstance(kg, _LegacyKGAdapter)
    assert kg.drug_ids == {"DB00001", "DB00002"}
    assert kg.name_dict("enzyme") == legacy_kb["dbid_2_enzymes"]


# PairDataset API regressions.


class TestGetTrainNegativesContract:
    """Lock the public contract of :meth:`PairDataset.get_train_negatives`."""

    def test_modern_path_returns_two_columns(self, toy_splits, tmp_path):
        """The modern API returns only drug_a_id and drug_b_id columns."""
        from coldddi.data.dataset import PairDataset
        from coldddi.data.negatives import build_train_negatives_epochs

        root = tmp_path / "release_root"
        (root / "filtered").mkdir(parents=True)
        for fname in (
            "drugs.csv", "ddi_edges.csv",
            "drug_enzymes.csv", "drug_targets.csv",
            "drug_transporters.csv", "drug_carriers.csv", "drug_pathways.csv",
        ):
            (root / "filtered" / fname).write_bytes((TOY_FILTERED / fname).read_bytes())
        toy_splits.save(root / "splits" / "seed42")

        # Pre-bake one epoch
        epochs = build_train_negatives_epochs(toy_splits, base_seed=42, n_epochs=1)
        train_neg_dir = root / "splits" / "seed42" / "train_negatives"
        train_neg_dir.mkdir()
        for i, df in enumerate(epochs):
            df.to_parquet(train_neg_dir / f"epoch_{i}.parquet", index=False)

        ds = PairDataset.from_release_dir(root, seed=42)
        out = ds.get_train_negatives(0)
        assert list(out.columns) == ["drug_a_id", "drug_b_id"]

    @pytest.mark.skipif(
        not _LEGACY_PKL.is_file(), reason="Legacy 800drug pkl not present"
    )
    def test_legacy_path_returns_two_columns(self):
        """Legacy train negatives also return only canonical pair columns."""
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        out = ds.get_train_negatives(0)
        assert list(out.columns) == ["drug_a_id", "drug_b_id"]

    @pytest.mark.skipif(
        not _LEGACY_PKL.is_file(), reason="Legacy 800drug pkl not present"
    )
    def test_legacy_epoch_out_of_range_falls_through(self):
        """Sample fresh negatives when the requested legacy epoch is absent."""
        from coldddi.data.dataset import PairDataset

        ds = PairDataset.from_pkl(_LEGACY_PKL)
        n_legacy = len(ds.legacy_bundle.extra.get("train_neg_epochs", []))
        # Request an epoch one past the legacy bundle's range.
        out = ds.get_train_negatives(epoch=n_legacy + 5)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["drug_a_id", "drug_b_id"]
        assert len(out) == len(ds.splits.train)


class TestLegacyKBStrictSchema:
    """Legacy KG promotion requires all five tables with valid columns."""

    def test_partial_my_x_list_falls_back_to_adapter(self):
        from coldddi.data.dataset import _build_kg_from_legacy_kb, _LegacyKGAdapter

        # Only 3 of 5 tables present — must fall back to adapter, not
        # silently build a KG with empty carriers/pathways.
        kb = {
            "my_enzyme_list": pd.DataFrame(
                {"drugbank_id": ["DB1"], "enzyme_id": ["E1"], "enzyme_name": ["X"], "organism": ["Humans"], "action": [""]}
            ),
            "my_target_list": pd.DataFrame(
                {"drugbank_id": ["DB1"], "target_id": ["T1"], "target_name": ["Y"], "organism": ["Humans"], "action": [""]}
            ),
            "my_transporter_list": pd.DataFrame(
                {"drugbank_id": ["DB1"], "transporter_id": ["TR1"], "transporter_name": ["Z"], "organism": ["Humans"], "action": [""]}
            ),
            # missing my_carrier_list and my_pathway_list
        }
        kg = _build_kg_from_legacy_kb(kb)
        assert isinstance(kg, _LegacyKGAdapter)

    def test_my_x_list_with_wrong_columns_falls_back(self):
        from coldddi.data.dataset import _build_kg_from_legacy_kb, _LegacyKGAdapter

        kb = {
            "my_enzyme_list": pd.DataFrame({"foo": [], "bar": []}),
            "my_target_list": pd.DataFrame(),
            "my_transporter_list": pd.DataFrame(),
            "my_carrier_list": pd.DataFrame(),
            "my_pathway_list": pd.DataFrame(),
        }
        kg = _build_kg_from_legacy_kb(kb)
        assert isinstance(kg, _LegacyKGAdapter)


class TestLegacyKGAdapterRepr:
    """Adapter repr stays short even for large KB dictionaries."""

    def test_repr_truncates_for_many_keys(self):
        from coldddi.data.dataset import _LegacyKGAdapter

        # Simulate a drug-id-keyed dict with 5,000 keys
        big_kb = {f"DB{i:05d}": ["x"] for i in range(5000)}
        adapter = _LegacyKGAdapter(big_kb)
        r = repr(adapter)
        # Must encode the size and a small sample, not all 5000 keys
        assert "n_keys=5000" in r
        assert len(r) < 300

    def test_repr_short_for_small_kb(self):
        from coldddi.data.dataset import _LegacyKGAdapter

        adapter = _LegacyKGAdapter({"dbid_2_enzymes": {}, "dbid_2_targets": {}})
        r = repr(adapter)
        assert "keys=" in r
        assert "dbid_2_enzymes" in r
