"""Test fair and legacy splitting against paper Table B.1.

Fair is the default: G1 has int(N * g1_ratio) drugs, S0 uses 90/5/5,
and S1/S2 use 50/50. Explicit drug_ratio selects legacy semantics.
Manifests preserve protocol/ratio; missing protocol fields mean legacy.
Full-data counts are checked when reconstructed DrugBank data is available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_EDGES_CSV = REPO_ROOT / "data" / "private" / "intermediate" / "filtered" / "ddi_edges.csv"
TOY_EDGES_CSV = REPO_ROOT / "data" / "public" / "intermediate" / "filtered" / "ddi_edges.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Default protocol is "fair"


class TestDefaultProtocolIsFair:
    def test_module_constants(self):
        from coldddi.data.splits import (
            DEFAULT_G1_RATIO,
            DEFAULT_PROTOCOL,
            FAIR_S0_TRAIN_SIZE,
            FAIR_VAL_FRACTION,
            PROTOCOLS,
        )

        assert DEFAULT_PROTOCOL == "fair"
        assert DEFAULT_G1_RATIO == 0.8
        assert FAIR_S0_TRAIN_SIZE == 0.9
        assert FAIR_VAL_FRACTION == 0.5
        assert set(PROTOCOLS) == {"fair", "legacy"}

    def test_build_splits_no_kwargs_uses_fair(self):
        from coldddi.data.splits import build_splits

        # Synthetic edges that exercise all three pools.
        # 20 drugs → with g1_ratio=0.8, |G1|=16, |G2|=4.
        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(20) for b in range(a + 1, 20)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42)
        assert s.protocol == "fair"
        assert len(s.g1_drugs) == 16    # int(20 * 0.8)
        assert len(s.g2_drugs) == 4


# Fair protocol matches paper Table B.1 byte-exactly


@pytest.mark.skipif(
    not FULL_EDGES_CSV.is_file(),
    reason="Full DrugBank ddi_edges.csv not present — run reconstruct.py first.",
)
class TestFairProtocolMatchesPaperTableB1:
    """Match Table B.1's per-seed positive counts on the 1,900-drug benchmark."""

    PAPER_SEED_COUNTS: dict[int, dict[str, int]] = {
        42: {
            "train": 325_336,
            "val_s0": 18_074, "test_s0": 18_075,
            "val_s1": 90_785, "test_s1": 90_786,
            "val_s2": 11_337, "test_s2": 11_338,
        },
        43: {
            "train": 329_976,
            "val_s0": 18_332, "test_s0": 18_333,
            "val_s1": 88_884, "test_s1": 88_885,
            "val_s2": 10_660, "test_s2": 10_661,
        },
        44: {
            "train": 324_297,
            "val_s0": 18_016, "test_s0": 18_017,
            "val_s1": 91_190, "test_s1": 91_190,
            "val_s2": 11_510, "test_s2": 11_511,
        },
    }

    @pytest.fixture(scope="class")
    def edges(self):
        return pd.read_csv(
            FULL_EDGES_CSV, usecols=["drug_a_id", "drug_b_id"],
        )

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_per_seed_counts_match_paper(self, edges, seed):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=seed)
        actual = {name: len(df) for name, df in s.items()}
        expected = self.PAPER_SEED_COUNTS[seed]
        for k, v in expected.items():
            assert actual[k] == v, (
                f"seed={seed} {k}: release {actual[k]} != paper {v}"
            )

    def test_g1_g2_sizes_match_paper(self, edges):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42)
        assert len(s.g1_drugs) == 1520
        assert len(s.g2_drugs) == 380

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_total_positives_565731(self, edges, seed):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=seed)
        assert s.total_pairs() == 565_731


# G1 partition formula


class TestPartitionFormula:
    def test_fair_uses_multiplicative_formula(self):
        from coldddi.data.splits import _partition_drugs_fair

        drugs = [f"DB{i:04d}" for i in range(1000)]
        g1, g2 = _partition_drugs_fair(drugs, seed=42, g1_ratio=0.8)
        # int(1000 * 0.8) = 800
        assert len(g1) == 800
        assert len(g2) == 200

    def test_legacy_uses_divisive_formula(self):
        from coldddi.data.splits import _partition_drugs

        drugs = [f"DB{i:04d}" for i in range(1000)]
        g1, g2 = _partition_drugs(drugs, seed=42, drug_ratio=1.5)
        # int(1000 // 1.5) = 666
        assert len(g1) == 666
        assert len(g2) == 334

    def test_fair_protocol_g1_ratio_kwarg_drives_size(self):
        """``g1_ratio=0.7`` should produce 70%/30%, not 80%/20%."""
        from coldddi.data.splits import build_splits

        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(20) for b in range(a + 1, 20)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42, g1_ratio=0.7)
        # int(20 * 0.7) = 14
        assert len(s.g1_drugs) == 14
        assert len(s.g2_drugs) == 6
        assert s.g1_ratio == 0.7


# S0/S1/S2 within-pool ratios under fair protocol


@pytest.mark.skipif(
    not FULL_EDGES_CSV.is_file(),
    reason="Full DrugBank ddi_edges.csv not present.",
)
class TestFairProtocolPoolRatios:
    """Paper App B.1 line 25: S0 90/5/5, S1/S2 50/50."""

    @pytest.fixture(scope="class")
    def edges(self):
        return pd.read_csv(
            FULL_EDGES_CSV, usecols=["drug_a_id", "drug_b_id"],
        )

    def test_s0_split_is_90_5_5(self, edges):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42)
        s0_total = len(s.train) + len(s.val_s0) + len(s.test_s0)
        assert len(s.train) / s0_total == pytest.approx(0.9, abs=1e-3)
        assert len(s.val_s0) / s0_total == pytest.approx(0.05, abs=1e-3)
        assert len(s.test_s0) / s0_total == pytest.approx(0.05, abs=1e-3)

    def test_s1_split_is_50_50(self, edges):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42)
        # val_s1 + test_s1 = full S1 pool; should be ≈ 50/50.
        s1_total = len(s.val_s1) + len(s.test_s1)
        assert len(s.val_s1) / s1_total == pytest.approx(0.5, abs=1e-3)
        assert len(s.test_s1) / s1_total == pytest.approx(0.5, abs=1e-3)

    def test_s2_split_is_50_50(self, edges):
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42)
        s2_total = len(s.val_s2) + len(s.test_s2)
        assert len(s.val_s2) / s2_total == pytest.approx(0.5, abs=1e-3)
        assert len(s.test_s2) / s2_total == pytest.approx(0.5, abs=1e-3)


# Back-compat: explicit drug_ratio routes to legacy


class TestLegacyBackCompat:
    def test_explicit_drug_ratio_routes_to_legacy(self):
        """Explicit drug_ratio selects legacy behavior when protocol is omitted."""
        from coldddi.data.splits import build_splits

        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(30) for b in range(a + 1, 30)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42, drug_ratio=1.5)
        assert s.protocol == "legacy"
        assert s.drug_ratio == 1.5
        # |G1| = 30 // 1.5 = 20  (legacy formula)
        assert len(s.g1_drugs) == 20
        assert len(s.g2_drugs) == 10

    def test_explicit_protocol_legacy_keeps_legacy(self):
        from coldddi.data.splits import build_splits

        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(30) for b in range(a + 1, 30)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42, protocol="legacy")
        assert s.protocol == "legacy"
        assert len(s.g1_drugs) == 20    # 30 // 1.5

    def test_invalid_protocol_raises(self):
        from coldddi.data.splits import build_splits

        edges = pd.DataFrame(
            [("DBA", "DBB")], columns=["drug_a_id", "drug_b_id"],
        )
        with pytest.raises(ValueError, match="protocol must be"):
            build_splits(edges, seed=42, protocol="bogus")


# Protocol conflicts.


class TestKnobConflictDetection:
    """Distinguish omitted protocol from explicit fair and reject conflicting kwargs."""

    @pytest.fixture
    def edges(self):
        return pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(20) for b in range(a + 1, 20)],
            columns=["drug_a_id", "drug_b_id"],
        )

    def test_explicit_fair_plus_drug_ratio_raises(self, edges):
        """Explicit fair with drug_ratio raises instead of selecting legacy."""
        from coldddi.data.splits import build_splits

        with pytest.raises(ValueError, match="legacy kwargs"):
            build_splits(edges, seed=42, protocol="fair", drug_ratio=1.5)

    def test_explicit_fair_plus_val_ratio_raises(self, edges):
        from coldddi.data.splits import build_splits

        with pytest.raises(ValueError, match="legacy kwargs"):
            build_splits(edges, seed=42, protocol="fair", val_ratio=0.1)

    def test_explicit_fair_plus_s0_train_budget_raises(self, edges):
        from coldddi.data.splits import build_splits

        with pytest.raises(ValueError, match="legacy kwargs"):
            build_splits(edges, seed=42, protocol="fair", s0_train_budget=100)

    def test_explicit_legacy_plus_g1_ratio_raises(self, edges):
        from coldddi.data.splits import build_splits

        with pytest.raises(ValueError, match="g1_ratio"):
            build_splits(edges, seed=42, protocol="legacy", g1_ratio=0.8)

    def test_implicit_fair_with_g1_ratio_works(self, edges):
        """No protocol kwarg + explicit g1_ratio → fair (default)."""
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42, g1_ratio=0.7)
        assert s.protocol == "fair"
        assert s.g1_ratio == 0.7

    def test_legacy_plus_val_ratio_works(self, edges):
        """legacy is allowed to take val_ratio."""
        from coldddi.data.splits import build_splits

        s = build_splits(edges, seed=42, protocol="legacy", val_ratio=0.1)
        assert s.protocol == "legacy"
        assert s.val_ratio == 0.1


# Manifest round-trip


class TestManifestRoundTrip:
    def test_fair_protocol_round_trip(self, tmp_path):
        from coldddi.data.splits import SplitFolds, build_splits

        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(20) for b in range(a + 1, 20)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42)
        s.save(tmp_path / "out")

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["protocol"] == "fair"
        assert manifest["g1_ratio"] == 0.8

        s2 = SplitFolds.from_dir(tmp_path / "out")
        assert s2.protocol == "fair"
        assert s2.g1_ratio == 0.8
        assert list(s2.g1_drugs) == list(s.g1_drugs)
        assert list(s2.g2_drugs) == list(s.g2_drugs)

    def test_legacy_protocol_round_trip(self, tmp_path):
        from coldddi.data.splits import SplitFolds, build_splits

        edges = pd.DataFrame(
            [(f"DB{a:04d}", f"DB{b:04d}") for a in range(30) for b in range(a + 1, 30)],
            columns=["drug_a_id", "drug_b_id"],
        )
        s = build_splits(edges, seed=42, drug_ratio=1.5)
        s.save(tmp_path / "out")

        s2 = SplitFolds.from_dir(tmp_path / "out")
        assert s2.protocol == "legacy"
        assert s2.drug_ratio == 1.5

    def test_pre_audit_manifest_without_protocol_field_reads_as_legacy(
        self, tmp_path,
    ):
        """Manifests without a protocol field load as legacy bundles."""
        from coldddi.data.splits import SPLIT_NAMES, SplitFolds

        # Build a minimal manifest that lacks 'protocol' field.
        out = tmp_path / "preaudit"
        out.mkdir()
        # Empty parquets per split name.
        empty_df = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        for name in SPLIT_NAMES:
            empty_df.to_parquet(out / f"{name}.parquet", index=False)
        manifest = {
            "seed": 42,
            "drug_ratio": 1.5,
            "val_ratio": 0.1,
            "n_pairs": {name: 0 for name in SPLIT_NAMES},
            "g1_drugs": ["DBA"],
            "g2_drugs": ["DBB"],
        }
        (out / "manifest.json").write_text(json.dumps(manifest))
        s = SplitFolds.from_dir(out)
        assert s.protocol == "legacy"


# Fair-protocol structural invariants


@pytest.mark.skipif(
    not TOY_EDGES_CSV.is_file(),
    reason="Toy ddi_edges.csv not present",
)
class TestFairProtocolStructuralInvariants:
    """Same cold-start invariants the legacy splitter satisfies must
    hold under the fair protocol: train ⊆ G1xG1, no train↔val/test
    overlap, S1 cross-pool, S2 G2xG2."""

    @pytest.fixture(scope="class")
    def splits(self):
        from coldddi.data.splits import build_splits

        edges = pd.read_csv(
            TOY_EDGES_CSV, usecols=["drug_a_id", "drug_b_id"],
        )
        return build_splits(edges, seed=42)

    def test_train_is_g1_g1(self, splits):
        g1 = set(splits.g1_drugs)
        a = splits.train["drug_a_id"].astype(str)
        b = splits.train["drug_b_id"].astype(str)
        assert a.isin(g1).all() and b.isin(g1).all()

    def test_train_disjoint_from_every_holdout(self, splits):
        def canon(df):
            out = set()
            for a, b in zip(
                df["drug_a_id"].astype(str),
                df["drug_b_id"].astype(str),
            ):
                if a > b:
                    a, b = b, a
                out.add((a, b))
            return out

        train = canon(splits.train)
        for n in (
            "val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2",
        ):
            other = canon(getattr(splits, n))
            assert train.isdisjoint(other), f"train overlaps {n}"

    def test_test_s2_is_g2_g2(self, splits):
        g2 = set(splits.g2_drugs)
        a = splits.test_s2["drug_a_id"].astype(str)
        b = splits.test_s2["drug_b_id"].astype(str)
        if len(a):
            assert a.isin(g2).all() and b.isin(g2).all()

    def test_test_s1_is_cross(self, splits):
        g1, g2 = set(splits.g1_drugs), set(splits.g2_drugs)
        for _, r in splits.test_s1.iterrows():
            a, b = str(r["drug_a_id"]), str(r["drug_b_id"])
            assert (a in g1 and b in g2) or (a in g2 and b in g1)
