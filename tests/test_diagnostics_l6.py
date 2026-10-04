"""Test indicator math and bucket conventions against upstream
``Code-Released/exps/sec5-3/2_indicators/compute_*.py``:

ALL contains positive base pairs only; per-bucket rows use PK-A/B and PD-A/B.
KPS-F needs only R0/base predictions, including for single-modality baselines.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Bucket lookup.

class TestBuckets:
    def test_lookup_from_dataframe(self):
        from coldddi.diagnostics.buckets import build_bucket_lookup

        ab = pd.DataFrame({
            "drug_a_id":      ["DB1", "DB2", "DB3", "DB4", "DB5"],
            "drug_b_id":      ["DBA", "DBB", "DBC", "DBD", "DBE"],
            "pk_pd_label":    ["PK",  "PK",  "PD",  "PD",  "Mixed"],
            "has_key_entity": [True,  False, True,  False, True],
        })
        lookup = build_bucket_lookup(ab)
        assert lookup.bucket("DB1", "DBA") == "PK-A"
        assert lookup.bucket("DB2", "DBB") == "PK-B"
        assert lookup.bucket("DB3", "DBC") == "PD-A"
        assert lookup.bucket("DB4", "DBD") == "PD-B"
        assert lookup.bucket("DB5", "DBE") == "Other"  # Mixed → Other
        assert lookup.bucket("XX", "YY") == "Other"

    def test_reverse_order_fallback(self):
        from coldddi.diagnostics.buckets import build_bucket_lookup

        ab = pd.DataFrame({
            "drug_a_id": ["DB1"], "drug_b_id": ["DB2"],
            "pk_pd_label": ["PK"], "has_key_entity": [True],
        })
        lookup = build_bucket_lookup(ab)
        assert lookup.bucket("DB2", "DB1") == "PK-A"

    def test_missing_columns_raises(self):
        from coldddi.diagnostics.buckets import build_bucket_lookup

        with pytest.raises(ValueError, match="missing required columns"):
            build_bucket_lookup(pd.DataFrame({"drug_a_id": [], "drug_b_id": []}))


# ALL-bucket convention is "positives only"

class TestALLBucketConvention:
    """ALL includes only positive base pairs, matching upstream _agg_buckets."""

    def setup_method(self):
        from coldddi.diagnostics.kps_swap import SwapTriple

        # 3 base pairs in test_s2 — 2 positive, 1 negative.
        # We build 1 swap triple per base pair.
        self.swap = [
            SwapTriple("DBA", "DBC", "DBB", label_uv=1),  # pos
            SwapTriple("DBB", "DBC", "DBA", label_uv=1),  # pos
            SwapTriple("DBD", "DBC", "DBB", label_uv=0),  # neg
        ]
        self.bucket_fn = lambda a, b: {
            ("DBA", "DBC"): "PK-A",
            ("DBB", "DBC"): "PK-A",
            ("DBD", "DBC"): "PK-A",
        }.get((a, b), "")

        # R0 predictions for all pairs in the swap table.
        self.r0 = {
            ("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5,
            ("DBD", "DBC"): 0.2,
        }

    def test_all_bucket_is_positives_only(self):
        """ALL excludes the negative base pair; PK-A includes it, as upstream does.
        """
        from coldddi.diagnostics.indicators import compute_indicators

        df = compute_indicators(
            {"R0": self.r0}, self.swap, bucket_fn=self.bucket_fn,
        )
        # PK-A: 3 deltas → |0.9-0.5| (pos), |0.5-0.9| (pos), |0.2-0.5| (neg)
        pk_a = df.query("indicator == 'KPS-F' and bucket == 'PK-A'").iloc[0]
        assert pk_a["n"] == 3
        # ALL = only positives = 2 rows → |0.9-0.5| + |0.5-0.9| → mean = 0.4
        all_row = df.query("indicator == 'KPS-F' and bucket == 'ALL'").iloc[0]
        assert all_row["n"] == 2
        assert all_row["value"] == pytest.approx(0.4)


# "Other" bucket is NOT emitted (upstream parity)

class TestNoOtherBucket:
    def test_no_other_bucket_for_kps_f(self):
        """Empty bucket labels contribute only positives to ALL, never an Other row.
        """
        from coldddi.diagnostics.indicators import compute_indicators
        from coldddi.diagnostics.kps_swap import SwapTriple

        swap = [SwapTriple("DBA", "DBC", "DBB", label_uv=1)]
        r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5}
        df = compute_indicators(
            {"R0": r0}, swap, bucket_fn=lambda a, b: "",
        )
        kf = df[df["indicator"] == "KPS-F"]
        # KPS-F rows from EMPTY-bucket data: only the ALL aggregate.
        assert set(kf["bucket"].unique()) == {"ALL"}
        # The "Other" bucket name is not emitted anywhere in the table.
        assert "Other" not in set(df["bucket"].unique())


# Empty / NaN-rich primary bucket rows

class TestEmptyBuckets:
    def test_empty_primary_bucket_dropped(self):
        """Omit empty primary buckets; NaN blocks mean a whole condition is missing."""
        from coldddi.diagnostics.indicators import compute_indicators
        from coldddi.diagnostics.kps_swap import SwapTriple

        # Only PK-A pairs exist in the data; PK-B/PD-A/PD-B never appear.
        swap = [SwapTriple("DBA", "DBC", "DBB", label_uv=1)]
        r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5}
        df = compute_indicators(
            {"R0": r0}, swap, bucket_fn=lambda a, b: "PK-A",
        )
        pk_a = df.query("indicator == 'KPS-F' and bucket == 'PK-A'")
        all_ = df.query("indicator == 'KPS-F' and bucket == 'ALL'")
        # PK-A exists; PK-B/PD-A/PD-B do not.
        assert len(pk_a) == 1
        for missing_bk in ("PK-B", "PD-A", "PD-B"):
            assert df.query(
                f"indicator == 'KPS-F' and bucket == '{missing_bk}'"
            ).empty


# Single-modality baseline path (only R0 / "base")

class TestSingleModalityBaselinePath:
    def test_only_kps_f_computed(self):
        """When only R0 is provided, KPS-F has values; the channel and
        KSAI indicators come back as NaN row blocks (one NaN row per
        primary bucket + ALL)."""
        from coldddi.diagnostics.indicators import (
            LLM_INDICATOR_NAMES, compute_indicators,
        )
        from coldddi.diagnostics.kps_swap import SwapTriple

        swap = [SwapTriple("DBA", "DBC", "DBB", label_uv=1)]
        r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5}
        df = compute_indicators(
            {"R0": r0}, swap, bucket_fn=lambda a, b: "PK-A",
        )
        # KPS-F: non-NaN
        kf = df.query("indicator == 'KPS-F' and bucket == 'PK-A'").iloc[0]
        assert not math.isnan(kf["value"])
        # Every other LLM indicator: NaN rows (one per primary bucket + ALL).
        for ind in LLM_INDICATOR_NAMES:
            if ind == "KPS-F":
                continue
            sub = df.query(f"indicator == '{ind}'")
            assert len(sub) == 5, f"{ind}: expected 5 NaN rows, got {len(sub)}"
            assert sub["value"].isna().all(), f"{ind}: some rows are non-NaN"
            assert (sub["n"] == 0).all()


# LLM full path: 7 indicators, math correctness

class TestLLMFullPath:
    def setup_method(self):
        from coldddi.diagnostics.kps_swap import SwapTriple

        # 2 positive + 1 negative base pair, all PK-A.
        self.swap = [
            SwapTriple("DBA", "DBC", "DBE", label_uv=1),
            SwapTriple("DBB", "DBC", "DBE", label_uv=1),
            SwapTriple("DBD", "DBC", "DBE", label_uv=0),
        ]
        self.bucket_fn = lambda a, b: "PK-A"

        self.r0 = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5,
                   ("DBD", "DBC"): 0.2, ("DBE", "DBC"): 0.3}
        self.r1 = {("DBA", "DBC"): 0.7, ("DBB", "DBC"): 0.3,
                   ("DBD", "DBC"): 0.0, ("DBE", "DBC"): 0.1}
        self.r2 = {("DBA", "DBC"): 0.5, ("DBB", "DBC"): 0.1,
                   ("DBD", "DBC"): 0.0, ("DBE", "DBC"): 0.0}
        self.r3 = {("DBA", "DBC"): 0.4, ("DBB", "DBC"): 0.0,
                   ("DBD", "DBC"): 0.0, ("DBE", "DBC"): 0.0}

    def test_kps_name_positives_only_for_all(self):
        from coldddi.diagnostics.indicators import compute_indicators

        df = compute_indicators(
            {"R0": self.r0, "R1": self.r1},
            self.swap, bucket_fn=self.bucket_fn,
        )
        # Base pairs (deduped from swap): DBA, DBB, DBD (3 unique).
        # PK-A row counts all 3; ALL row counts only positives (DBA, DBB).
        # |R0 - R1| for each: 0.2, 0.2, 0.2.
        pk_a = df.query("indicator == 'KPS-Name' and bucket == 'PK-A'").iloc[0]
        assert pk_a["n"] == 3
        assert pk_a["value"] == pytest.approx(0.2)
        all_ = df.query("indicator == 'KPS-Name' and bucket == 'ALL'").iloc[0]
        assert all_["n"] == 2  # positives only
        assert all_["value"] == pytest.approx(0.2)

    def test_kps_kg_named_alias(self):
        """KPS-KG-Named must equal KPS-KG numerically and in n."""
        from coldddi.diagnostics.indicators import compute_indicators

        df = compute_indicators(
            {"R0": self.r0, "R2": self.r2},
            self.swap, bucket_fn=self.bucket_fn,
        )
        for bk in ("PK-A", "ALL"):
            a = df.query(f"indicator == 'KPS-KG' and bucket == '{bk}'").iloc[0]
            b = df.query(f"indicator == 'KPS-KG-Named' and bucket == '{bk}'").iloc[0]
            assert a["value"] == pytest.approx(b["value"])
            assert a["n"] == b["n"]

    def test_ksai_masking_sign(self):
        """KSAI = (|R1-R3| - |R0-R2|) per pair; verify per-pair math
        and the positives-only ALL aggregate."""
        from coldddi.diagnostics.indicators import compute_indicators

        df = compute_indicators(
            {"R0": self.r0, "R1": self.r1, "R2": self.r2, "R3": self.r3},
            self.swap, bucket_fn=self.bucket_fn,
        )
        # Per pair:
        #   DBA: |0.7-0.4| - |0.9-0.5| = 0.3 - 0.4 = -0.1  (positive)
        #   DBB: |0.3-0.0| - |0.5-0.1| = 0.3 - 0.4 = -0.1  (positive)
        #   DBD: |0.0-0.0| - |0.2-0.0| = 0.0 - 0.2 = -0.2  (negative)
        pk_a = df.query("indicator == 'KSAI' and bucket == 'PK-A'").iloc[0]
        assert pk_a["n"] == 3
        assert pk_a["value"] == pytest.approx((-0.1 + -0.1 + -0.2) / 3)
        all_ = df.query("indicator == 'KSAI' and bucket == 'ALL'").iloc[0]
        assert all_["n"] == 2  # positives only
        assert all_["value"] == pytest.approx((-0.1 + -0.1) / 2)


# Baseline channel indicators (MKG-FENN / TIGER)

class TestBaselineChannelIndicators:
    def setup_method(self):
        from coldddi.diagnostics.kps_swap import SwapTriple

        self.swap = [
            SwapTriple("DBA", "DBC", "DBB", label_uv=1),
            SwapTriple("DBB", "DBC", "DBA", label_uv=1),
        ]
        self.bucket_fn = lambda a, b: "PK-A"
        self.base     = {("DBA", "DBC"): 0.9, ("DBB", "DBC"): 0.5}
        self.mask_mol = {("DBA", "DBC"): 0.6, ("DBB", "DBC"): 0.2}
        self.mask_kg  = {("DBA", "DBC"): 0.7, ("DBB", "DBC"): 0.3}

    def test_three_indicators_emitted(self):
        from coldddi.diagnostics.indicators import (
            BASELINE_CHANNEL_INDICATOR_NAMES,
            compute_baseline_channel_indicators,
        )

        df = compute_baseline_channel_indicators(
            {"base": self.base, "mask_mol": self.mask_mol, "mask_kg": self.mask_kg},
            self.swap, bucket_fn=self.bucket_fn,
        )
        emitted = set(df["indicator"].unique())
        assert emitted == set(BASELINE_CHANNEL_INDICATOR_NAMES)

        # KPS-mol = |base - mask_mol| per pair = 0.3, 0.3 → mean 0.3
        pk_a = df.query("indicator == 'KPS-mol' and bucket == 'PK-A'").iloc[0]
        assert pk_a["value"] == pytest.approx(0.3)

    def test_single_modality_baseline_only_kpsf(self):
        """Pass only ``base`` (no mol/kg masks): KPS-F has values,
        KPS-mol and KPS-KG come back as NaN row blocks."""
        from coldddi.diagnostics.indicators import (
            compute_baseline_channel_indicators,
        )

        df = compute_baseline_channel_indicators(
            {"base": self.base},
            self.swap, bucket_fn=self.bucket_fn,
        )
        kf = df.query("indicator == 'KPS-F' and bucket == 'PK-A'").iloc[0]
        assert not math.isnan(kf["value"])
        for ind in ("KPS-mol", "KPS-KG"):
            sub = df.query(f"indicator == '{ind}'")
            assert sub["value"].isna().all()
            assert (sub["n"] == 0).all()


# A-B gap (paper-headline)

class TestABGap:
    def test_gap_formula(self):
        """``(PK-A + PD-A)/2 - (PK-B + PD-B)/2``."""
        from coldddi.diagnostics.indicators import compute_ab_gap

        df = pd.DataFrame([
            {"indicator": "KPS-F", "bucket": "PK-A", "value": 0.30},
            {"indicator": "KPS-F", "bucket": "PK-B", "value": 0.10},
            {"indicator": "KPS-F", "bucket": "PD-A", "value": 0.40},
            {"indicator": "KPS-F", "bucket": "PD-B", "value": 0.20},
        ])
        gap = compute_ab_gap(df, "KPS-F")
        assert gap == pytest.approx((0.30 + 0.40) / 2 - (0.10 + 0.20) / 2)

    def test_missing_bucket_returns_nan(self):
        from coldddi.diagnostics.indicators import compute_ab_gap

        df = pd.DataFrame([
            {"indicator": "KPS-F", "bucket": "PK-A", "value": 0.30},
        ])
        assert math.isnan(compute_ab_gap(df, "KPS-F"))

    def test_nan_value_propagates(self):
        from coldddi.diagnostics.indicators import compute_ab_gap

        df = pd.DataFrame([
            {"indicator": "KPS-F", "bucket": "PK-A", "value": float("nan")},
            {"indicator": "KPS-F", "bucket": "PK-B", "value": 0.10},
            {"indicator": "KPS-F", "bucket": "PD-A", "value": 0.40},
            {"indicator": "KPS-F", "bucket": "PD-B", "value": 0.20},
        ])
        assert math.isnan(compute_ab_gap(df, "KPS-F"))


# Directional lookup

class TestDirectionalLookup:
    def test_swap_pair_stored_in_reverse(self):
        """If R0 has ``(b, a)`` but swap_candidate carries ``(a, b)``,
        the indicator MUST still find the prediction."""
        from coldddi.diagnostics.indicators import compute_indicators
        from coldddi.diagnostics.kps_swap import SwapTriple

        swap = [SwapTriple("DBA", "DBC", "DBB", label_uv=1)]
        # R0 stores both pairs in REVERSE order:
        r0 = {("DBC", "DBA"): 0.9, ("DBC", "DBB"): 0.4}
        df = compute_indicators(
            {"R0": r0}, swap, bucket_fn=lambda a, b: "PK-A",
        )
        pk_a = df.query("indicator == 'KPS-F' and bucket == 'PK-A'").iloc[0]
        assert pk_a["n"] == 1
        assert pk_a["value"] == pytest.approx(0.5)


# KPS-Swap

def _make_tiny_dataset_for_swap():
    from coldddi.data.dataset import PairDataset
    from coldddi.data.kg import KnowledgeGraph
    from coldddi.data.splits import SplitFolds

    drugs = pd.DataFrame({
        "drugbank_id": ["DBA", "DBB", "DBC", "DBD"],
        "name":        ["A", "B", "C", "D"],
        "smiles":      ["CC", "CCC", "CCO", "CC=O"],
        "type":        ["small molecule"] * 4,
        "groups":      ["approved"] * 4,
    })
    empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
    splits = SplitFolds(
        train=empty, val_s0=empty, val_s1=empty, val_s2=empty,
        test_s0=empty, test_s1=empty,
        test_s2=pd.DataFrame({"drug_a_id": ["DBA"], "drug_b_id": ["DBC"]}),
        g1_drugs=["DBA"], g2_drugs=["DBB", "DBC", "DBD"], seed=42,
    )
    e = pd.DataFrame(columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"])
    t = pd.DataFrame(columns=["drugbank_id", "target_id", "target_name", "organism", "action"])
    tr = pd.DataFrame(columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"])
    c = pd.DataFrame(columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"])
    p = pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"])
    kg = KnowledgeGraph(enzymes=e, targets=t, transporters=tr, carriers=c, pathways=p)
    ds = PairDataset(edges=empty, splits=splits, kg=kg, drugs=drugs)
    neg = pd.DataFrame({"drug_a_id": ["DBB", "DBD"], "drug_b_id": ["DBC", "DBC"]})

    def _get_neg(s):
        return neg if s == "test_s2" else empty
    ds.get_negatives = _get_neg  # type: ignore[method-assign]
    return ds


class TestKPSSwap:
    def test_both_directions_emitted(self):
        """Generate both base-pair directions, matching upstream build_kps_data.py:183-187."""
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        # Richer fixture so both directions yield triples:
        # 4 G2 drugs, 2 positives + 2 negatives in test_s2.
        drugs = pd.DataFrame({
            "drugbank_id": ["DBA", "DBB", "DBC", "DBD"],
            "name":   ["A", "B", "C", "D"],
            "smiles": ["CC", "CCC", "CCO", "CC=O"],
            "type":   ["small molecule"] * 4,
            "groups": ["approved"] * 4,
        })
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        pos = pd.DataFrame({
            "drug_a_id": ["DBA", "DBC"],
            "drug_b_id": ["DBB", "DBD"],
        })
        neg = pd.DataFrame({
            "drug_a_id": ["DBA", "DBB"],
            "drug_b_id": ["DBC", "DBD"],
        })
        splits = SplitFolds(
            train=empty, val_s0=empty, val_s1=empty, val_s2=empty,
            test_s0=empty, test_s1=empty, test_s2=pos,
            g1_drugs=[], g2_drugs=["DBA", "DBB", "DBC", "DBD"], seed=42,
        )
        kg = KnowledgeGraph(
            enzymes=pd.DataFrame(columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"]),
            targets=pd.DataFrame(columns=["drugbank_id", "target_id", "target_name", "organism", "action"]),
            transporters=pd.DataFrame(columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"]),
            carriers=pd.DataFrame(columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"]),
            pathways=pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"]),
        )
        ds = PairDataset(edges=empty, splits=splits, kg=kg, drugs=drugs)
        ds.get_negatives = lambda s: neg if s == "test_s2" else empty  # type: ignore[method-assign]

        trips = build_swap_candidates(
            ds, source_split="test_s2", search_pool_splits=("test_s2",),
        )
        v_seen = {t.qb for t in trips}
        u_seen = {t.qa for t in trips}
        # Both DBB (from canonical) and DBA (from reverse) must appear as v.
        assert "DBB" in v_seen, f"canonical-direction triples missing; v_seen={v_seen}"
        assert "DBA" in v_seen, (
            f"reverse-direction triples missing — upstream 'expectation' "
            f"step not emitting both directions; v_seen={v_seen}"
        )
        # Symmetric: both DBA (canonical) and DBB (reverse) appear as u.
        assert "DBA" in u_seen and "DBB" in u_seen, f"u_seen={u_seen}"

    def test_drug_pool_filter(self):
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        ds = _make_tiny_dataset_for_swap()
        trips = build_swap_candidates(
            ds, source_split="test_s2", drug_pool={"DBB"},
            search_pool_splits=("test_s2",),
        )
        assert all(t.qa_prime == "DBB" for t in trips)

    def test_unknown_split_raises(self):
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        ds = _make_tiny_dataset_for_swap()
        with pytest.raises(ValueError, match="not in dataset"):
            build_swap_candidates(ds, source_split="val_s99")

    def test_s1_setting_preserved(self):
        """Keep S1 swaps cross-partition in both directions.

        Replace a G1 head with G1 and a G2 head with G2; forcing both
        replacements into G2 would turn the canonical S1 pair into S2.
        """
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        drugs = pd.DataFrame({
            "drugbank_id": ["G1A", "G1B", "G2A", "G2B", "G2C"],
            "name":   ["A", "B", "C", "D", "E"],
            "smiles": ["CC", "CCC", "CCO", "CC=O", "C=C"],
            "type":   ["small molecule"] * 5,
            "groups": ["approved"] * 5,
        })
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        # test_s1 base pair (G1A, G2A) — one drug from each partition.
        # Also seed a few candidates: (G1B, G2A) and (G1A, G2B), (G1B, G2B).
        pos = pd.DataFrame({
            "drug_a_id": ["G1A"], "drug_b_id": ["G2A"],
        })
        neg = pd.DataFrame({
            "drug_a_id": ["G1B", "G1A", "G1B", "G2C"],
            "drug_b_id": ["G2A", "G2B", "G2B", "G2A"],
        })
        splits = SplitFolds(
            train=empty, val_s0=empty, val_s1=empty, val_s2=empty,
            test_s0=empty, test_s1=pos, test_s2=empty,
            g1_drugs=["G1A", "G1B"], g2_drugs=["G2A", "G2B", "G2C"], seed=42,
        )
        kg = KnowledgeGraph(
            enzymes=pd.DataFrame(columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"]),
            targets=pd.DataFrame(columns=["drugbank_id", "target_id", "target_name", "organism", "action"]),
            transporters=pd.DataFrame(columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"]),
            carriers=pd.DataFrame(columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"]),
            pathways=pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"]),
        )
        ds = PairDataset(edges=empty, splits=splits, kg=kg, drugs=drugs)
        ds.get_negatives = lambda s: neg if s == "test_s1" else empty  # type: ignore[method-assign]

        trips = build_swap_candidates(
            ds, source_split="test_s1",
            search_pool_splits=("test_s1",),
        )

        # For every triple, the swap pair (u', v) must preserve the S1
        # setting — i.e. exactly one of {u', v} is in G1 and one in G2.
        g1 = set(splits.g1_drugs)
        g2 = set(splits.g2_drugs)
        for t in trips:
            u_prime_in_g1 = t.qa_prime in g1
            v_in_g1 = t.qb in g1
            u_prime_in_g2 = t.qa_prime in g2
            v_in_g2 = t.qb in g2
            assert u_prime_in_g1 != v_in_g1, (
                f"S1 setting violated: u'={t.qa_prime} in_G1={u_prime_in_g1} "
                f"and v={t.qb} in_G1={v_in_g1} are not opposite partitions"
            )
            assert u_prime_in_g2 != v_in_g2

    def test_no_self_pair_swaps(self):
        """Reject self-pair swaps (u' == v), even if the search pool includes them."""
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        drugs = pd.DataFrame({
            "drugbank_id": ["DBA", "DBB", "DBC"],
            "name":   ["A", "B", "C"], "smiles": ["CC", "CCC", "CCO"],
            "type":   ["small molecule"] * 3, "groups": ["approved"] * 3,
        })
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])

        # base pair (DBA, DBB), positive.
        pos = pd.DataFrame({"drug_a_id": ["DBA"], "drug_b_id": ["DBB"]})
        # A self-loop negative would create a self-pair swap without the u' != v guard.
        neg = pd.DataFrame({
            "drug_a_id": ["DBB", "DBC"],
            "drug_b_id": ["DBB", "DBB"],
        })
        splits = SplitFolds(
            train=empty, val_s0=empty, val_s1=empty, val_s2=empty,
            test_s0=empty, test_s1=empty, test_s2=pos,
            g1_drugs=[], g2_drugs=["DBA", "DBB", "DBC"], seed=42,
        )
        kg = KnowledgeGraph(
            enzymes=pd.DataFrame(columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"]),
            targets=pd.DataFrame(columns=["drugbank_id", "target_id", "target_name", "organism", "action"]),
            transporters=pd.DataFrame(columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"]),
            carriers=pd.DataFrame(columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"]),
            pathways=pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"]),
        )
        ds = PairDataset(edges=empty, splits=splits, kg=kg, drugs=drugs)
        ds.get_negatives = lambda s: neg if s == "test_s2" else empty  # type: ignore[method-assign]

        trips = build_swap_candidates(
            ds, source_split="test_s2",
            search_pool_splits=("test_s2",),
        )
        # NO triple may have u' == v.
        for t in trips:
            assert t.qa_prime != t.qb, (
                f"Self-pair swap leaked through: ({t.qa}, {t.qb}, {t.qa_prime})"
            )

    def test_explicit_drug_pool_overrides_auto(self):
        """An explicit drug_pool bypasses per-direction selection for ablations."""
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        ds = _make_tiny_dataset_for_swap()
        trips = build_swap_candidates(
            ds, source_split="test_s2",
            drug_pool={"DBB"},
            search_pool_splits=("test_s2",),
        )
        # With the explicit pool, every u' must be DBB.
        assert all(t.qa_prime == "DBB" for t in trips)

    def test_reverse_orientation_indexed(self):
        from coldddi.data.dataset import PairDataset
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.data.splits import SplitFolds
        from coldddi.diagnostics.kps_swap import build_swap_candidates

        drugs = pd.DataFrame({
            "drugbank_id": ["DBA", "DBB", "DBC"],
            "name":   ["A", "B", "C"], "smiles": ["CC", "CCC", "CCO"],
            "type":   ["small molecule"] * 3, "groups": ["approved"] * 3,
        })
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        splits = SplitFolds(
            train=empty, val_s0=empty, val_s1=empty, val_s2=empty,
            test_s0=empty, test_s1=empty,
            test_s2=pd.DataFrame({"drug_a_id": ["DBA"], "drug_b_id": ["DBC"]}),
            g1_drugs=[], g2_drugs=["DBA", "DBB", "DBC"], seed=42,
        )
        kg = KnowledgeGraph(
            enzymes=pd.DataFrame(columns=["drugbank_id", "enzyme_id", "enzyme_name", "organism", "action"]),
            targets=pd.DataFrame(columns=["drugbank_id", "target_id", "target_name", "organism", "action"]),
            transporters=pd.DataFrame(columns=["drugbank_id", "transporter_id", "transporter_name", "organism", "action"]),
            carriers=pd.DataFrame(columns=["drugbank_id", "carrier_id", "carrier_name", "organism", "action"]),
            pathways=pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"]),
        )
        ds = PairDataset(edges=empty, splits=splits, kg=kg, drugs=drugs)
        # Reverse: negative stored as (DBC, DBB)
        neg = pd.DataFrame({"drug_a_id": ["DBC"], "drug_b_id": ["DBB"]})

        def _get_neg(s):
            return neg if s == "test_s2" else empty
        ds.get_negatives = _get_neg  # type: ignore[method-assign]

        trips = build_swap_candidates(
            ds, source_split="test_s2", search_pool_splits=("test_s2",),
        )
        candidates = {t.qa_prime for t in trips if t.qa == "DBA" and t.qb == "DBC"}
        assert "DBB" in candidates
