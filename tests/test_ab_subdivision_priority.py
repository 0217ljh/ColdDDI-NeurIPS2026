"""Test deterministic, DrugBank-ranked PK-A and PD-A entity selection.

Ranks use first occurrence in DrugBank XML, not hard-coded enzyme preferences.
PK sorts by (rank_a + rank_b, name, id) within the first eligible bucket.
PD sorts by confidence, DDI-type relevance, DrugBank rank, then (name, id).
The final tie-breakers prevent hash-seeded set order from changing the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_FILTERED = REPO_ROOT / "data" / "private" / "intermediate" / "filtered"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# First-occurrence ranks.


class TestFirstOccurrenceRanks:
    """Ranks follow DrugBank XML order; duplicate IDs retain their first rank."""

    def test_distinct_entities_ranked_in_order(self):
        from coldddi.annotations.ab_subdivision import _first_occurrence_ranks

        entries = [
            {"id": "E_A", "name": "First", "type": "enzyme", "action": "i"},
            {"id": "E_B", "name": "Second", "type": "enzyme", "action": "i"},
            {"id": "E_C", "name": "Third", "type": "enzyme", "action": "i"},
        ]
        assert _first_occurrence_ranks(entries) == {"E_A": 0, "E_B": 1, "E_C": 2}

    def test_duplicate_id_keeps_first_occurrence(self):
        """Repeated polypeptides with different actions keep their first rank."""
        from coldddi.annotations.ab_subdivision import _first_occurrence_ranks

        entries = [
            {"id": "E_A", "name": "n", "type": "enzyme", "action": "inhibitor"},
            {"id": "E_B", "name": "n", "type": "enzyme", "action": "inhibitor"},
            {"id": "E_A", "name": "n", "type": "enzyme", "action": "substrate"},
        ]
        assert _first_occurrence_ranks(entries) == {"E_A": 0, "E_B": 1}

    def test_empty(self):
        from coldddi.annotations.ab_subdivision import _first_occurrence_ranks

        assert _first_occurrence_ranks([]) == {}

    def test_cyp_priority_constant_removed(self):
        """Hard-coded enzyme priorities must not override DrugBank ranks."""
        from coldddi.annotations import ab_subdivision

        assert not hasattr(ab_subdivision, "CYP_PRIORITY"), (
            "CYP_PRIORITY reintroduced — the rank-based design has "
            "been overridden; see this test file's docstring"
        )
        assert not hasattr(ab_subdivision, "_enzyme_priority"), (
            "_enzyme_priority reintroduced — the rank-based design "
            "has been overridden; see this test file's docstring"
        )


# PK candidate selection.


class TestSyntheticPkRankSelection:
    """PK candidates follow DrugBank rank order."""

    def _pk_best(self, enzyme_idx: dict, *, ddi_type: str = "metabolism") -> dict:
        from coldddi.annotations.ab_subdivision import _find_key_entity_pk

        return _find_key_entity_pk(
            "DB_X", "DB_Y", ddi_type,
            "drug X", "drug Y",
            enzyme_idx, {}, {},
        )

    def test_lowest_joint_rank_wins(self):
        """Enzyme E_A is rank 0 for both drugs (sum = 0); enzyme E_B
        is rank 1 for both drugs (sum = 2).  E_A must win."""
        enzyme_idx = {
            "DB_X": [
                {"id": "E_A", "name": "Alpha Enzyme", "type": "enzyme",
                 "action": "inhibitor"},
                {"id": "E_B", "name": "Beta Enzyme", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_A", "name": "Alpha Enzyme", "type": "enzyme",
                 "action": "substrate"},
                {"id": "E_B", "name": "Beta Enzyme", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        best = self._pk_best(enzyme_idx)
        assert best is not None
        assert best["key_entity_id"] == "E_A"

    def test_rank_uses_joint_sum_not_per_drug(self):
        """E_A: rank 0 for X, rank 2 for Y → sum 2.
        E_B: rank 1 for X, rank 0 for Y → sum 1.  E_B wins."""
        enzyme_idx = {
            "DB_X": [
                {"id": "E_A", "name": "First", "type": "enzyme",
                 "action": "inhibitor"},
                {"id": "E_B", "name": "Second", "type": "enzyme",
                 "action": "inhibitor"},
                {"id": "E_C", "name": "Third", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_B", "name": "Second", "type": "enzyme",
                 "action": "substrate"},
                {"id": "E_C", "name": "Third", "type": "enzyme",
                 "action": "substrate"},
                {"id": "E_A", "name": "First", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        best = self._pk_best(enzyme_idx)
        assert best is not None
        assert best["key_entity_id"] == "E_B"

    def test_curator_rank_overrides_alphabetical(self):
        """DrugBank rank takes precedence over alphabetical order."""
        enzyme_idx = {
            "DB_X": [
                # Z-enzyme is curator rank 0 for X.
                {"id": "E_Z", "name": "Z-prime enzyme", "type": "enzyme",
                 "action": "inhibitor"},
                {"id": "E_A", "name": "Alpha enzyme", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_Z", "name": "Z-prime enzyme", "type": "enzyme",
                 "action": "substrate"},
                {"id": "E_A", "name": "Alpha enzyme", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        best = self._pk_best(enzyme_idx)
        assert best is not None
        # Z-prime's joint rank 0 beats Alpha's 2 despite alphabetical order.
        assert best["key_entity_id"] == "E_Z"

    def test_equal_rank_falls_back_to_name(self):
        """Equal joint ranks use the deterministic (name, id) tie-breaker."""
        enzyme_idx = {
            "DB_X": [
                {"id": "E_A", "name": "Beta enzyme", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_A", "name": "Beta enzyme", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        # Single candidate path.
        best = self._pk_best(enzyme_idx)
        assert best["key_entity_id"] == "E_A"

    def test_subtype_short_circuit_pinned(self):
        """Rank candidates only within the first eligible PK bucket.

        As in the legacy script, metabolism considers enzymes only; a
        better-ranked transporter must not displace an eligible enzyme.
        """
        from coldddi.annotations.ab_subdivision import _find_key_entity_pk

        enzyme_idx = {
            "DB_X": [
                {"id": "E_LOW", "name": "Low rank enzyme", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                # 99 dummy rank-fillers shift E_LOW to rank 99.
                *[
                    {"id": f"PAD_{i}", "name": f"Pad {i}", "type": "enzyme",
                     "action": "inhibitor"}
                    for i in range(99)
                ],
                {"id": "E_LOW", "name": "Low rank enzyme", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        transport_idx = {
            "DB_X": [
                # Transporter shared and rank-0 for both drugs.
                {"id": "T_TOP", "name": "Top transporter", "type": "transporter",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "T_TOP", "name": "Top transporter", "type": "transporter",
                 "action": "substrate"},
            ],
        }
        best = _find_key_entity_pk(
            "DB_X", "DB_Y", "metabolism",   # bucket order: [enzyme]
            "drug X", "drug Y",
            enzyme_idx, transport_idx, {},
        )
        # Enzyme bucket short-circuits → low-ranked enzyme wins,
        # not the rank-0/0 transporter.
        assert best is not None
        assert best["key_entity_type"] == "enzyme"
        assert best["key_entity_id"] == "E_LOW"

    def test_duplicate_id_multiple_actions_picks_compatible_pair(self):
        """Try all repeated-enzyme action pairs while retaining first-occurrence rank."""
        from coldddi.annotations.ab_subdivision import _find_key_entity_pk

        enzyme_idx = {
            "DB_X": [
                # Only an incompatible-with-Y action listed first.
                {"id": "E_A", "name": "First enzyme", "type": "enzyme",
                 "action": "substrate"},
                # Second listing of same enzyme: this one pairs with Y.
                {"id": "E_A", "name": "First enzyme", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_A", "name": "First enzyme", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        best = _find_key_entity_pk(
            "DB_X", "DB_Y", "metabolism",
            "drug X", "drug Y",
            enzyme_idx, {}, {},
        )
        assert best is not None
        assert best["key_entity_id"] == "E_A"
        # Compatible PK role: inhibitor (X) × substrate (Y).
        assert best["action_drug_a"] == "inhibitor"
        assert best["action_drug_b"] == "substrate"

    def test_candidates_field_keeps_full_list(self):
        """Keep alternative candidates in rank order in key_entity_candidates."""
        import json as _json

        enzyme_idx = {
            "DB_X": [
                {"id": "E_A", "name": "First", "type": "enzyme",
                 "action": "inhibitor"},
                {"id": "E_B", "name": "Second", "type": "enzyme",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "E_A", "name": "First", "type": "enzyme",
                 "action": "substrate"},
                {"id": "E_B", "name": "Second", "type": "enzyme",
                 "action": "substrate"},
            ],
        }
        best = self._pk_best(enzyme_idx)
        assert best["key_entity_id"] == "E_A"
        rest = _json.loads(best["key_entity_candidates"])
        assert isinstance(rest, list)
        assert len(rest) == 1
        assert rest[0]["key_entity_id"] == "E_B"
        # Private ranking fields must not leak into CSV.
        assert "_drugbank_rank" not in rest[0]
        assert "_drugbank_rank" not in best


# Cross-process determinism.


class TestPkPickAcrossHashSeeds:
    """Entity selection must be invariant to subprocess PYTHONHASHSEED values."""

    def test_pick_is_reproducible_across_hash_seeds(self):
        import json
        import os
        import subprocess
        import sys as _sys

        # Equal joint ranks isolate the (name, id) tie-breaker across hash seeds.
        snippet = """
import sys, json
sys.path.insert(0, %r)
from coldddi.annotations.ab_subdivision import _find_key_entity_pk
enzyme_idx = {
    'DB_X': [
        {'id': 'E_A', 'name': 'Alpha enzyme', 'type': 'enzyme',
         'action': 'inhibitor'},
        {'id': 'E_B', 'name': 'Beta enzyme', 'type': 'enzyme',
         'action': 'inhibitor'},
        {'id': 'E_C', 'name': 'Gamma enzyme', 'type': 'enzyme',
         'action': 'inhibitor'},
    ],
    'DB_Y': [
        {'id': 'E_A', 'name': 'Alpha enzyme', 'type': 'enzyme',
         'action': 'substrate'},
        {'id': 'E_B', 'name': 'Beta enzyme', 'type': 'enzyme',
         'action': 'substrate'},
        {'id': 'E_C', 'name': 'Gamma enzyme', 'type': 'enzyme',
         'action': 'substrate'},
    ],
}
best = _find_key_entity_pk(
    'DB_X', 'DB_Y', 'metabolism',
    'drug X', 'drug Y',
    enzyme_idx, {}, {},
)
print(json.dumps({'name': best['key_entity_name'], 'id': best['key_entity_id']}))
""" % str(REPO_ROOT)

        results: list[dict] = []
        for seed in (0, 1, 42, 12345):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            r = subprocess.run(
                [_sys.executable, "-c", snippet],
                capture_output=True, text=True, env=env, check=True,
            )
            results.append(json.loads(r.stdout.strip()))
        first = results[0]
        for r in results[1:]:
            assert r == first, (
                f"PYTHONHASHSEED affects _find_key_entity_pk's pick — "
                f"deterministic tiebreaker has regressed.  Got {results}"
            )


# PD selection: confidence, relevance, then rank.


class TestSyntheticPdRankTiebreak:
    """PD uses DrugBank rank after confidence and DDI-type relevance."""

    def _pd_best(self, target_idx: dict, *, ddi_type: str = "bleeding") -> dict:
        from coldddi.annotations.ab_subdivision import _find_key_entity_pd

        return _find_key_entity_pd(
            "DB_X", "DB_Y", ddi_type,
            "drug X", "drug Y", target_idx,
        )

    def test_confidence_still_beats_rank(self):
        """``high`` confidence (inhibitor-inhibitor) must beat
        ``low`` confidence (agonist-inhibitor) even if the high-
        confidence target has a worse DrugBank rank."""
        target_idx = {
            "DB_X": [
                # Low-confidence target listed first (rank 0).
                {"id": "T_LOW", "name": "Low conf target", "type": "target",
                 "action": "agonist"},
                {"id": "T_HIGH", "name": "High conf target", "type": "target",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "T_LOW", "name": "Low conf target", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_HIGH", "name": "High conf target", "type": "target",
                 "action": "inhibitor"},
            ],
        }
        best = self._pd_best(target_idx)
        assert best is not None
        assert best["key_entity_id"] == "T_HIGH"
        assert best["confidence"] == "high"

    def test_rank_breaks_tie_when_confidence_and_relevance_equal(self):
        """Both targets share confidence=high + identical relevance
        (neither matches bleeding-keywords).  The lower DrugBank
        joint rank wins."""
        target_idx = {
            "DB_X": [
                # T_A: rank 0 for X.
                {"id": "T_A", "name": "Zeta target", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_B", "name": "Alpha target", "type": "target",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "T_A", "name": "Zeta target", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_B", "name": "Alpha target", "type": "target",
                 "action": "inhibitor"},
            ],
        }
        best = self._pd_best(target_idx)
        assert best is not None
        # Joint rank: T_A = 0+0 = 0, T_B = 1+1 = 2 → T_A wins
        # despite "Zeta" sorting AFTER "Alpha" by name.
        assert best["key_entity_id"] == "T_A"
        assert best["key_entity_name"] == "Zeta target"

    def test_pd_relevance_beats_rank(self):
        """When a target matches the DDI type's keyword list, the
        relevance bucket pulls it ahead of any DrugBank-rank
        advantage on a less-relevant target."""
        target_idx = {
            "DB_X": [
                # Rank-0 generic target; no keyword overlap with
                # "bleeding".
                {"id": "T_GEN", "name": "Generic protein", "type": "target",
                 "action": "inhibitor"},
                # Rank-1 coagulation factor; matches bleeding kw.
                {"id": "T_COAG", "name": "Coagulation factor X",
                 "type": "target", "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "T_GEN", "name": "Generic protein", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_COAG", "name": "Coagulation factor X",
                 "type": "target", "action": "inhibitor"},
            ],
        }
        best = self._pd_best(target_idx)
        assert best is not None
        assert best["key_entity_id"] == "T_COAG"

    def test_pd_pick_reproducible_across_hash_seeds(self):
        """PD selection is invariant to subprocess PYTHONHASHSEED values."""
        import json
        import os
        import subprocess
        import sys as _sys

        snippet = """
import sys, json
sys.path.insert(0, %r)
from coldddi.annotations.ab_subdivision import _find_key_entity_pd
# Three targets that all share confidence='high' (inhibitor-
# inhibitor), identical relevance (no DDI-type keyword match), and
# identical DrugBank rank (all rank 0 for each drug) -- the only
# deterministic key remaining is (name, id).
target_idx = {
    'DB_X': [
        {'id': 'T_A', 'name': 'Alpha target', 'type': 'target',
         'action': 'inhibitor'},
        {'id': 'T_B', 'name': 'Beta target', 'type': 'target',
         'action': 'inhibitor'},
        {'id': 'T_C', 'name': 'Gamma target', 'type': 'target',
         'action': 'inhibitor'},
    ],
    'DB_Y': [
        {'id': 'T_A', 'name': 'Alpha target', 'type': 'target',
         'action': 'inhibitor'},
        {'id': 'T_B', 'name': 'Beta target', 'type': 'target',
         'action': 'inhibitor'},
        {'id': 'T_C', 'name': 'Gamma target', 'type': 'target',
         'action': 'inhibitor'},
    ],
}
best = _find_key_entity_pd(
    'DB_X', 'DB_Y', 'unspecified ddi type',
    'drug X', 'drug Y', target_idx,
)
print(json.dumps({'name': best['key_entity_name'], 'id': best['key_entity_id']}))
""" % str(REPO_ROOT)

        results: list[dict] = []
        for seed in (0, 1, 42, 12345):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            r = subprocess.run(
                [_sys.executable, "-c", snippet],
                capture_output=True, text=True, env=env, check=True,
            )
            results.append(json.loads(r.stdout.strip()))
        first = results[0]
        for r in results[1:]:
            assert r == first, (
                f"PYTHONHASHSEED affects _find_key_entity_pd's pick — "
                f"deterministic tiebreaker has regressed.  Got {results}"
            )

    def test_pd_pick_reproducible(self):
        target_idx = {
            "DB_X": [
                {"id": "T_A", "name": "Target Two", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_B", "name": "Target One", "type": "target",
                 "action": "inhibitor"},
            ],
            "DB_Y": [
                {"id": "T_A", "name": "Target Two", "type": "target",
                 "action": "inhibitor"},
                {"id": "T_B", "name": "Target One", "type": "target",
                 "action": "inhibitor"},
            ],
        }
        from coldddi.annotations.ab_subdivision import _find_key_entity_pd

        b1 = _find_key_entity_pd(
            "DB_X", "DB_Y", "bleeding", "drug X", "drug Y", target_idx,
        )
        b2 = _find_key_entity_pd(
            "DB_X", "DB_Y", "bleeding", "drug X", "drug Y", target_idx,
        )
        assert b1["key_entity_id"] == b2["key_entity_id"]
        assert b1["key_entity_name"] == b2["key_entity_name"]
        # The strip-helper must remove BOTH private fields.
        assert "_drugbank_rank" not in b1
        assert "_relevance" not in b1


# End-to-end determinism on real data.


@pytest.mark.skipif(
    not (FULL_FILTERED / "drug_enzymes.csv").is_file(),
    reason="Full DrugBank filtered data missing — run reconstruct.py first.",
)
class TestAbSubdivisionIsDeterministic:
    """Two runs on full filtered data produce identical top entities and counts."""

    def _run_once(self):
        from coldddi.annotations.ab_subdivision import run_ab_subdivision
        from coldddi.annotations.pkpd_keywords import label_ddi_types

        edges = pd.read_csv(FULL_FILTERED / "ddi_edges.csv")
        drugs = pd.read_csv(
            FULL_FILTERED / "drugs.csv", usecols=["drugbank_id", "name"],
        )
        drug_id_to_name = dict(zip(drugs["drugbank_id"], drugs["name"]))
        pkpd = label_ddi_types(edges["ddi_type"].dropna().astype(str))
        key_df, _ = run_ab_subdivision(
            ddi_edges=edges, pk_pd_labels=pkpd,
            enzymes_csv=FULL_FILTERED / "drug_enzymes.csv",
            targets_csv=FULL_FILTERED / "drug_targets.csv",
            transporters_csv=FULL_FILTERED / "drug_transporters.csv",
            carriers_csv=FULL_FILTERED / "drug_carriers.csv",
            drug_id_to_name=drug_id_to_name, verbose=False,
        )
        pka = key_df[(key_df["pk_pd_label"] == "PK") & (key_df["has_key_entity"])]
        return pka["key_entity_name"].value_counts()

    def test_same_process_two_runs_identical(self):
        c1 = self._run_once()
        c2 = self._run_once()
        pd.testing.assert_series_equal(c1, c2)


# Paper Appendix A.3 qualitative claims.


@pytest.mark.skipif(
    not (FULL_FILTERED / "drug_enzymes.csv").is_file(),
    reason="Full DrugBank filtered data missing — run reconstruct.py first.",
)
class TestPkAQualitativeDominance:
    """Preserve Appendix A.3's CYP3A4 dominance without fixing version-specific percentages."""

    @pytest.fixture(scope="class")
    def pka_counts(self):
        from coldddi.annotations.ab_subdivision import run_ab_subdivision
        from coldddi.annotations.pkpd_keywords import label_ddi_types

        edges = pd.read_csv(FULL_FILTERED / "ddi_edges.csv")
        drugs = pd.read_csv(
            FULL_FILTERED / "drugs.csv", usecols=["drugbank_id", "name"],
        )
        drug_id_to_name = dict(zip(drugs["drugbank_id"], drugs["name"]))
        pkpd = label_ddi_types(edges["ddi_type"].dropna().astype(str))
        key_df, _ = run_ab_subdivision(
            ddi_edges=edges, pk_pd_labels=pkpd,
            enzymes_csv=FULL_FILTERED / "drug_enzymes.csv",
            targets_csv=FULL_FILTERED / "drug_targets.csv",
            transporters_csv=FULL_FILTERED / "drug_transporters.csv",
            carriers_csv=FULL_FILTERED / "drug_carriers.csv",
            drug_id_to_name=drug_id_to_name, verbose=False,
        )
        pka = key_df[(key_df["pk_pd_label"] == "PK") & (key_df["has_key_entity"])]
        return pka["key_entity_name"].value_counts()

    def test_cyp3a4_is_top1(self, pka_counts):
        """CYP3A4 must still be the most common PK-A mediator."""
        assert pka_counts.index[0] == "Cytochrome P450 3A4"

    def test_cyp3a4_dominant_share(self, pka_counts):
        """Use a loose 30% CYP3A4 lower bound to allow DrugBank version changes."""
        total = int(pka_counts.sum())
        cyp3a4 = int(pka_counts.get("Cytochrome P450 3A4", 0))
        share = cyp3a4 / total
        assert share > 0.30, (
            f"CYP3A4 share dropped to {share:.1%}; the 'CYP3A4 "
            "dominates PK-A' paper claim no longer holds"
        )

    def test_top5_qualitative_membership(self, pka_counts):
        """Top-five mediators remain CYP, transporter, or UGT family members."""
        top5 = list(pka_counts.head(5).index)
        # Every top-5 entity must be either a CYP or a recognised
        # transporter / UGT family member.
        recognised_substrings = ("Cytochrome P450 ", "ABC", "UDP-glucuronosyl")
        for name in top5:
            assert any(s in name for s in recognised_substrings), (
                f"Unexpected entity in PK-A top-5: {name!r}.  Top-5 "
                f"observed: {top5}.  Either the rank-based selection "
                f"drifted or the DrugBank version changed materially."
            )
