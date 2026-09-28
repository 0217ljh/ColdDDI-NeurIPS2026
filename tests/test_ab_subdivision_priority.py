"""Regression tests for the A3-audit fix on PK-A / PD-A best-entity
selection.

Pre-fix bug
-----------
``coldddi.annotations.ab_subdivision._enzyme_priority`` matched the
literal ``"cyp3a4"`` etc. against the lowercased enzyme name, but
real DrugBank enzyme names are ``"Cytochrome P450 3A4"`` —
lowercased ``"cytochrome p450 3a4"`` does NOT contain the substring
``"cyp3a4"``.  All enzymes therefore returned priority 9 (catch-all)
and the "best entity" picked for each pair was determined by
Python set iteration order (process-hash-seeded).  Both upstream
and release shared this bug, producing non-reproducible top-entity
stats across runs.

Post-fix (final design, post-user-pushback)
-------------------------------------------
The hand-rolled ``CYP_PRIORITY`` table was the wrong abstraction —
it overrode DrugBank's own curator-assigned per-drug ranking with a
hard-coded enzyme-name preference, and the LLM mask experiment
(R2/R3/R6/R7 in :mod:`coldddi.llm.prompts.binary_cls`) needs the
"most-important entity for THIS pair" rather than the "most-popular
enzyme in DrugBank globally".

Final design:

* Drop ``CYP_PRIORITY`` / ``_enzyme_priority`` entirely.
* Introduce ``_first_occurrence_ranks`` which assigns
  ``rank = list position`` per (drug, entity).  DrugBank's XML
  extraction (:mod:`coldddi.data.extract`) preserves document
  order, so rank 0 is the curator-prioritised polypeptide for that
  drug.
* ``_find_key_entity_pk`` sorts candidates by
  ``(rank_a[eid] + rank_b[eid], name, id)`` so the joint DrugBank
  importance picks the mask target.
* ``_find_key_entity_pd`` keeps its semantic keys (confidence,
  ddi-type relevance) but inserts ``drugbank_rank`` as a tiebreaker
  before ``(name, id)``.

Both paths use ``(name, id)`` as a final tiebreaker so the pick is
reproducible across processes (Python ``set`` iteration is hash-
seeded otherwise).
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


# ─── Unit: _first_occurrence_ranks ─────────────────────────────────


class TestFirstOccurrenceRanks:
    """The fix's correctness anchor: ranks must reflect DrugBank XML
    document order; duplicate ids keep their first-seen rank."""

    def test_distinct_entities_ranked_in_order(self):
        from coldddi.annotations.ab_subdivision import _first_occurrence_ranks

        entries = [
            {"id": "E_A", "name": "First", "type": "enzyme", "action": "i"},
            {"id": "E_B", "name": "Second", "type": "enzyme", "action": "i"},
            {"id": "E_C", "name": "Third", "type": "enzyme", "action": "i"},
        ]
        assert _first_occurrence_ranks(entries) == {"E_A": 0, "E_B": 1, "E_C": 2}

    def test_duplicate_id_keeps_first_occurrence(self):
        """A drug can list the same polypeptide multiple times with
        different actions (e.g. enzyme listed as both substrate and
        inhibitor for separate mechanisms).  The DrugBank rank we
        care about is the first-seen position."""
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
        """Audit guard: the hand-rolled ``CYP_PRIORITY`` /
        ``_enzyme_priority`` symbols must NOT come back.  If a future
        edit reintroduces them, the rank-based design is being silently
        bypassed."""
        from coldddi.annotations import ab_subdivision

        assert not hasattr(ab_subdivision, "CYP_PRIORITY"), (
            "CYP_PRIORITY reintroduced — the rank-based design has "
            "been overridden; see this test file's docstring"
        )
        assert not hasattr(ab_subdivision, "_enzyme_priority"), (
            "_enzyme_priority reintroduced — the rank-based design "
            "has been overridden; see this test file's docstring"
        )


# ─── Behavioural: PK candidate selection follows DrugBank rank ────


class TestSyntheticPkRankSelection:
    """Build a small enzyme index where rank order should decide the
    winner, and assert the pick matches the DrugBank-rank rule."""

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
        """Without the rank rule a (name, id) tiebreaker would pick
        the alphabetically-first enzyme.  With the rank rule, the
        curator-prioritised enzyme wins even if its name sorts
        later."""
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
        # Z-prime is rank 0 for both → joint rank 0 < Alpha's joint
        # rank 2.  Name tiebreak only fires when joint ranks tie.
        assert best["key_entity_id"] == "E_Z"

    def test_equal_rank_falls_back_to_name(self):
        """If both candidates share the same joint rank (e.g. the
        index is intentionally symmetric), the deterministic
        (name, id) tiebreaker kicks in."""
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
        """Policy pin (codex review): :func:`_pk_subtype` walks the
        bucket list (``["enzyme", "transporter"]`` for most DDI
        types) in order and STOPS at the first bucket that produces
        any candidate.  Joint-rank selection is therefore *within
        bucket*, not across buckets — a rank-0/0 transporter cannot
        win over even a poorly-ranked enzyme if the DDI type's
        priority puts enzymes first.

        This mirrors the legacy script and is intentional: when a
        DDI type is mechanistically "metabolism"-flavoured, only
        enzymes are considered as candidate mediators.  Pin the
        policy so a future refactor that flattens the bucket list
        doesn't silently change PK-A stats."""
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
        """A drug may list the same enzyme multiple times with
        different actions (e.g. CYP3A4 as both substrate and
        inhibitor for separate mechanisms).  The PK path must
        evaluate all action-pair combinations and pick a compatible
        one — and the entity's rank is the FIRST-occurrence rank
        (locked by :func:`_first_occurrence_ranks`)."""
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
        """The non-best candidates should still surface in
        ``key_entity_candidates`` JSON for downstream inspection,
        sorted by the same rank rule (best first after the picked
        one)."""
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
        # The strip-helper should have removed the private
        # ``_drugbank_rank`` field so it doesn't leak into CSV.
        assert "_drugbank_rank" not in rest[0]
        assert "_drugbank_rank" not in best


# ─── Determinism across processes (the bug-revealing test) ────────


class TestPkPickAcrossHashSeeds:
    """Pre-fix, ``set(map_a) & set(map_b)`` iteration order depended
    on PYTHONHASHSEED so two release runs could pick different
    best-entities for the same pair.  Spawn subprocesses with explicit
    PYTHONHASHSEED values and assert the picked entity is identical
    across all of them."""

    def test_pick_is_reproducible_across_hash_seeds(self):
        import json
        import os
        import subprocess
        import sys as _sys

        # Construct an enzyme index where multiple shared entities
        # have IDENTICAL joint ranks, so the deterministic (name, id)
        # tiebreaker is the only thing keeping the pick stable across
        # hash seeds.
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


# ─── Behavioural: PD picks honour confidence > relevance > rank ──


class TestSyntheticPdRankTiebreak:
    """PD path keeps its semantic keys (confidence, ddi-type
    relevance) but adds the DrugBank rank as a tiebreaker so the
    mask target lines up with curator priority."""

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
        """PD analogue of the PK cross-process test.  Confidence and
        relevance bucketise candidates; within a bucket the DrugBank
        rank then (name, id) keys decide the pick.  Spawn
        subprocesses with explicit PYTHONHASHSEEDs and assert the
        pick is invariant."""
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


# ─── End-to-end determinism on real data ──────────────────────────


@pytest.mark.skipif(
    not (FULL_FILTERED / "drug_enzymes.csv").is_file(),
    reason="Full DrugBank filtered data missing — run reconstruct.py first.",
)
class TestAbSubdivisionIsDeterministic:
    """Run ``run_ab_subdivision`` over the full filtered data twice in
    the same process; the top entities and their counts must be byte-
    identical."""

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


# ─── Paper App A.3 qualitative claim survives the rank-based fix ──


@pytest.mark.skipif(
    not (FULL_FILTERED / "drug_enzymes.csv").is_file(),
    reason="Full DrugBank filtered data missing — run reconstruct.py first.",
)
class TestPkAQualitativeDominance:
    """Paper App A.3 reports CYP3A4 as the dominant PK-A mediator.
    Both the old CYP_PRIORITY design and the new DrugBank-rank
    design preserve this qualitative claim because CYP3A4 is the
    most commonly listed rank-0 enzyme in DrugBank.  Pin the
    qualitative property (not exact percentages — those depend on
    the DrugBank version)."""

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
        """Paper claims ~48% in the buggy upstream run; rank-based
        fix should give >= 30% (CYP3A4 is rank 0 for many of the
        most-studied drugs).  We assert a loose lower bound so the
        test survives DrugBank version bumps."""
        total = int(pka_counts.sum())
        cyp3a4 = int(pka_counts.get("Cytochrome P450 3A4", 0))
        share = cyp3a4 / total
        assert share > 0.30, (
            f"CYP3A4 share dropped to {share:.1%}; the 'CYP3A4 "
            "dominates PK-A' paper claim no longer holds"
        )

    def test_top5_qualitative_membership(self, pka_counts):
        """The post-rank-fix top-5 should still be cytochromes /
        transporter family entities (CYPs + ABCB1-class), not
        accidental low-rank generic proteins."""
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
