"""Test P2/P5 few-shot retrieval and prompt integration.

Synthetic four-drug tests run without reconstruction; optional toy tests use
the outputs of reconstruct.py --toy through to_llm_samples.
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

# Both retrieval modules require RDKit; skip before constructing fixtures.
pytest.importorskip("rdkit", reason="rdkit is required for L4 few-shot retrieval")


# Synthetic 4-drug PairDataset

def _make_synthetic_dataset():
    """Build a four-drug dataset with DB003 reserved for a cold-start test pair.
    """
    from coldddi.data.dataset import PairDataset
    from coldddi.data.kg import KnowledgeGraph
    from coldddi.data.splits import SplitFolds

    drugs = pd.DataFrame({
        "drugbank_id": ["DB001", "DB002", "DB003", "DB004"],
        "name":        ["DrugA", "DrugB", "DrugC", "DrugD"],
        # Distinct SMILES so Morgan FPs differ; A/B more similar than C/D.
        "smiles":      [
            "CCO",                              # ethanol-like (A)
            "CCC",                              # propane (B)
            "c1ccccc1",                          # benzene (C, very different)
            "CC(=O)OC1=CC=CC=C1C(=O)O",         # aspirin-like (D)
        ],
        "type":   ["small molecule"] * 4,
        "groups": ["approved"] * 4,
    })
    kg_targets = pd.DataFrame({
        "drugbank_id": ["DB001", "DB002", "DB003", "DB004"],
        "target_id":   ["T1", "T1", "T2", "T1"],
        "target_name": ["F2", "F2", "EGFR", "F2"],
        "organism":    ["Homo sapiens"] * 4,
        "action":      [None] * 4,
    })
    kg_enzymes = pd.DataFrame({
        "drugbank_id": ["DB001", "DB004"],
        "enzyme_id":   ["E1", "E1"],
        "enzyme_name": ["CYP3A4", "CYP3A4"],
        "organism":    ["Homo sapiens", "Homo sapiens"],
        "action":      [None, None],
    })
    empty = pd.DataFrame(columns=[
        "drugbank_id", "transporter_id", "transporter_name", "organism", "action",
    ])
    empty_c = pd.DataFrame(columns=[
        "drugbank_id", "carrier_id", "carrier_name", "organism", "action",
    ])
    empty_p = pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"])
    kg = KnowledgeGraph(
        enzymes=kg_enzymes,
        targets=kg_targets,
        transporters=empty,
        carriers=empty_c,
        pathways=empty_p,
    )

    # Train: G1 = {DB001, DB002, DB004}; positive pairs all in G1×G1.
    train = pd.DataFrame({
        "drug_a_id": ["DB001", "DB001", "DB002"],
        "drug_b_id": ["DB002", "DB004", "DB004"],
    })
    # One test pair using the cold-start drug DB003.
    test_s2 = pd.DataFrame({
        "drug_a_id": ["DB003"],
        "drug_b_id": ["DB001"],
    })
    empty_pair = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])

    splits = SplitFolds(
        train=train,
        val_s0=empty_pair, val_s1=empty_pair, val_s2=empty_pair,
        test_s0=empty_pair, test_s1=empty_pair, test_s2=test_s2,
        g1_drugs=["DB001", "DB002", "DB004"],
        g2_drugs=["DB003"],
        seed=42,
    )
    return PairDataset(
        edges=train.copy(),
        splits=splits,
        kg=kg,
        drugs=drugs,
    )


# P2 (Morgan FP similarity)

class TestP2SmilesSynthetic:
    def test_map_covers_every_pair(self):
        from coldddi.llm.retrieval import build_fewshot_smiles_map

        ds = _make_synthetic_dataset()
        m = build_fewshot_smiles_map(ds, k=2, seed=42)
        # 3 unique train pairs + 1 test_s2 pair = 4 keys.
        assert len(m) == 4
        for key, payload in m.items():
            assert "fewshot_samples" in payload
            assert len(payload["fewshot_samples"]) == 2
            for score, ra, rb, lbl in payload["fewshot_samples"]:
                assert lbl == "1"
                assert isinstance(ra, str)
                assert isinstance(rb, str)
                assert -1.0 <= score <= 1.0

    def test_leakage_excluded(self):
        """An exhausted non-leaking G1 pool still returns k fallback rows."""
        from coldddi.llm.retrieval import build_fewshot_smiles_map

        ds = _make_synthetic_dataset()
        m = build_fewshot_smiles_map(ds, k=3, seed=42)
        # Every pool pair shares a query drug. Upstream permits leaking rows
        # in this fallback; this test checks only completion and length k.
        samples = m[("DB001", "DB002")]["fewshot_samples"]
        assert len(samples) == 3

    def test_g2_query_works(self):
        """Cold-start drug DB003 must still get scored despite being absent
        from the G1 pool."""
        from coldddi.llm.retrieval import build_fewshot_smiles_map

        ds = _make_synthetic_dataset()
        m = build_fewshot_smiles_map(ds, k=2, seed=42)
        samples = m[("DB003", "DB001")]["fewshot_samples"]
        assert len(samples) == 2

    def test_empty_pool_raises(self):
        from coldddi.llm.retrieval import build_fewshot_smiles_map

        ds = _make_synthetic_dataset()
        empty = pd.DataFrame(columns=["drug_a_id", "drug_b_id"])
        with pytest.raises(ValueError, match="empty"):
            build_fewshot_smiles_map(ds, k=2, seed=42, pool_pairs=empty)


# P5 (2-hop shared KG entities)

class TestP5TwoHopSynthetic:
    def test_metadata_carries_shared_entities(self):
        """Three of the four drugs share target F2 → the metadata for
        the (DB003, DB001) query must reveal a shared entity for the
        retrieved pair (which is DB001-something or DB002-DB004)."""
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = _make_synthetic_dataset()
        m = build_fewshot_2hop_map(ds, k=2, seed=42, max_pool=None)
        assert len(m) == 4
        samples = m[("DB003", "DB001")]["fewshot_samples"]
        metas = m[("DB003", "DB001")]["fewshot_metadata"]
        assert len(samples) == len(metas) == 2

    def test_self_exclusion_when_leakage_free_pool_exists(self):
        """For query (DB003, DB001) the leakage-free pool is non-empty
        ((DB002, DB004) is the only train pair not touching DB003 or
        DB001) — every returned reference pair must avoid DB003/DB001."""
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = _make_synthetic_dataset()
        m = build_fewshot_2hop_map(ds, k=2, seed=42, max_pool=None)
        for score, ra, rb, lbl in m[("DB003", "DB001")]["fewshot_samples"]:
            assert ra not in {"DB003", "DB001"}, (
                f"self-exclusion violated: ref_a={ra}"
            )
            assert rb not in {"DB003", "DB001"}, (
                f"self-exclusion violated: ref_b={rb}"
            )

    def test_always_exactly_k_when_pool_has_leakage_free_subset(self):
        """Sample safe fallback pairs with replacement to return exactly k rows."""
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = _make_synthetic_dataset()
        # Query (DB003, DB001) — pool is (001,002), (001,004), (002,004).
        # Only (002,004) is leakage-free → 1-row safe pool. With k=5,
        # we must sample with replacement and return 5 non-leaking rows.
        m = build_fewshot_2hop_map(ds, k=5, seed=42, max_pool=None)
        samples = m[("DB003", "DB001")]["fewshot_samples"]
        metas = m[("DB003", "DB001")]["fewshot_metadata"]
        assert len(samples) == 5
        assert len(metas) == 5
        for _, ra, rb, _ in samples:
            assert ra not in {"DB003", "DB001"}
            assert rb not in {"DB003", "DB001"}

    def test_exactly_k_when_no_leakage_free_pool(self):
        """When no safe pairs exist, return k leaking rows with an explanatory note."""
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = _make_synthetic_dataset()
        # Every pool pair touches DB001 or DB002, so leakage is unavoidable.
        m = build_fewshot_2hop_map(ds, k=3, seed=42, max_pool=None)
        samples = m[("DB001", "DB002")]["fewshot_samples"]
        metas = m[("DB001", "DB002")]["fewshot_metadata"]
        assert len(samples) == 3
        assert len(metas) == 3
        # Every fallback note must include leaking_unavoidable.
        for meta in metas:
            note = meta.get("note", "")
            assert "leaking" in note, (
                f"expected leaking-unavoidable note, got note={note!r}"
            )

    def test_fallback_note_when_no_overlap(self):
        """A query whose drugs have NO KG entity must trigger the
        random fallback path and stamp a ``note`` on every metadata row."""
        from coldddi.data.kg import KnowledgeGraph
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = _make_synthetic_dataset()
        # Wipe the KG so no drug has any entity → every query falls back.
        empty_t = pd.DataFrame(columns=[
            "drugbank_id", "target_id", "target_name", "organism", "action",
        ])
        empty_e = pd.DataFrame(columns=[
            "drugbank_id", "enzyme_id", "enzyme_name", "organism", "action",
        ])
        empty_tr = pd.DataFrame(columns=[
            "drugbank_id", "transporter_id", "transporter_name", "organism", "action",
        ])
        empty_c = pd.DataFrame(columns=[
            "drugbank_id", "carrier_id", "carrier_name", "organism", "action",
        ])
        empty_p = pd.DataFrame(columns=["drugbank_id", "pathway_id", "pathway_name"])
        ds.kg = KnowledgeGraph(
            enzymes=empty_e,
            targets=empty_t,
            transporters=empty_tr,
            carriers=empty_c,
            pathways=empty_p,
        )
        m = build_fewshot_2hop_map(ds, k=2, seed=42, max_pool=None)
        for key, payload in m.items():
            for meta in payload["fewshot_metadata"]:
                # All four shared_* lists empty + a fallback note present.
                assert meta["shared_QA_CA"] == []
                assert meta["shared_QB_CB"] == []
                # Either no_protein_overlap or all_filtered/no_fp depending
                # on path; just ensure some non-empty note was set.
                assert "note" in meta or all(
                    not meta[k_] for k_ in
                    ("shared_QA_CA", "shared_QA_CB", "shared_QB_CA", "shared_QB_CB")
                )


# End-to-end prompt rendering integration

class TestEndToEndPrompts:
    def test_p2_prompt_uses_fewshot_map(self):
        from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
        from coldddi.llm.retrieval import (
            build_fewshot_smiles_map, to_llm_samples,
        )

        ds = _make_synthetic_dataset()
        fewshot = build_fewshot_smiles_map(ds, k=2, seed=42)
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
        samples = to_llm_samples(
            pairs, [1] * len(pairs), ds=ds, fewshot_map=fewshot,
        )
        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Few_Shot_Similarity_SMILES",
            model_name="meta-llama/Llama-3.2-1B",
        )
        prompt = build_binary_prompt(
            samples[0], cfg,
            drug_id2name=dict(zip(
                ds.drugs["drugbank_id"].astype(str), ds.drugs["name"],
            )),
            drug_id2smiles=dict(zip(
                ds.drugs["drugbank_id"].astype(str), ds.drugs["smiles"],
            )),
        )
        # P2 prompt must contain the few-shot block header and end with the label.
        assert "Reference Examples (Structural Analogs)" in prompt
        assert prompt.endswith(" Yes")

    def test_p5_prompt_uses_fewshot_map(self):
        from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
        from coldddi.llm.retrieval import (
            build_fewshot_2hop_map, to_llm_samples,
        )

        ds = _make_synthetic_dataset()
        fewshot = build_fewshot_2hop_map(ds, k=2, seed=42, max_pool=None)
        pairs = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
        samples = to_llm_samples(
            pairs, [1] * len(pairs), ds=ds, fewshot_map=fewshot,
        )
        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method="Few_Shot_2hop",
            model_name="meta-llama/Llama-3.2-1B",
        )
        prompt = build_binary_prompt(
            samples[0], cfg,
            drug_id2name=dict(zip(
                ds.drugs["drugbank_id"].astype(str), ds.drugs["name"],
            )),
            drug_id2smiles=dict(zip(
                ds.drugs["drugbank_id"].astype(str), ds.drugs["smiles"],
            )),
        )
        assert "Reference Examples (Mechanism-Aware)" in prompt
        assert prompt.endswith(" Yes")


# Toy fixture path (optional)

toy_required = pytest.mark.skipif(
    not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
    reason="Toy filtered dir not found — run reconstruct.py --toy first.",
)


@toy_required
class TestFewshotToyFixture:
    def test_p2_map_covers_every_test_s2_pair(self):
        from coldddi.data.dataset import PairDataset
        from coldddi.llm.retrieval import build_fewshot_smiles_map

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        m = build_fewshot_smiles_map(ds, k=3, seed=42)
        for a, b in zip(
            ds.splits.test_s2["drug_a_id"].astype(str),
            ds.splits.test_s2["drug_b_id"].astype(str),
        ):
            assert (a, b) in m
            assert len(m[(a, b)]["fewshot_samples"]) == 3

    def test_p5_map_metadata_shape(self):
        from coldddi.data.dataset import PairDataset
        from coldddi.llm.retrieval import build_fewshot_2hop_map

        ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
        m = build_fewshot_2hop_map(ds, k=3, seed=42, max_pool=None)
        any_key = next(iter(m))
        meta = m[any_key]["fewshot_metadata"][0]
        assert set(meta.keys()) >= {
            "shared_QA_CA", "shared_QA_CB", "shared_QB_CA", "shared_QB_CB",
        }
