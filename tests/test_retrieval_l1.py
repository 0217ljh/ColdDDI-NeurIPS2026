"""Test retrieval and views with synthetic and reconstructed toy data.

Synthetic tests do not need generated fixtures; toy tests skip when the
intermediate directory from reconstruct.py --toy is missing.
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


# Synthetic fixtures (no toy dataset required)

@pytest.fixture
def synthetic_kg():
    from coldddi.data.kg import KnowledgeGraph

    enzymes = pd.DataFrame({
        "drugbank_id": ["DB001", "DB001", "DB001", "DB001", "DB002"],
        "enzyme_id":   ["E1", "E2", "E3", "E4", "E1"],
        "enzyme_name": ["CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP3A4"],
        "organism":    ["Homo sapiens"] * 5,
        "action":      [None] * 5,
    })
    targets = pd.DataFrame({
        "drugbank_id": ["DB001", "DB002", "DB002"],
        "target_id":   ["T1", "T2", "T3"],
        "target_name": ["F2", "EGFR", "ESR1"],
        "organism":    ["Homo sapiens"] * 3,
        "action":      [None] * 3,
    })
    transporters = pd.DataFrame({
        "drugbank_id": ["DB001"],
        "transporter_id": ["TR1"],
        "transporter_name": ["ABCB1"],
        "organism": ["Homo sapiens"],
        "action": [None],
    })
    carriers = pd.DataFrame({
        "drugbank_id": ["DB002"],
        "carrier_id": ["C1"],
        "carrier_name": ["ALB"],
        "organism": ["Homo sapiens"],
        "action": [None],
    })
    pathways = pd.DataFrame({
        "drugbank_id": ["DB001", "DB002"],
        "pathway_id":  ["P1", "P2"],
        "pathway_name": ["Coagulation cascade", "EGFR signaling"],
    })
    return KnowledgeGraph(
        enzymes=enzymes,
        targets=targets,
        transporters=transporters,
        carriers=carriers,
        pathways=pathways,
    )


@pytest.fixture
def synthetic_drugs():
    return pd.DataFrame({
        "drugbank_id": ["DB001", "DB002"],
        "name":   ["Lepirudin", "Cetuximab"],
        "smiles": ["CC[C@H](C)C", "CN(C)CC"],
    })


class TestSubgraphSynthetic:
    def test_top3_truncation(self, synthetic_kg, synthetic_drugs):
        """DB001 has 4 enzymes; top-3 must keep exactly 3 (the first
        three in the table order), full KG must keep all 4."""
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        sm3 = build_subgraph_map(synthetic_kg, synthetic_drugs, topk=3)
        smF = build_subgraph_map(synthetic_kg, synthetic_drugs, topk=None)
        assert sm3.data["DB001"]["enzymes"] == ["CYP3A4", "CYP2D6", "CYP2C9"]
        assert smF.data["DB001"]["enzymes"] == [
            "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2",
        ]

    def test_unknown_fallback(self, synthetic_kg, synthetic_drugs):
        """DB001 has no carrier in the synthetic KG; the slot must be
        ['unknown'] so the prompt builder's singularised branch fires."""
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        sm = build_subgraph_map(synthetic_kg, synthetic_drugs, topk=3)
        assert sm.data["DB001"]["carriers"] == ["unknown"]
        assert sm.data["DB002"]["transporters"] == ["unknown"]

    def test_smiles_single_element_list(self, synthetic_kg, synthetic_drugs):
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        sm = build_subgraph_map(synthetic_kg, synthetic_drugs, topk=3)
        assert sm.data["DB001"]["smiles"] == ["CC[C@H](C)C"]
        assert sm.data["DB002"]["smiles"] == ["CN(C)CC"]

    def test_get_neighbors_block_shape(self, synthetic_kg, synthetic_drugs):
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        sm = build_subgraph_map(synthetic_kg, synthetic_drugs, topk=3)
        block = sm.get_neighbors_block("DB001", "DB002")
        assert set(block["neighbors"].keys()) == {
            "transporters", "pathways", "targets",
            "enzymes", "carriers", "smiles",
        }
        for stype, ab in block["neighbors"].items():
            assert {"A", "B"} <= set(ab.keys())

    def test_negative_topk_rejected(self, synthetic_kg, synthetic_drugs):
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        with pytest.raises(ValueError, match="non-negative"):
            build_subgraph_map(synthetic_kg, synthetic_drugs, topk=-1)


# Toy fixture path (depends on reconstruct.py --toy)

toy_required = pytest.mark.skipif(
    not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
    reason="Toy filtered dir not found — run reconstruct.py --toy first.",
)


@pytest.fixture(scope="module")
def toy_dataset():
    pytest.importorskip("rdkit")  # PairDataset loads SSP-related modules
    from coldddi.data.dataset import PairDataset

    return PairDataset.from_release_dir(TOY_RELEASE, seed=42)


@toy_required
class TestSubgraphToyFixture:
    def test_build_top3(self, toy_dataset):
        from coldddi.llm.retrieval.kg_subgraph import (
            SUBGRAPH_ENTITY_TYPES,
            build_subgraph_map,
        )

        sm = build_subgraph_map(
            toy_dataset.kg, toy_dataset.drugs, topk=3
        )
        assert sm.selected_entities == SUBGRAPH_ENTITY_TYPES
        seen = set()
        for _, df in toy_dataset.splits.items():
            seen.update(df["drug_a_id"].astype(str))
            seen.update(df["drug_b_id"].astype(str))
        assert seen <= set(sm.data.keys())
        for did, row in sm.data.items():
            for stype, vals in row.items():
                if stype == "smiles":
                    assert len(vals) == 1, f"{did}.smiles must be 1 element"
                else:
                    assert len(vals) <= 3, f"{did}.{stype} exceeds top-3 cap"
                assert vals, f"{did}.{stype} should fall back to ['unknown']"

    def test_top3_truncates_at_least_one_drug(self, toy_dataset):
        from coldddi.llm.retrieval.kg_subgraph import build_subgraph_map

        sm_full = build_subgraph_map(
            toy_dataset.kg, toy_dataset.drugs, topk=None
        )
        sm_top3 = build_subgraph_map(
            toy_dataset.kg, toy_dataset.drugs, topk=3
        )
        truncated = False
        for did, full in sm_full.data.items():
            top3 = sm_top3.data[did]
            for stype in full:
                if stype == "smiles":
                    continue
                if len(full[stype]) > len(top3[stype]):
                    truncated = True
                    break
            if truncated:
                break
        assert truncated, "toy KG has no drug with >3 neighbours in any type"

    def test_to_llm_samples_with_subgraph(self, toy_dataset):
        from coldddi.llm.prompts import (
            PromptBuildConfig, build_binary_prompt,
        )
        from coldddi.llm.retrieval import build_subgraph_map, to_llm_samples

        sm = build_subgraph_map(toy_dataset.kg, toy_dataset.drugs, topk=3)
        pairs = toy_dataset.splits.test_s2[["drug_a_id", "drug_b_id"]].head(5)
        labels = [1] * len(pairs)
        samples = to_llm_samples(
            pairs, labels, ds=toy_dataset, subgraph_map=sm
        )
        assert len(samples) == 5
        for s in samples:
            assert "subgraph_1hop" in s
            assert s["label"] == 1
            cfg = PromptBuildConfig(
                task_name="Binary_cls",
                method="One_Hop_Subgraph_Sequence",
                model_name="Llama-3.2-1B",
            )
            prompt = build_binary_prompt(
                s, cfg,
                drug_id2name={
                    s["drug_a_id"]: s["drug_a_name"],
                    s["drug_b_id"]: s["drug_b_name"],
                },
                drug_id2smiles={},
                key_entity_map={},
            )
            assert prompt.endswith(" Yes")
