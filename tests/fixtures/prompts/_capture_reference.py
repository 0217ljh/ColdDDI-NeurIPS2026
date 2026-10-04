"""Capture byte-exact prompts from upstream
``Version_1_1/dataloader/prompts/binary_cls.py`` for parity tests.

Usage::

    export COLDDDI_UPSTREAM_V11=/path/to/Version_1_1
    python tests/fixtures/prompts/_capture_reference.py

Each JSON fixture stores the inputs, configuration, seed, and rendered prompt.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The upstream checkout is needed only when regenerating the shipped fixtures.
_v11_env = os.environ.get("COLDDDI_UPSTREAM_V11")
if not _v11_env:
    raise RuntimeError(
        "Set COLDDDI_UPSTREAM_V11 to your local Version_1_1 checkout "
        "to regenerate fixtures (the shipped JSON fixtures are already "
        "checked in; you only need this script for re-capture)."
    )
V11_ROOT = Path(_v11_env)
if str(V11_ROOT) not in sys.path:
    sys.path.insert(0, str(V11_ROOT))

from dataloader.prompts.binary_cls import build_binary_prompt  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Shared inputs for all methods.
DRUG_A_ID = "DB00001"
DRUG_B_ID = "DB00002"
DRUG_A_NAME = "Lepirudin"
DRUG_B_NAME = "Cetuximab"
DRUG_A_SMILES = "CC[C@H](C)[C@H](NC(=O)C)C(=O)O"
DRUG_B_SMILES = "CN(C)CCC(=O)O"

KB = {
    "drug_id2name": {
        DRUG_A_ID: DRUG_A_NAME,
        DRUG_B_ID: DRUG_B_NAME,
        "DB00003": "Aspirin",
        "DB00004": "Warfarin",
        "DB00005": "Metformin",
        "DB00006": "Ibuprofen",
    },
    "drug_id2smiles": {
        DRUG_A_ID: DRUG_A_SMILES,
        DRUG_B_ID: DRUG_B_SMILES,
        "DB00003": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "DB00004": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "DB00005": "CN(C)C(=N)N=C(N)N",
        "DB00006": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    },
}

NEIGHBORS = {
    "neighbors": {
        "transporters": {"A": ["SLC22A1", "ABCB1"], "B": ["unknown"]},
        "pathways":     {"A": ["Coagulation cascade"], "B": ["EGFR signaling"]},
        "targets":      {"A": ["F2", "F10"], "B": ["EGFR"]},
        "enzymes":      {"A": ["CYP3A4", "CYP2D6"], "B": ["CYP2C9"]},
        "carriers":     {"A": ["ALB"], "B": ["unknown"]},
        "smiles":       {"A": [DRUG_A_SMILES], "B": [DRUG_B_SMILES]},
    }
}

# Few-shot samples: (score, ref_a, ref_b, label_str).
FEWSHOT_SAMPLES = [
    (0.91, "DB00003", "DB00004", "1"),
    (0.85, "DB00005", "DB00006", "0"),
    (0.80, "DB00003", "DB00006", "1"),
]

# Few-shot 2-hop metadata, one per fewshot sample.
FEWSHOT_2HOP_META = [
    {
        "shared_QA_CA": [("F2", "targets")],
        "shared_QB_CB": [("EGFR", "targets")],
        "shared_QA_CB": [],
        "shared_QB_CA": [],
    },
    {
        "shared_QA_CA": [],
        "shared_QB_CB": [],
        "shared_QA_CB": [("CYP3A4", "enzymes")],
        "shared_QB_CA": [("ALB", "carriers")],
    },
    {
        "shared_QA_CA": [],
        "shared_QB_CB": [],
        "shared_QA_CB": [],
        "shared_QB_CA": [],
    },
]

# Key entity map used by the masking variants (R4/R5/R6/R7).
# A-group case (has_key_entity=True): the named key entity is replaced.
KEY_ENTITY_MAP_A = {
    (DRUG_A_ID, DRUG_B_ID): {
        "key_entity_name": "F2",
        "key_entity_type": "targets",
        "has_key_entity": True,
    },
}
# B-group case (has_key_entity=False): no specific entity is known, so
# the masking falls back to "first non-unknown per type".
KEY_ENTITY_MAP_B = {
    (DRUG_A_ID, DRUG_B_ID): {
        "key_entity_name": "",
        "key_entity_type": "",
        "has_key_entity": False,
    },
}


def _make_cfg(method: str, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        Task_Name="Binary_cls",
        Method=method,
        model_name=model_name,
    )


def _make_bundle(key_entity_map=None):
    return SimpleNamespace(
        extra={
            "kb": KB,
            "key_entity_map": key_entity_map or KEY_ENTITY_MAP_A,
        }
    )


def _base_sample(label: int) -> dict:
    return {
        "drug_a_id": DRUG_A_ID,
        "drug_b_id": DRUG_B_ID,
        "drug_a_name": DRUG_A_NAME,
        "drug_b_name": DRUG_B_NAME,
        "drugA_name": DRUG_A_NAME,  # alias used by zero_shot branch
        "drugB_name": DRUG_B_NAME,
        "label": label,
    }


METHODS = [
    # (paper_name,   internal Method string fed to cfg.Method)
    ("P1_zero_shot",                          "Zero_Shot_Sequence"),
    ("P2_few_shot_similarity_smiles",         "Few_Shot_Similarity_SMILES"),
    ("P3_one_hop_subgraph_single",            "One_Hop_Subgraph_Single"),
    ("P4_one_hop_subgraph_sequence",          "One_Hop_Subgraph_Sequence"),
    ("P5_few_shot_2hop",                      "Few_Shot_2hop"),
    # Masking variants (R0 is identical to P4)
    ("R1_ohs_mask_name",                      "OHS_Mask_Name"),
    ("R2_ohs_full",                           "OHS_Full"),
    ("R3_ohs_full_mask_name",                 "OHS_Full_Mask_Name"),
    ("R4_ohs_full_mask_entity",               "OHS_Full_Mask_Entity"),
    ("R5_ohs_full_mask_name_entity",          "OHS_Full_Mask_Name_Entity"),
    ("R6_ohs_mask_entity",                    "OHS_Mask_Entity"),
    ("R7_ohs_mask_name_entity",               "OHS_Mask_Name_Entity"),
]

#: Paper families cover every method and both labels for byte-exact parity.
PAPER_MODEL_FAMILIES = [
    ("llama", "/tmp/mydata/Models/Hub/Llama-3.2-1B"),
    ("qwen",  "/tmp/mydata/Models/Hub/Qwen2.5-3B"),
    ("gemma", "/tmp/mydata/Models/Hub/gemma-3-1b-pt"),
]

#: Other families need only P1: chat formatting is independent of the method.
NON_PAPER_MODEL_FAMILIES = [
    ("mistral",  "mistralai/Mistral-7B-Instruct"),
    ("deepseek", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("chatglm",  "THUDM/chatglm3-6b"),
    ("baichuan", "baichuan-inc/Baichuan2-7B"),
]


def _build_sample(method_internal: str, label: int) -> dict:
    s = _base_sample(label)
    # Methods that consume subgraph_1hop:
    if method_internal in (
        "One_Hop_Subgraph_Single",
        "One_Hop_Subgraph_Sequence",
        "One_Hop_Subgraph_Sequence_Mask_Name",
        "One_Hop_Subgraph_Sequence_Mask_PK",
        "OHS_Mask_Name", "OHS_Mask_Entity", "OHS_Mask_Name_Entity",
        "OHS_Full", "OHS_Full_Mask_Name",
        "OHS_Full_Mask_Entity", "OHS_Full_Mask_Name_Entity",
    ):
        s["subgraph_1hop"] = NEIGHBORS
    if method_internal == "Few_Shot_Similarity_SMILES":
        s["fewshot_samples"] = list(FEWSHOT_SAMPLES)
    if method_internal == "Few_Shot_2hop":
        s["fewshot_samples"] = list(FEWSHOT_SAMPLES)
        s["fewshot_metadata"] = list(FEWSHOT_2HOP_META)
    return s


def _capture_one(
    *,
    paper_name: str,
    method_internal: str,
    family: str,
    model_name: str,
    label: int,
    key_entity_map: dict,
    ke_variant: str,  # "A" or "B" — appended to filename only for masking variants
) -> str:
    np.random.seed(20260511)
    cfg = _make_cfg(method_internal, model_name)
    bundle = _make_bundle(key_entity_map=key_entity_map)
    sample = _build_sample(method_internal, label)
    sample_input_snapshot = copy.deepcopy(sample)
    prompt = build_binary_prompt(sample, cfg, bundle=bundle)

    payload = {
        "paper_name": paper_name,
        "method_internal": method_internal,
        "model_family": family,
        "model_name": model_name,
        "label": label,
        "ke_variant": ke_variant,
        "sample": sample_input_snapshot,
        "kb": KB,
        "key_entity_map": {
            f"{a}|{b}": v for (a, b), v in key_entity_map.items()
        },
        "fewshot_samples": (
            FEWSHOT_SAMPLES
            if method_internal in ("Few_Shot_Similarity_SMILES", "Few_Shot_2hop")
            else None
        ),
        "fewshot_metadata": (
            FEWSHOT_2HOP_META if method_internal == "Few_Shot_2hop" else None
        ),
        "np_seed": 20260511,
        "prompt": prompt,
    }

    # Only entity-masking variants need a suffix; keep other fixture names stable.
    suffix = ""
    if method_internal in (
        "OHS_Mask_Entity", "OHS_Mask_Name_Entity",
        "OHS_Full_Mask_Entity", "OHS_Full_Mask_Name_Entity",
    ):
        suffix = f"__ke{ke_variant}"
    fname = f"{paper_name}__{family}__label{label}{suffix}.json"
    with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fname


def main():
    n = 0

    # Paper families cover every method and label, with A/B entity-mask variants.
    entity_mask_methods = {
        "OHS_Mask_Entity",
        "OHS_Mask_Name_Entity",
        "OHS_Full_Mask_Entity",
        "OHS_Full_Mask_Name_Entity",
    }
    for paper_name, method_internal in METHODS:
        for family, model_name in PAPER_MODEL_FAMILIES:
            for label in (0, 1):
                if method_internal in entity_mask_methods:
                    for variant, ke_map in (("A", KEY_ENTITY_MAP_A),
                                             ("B", KEY_ENTITY_MAP_B)):
                        _capture_one(
                            paper_name=paper_name,
                            method_internal=method_internal,
                            family=family,
                            model_name=model_name,
                            label=label,
                            key_entity_map=ke_map,
                            ke_variant=variant,
                        )
                        n += 1
                else:
                    _capture_one(
                        paper_name=paper_name,
                        method_internal=method_internal,
                        family=family,
                        model_name=model_name,
                        label=label,
                        key_entity_map=KEY_ENTITY_MAP_A,
                        ke_variant="A",
                    )
                    n += 1

    # Other families cover P1 only.
    for family, model_name in NON_PAPER_MODEL_FAMILIES:
        for label in (0, 1):
            _capture_one(
                paper_name="P1_zero_shot",
                method_internal="Zero_Shot_Sequence",
                family=family,
                model_name=model_name,
                label=label,
                key_entity_map=KEY_ENTITY_MAP_A,
                ke_variant="A",
            )
            n += 1

    print(f"[capture] wrote {n} reference fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
