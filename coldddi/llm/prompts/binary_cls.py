"""Build binary DDI prompts from task, method, context and pair information.

Task instructions form the system message; method instructions, examples,
output constraints and the query form the user message. The assistant
message carries the training answer or an empty inference slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from coldddi.llm.prompts.chat_formatter import format_messages_for_model


# Task instructions

TASK_DISPLAY_NAME: dict[str, str] = {
    "Binary_cls": "Binary Classification of Drug-Drug Interactions (DDI)",
}

TASK_INSTRUCTION: dict[str, str] = {
    "Binary_cls": (
        "You are an expert pharmacologist and clinical research assistant. "
        "Your objective is to determine whether a clinically significant "
        "drug-drug interaction (DDI) exists between the two provided drugs "
        "identified by the '[Prediction]' tag. "
        "Consider potential pharmacokinetic interactions and pharmacodynamic "
        "interactions in your assessment."
    ),
}


# Method instructions

_OHS_KG_INSTRUCTION = (
    "This inference is augmented by external biomedical knowledge transformed "
    "into natural language descriptions. "
    "Structured Knowledge Graph facts regarding each drug have been linearized "
    "into coherent sentences grouping biological associations by category "
    "(e.g., whose targets are xxx, xxx, and xxx). "
    "Utilize these semantic descriptions to identify potential interaction "
    "mechanisms.\n"
)

_OHS_KG_DESC_INSTRUCTION = (
    "This inference is augmented by external biomedical knowledge combined "
    "with a per-drug clinical pharmacology description. "
    "Structured Knowledge Graph facts regarding each drug have been linearized "
    "into coherent sentences grouping biological associations by category "
    "(e.g., whose targets are xxx, xxx, and xxx). "
    "In addition, each drug is accompanied by a 150-200 word clinical "
    "description covering pharmacological class, primary indications, "
    "physiological (organ / tissue-level) mechanism of action, dosing, "
    "common adverse effects, and contraindications. "
    "The description is generated independently of the Knowledge Graph and "
    "does not restate protein-level facts already present in the graph. "
    "Utilize both the graph facts and the clinical description together to "
    "identify potential interaction mechanisms.\n"
)

_DESC_ONLY_INSTRUCTION = (
    "This inference is augmented by a per-drug clinical pharmacology "
    "description. Each drug is accompanied by a 150-200 word clinical "
    "description covering pharmacological class, primary indications, "
    "physiological (organ / tissue-level) mechanism of action, dosing, "
    "common adverse effects, and contraindications. No structured "
    "biomedical knowledge graph (targets, enzymes, transporters, "
    "carriers, pathways) is provided. The description is generated "
    "independently of any protein-level knowledge graph and does not "
    "contain protein / enzyme / transporter / carrier or interacting-drug "
    "names. Utilize this clinical description together with your internal "
    "pharmacological knowledge to identify potential interaction "
    "mechanisms.\n"
)

METHOD_PROMPT: dict[str, str] = {
    "zero_shot": (
        "This is a Zero-Shot inference mode. No external context, drug "
        "descriptions, or labeled examples are provided. You must rely "
        "exclusively on your pre-trained internal knowledge regarding drug "
        "mechanisms and chemical knowledge to make the prediction.\n"
    ),
    "few_shot_similarity_smiles": (
        "This is a Few-Shot inference task enhanced by molecular structural "
        "similarity. You are provided with reference examples selected based "
        "on high SMILES similarity (Morgan fingerprints) to the target drug "
        "pair. Then you can infer the interaction status of the target pair "
        "by analyzing the shared molecular substructures and interaction "
        "patterns observed in these structural analogs.\n"
    ),
    "one_hop_subgraph_single": (
        "This inference is augmented by external biomedical knowledge. "
        "For each drug, you are provided with a set of explicit 1-hop "
        "Knowledge Graph Triples (Subject, Relation, Object) representing "
        "its immediate biological associations (e.g., (Drug A, has targets, "
        "xxx target)). Integrate these structured facts with your internal "
        "knowledge to assess the likelihood of an interaction.\n"
    ),
    # P4 masking variants share the same instruction.
    "one_hop_subgraph_sequence":       _OHS_KG_INSTRUCTION,
    "one_hop_subgraph_sequence_mask_name": _OHS_KG_INSTRUCTION,
    "one_hop_subgraph_sequence_mask_pk":   _OHS_KG_INSTRUCTION,
    "ohs_mask_name":              _OHS_KG_INSTRUCTION,
    "ohs_mask_entity":            _OHS_KG_INSTRUCTION,
    "ohs_mask_name_entity":       _OHS_KG_INSTRUCTION,
    "ohs_full":                   _OHS_KG_INSTRUCTION,
    "ohs_full_mask_name":         _OHS_KG_INSTRUCTION,
    "ohs_full_mask_entity":       _OHS_KG_INSTRUCTION,
    "ohs_full_mask_name_entity":  _OHS_KG_INSTRUCTION,
    "one_hop_subgraph_sequence_desc": _OHS_KG_DESC_INSTRUCTION,
    "desc_only": _DESC_ONLY_INSTRUCTION,
    "few_shot_2hop": (
        "This task utilizes a Mechanism-Aware Few-Shot strategy. "
        "The provided reference examples are not selected randomly, they "
        "are retrieved based on shared biological entities (targets, "
        "transporters, enzymes) from a biomedical Knowledge Graph. "
        "Specifically, the reference drugs share critical protein "
        "associations with the query drugs (with tag '[Prediction]'), "
        "implying a potential 2nd-order mechanistic similarity (Query Drug "
        "-> Shared Protein <- Reference Drug). Use these mechanistically "
        "related examples to infer the interaction status of the target "
        "pair.\n"
    ),
}


# Answer constraint

OUTPUT_CONSTRAINT_TEXT = (
    "Output Constraint: Respond strictly with exactly one token: ' Yes' or "
    "' No'. Do not provide any explanations, confidence scores, punctuation, "
    "or additional text.\n"
)


# Configuration and method aliases

@dataclass
class PromptBuildConfig:
    """Prompt configuration and optional per-drug descriptions."""

    task_name: str = "Binary_cls"
    method: str = "Zero_Shot_Sequence"
    model_name: str = ""
    # P6/P7 use extra["drug_id2description"]; P1--P5 leave this empty.
    extra: dict[str, Any] = field(default_factory=dict)


_METHOD_ALIASES: dict[str, str] = {
    # Map public method names to prompt keys.
    "Zero_Shot":                              "zero_shot",
    "Zero_Shot_Sequence":                     "zero_shot",
    "Few_Shot_Similarity_SMILES":             "few_shot_similarity_smiles",
    "One_Hop_Subgraph_Single":                "one_hop_subgraph_single",
    "One_Hop_Subgraph_Sequence":              "one_hop_subgraph_sequence",
    "One_Hop_Subgraph_Sequence_Desc":         "one_hop_subgraph_sequence_desc",
    "Desc_Only":                              "desc_only",
    "One_Hop_Subgraph_Sequence_Mask_Name":    "one_hop_subgraph_sequence_mask_name",
    "exp_100d_One_Hop_Subgraph_Sequence_Mask_Name": "one_hop_subgraph_sequence_mask_name",
    "One_Hop_Subgraph_Sequence_Mask_PK":      "one_hop_subgraph_sequence_mask_pk",
    "exp_100d_One_Hop_Subgraph_Sequence_Mask_PK":   "one_hop_subgraph_sequence_mask_pk",
    "OHS_Mask_Name":                          "ohs_mask_name",
    "OHS_Mask_Entity":                        "ohs_mask_entity",
    "OHS_Mask_Name_Entity":                   "ohs_mask_name_entity",
    "OHS_Full":                               "ohs_full",
    "OHS_Full_Mask_Name":                     "ohs_full_mask_name",
    "OHS_Full_Mask_Entity":                   "ohs_full_mask_entity",
    "OHS_Full_Mask_Name_Entity":              "ohs_full_mask_name_entity",
    "Few_Shot_2hop":                          "few_shot_2hop",
}


def canon_method(method: str) -> str:
    """Resolve a method alias, or return the lowercase input."""
    if not method:
        return "zero_shot"
    if method in _METHOD_ALIASES:
        return _METHOD_ALIASES[method]
    return method.lower()


def infer_model_family(model_name: str) -> str:
    """Map a model name to qwen, llama, gemma or mistral.

    Unknown names, including ChatGLM, Baichuan and DeepSeek, default to qwen.
    Direct calls to ``format_messages_for_model`` instead default to Llama-3.
    """
    m = (model_name or "").lower()
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    if "gemma" in m:
        return "gemma"
    if "mistral" in m:
        return "mistral"
    return "qwen"


# Subgraph rendering

def _build_line(prefix: str, name: str, d: dict, keys: list[str]) -> str:
    parts = [f"{prefix}{name}\n"]
    for ent in keys:
        val = d.get(ent, "unknown")
        if val == "unknown":
            # Singularise the entity type when value is unknown
            # ("transporters" → "transporter is unknown").
            parts.append(f"whose {ent[:-1]} is unknown")
            continue
        parts.append(f"whose {ent} are {val}")
    return parts[0] + "\n".join(parts[1:]) + "."


def _mask_first_valid(entity_list: list) -> list:
    """Replace the first non-``unknown`` element with ``[ENTITY]``."""
    result = list(entity_list)
    for i, e in enumerate(result):
        if e != "unknown":
            result[i] = "[ENTITY]"
            break
    return result


def _apply_entity_mask(
    neighbors: dict,
    drug_a_id: str,
    drug_b_id: str,
    key_entity_map: dict | None,
) -> dict:
    """Copy neighbors and mask key entities with ``[ENTITY]``.

    For Type A, mask the confirmed entity in each non-SMILES type. For Type B,
    mask the first non-``unknown`` entry per type and drug.
    """
    ke_map = key_entity_map or {}
    ke_info = ke_map.get((drug_a_id, drug_b_id), None)

    masked = {
        stype: {"A": list(v["A"]), "B": list(v["B"])}
        for stype, v in neighbors.items()
    }

    if ke_info and ke_info.get("has_key_entity"):
        ke_name = str(ke_info.get("key_entity_name", ""))
        for stype, sides in masked.items():
            if stype == "smiles":
                continue
            for side in ("A", "B"):
                sides[side] = [
                    "[ENTITY]" if e == ke_name else e for e in sides[side]
                ]
    else:
        for stype, sides in masked.items():
            if stype == "smiles":
                continue
            for side in ("A", "B"):
                sides[side] = _mask_first_valid(sides[side])
    return masked


# Query formatting

def _format_pair(
    sample: dict,
    method: str,
    *,
    drug_id2smiles: dict | None = None,
    key_entity_map: dict | None = None,
) -> str:
    """Render the method-specific prediction block."""
    a_name = sample.get("drugA_name", None)
    b_name = sample.get("drugB_name", None)
    if a_name is None or b_name is None:
        a_name = sample.get("drug_a_name", "")
        b_name = sample.get("drug_b_name", "")
        if a_name is None or b_name is None:
            raise ValueError("drug_a_name or drug_b_name is None")

    smiles_map = drug_id2smiles or {}
    a_smiles = smiles_map.get(sample.get("drug_a_id"), None)
    b_smiles = smiles_map.get(sample.get("drug_b_id"), None)

    if method == "zero_shot":
        return (
            f"### [Prediction]:\n"
            f"Drug A: {a_name}\n"
            f"Drug B: {b_name}\n"
            f"### Answer: "
        )

    if method == "few_shot_similarity_smiles":
        return (
            f"### [Prediction]:\n"
            f"Drug A: {a_name}, SMILES: {a_smiles}\n"
            f"Drug B: {b_name}, SMILES: {b_smiles}\n"
            f"### Answer: "
        )

    if method == "one_hop_subgraph_single":
        raw = sample.get("subgraph_1hop", "") or ""
        A: list[tuple[str, str, str]] = []
        B: list[tuple[str, str, str]] = []
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A.extend((a_name, f"has {key}", t) for t in raw["neighbors"][key]["A"])
            B.extend((b_name, f"has {key}", t) for t in raw["neighbors"][key]["B"])
        A_text = ", ".join(f"({h}, {r}, {t})" for (h, r, t) in A)
        B_text = ", ".join(f"({h}, {r}, {t})" for (h, r, t) in B)
        return (
            f"### [Prediction]:\n"
            f"Drug A: {a_name}\n"
            f"Drug A's Facts: {A_text}\n"
            f"Drug B: {b_name}\n"
            f"Drug B's Facts: {B_text}\n"
            f"### Answer: "
        )

    if method == "one_hop_subgraph_sequence":
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict: dict[str, str] = {}
        B_dict: dict[str, str] = {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
            B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", a_name, A_dict, keys)
        line_b = _build_line("Drug B: ", b_name, B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "one_hop_subgraph_sequence_desc":
        raw = sample["subgraph_1hop"]
        keys = list(raw["neighbors"].keys())
        a_dict = {key: ", ".join(raw["neighbors"][key]["A"]) for key in keys}
        b_dict = {key: ", ".join(raw["neighbors"][key]["B"]) for key in keys}
        line_a = _build_line("Drug A: ", a_name, a_dict, keys)
        line_b = _build_line("Drug B: ", b_name, b_dict, keys)
        a_head, a_sep, a_tail = line_a.partition("\n")
        b_head, b_sep, b_tail = line_b.partition("\n")
        line_a = f"{a_head}\nClinical description: {sample['drug_a_description'].strip()}{a_sep}{a_tail}"
        line_b = f"{b_head}\nClinical description: {sample['drug_b_description'].strip()}{b_sep}{b_tail}"
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "desc_only":
        return (
            f"### [Prediction]:\n"
            f"Drug A: {a_name}\n"
            f"Clinical description: {sample['drug_a_description'].strip()}\n"
            f"Drug B: {b_name}\n"
            f"Clinical description: {sample['drug_b_description'].strip()}\n"
            f"### Answer: "
        )

    if method == "one_hop_subgraph_sequence_mask_name":
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict, B_dict = {}, {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
            B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", "[DRUG_A]", A_dict, keys)
        line_b = _build_line("Drug B: ", "[DRUG_B]", B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "one_hop_subgraph_sequence_mask_pk":
        # Hide enzyme and transporter values.
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict, B_dict = {}, {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            if key in ("enzymes", "transporters"):
                A_dict[key] = "unknown"
                B_dict[key] = "unknown"
            else:
                A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
                B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", a_name, A_dict, keys)
        line_b = _build_line("Drug B: ", b_name, B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    # R1--R7 masking variants

    if method == "ohs_mask_name":
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict, B_dict = {}, {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
            B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", "[DRUG_A]", A_dict, keys)
        line_b = _build_line("Drug B: ", "[DRUG_B]", B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_full":
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict, B_dict = {}, {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
            B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", a_name, A_dict, keys)
        line_b = _build_line("Drug B: ", b_name, B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_full_mask_name":
        raw = sample.get("subgraph_1hop", "") or ""
        A_dict, B_dict = {}, {}
        keys = list(raw["neighbors"].keys())
        for key in keys:
            A_dict[key] = ", ".join(raw["neighbors"][key]["A"])
            B_dict[key] = ", ".join(raw["neighbors"][key]["B"])
        line_a = _build_line("Drug A: ", "[DRUG_A]", A_dict, keys)
        line_b = _build_line("Drug B: ", "[DRUG_B]", B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_full_mask_entity":
        raw = sample.get("subgraph_1hop", "") or ""
        drug_a_id = str(sample.get("drug_a_id", ""))
        drug_b_id = str(sample.get("drug_b_id", ""))
        masked_nb = _apply_entity_mask(
            raw["neighbors"], drug_a_id, drug_b_id, key_entity_map
        )
        A_dict, B_dict = {}, {}
        keys = list(masked_nb.keys())
        for key in keys:
            A_dict[key] = ", ".join(masked_nb[key]["A"])
            B_dict[key] = ", ".join(masked_nb[key]["B"])
        line_a = _build_line("Drug A: ", a_name, A_dict, keys)
        line_b = _build_line("Drug B: ", b_name, B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_full_mask_name_entity":
        raw = sample.get("subgraph_1hop", "") or ""
        drug_a_id = str(sample.get("drug_a_id", ""))
        drug_b_id = str(sample.get("drug_b_id", ""))
        masked_nb = _apply_entity_mask(
            raw["neighbors"], drug_a_id, drug_b_id, key_entity_map
        )
        A_dict, B_dict = {}, {}
        keys = list(masked_nb.keys())
        for key in keys:
            A_dict[key] = ", ".join(masked_nb[key]["A"])
            B_dict[key] = ", ".join(masked_nb[key]["B"])
        line_a = _build_line("Drug A: ", "[DRUG_A]", A_dict, keys)
        line_b = _build_line("Drug B: ", "[DRUG_B]", B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_mask_entity":
        raw = sample.get("subgraph_1hop", "") or ""
        drug_a_id = str(sample.get("drug_a_id", ""))
        drug_b_id = str(sample.get("drug_b_id", ""))
        masked_nb = _apply_entity_mask(
            raw["neighbors"], drug_a_id, drug_b_id, key_entity_map
        )
        A_dict, B_dict = {}, {}
        keys = list(masked_nb.keys())
        for key in keys:
            A_dict[key] = ", ".join(masked_nb[key]["A"])
            B_dict[key] = ", ".join(masked_nb[key]["B"])
        line_a = _build_line("Drug A: ", a_name, A_dict, keys)
        line_b = _build_line("Drug B: ", b_name, B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "ohs_mask_name_entity":
        raw = sample.get("subgraph_1hop", "") or ""
        drug_a_id = str(sample.get("drug_a_id", ""))
        drug_b_id = str(sample.get("drug_b_id", ""))
        masked_nb = _apply_entity_mask(
            raw["neighbors"], drug_a_id, drug_b_id, key_entity_map
        )
        A_dict, B_dict = {}, {}
        keys = list(masked_nb.keys())
        for key in keys:
            A_dict[key] = ", ".join(masked_nb[key]["A"])
            B_dict[key] = ", ".join(masked_nb[key]["B"])
        line_a = _build_line("Drug A: ", "[DRUG_A]", A_dict, keys)
        line_b = _build_line("Drug B: ", "[DRUG_B]", B_dict, keys)
        return (
            f"### [Prediction]:\n"
            f"{line_a}\n"
            f"{line_b}\n"
            f"### Answer: "
        )

    if method == "few_shot_2hop":
        return (
            f"### [Prediction]:\n"
            f"Query Drug A: {a_name}\n"
            f"Query Drug B: {b_name}\n"
            f"### Answer: "
        )

    return ""


# Few-shot examples

def _fewshot_similarity_smiles_block(sample: dict, drug_id2name: dict, drug_id2smiles: dict) -> str:
    fs_samples = sample.get("fewshot_samples", None)
    if not fs_samples:
        return ""
    # Shuffle a copy to leave shared few-shot lists unchanged.
    fs_samples = list(fs_samples)
    np.random.shuffle(fs_samples)
    lines = ["### Reference Examples (Structural Analogs):\n"]
    for idx, k in enumerate(fs_samples):
        ref_a_id, ref_b_id = k[1], k[2]
        label = "Yes" if k[3] == "1" else "No"
        a_name = drug_id2name[ref_a_id]
        b_name = drug_id2name[ref_b_id]
        a_smiles = drug_id2smiles[ref_a_id]
        b_smiles = drug_id2smiles[ref_b_id]
        lines.append(
            f"### [Example {idx+1}]\n"
            f"Drug A: {a_name}, SMILES: {a_smiles}\n"
            f"Drug B: {b_name}, SMILES: {b_smiles}\n"
            f"### Answer: {label}\n"
        )
    return "\n".join(lines) + "\n"


def _fewshot_2hop_block(sample: dict, drug_id2name: dict) -> str:
    fs_samples = sample.get("fewshot_samples", []) or []
    fs_metadata = sample.get("fewshot_metadata", []) or []
    if not fs_samples:
        return ""

    # Shuffle samples and metadata together without modifying shared input lists.
    if fs_metadata and len(fs_samples) == len(fs_metadata):
        combined = list(zip(fs_samples, fs_metadata))
        np.random.shuffle(combined)
        fs_samples, fs_metadata = zip(*combined)
    else:
        fs_samples = list(fs_samples)
        np.random.shuffle(fs_samples)
        fs_metadata = [{}] * len(fs_samples)

    lines = ["### Reference Examples (Mechanism-Aware):\n"]

    def _format_path(query_label: str, ref_drug: str, matches_list) -> str:
        if not matches_list:
            return f"({query_label} -> {{Structural Similarity}} <- {ref_drug})"
        i = np.random.choice(len(matches_list))
        entity_name, entity_type = matches_list[i]
        return f"({query_label} -> {{{entity_type}: {entity_name}}} <- {ref_drug})"

    for idx, (k, meta) in enumerate(zip(fs_samples, fs_metadata)):
        ref_a_id, ref_b_id = k[1], k[2]
        label = "Yes" if k[3] == "1" else "No"
        ref_a_name = drug_id2name.get(ref_a_id, "Unknown")
        ref_b_name = drug_id2name.get(ref_b_id, "Unknown")

        m = meta if isinstance(meta, dict) else {}
        all_matches = (
            m.get("shared_QA_CA", [])
            + m.get("shared_QA_CB", [])
            + m.get("shared_QB_CA", [])
            + m.get("shared_QB_CB", [])
        )

        if not all_matches:
            path_a = _format_path("Query Drug A", ref_a_name, [])
            path_b = _format_path("Query Drug B", ref_b_name, [])
        else:
            straight = len(m.get("shared_QA_CA", [])) + len(m.get("shared_QB_CB", []))
            cross = len(m.get("shared_QA_CB", [])) + len(m.get("shared_QB_CA", []))
            if straight >= cross:
                path_a = _format_path("Query Drug A", ref_a_name, m.get("shared_QA_CA", []))
                path_b = _format_path("Query Drug B", ref_b_name, m.get("shared_QB_CB", []))
            else:
                path_a = _format_path("Query Drug A", ref_b_name, m.get("shared_QA_CB", []))
                path_b = _format_path("Query Drug B", ref_a_name, m.get("shared_QB_CA", []))

        lines.append(
            f"### [Example {idx+1}]\n"
            f"Drug A: {ref_a_name}\n"
            f"Drug B: {ref_b_name}\n"
            f"Matching Evidence: {path_a}; {path_b}\n"
            f"### Answer: {label}\n"
        )
    return "\n".join(lines) + "\n"


def _build_prompt_4(
    method: str,
    sample: dict,
    *,
    drug_id2name: dict,
    drug_id2smiles: dict,
) -> str:
    if method == "few_shot_similarity_smiles":
        return _fewshot_similarity_smiles_block(sample, drug_id2name, drug_id2smiles)
    if method == "few_shot_2hop":
        return _fewshot_2hop_block(sample, drug_id2name)
    # All other methods (zero_shot, one_hop_*, ohs_*, mask_*) have no aux.
    return ""


# Prompt assembly

def build_binary_prompt(
    sample: dict,
    cfg: PromptBuildConfig,
    *,
    drug_id2name: dict | None = None,
    drug_id2smiles: dict | None = None,
    key_entity_map: dict | None = None,
    assistant_content: str | None = None,
) -> str:
    """Build the chat-formatted prompt for one DDI pair.

    ``sample`` follows ``to_llm_samples``. Few-shot methods use the name and
    SMILES lookups; R4--R7 use ``key_entity_map`` entries keyed by drug pairs
    with ``key_entity_name``, ``key_entity_type`` and ``has_key_entity``.

    With ``assistant_content=None``, derive ``" Yes"`` or ``" No"`` from the
    label (a missing label gives one space). Pass ``""`` for inference so
    scoring starts immediately after the assistant header.
    """
    task = cfg.task_name or "Binary_cls"
    task_display = TASK_DISPLAY_NAME.get(task, task)
    method = canon_method(cfg.method)

    if method in ("one_hop_subgraph_sequence_desc", "desc_only"):
        # This path is shared by training, checkpoint scoring and inference.
        sample = dict(sample)
        descriptions = cfg.extra.get("drug_id2description", {})
        for side in ("a", "b"):
            drug_id = str(sample.get(f"drug_{side}_id", ""))
            field_name = f"drug_{side}_description"
            description = descriptions.get(drug_id, sample.get(field_name))
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Missing clinical description for {drug_id!r} ({cfg.method}).")
            sample[field_name] = description

    drug_id2name = drug_id2name or {}
    drug_id2smiles = drug_id2smiles or {}

    prompt_1 = f"Task: {task_display}\n"
    prompt_2 = TASK_INSTRUCTION.get(task, TASK_INSTRUCTION.get("Binary_cls", ""))
    prompt_3 = METHOD_PROMPT.get(method, "")
    prompt_4 = _build_prompt_4(
        method, sample,
        drug_id2name=drug_id2name,
        drug_id2smiles=drug_id2smiles,
    )
    prompt_5 = OUTPUT_CONSTRAINT_TEXT
    prompt_6 = _format_pair(
        sample, method,
        drug_id2smiles=drug_id2smiles,
        key_entity_map=key_entity_map,
    )

    system_content = "".join(p for p in (prompt_1, prompt_2) if p).strip()
    user_content = "".join(p for p in (prompt_3, prompt_4, prompt_5, prompt_6) if p).strip()

    if assistant_content is None:
        # Use the sample label as the training answer.
        response = sample.get("label", "")
        if response == 1:
            response = "Yes"
        elif response == 0:
            response = "No"
        asst_content = f" {response}"
    else:
        asst_content = assistant_content

    # Resolve the family first so unknown models use the qwen fallback.
    family = infer_model_family(cfg.model_name)
    return format_messages_for_model(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        family,
    )


__all__ = [
    "PromptBuildConfig",
    "build_binary_prompt",
    "canon_method",
    "infer_model_family",
    "TASK_INSTRUCTION",
    "METHOD_PROMPT",
    "OUTPUT_CONSTRAINT_TEXT",
]
