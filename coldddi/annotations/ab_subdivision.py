"""A/B subdivision protocol (Appendix A.3).

For each positive DDI pair, attempt to find a shared biomedical *key
entity* whose participation explains the interaction:

* **PK** pairs → search for a shared enzyme / transporter / carrier whose
  drug-side actions form a recognized PK role pattern (e.g.
  ``inhibitor`` × ``substrate``). The entity name is reported alongside
  a textual mechanism chain.
* **PD** pairs → search for a shared human target whose drug-side
  actions form a convergent / opposing pharmacological pattern.
* **Mixed** pairs (very rare) → try PK first, then PD.

Pairs with a found entity are labelled **Type A** (``has_key_entity =
True``); the rest are **Type B**.

The implementation mirrors the legacy
``scripts/Label_ddi_for_key_entities/find_ddi_key_entities.py`` but is
streamlined for the release package and exposes a CLI driven by
:func:`run_ab_subdivision`.

Inputs
------
- ``ddi_edges.csv``   — must include columns ``drug_a_id, drug_b_id, ddi_type``.
- ``ddi_pk_pd_labels.csv`` — output of :mod:`coldddi.annotations.pkpd_keywords`.
- Four entity tables (one row per drug-entity binding, organism filtered to
  ``Humans``): ``drug_enzymes.csv``, ``drug_targets.csv``,
  ``drug_transporters.csv``, ``drug_carriers.csv``.

Outputs
-------
- ``ddi_key_entities.csv`` — every DDI pair with mechanism-chain columns and
  a boolean ``has_key_entity``.
- ``ddi_key_entities_type_summary.csv`` — per-``ddi_type`` coverage stats.

CLI
---
``python -m coldddi.annotations.ab_subdivision --ddi-edges PATH ...``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# PK role patterns
# ---------------------------------------------------------------------

PK_ROLE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("inhibitor", "substrate"),
        ("inducer", "substrate"),
        ("substrate", "inhibitor"),
        ("substrate", "inducer"),
        ("inhibitor", "inhibitor"),
    }
)


def _canonical_pk_pattern(action_a: str, action_b: str) -> str:
    if action_a in ("inhibitor", "inducer") and action_b == "substrate":
        return f"{action_a}-substrate"
    if action_b in ("inhibitor", "inducer") and action_a == "substrate":
        return f"{action_b}-substrate"
    if action_a == "inhibitor" and action_b == "inhibitor":
        return "inhibitor-inhibitor"
    return f"{action_a}-{action_b}"


PK_SUBTYPE_PRIORITY: dict[str, list[str]] = {
    "metabolism": ["enzyme"],
    "serum concentration": ["enzyme", "transporter"],
    "bioavailability": ["enzyme", "transporter"],
    "excretion": ["transporter"],
    "absorption": ["transporter"],
    "protein binding": ["carrier"],
}

def _pk_subtype(ddi_type: str) -> list[str]:
    text = ddi_type.lower()
    for keyword, priority in PK_SUBTYPE_PRIORITY.items():
        if keyword in text:
            return priority
    return ["enzyme", "transporter"]


def _first_occurrence_ranks(entries: list[dict]) -> dict[str, int]:
    """Return ``{entity_id: rank}`` where ``rank`` is the entity's
    first-occurrence position in ``entries`` (0-indexed).

    DrugBank's XML iterates ``<enzymes>/<enzyme>``,
    ``<targets>/<target>``, ``<transporters>/<transporter>``, and
    ``<carriers>/<carrier>`` in curator document order.  Rank 0
    therefore corresponds to the curator-prioritised polypeptide for
    that drug.  :func:`coldddi.data.extract._polypeptide_entities`
    preserves that order (it uses ``container.findall(item_tag)``),
    so a single pass over the per-drug entry list reproduces the
    DrugBank ranking that downstream tooling (e.g. the LLM mask
    experiment in :mod:`coldddi.llm.prompts.binary_cls`) relies on.

    HISTORICAL NOTE: pre-audit, PK best-entity selection used a
    hand-rolled ``CYP_PRIORITY`` substring table.  That table never
    matched the actual DrugBank enzyme name format ("Cytochrome P450
    3A4") so all enzymes fell through to a catch-all bucket and the
    "best" candidate was determined by Python set iteration order
    (process-hash-dependent).  The DrugBank-rank approach replaces
    the broken table with the curator's own importance ordering and
    is deterministic across processes.
    """
    ranks: dict[str, int] = {}
    for i, e in enumerate(entries):
        ranks.setdefault(e["id"], i)
    return ranks


# ---------------------------------------------------------------------
# PD role patterns
# ---------------------------------------------------------------------

PD_PAIR_CONFIDENCE: dict[tuple[str, str], str] = {
    # high-confidence convergent
    ("inhibitor", "inhibitor"): "high",
    ("agonist", "agonist"): "high",
    ("antagonist", "antagonist"): "high",
    ("inhibitor", "antagonist"): "high",
    ("antagonist", "inhibitor"): "high",
    ("agonist", "activator"): "high",
    ("activator", "agonist"): "high",
    ("activator", "activator"): "high",
    ("positive allosteric modulator", "positive allosteric modulator"): "high",
    ("inverse agonist", "antagonist"): "high",
    ("antagonist", "inverse agonist"): "high",
    ("partial agonist", "agonist"): "high",
    ("agonist", "partial agonist"): "high",
    ("partial agonist", "antagonist"): "high",
    ("antagonist", "partial agonist"): "high",
    ("binder", "antagonist"): "high",
    ("antagonist", "binder"): "high",
    ("binder", "agonist"): "high",
    ("agonist", "binder"): "high",
    ("binder", "inhibitor"): "high",
    ("inhibitor", "binder"): "high",
    ("ligand", "ligand"): "high",
    # low-confidence (modulator pairs / opposing)
    ("agonist", "modulator"): "low",
    ("modulator", "agonist"): "low",
    ("antagonist", "modulator"): "low",
    ("modulator", "antagonist"): "low",
    ("inhibitor", "modulator"): "low",
    ("modulator", "inhibitor"): "low",
    ("binder", "modulator"): "low",
    ("modulator", "binder"): "low",
    ("modulator", "partial agonist"): "low",
    ("partial agonist", "modulator"): "low",
    ("inhibitor", "agonist"): "low",
    ("agonist", "inhibitor"): "low",
    ("agonist", "antagonist"): "low",
    ("antagonist", "agonist"): "low",
    ("activator", "antagonist"): "low",
    ("antagonist", "activator"): "low",
    ("activator", "inhibitor"): "low",
    ("inhibitor", "activator"): "low",
}

PD_TYPE_TARGET_KEYWORDS: dict[str, list[str]] = {
    "anticoagulant": ["thrombin", "coagulation", "factor", "platelet", "fibrin"],
    "qtc prolongation": ["kcnh2", "herg", "potassium channel", "cardiac", "ion channel"],
    "arrhythmia": ["kcnh2", "herg", "cardiac", "sodium channel", "potassium"],
    "bradycardia": ["beta", "adrenergic", "cholinergic", "muscarinic"],
    "tachycardia": ["adrenergic", "dopamine", "thyroid"],
    "bleeding": ["coagulation", "platelet", "thrombin", "fibrin"],
    "hemorrhage": ["coagulation", "platelet", "anticoagulant"],
    "hypotension": ["adrenergic", "angiotensin", "vasopressin", "renin", "alpha"],
    "hypertension": ["adrenergic", "angiotensin", "renin", "alpha"],
    "cns depression": ["gaba", "opioid", "serotonin", "histamine", "nmda", "glutamate"],
    "sedation": ["gaba", "opioid", "histamine", "benzodiazepine"],
    "serotonin syndrome": ["serotonin", "5-hydroxytryptamine", "5-ht"],
    "myopathy": ["hmg-coa", "statin", "coq10", "myopathy"],
    "nephrotoxicity": ["renal", "kidney", "proximal tubule"],
    "seizure": ["gaba", "sodium channel", "glutamate", "nmda"],
    "extrapyramidal": ["dopamine", "d2", "d1"],
    "analgesic": ["opioid", "cyclooxygenase", "prostaglandin", "cox"],
    "antihypertensive": ["adrenergic", "angiotensin", "calcium channel"],
    "antidepressant": ["serotonin", "noradrenaline", "dopamine"],
    "immunosuppression": ["calcineurin", "mtor", "interleukin"],
    "sedative": ["gaba", "histamine", "benzodiazepine"],
    "respiratory depression": ["opioid", "gaba", "respiratory"],
    "myelosuppression": ["dna", "topoisomerase", "dihydrofolate"],
    "hypoglycemia": ["insulin", "glucose", "sulfonylurea"],
    "hyperkalemia": ["potassium", "aldosterone", "renin"],
    "neuromuscular blockade": ["acetylcholine", "nicotinic", "neuromuscular"],
}


def _pd_target_relevance(ddi_type: str, target_name: str) -> int:
    t = ddi_type.lower()
    n = target_name.lower()
    for ddi_kw, target_kws in PD_TYPE_TARGET_KEYWORDS.items():
        if ddi_kw in t:
            for i, kw in enumerate(target_kws):
                if kw in n:
                    return i
    return 99


# ---------------------------------------------------------------------
# Entity index loader
# ---------------------------------------------------------------------


def _load_entity_index(
    csv_path: Path,
    id_col: str,
    name_col: str,
    entity_type: str,
    *,
    required: bool = True,
) -> dict[str, list[dict]]:
    """Load an entity table into ``{drug_id: [{id,name,type,action}, ...]}``.

    A missing CSV with ``required=True`` raises :class:`FileNotFoundError`;
    silently degrading would cause the whole pipeline to label every pair
    as Type-B and quietly destroy the Type-A coverage statistics.
    Optional inputs (currently only ``carriers``) may set
    ``required=False`` to allow the empty index.
    """
    if not csv_path.exists():
        if required:
            raise FileNotFoundError(
                f"Required entity CSV not found: {csv_path}. "
                f"Run `python -m coldddi.data.extract` first to produce it."
            )
        return {}
    df = pd.read_csv(csv_path)
    missing = {"drugbank_id", id_col, name_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Entity CSV {csv_path} missing required columns: {sorted(missing)}"
        )
    if "organism" in df.columns:
        df = df[df["organism"] == "Humans"]
    out: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        action = str(row.get("action", "")).strip().lower() if "action" in row else ""
        out[row["drugbank_id"]].append(
            {
                "id": row[id_col],
                "name": row[name_col],
                "type": entity_type,
                "action": action,
            }
        )
    return dict(out)


# ---------------------------------------------------------------------
# Per-pair search
# ---------------------------------------------------------------------


def _find_key_entity_pk(
    drug_a: str,
    drug_b: str,
    ddi_type: str,
    name_a: str,
    name_b: str,
    enzyme_idx: dict,
    transport_idx: dict,
    carrier_idx: dict,
) -> dict | None:
    candidates: list[dict] = []
    for entity_type in _pk_subtype(ddi_type):
        if entity_type == "enzyme":
            source = enzyme_idx
        elif entity_type == "transporter":
            source = transport_idx
        else:
            source = carrier_idx

        entries_a = source.get(drug_a, [])
        entries_b = source.get(drug_b, [])
        # DrugBank-XML document-order rank per (drug, entity).  Lower
        # rank == higher curator-assigned importance for that drug.
        rank_a = _first_occurrence_ranks(entries_a)
        rank_b = _first_occurrence_ranks(entries_b)

        map_a: dict[str, list[dict]] = defaultdict(list)
        map_b: dict[str, list[dict]] = defaultdict(list)
        for e in entries_a:
            map_a[e["id"]].append(e)
        for e in entries_b:
            map_b[e["id"]].append(e)

        for eid in set(map_a) & set(map_b):
            for ea in map_a[eid]:
                for eb in map_b[eid]:
                    pair = (ea["action"], eb["action"])
                    if pair not in PK_ROLE_PAIRS:
                        continue
                    pattern = _canonical_pk_pattern(*pair)
                    verb = "transports" if entity_type == "transporter" else "metabolizes"
                    if pair[0] in ("inhibitor", "inducer") and pair[1] == "substrate":
                        chain = (
                            f"{name_a} --[{pair[0]}]--> {ea['name']} "
                            f"--[substrate: {verb}]--> {name_b}"
                        )
                    elif pair[1] in ("inhibitor", "inducer") and pair[0] == "substrate":
                        chain = (
                            f"{name_b} --[{pair[1]}]--> {eb['name']} "
                            f"--[substrate: {verb}]--> {name_a}"
                        )
                    else:
                        chain = (
                            f"{name_a} --[{pair[0]}]--> {ea['name']} "
                            f"<--[{pair[1]}]-- {name_b}"
                        )
                    # Joint DrugBank importance: prefer entities that
                    # rank highly for BOTH drugs.  Summing the per-drug
                    # ranks is monotone in either drug's rank, so a
                    # candidate that is rank-0 for one side and rank-0
                    # for the other always beats any mixed pair.
                    drugbank_rank = rank_a[eid] + rank_b[eid]
                    candidates.append(
                        {
                            "key_entity_id": eid,
                            "key_entity_name": ea["name"],
                            "key_entity_type": entity_type,
                            "action_drug_a": pair[0],
                            "action_drug_b": pair[1],
                            "match_pattern": pattern,
                            "mechanism_chain": chain,
                            "chain_type": f"PK_through_{entity_type}",
                            "confidence": "high",
                            "_drugbank_rank": drugbank_rank,
                        }
                    )
        if candidates:
            break

    if not candidates:
        return None
    # Deterministic ordering: (drugbank_rank, name, id).  The primary
    # key is the joint DrugBank XML document-order rank — DrugBank
    # curates the per-drug polypeptide list so that the most
    # mechanistically important entry comes first, so the entity
    # ranked highest by BOTH drugs is the natural mask target for
    # the LLM mask experiments (R2/R3/R6/R7 in
    # :mod:`coldddi.llm.prompts.binary_cls`).  (name, id) break any
    # remaining ties so the pick is reproducible across processes
    # (set iteration is Python-hash-seeded otherwise).
    candidates.sort(
        key=lambda x: (
            x["_drugbank_rank"],
            x["key_entity_name"],
            x["key_entity_id"],
        )
    )
    best = candidates[0]
    best.pop("_drugbank_rank")
    rest = [
        {k: v for k, v in c.items() if k != "_drugbank_rank"}
        for c in candidates[1:]
    ]
    best["key_entity_candidates"] = json.dumps(rest) if rest else "[]"
    return best


def _find_key_entity_pd(
    drug_a: str,
    drug_b: str,
    ddi_type: str,
    name_a: str,
    name_b: str,
    target_idx: dict,
) -> dict | None:
    entries_a = target_idx.get(drug_a, [])
    entries_b = target_idx.get(drug_b, [])
    # DrugBank-XML document-order rank per (drug, target).  Lower
    # rank == higher curator-assigned importance for that drug.
    rank_a = _first_occurrence_ranks(entries_a)
    rank_b = _first_occurrence_ranks(entries_b)

    map_a: dict[str, list[dict]] = defaultdict(list)
    map_b: dict[str, list[dict]] = defaultdict(list)
    for e in entries_a:
        map_a[e["id"]].append(e)
    for e in entries_b:
        map_b[e["id"]].append(e)

    candidates: list[dict] = []
    for tid in set(map_a) & set(map_b):
        best_conf: str | None = None
        best_aa = best_ab = ""
        for ea in map_a[tid]:
            for eb in map_b[tid]:
                conf = PD_PAIR_CONFIDENCE.get((ea["action"], eb["action"]))
                if conf is None:
                    continue
                if best_conf is None or (best_conf == "low" and conf == "high"):
                    best_conf = conf
                    best_aa, best_ab = ea["action"], eb["action"]
        if best_conf is None:
            continue
        tname = map_a[tid][0]["name"]
        chain = f"{name_a} --[{best_aa}]--> {tname} <--[{best_ab}]-- {name_b}"
        chain_type = (
            "PD_convergent_target"
            if best_aa == best_ab or best_conf == "high"
            else "PD_opposing_target"
        )
        candidates.append(
            {
                "key_entity_id": tid,
                "key_entity_name": tname,
                "key_entity_type": "target",
                "action_drug_a": best_aa,
                "action_drug_b": best_ab,
                "match_pattern": f"{best_aa}-{best_ab}",
                "mechanism_chain": chain,
                "chain_type": chain_type,
                "confidence": best_conf,
                "_relevance": _pd_target_relevance(ddi_type, tname),
                "_drugbank_rank": rank_a[tid] + rank_b[tid],
            }
        )
    if not candidates:
        return None
    # Deterministic ordering:
    #   (confidence, ddi-type relevance, drugbank rank, name, id)
    # Semantic keys come first: high-confidence convergence beats
    # low-confidence, then DDI-type-specific target relevance (e.g.
    # for "QTc prolongation" prefer KCNH2/hERG-like targets).  Among
    # semantically equivalent candidates we fall back to the DrugBank
    # XML rank so the "best target" matches the curator's per-drug
    # priority (same notion as the PK path, important for the mask
    # experiment).  (name, id) break any remaining ties so the pick
    # is reproducible across processes (set iteration is Python-
    # hash-seeded otherwise).
    candidates.sort(
        key=lambda x: (
            0 if x["confidence"] == "high" else 1,
            x["_relevance"],
            x["_drugbank_rank"],
            x["key_entity_name"],
            x["key_entity_id"],
        )
    )

    def _strip(c: dict) -> dict:
        return {k: v for k, v in c.items() if k not in ("_relevance", "_drugbank_rank")}

    best = _strip(candidates[0])
    rest = [_strip(c) for c in candidates[1:]]
    best["key_entity_candidates"] = json.dumps(rest) if rest else "[]"
    return best


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def run_ab_subdivision(
    *,
    ddi_edges: pd.DataFrame,
    pk_pd_labels: pd.DataFrame,
    enzymes_csv: Path,
    targets_csv: Path,
    transporters_csv: Path,
    carriers_csv: Path | None,
    drug_id_to_name: dict[str, str],
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the A/B subdivision and return (per-pair, per-type) DataFrames.

    Per-pair output columns include the input (``drug_a_id, drug_b_id,
    ddi_type, pk_pd_label``) plus ``key_entity_id, key_entity_name,
    key_entity_type, action_drug_a, action_drug_b, match_pattern,
    mechanism_chain, chain_type, confidence, has_key_entity,
    key_entity_candidates``.
    """
    enzyme_idx = _load_entity_index(
        enzymes_csv, "enzyme_id", "enzyme_name", "enzyme", required=True
    )
    target_idx = _load_entity_index(
        targets_csv, "target_id", "target_name", "target", required=True
    )
    transport_idx = _load_entity_index(
        transporters_csv, "transporter_id", "transporter_name", "transporter", required=True
    )
    carrier_idx: dict = (
        _load_entity_index(carriers_csv, "carrier_id", "carrier_name", "carrier", required=False)
        if carriers_csv is not None
        else {}
    )

    if verbose:
        print(
            f"[ab] entities loaded - enzymes: {len(enzyme_idx):,} drugs / "
            f"targets: {len(target_idx):,} / transporters: {len(transport_idx):,} / "
            f"carriers: {len(carrier_idx):,}",
            file=sys.stderr,
            flush=True,
        )

    merged = ddi_edges.merge(
        pk_pd_labels[["ddi_type", "pk_pd_label"]], on="ddi_type", how="left"
    )

    n_pk = n_pd = n_mixed = 0
    n_pk_found = n_pd_found = 0
    rows: list[dict] = []
    for i, row in merged.iterrows():
        a, b = row["drug_a_id"], row["drug_b_id"]
        ddi_type = row["ddi_type"]
        label = row.get("pk_pd_label", "Unknown")
        name_a = drug_id_to_name.get(a, a)
        name_b = drug_id_to_name.get(b, b)
        base = {
            "drug_a_id": a,
            "drug_b_id": b,
            "ddi_type": ddi_type,
            "pk_pd_label": label,
        }

        found: dict | None = None
        if label == "PK":
            n_pk += 1
            found = _find_key_entity_pk(
                a, b, ddi_type, name_a, name_b, enzyme_idx, transport_idx, carrier_idx
            )
            if found:
                n_pk_found += 1
        elif label == "PD":
            n_pd += 1
            found = _find_key_entity_pd(
                a, b, ddi_type, name_a, name_b, target_idx
            )
            if found:
                n_pd_found += 1
        elif label == "Mixed":
            n_mixed += 1
            found = _find_key_entity_pk(
                a, b, ddi_type, name_a, name_b, enzyme_idx, transport_idx, carrier_idx
            ) or _find_key_entity_pd(
                a, b, ddi_type, name_a, name_b, target_idx
            )

        if found:
            rows.append({**base, **found, "has_key_entity": True})
        else:
            rows.append(
                {
                    **base,
                    "key_entity_id": None,
                    "key_entity_name": None,
                    "key_entity_type": None,
                    "action_drug_a": None,
                    "action_drug_b": None,
                    "match_pattern": None,
                    "mechanism_chain": None,
                    "chain_type": None,
                    "confidence": None,
                    "has_key_entity": False,
                    "key_entity_candidates": "[]",
                }
            )

        if verbose and (i + 1) % 50_000 == 0:
            print(
                f"[ab] processed {i+1:,} / {len(merged):,} pairs",
                file=sys.stderr,
                flush=True,
            )

    per_pair_columns = [
        "drug_a_id",
        "drug_b_id",
        "ddi_type",
        "pk_pd_label",
        "key_entity_id",
        "key_entity_name",
        "key_entity_type",
        "action_drug_a",
        "action_drug_b",
        "match_pattern",
        "mechanism_chain",
        "chain_type",
        "confidence",
        "has_key_entity",
        "key_entity_candidates",
    ]
    per_pair = pd.DataFrame(rows, columns=per_pair_columns)
    if verbose:
        print(
            f"[ab] PK key-entity coverage: {n_pk_found}/{n_pk} | "
            f"PD: {n_pd_found}/{n_pd} | Mixed: {n_mixed}",
            file=sys.stderr,
            flush=True,
        )

    summary_columns = [
        "ddi_type",
        "pk_pd_label",
        "n_pairs",
        "n_with_key_entity",
        "coverage_pct",
        "top3_key_entities",
    ]
    if per_pair.empty:
        return per_pair, pd.DataFrame(columns=summary_columns)

    # type-level summary
    summary_rows: list[dict] = []
    for ddi_type, grp in per_pair.groupby("ddi_type"):
        n_pairs = len(grp)
        n_found = int(grp["has_key_entity"].sum())
        top = (
            grp[grp["has_key_entity"]]["key_entity_name"]
            .value_counts()
            .head(3)
        )
        summary_rows.append(
            {
                "ddi_type": ddi_type,
                "pk_pd_label": grp["pk_pd_label"].iloc[0],
                "n_pairs": n_pairs,
                "n_with_key_entity": n_found,
                "coverage_pct": round(n_found / n_pairs * 100, 1) if n_pairs else 0.0,
                "top3_key_entities": "; ".join(f"{k}({v})" for k, v in top.items()),
            }
        )
    per_type = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        .sort_values("n_pairs", ascending=False)
        .reset_index(drop=True)
    )
    return per_pair, per_type


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify the A/B mediating entity for each positive DDI pair.",
    )
    parser.add_argument("--ddi-edges", required=True, type=Path)
    parser.add_argument("--pkpd-labels", required=True, type=Path)
    parser.add_argument("--enzymes-csv", required=True, type=Path)
    parser.add_argument("--targets-csv", required=True, type=Path)
    parser.add_argument("--transporters-csv", required=True, type=Path)
    parser.add_argument("--carriers-csv", default=None, type=Path)
    parser.add_argument("--drugs-csv", required=True, type=Path,
                        help="drugs.csv with columns drugbank_id, name (used for chain text).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output ddi_key_entities.csv path.")
    parser.add_argument(
        "--summary-out",
        default=None,
        type=Path,
        help="Optional ddi_key_entities_type_summary.csv path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    edges = pd.read_csv(args.ddi_edges, usecols=["drug_a_id", "drug_b_id", "ddi_type"])
    labels = pd.read_csv(args.pkpd_labels)
    drug_df = pd.read_csv(args.drugs_csv, usecols=["drugbank_id", "name"])
    id2name = dict(zip(drug_df["drugbank_id"], drug_df["name"]))

    per_pair, per_type = run_ab_subdivision(
        ddi_edges=edges,
        pk_pd_labels=labels,
        enzymes_csv=args.enzymes_csv,
        targets_csv=args.targets_csv,
        transporters_csv=args.transporters_csv,
        carriers_csv=args.carriers_csv,
        drug_id_to_name=id2name,
        verbose=not args.quiet,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_pair.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(per_pair):,} pairs)")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        per_type.to_csv(args.summary_out, index=False)
        print(f"Wrote {args.summary_out}  ({len(per_type)} types)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
