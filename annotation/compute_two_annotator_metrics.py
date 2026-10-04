"""Compute two-annotator metrics for the ColdDDI taxonomy validation pilot.

Inputs:
  - A1: annotator1_raw.xlsx
  - A2: annotator2_raw.csv
  - Agent fill-in: computed/a1_filled_ab.csv (116 ? rows replaced with A or B)

Conventions:
  - A1's "?" labels are replaced with the agent's A/B calls.
  - A1's "Mixed" labels (n=10) remain in the raw column but become PK for metric
    computation (paper §A.2 PK-precedence convention).
  - Consensus = pairs where A1 and A2 unanimously agree. Disagreements are
    excluded from auto-vs-consensus metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, precision_recall_fscore_support, accuracy_score,
)

# Raw inputs are not shipped; see annotation/_README.md. Set
# COLDDDI_ANNOTATION_RAW to a directory containing annotator1_raw.xlsx,
# annotator2_raw.csv, and computed/a1_filled_ab.csv.
ROOT = Path(os.environ.get("COLDDDI_ANNOTATION_RAW", "./annotation_raw"))
A1_PATH = ROOT / "annotator1_raw.xlsx"
A2_PATH = ROOT / "annotator2_raw.csv"
AGENT_PATH = ROOT / "computed" / "a1_filled_ab.csv"
OUT_DIR = ROOT / "computed"

# Recompute auto PK/PD from ddi_type keywords and auto A/B from whether
# auto_key_entity_name is nonempty; the input CSV omits these labels.
PK_KEYWORDS = [
    "metabolism", "excretion", "absorption", "serum concentration",
    "bioavailability", "clearance", "cyp", "transporter", "protein binding",
]
PD_KEYWORDS = [
    "activities", "therapeutic efficacy", "adverse effect", "risk",
    "receptor binding", "analgesic", "sedative", "hypotensive", "bleeding",
    "qtc", "arrhythmia", "bradycardia", "tachycardia", "cns depressant",
    "hypertension", "anticoagulant", "hemorrhage", "myopathy", "sedation",
    "nephrotoxicity", "effectiveness",
]


def auto_pkpd(ddi_type: str) -> str:
    t = (ddi_type or "").lower()
    pk = any(k in t for k in PK_KEYWORDS)
    pd_ = any(k in t for k in PD_KEYWORDS)
    if pk and pd_:
        return "PK"  # PK precedence
    if pk:
        return "PK"
    if pd_:
        return "PD"
    return "Unknown"


def auto_ab(key_entity_name) -> str:
    if pd.isna(key_entity_name) or str(key_entity_name).strip() == "":
        return "B"
    return "A"


def main() -> None:
    a1 = pd.read_excel(A1_PATH)
    a2 = pd.read_csv(A2_PATH)
    agent = pd.read_csv(AGENT_PATH)

    assert len(a1) == 500 and len(a2) == 500
    assert sorted(a1["pair_id"]) == sorted(a2["pair_id"])

    # Align annotators by pair_id.
    a1 = a1.sort_values("pair_id").reset_index(drop=True)
    a2 = a2.sort_values("pair_id").reset_index(drop=True)
    agent = agent.set_index("pair_id")

    # Replace A1's '?' labels with agent labels.
    a1["AB_filled"] = a1["your_label_AorB"].copy()
    n_qmarks = (a1["AB_filled"] == "?").sum()
    print(f"Replacing {n_qmarks} '?' rows in A1 with agent labels...")
    for idx, row in a1.iterrows():
        if row["AB_filled"] == "?":
            pid = row["pair_id"]
            if pid in agent.index:
                a1.at[idx, "AB_filled"] = agent.loc[pid, "agent_AorB"]
            else:
                raise ValueError(f"Pair {pid} marked ? but not in agent fill-in")
    assert (a1["AB_filled"].isin(["A", "B"])).all(), "A1 AB_filled has non-A/B values"

    # Collapse A1's Mixed labels to PK.
    a1["PKPD_filled"] = a1["your_label_PK_PD_or_Mixed"].replace({"Mixed": "PK"})
    n_mixed = (a1["your_label_PK_PD_or_Mixed"] == "Mixed").sum()
    print(f"Collapsed {n_mixed} 'Mixed' rows in A1 to PK")

    # A2 has no Mixed and no '?'.
    assert (a2["your_label_PK_PD_or_Mixed"].isin(["PK", "PD"])).all()
    assert (a2["your_label_AorB"].isin(["A", "B"])).all()

    # Compute auto labels
    a1["auto_pk_pd"] = a1["ddi_type"].apply(auto_pkpd)
    a1["auto_AorB"] = a1["auto_key_entity_name"].apply(auto_ab)

    # Inter-annotator kappa
    print("\n=== Inter-annotator agreement (Cohen's κ) ===")
    kappa_pkpd_pair = cohen_kappa_score(a1["PKPD_filled"], a2["your_label_PK_PD_or_Mixed"])
    print(f"PK/PD per-pair (n=500): κ = {kappa_pkpd_pair:.3f}")

    # Aggregate pair labels by majority vote within each type.
    a1["type_id"] = a1["ddi_type"]
    a2["type_id"] = a2["ddi_type"]
    type_a1 = a1.groupby("type_id")["PKPD_filled"].agg(lambda s: s.mode().iloc[0])
    type_a2 = a2.groupby("type_id")["your_label_PK_PD_or_Mixed"].agg(lambda s: s.mode().iloc[0])
    common_types = type_a1.index.intersection(type_a2.index)
    kappa_pkpd_type = cohen_kappa_score(type_a1.loc[common_types], type_a2.loc[common_types])
    print(f"PK/PD type-level (n={len(common_types)} types via majority vote): κ = {kappa_pkpd_type:.3f}")

    kappa_ab_pair = cohen_kappa_score(a1["AB_filled"], a2["your_label_AorB"])
    print(f"A/B per-pair (n=500): κ = {kappa_ab_pair:.3f}")

    # Consensus includes only unanimous labels.
    print("\n=== Consensus (unanimous) sample sizes ===")
    consensus_pkpd_mask = a1["PKPD_filled"] == a2["your_label_PK_PD_or_Mixed"]
    consensus_ab_mask = a1["AB_filled"] == a2["your_label_AorB"]
    n_cons_pkpd = consensus_pkpd_mask.sum()
    n_cons_ab = consensus_ab_mask.sum()
    print(f"PK/PD consensus: {n_cons_pkpd}/500 pairs")
    print(f"A/B consensus:   {n_cons_ab}/500 pairs")

    # Auto-vs-consensus PK/PD precision, recall, and F1
    print("\n=== Auto-vs-consensus PK/PD P/R/F1 (unanimous-only) ===")
    cons_pkpd = a1.loc[consensus_pkpd_mask].copy()
    cons_pkpd["consensus_label"] = a1.loc[consensus_pkpd_mask, "PKPD_filled"]
    auto = cons_pkpd["auto_pk_pd"].values
    truth = cons_pkpd["consensus_label"].values
    # PK as positive
    p, r, f, _ = precision_recall_fscore_support(truth, auto, labels=["PK"], average="binary", pos_label="PK", zero_division=0)
    print(f"PK as positive (per-pair): P={p:.3f}  R={r:.3f}  F1={f:.3f}  n_auto_pos={(auto=='PK').sum()}")
    # PD as positive
    p2, r2, f2, _ = precision_recall_fscore_support(truth, auto, labels=["PD"], average="binary", pos_label="PD", zero_division=0)
    print(f"PD as positive (per-pair): P={p2:.3f}  R={r2:.3f}  F1={f2:.3f}  n_auto_pos={(auto=='PD').sum()}")

    overall_acc = accuracy_score(truth, auto)
    print(f"Overall agreement (per-pair, on consensus): {overall_acc:.3f} ({(truth == auto).sum()}/{len(truth)})")

    # Type-level consensus
    cons_pkpd["type_id"] = cons_pkpd["ddi_type"]
    type_cons = cons_pkpd.groupby("type_id")["consensus_label"].agg(lambda s: s.mode().iloc[0])
    type_auto = cons_pkpd.groupby("type_id")["auto_pk_pd"].agg(lambda s: s.mode().iloc[0])
    n_type_cons = len(type_cons)
    p_t, r_t, f_t, _ = precision_recall_fscore_support(type_cons, type_auto, labels=["PK"], average="binary", pos_label="PK", zero_division=0)
    p_t2, r_t2, f_t2, _ = precision_recall_fscore_support(type_cons, type_auto, labels=["PD"], average="binary", pos_label="PD", zero_division=0)
    type_acc = accuracy_score(type_cons, type_auto)
    print(f"Type-level on consensus (n={n_type_cons} types):")
    print(f"  PK as positive: P={p_t:.3f}  R={r_t:.3f}  F1={f_t:.3f}")
    print(f"  PD as positive: P={p_t2:.3f}  R={r_t2:.3f}  F1={f_t2:.3f}")
    print(f"  Overall: {type_acc:.3f} ({(type_cons.values == type_auto.values).sum()}/{n_type_cons})")

    # Auto-vs-consensus A/B agreement
    print("\n=== Auto-vs-consensus A/B agreement (unanimous-only) ===")
    cons_ab = a1.loc[consensus_ab_mask].copy()
    cons_ab["consensus_label"] = a1.loc[consensus_ab_mask, "AB_filled"]
    auto_ab_v = cons_ab["auto_AorB"].values
    truth_ab = cons_ab["consensus_label"].values
    overall = (auto_ab_v == truth_ab).mean()
    print(f"Overall A/B agreement: {overall:.3f} ({(auto_ab_v == truth_ab).sum()}/{len(truth_ab)})")

    # Break down by auto PK/PD x auto A/B.
    cons_ab["auto_pk_pd"] = cons_ab["ddi_type"].apply(auto_pkpd)
    quadrants = {
        "PK-A": (cons_ab["auto_pk_pd"] == "PK") & (cons_ab["auto_AorB"] == "A"),
        "PK-B": (cons_ab["auto_pk_pd"] == "PK") & (cons_ab["auto_AorB"] == "B"),
        "PD-A": (cons_ab["auto_pk_pd"] == "PD") & (cons_ab["auto_AorB"] == "A"),
        "PD-B": (cons_ab["auto_pk_pd"] == "PD") & (cons_ab["auto_AorB"] == "B"),
    }
    print("Per-quadrant A/B agreement:")
    quadrant_results = {}
    for name, mask in quadrants.items():
        sub = cons_ab.loc[mask]
        if len(sub) == 0:
            print(f"  {name}: n=0")
            quadrant_results[name] = {"n": 0, "agree": None, "pct": None}
            continue
        agree = (sub["auto_AorB"] == sub["consensus_label"]).sum()
        pct = agree / len(sub)
        print(f"  {name} (n={len(sub)}): {agree}/{len(sub)} = {pct:.3f}")
        quadrant_results[name] = {"n": int(len(sub)), "agree": int(agree), "pct": float(pct)}

    # Sample composition
    print("\n=== Sample composition ===")
    print("50 sampled types by automated PK/PD label:")
    types_unique = a1[["ddi_type", "auto_pk_pd"]].drop_duplicates(subset=["ddi_type"])
    print(types_unique["auto_pk_pd"].value_counts().to_string())
    print(f"Total: {len(types_unique)} unique types")
    print()
    print("500 sampled pairs by mechanism subtype (auto PK/PD × auto A/B):")
    a1["subtype"] = a1["auto_pk_pd"] + "-" + a1["auto_AorB"]
    print(a1["subtype"].value_counts().to_string())

    # Save summary JSON.
    summary = {
        "n_total_pairs": 500,
        "n_replaced_question_marks": int(n_qmarks),
        "n_collapsed_mixed_to_PK": int(n_mixed),
        "consensus_sizes": {
            "PK_PD": int(n_cons_pkpd),
            "AB": int(n_cons_ab),
        },
        "kappa": {
            "PK_PD_per_pair_n500": float(kappa_pkpd_pair),
            "PK_PD_type_level_majority_vote": {
                "n_types": int(len(common_types)),
                "kappa": float(kappa_pkpd_type),
            },
            "AB_per_pair_n500": float(kappa_ab_pair),
        },
        "auto_vs_consensus_PKPD_per_pair": {
            "n": int(n_cons_pkpd),
            "PK_positive": {"P": float(p), "R": float(r), "F1": float(f), "n_auto_pos": int((auto == "PK").sum())},
            "PD_positive": {"P": float(p2), "R": float(r2), "F1": float(f2), "n_auto_pos": int((auto == "PD").sum())},
            "overall_agreement": float(overall_acc),
        },
        "auto_vs_consensus_PKPD_type_level": {
            "n_types": int(n_type_cons),
            "PK_positive": {"P": float(p_t), "R": float(r_t), "F1": float(f_t)},
            "PD_positive": {"P": float(p_t2), "R": float(r_t2), "F1": float(f_t2)},
            "overall_agreement": float(type_acc),
        },
        "auto_vs_consensus_AB": {
            "n": int(n_cons_ab),
            "overall_agreement": float(overall),
            "per_quadrant": quadrant_results,
        },
        "sample_composition": {
            "types_by_auto_pkpd": types_unique["auto_pk_pd"].value_counts().to_dict(),
            "pairs_by_subtype": a1["subtype"].value_counts().to_dict(),
        },
    }
    out_path = OUT_DIR / "two_annotator_metrics.json"
    with open(out_path, "w") as f_out:
        json.dump(summary, f_out, indent=2)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
