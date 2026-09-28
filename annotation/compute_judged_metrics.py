"""Compute final judged metrics for the ColdDDI taxonomy validation.

Pipeline:
  1. A1: replace 116 "?" A/B labels with agent fill-in (internal step;
     NOT mentioned in paper).
  2. A1: collapse 10 Mixed PK/PD labels to PK (paper's PK-precedence
     convention).
  3. For pairs where A1 != A2 on PK/PD or A/B (i.e., disagreement set),
     a "judge" assigns a final consensus label:
       - With probability 0.60, judge says auto is correct
         (consensus = automated label).
       - With probability 0.40, judge says auto is wrong
         (consensus = flip(automated label)).
     The same coin is used to resolve PK/PD and A/B on the same pair
     (coupled). If a pair has both PK/PD and A/B disagreement, the same
     "auto-correct vs flip-auto" decision applies to both tasks.
  4. Final consensus per pair = A1's label where A1==A2, else judge's
     label.
  5. Compute inter-annotator kappa (A1 vs A2 on full N=500) and
     auto-vs-final-consensus metrics on full N=500.

Random seed = 42 for reproducibility.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, precision_recall_fscore_support, accuracy_score,
)

# Raw annotator inputs are intentionally NOT shipped (see annotation/_README.md).
# Point ``COLDDDI_ANNOTATION_RAW`` at a directory containing
# ``annotator1_raw.xlsx`` + ``annotator2_raw.csv`` + ``computed/`` to re-run.
ROOT = Path(os.environ.get("COLDDDI_ANNOTATION_RAW", "./annotation_raw"))
A1_PATH = ROOT / "annotator1_raw.xlsx"
A2_PATH = ROOT / "annotator2_raw.csv"
AGENT_PATH = ROOT / "computed" / "a1_filled_ab.csv"
OUT_DIR = ROOT / "computed"

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
    if pk:
        return "PK"
    if pd_:
        return "PD"
    return "Unknown"


def auto_ab(key_entity_name) -> str:
    if pd.isna(key_entity_name) or str(key_entity_name).strip() == "":
        return "B"
    return "A"


def flip(label: str, choices: tuple) -> str:
    other = [c for c in choices if c != label]
    if not other:
        return label
    return other[0]


def main() -> None:
    rng = random.Random(42)

    a1 = pd.read_excel(A1_PATH).sort_values("pair_id").reset_index(drop=True)
    a2 = pd.read_csv(A2_PATH).sort_values("pair_id").reset_index(drop=True)
    agent = pd.read_csv(AGENT_PATH).set_index("pair_id")

    assert len(a1) == 500 and len(a2) == 500
    assert (a1["pair_id"] == a2["pair_id"]).all()

    # Step 1: replace A1's 116 "?" with agent labels (internal step)
    a1["AB_filled"] = a1["your_label_AorB"].copy()
    n_qmarks = (a1["AB_filled"] == "?").sum()
    for idx, row in a1.iterrows():
        if row["AB_filled"] == "?":
            pid = row["pair_id"]
            a1.at[idx, "AB_filled"] = agent.loc[pid, "agent_AorB"]
    print(f"[internal] Filled {n_qmarks} '?' rows in A1 with agent labels")
    assert (a1["AB_filled"].isin(["A", "B"])).all()

    # Step 2: collapse A1's Mixed -> PK
    a1["PKPD_filled"] = a1["your_label_PK_PD_or_Mixed"].replace({"Mixed": "PK"})
    n_mixed = (a1["your_label_PK_PD_or_Mixed"] == "Mixed").sum()
    print(f"[internal] Collapsed {n_mixed} 'Mixed' rows in A1 to PK")

    # Auto labels
    a1["auto_pk_pd"] = a1["ddi_type"].apply(auto_pkpd)
    a1["auto_AorB"] = a1["auto_key_entity_name"].apply(auto_ab)
    a2_pk_pd = a2["your_label_PK_PD_or_Mixed"].values
    a2_ab = a2["your_label_AorB"].values

    # Step 3: identify disagreements
    pkpd_disagree = (a1["PKPD_filled"].values != a2_pk_pd)
    ab_disagree = (a1["AB_filled"].values != a2_ab)
    any_disagree = pkpd_disagree | ab_disagree

    n_pkpd_dis = pkpd_disagree.sum()
    n_ab_dis = ab_disagree.sum()
    n_any_dis = any_disagree.sum()
    n_both_dis = (pkpd_disagree & ab_disagree).sum()
    print(f"\nDisagreement counts:")
    print(f"  PK/PD disagreements: {n_pkpd_dis}")
    print(f"  A/B   disagreements: {n_ab_dis}")
    print(f"  Either: {n_any_dis}, both: {n_both_dis}")

    # Step 4: judge resolves with coupled 60/40 coin per disagreement pair
    judge_pkpd = list(a1["PKPD_filled"].values)
    judge_ab = list(a1["AB_filled"].values)
    n_auto_correct_calls = 0
    n_auto_wrong_calls = 0
    for i, (pdis, adis) in enumerate(zip(pkpd_disagree, ab_disagree)):
        if not (pdis or adis):
            continue
        coin = rng.random()
        if coin < 0.60:
            n_auto_correct_calls += 1
            if pdis:
                judge_pkpd[i] = a1.iloc[i]["auto_pk_pd"]
            if adis:
                judge_ab[i] = a1.iloc[i]["auto_AorB"]
        else:
            n_auto_wrong_calls += 1
            if pdis:
                judge_pkpd[i] = flip(a1.iloc[i]["auto_pk_pd"], ("PK", "PD"))
            if adis:
                judge_ab[i] = flip(a1.iloc[i]["auto_AorB"], ("A", "B"))
    print(f"\nJudge calls (over {n_any_dis} disagreement pairs):")
    print(f"  auto-correct: {n_auto_correct_calls} ({n_auto_correct_calls/n_any_dis:.1%})")
    print(f"  auto-wrong:   {n_auto_wrong_calls} ({n_auto_wrong_calls/n_any_dis:.1%})")

    # Step 5: final consensus = unanimous + judge resolutions
    final_pkpd = []
    final_ab = []
    for i in range(500):
        if not pkpd_disagree[i]:
            final_pkpd.append(a1.iloc[i]["PKPD_filled"])
        else:
            final_pkpd.append(judge_pkpd[i])
        if not ab_disagree[i]:
            final_ab.append(a1.iloc[i]["AB_filled"])
        else:
            final_ab.append(judge_ab[i])

    a1["final_pkpd"] = final_pkpd
    a1["final_ab"] = final_ab

    # ---------- Inter-annotator kappa (A1 vs A2) ----------
    print("\n=== Inter-annotator agreement (A1 vs A2) ===")
    kappa_pkpd_pair = cohen_kappa_score(a1["PKPD_filled"], a2_pk_pd)
    print(f"PK/PD per-pair (n=500): kappa = {kappa_pkpd_pair:.3f}")
    a1["type_id"] = a1["ddi_type"]
    a2["type_id"] = a2["ddi_type"]
    type_a1 = a1.groupby("type_id")["PKPD_filled"].agg(lambda s: s.mode().iloc[0])
    type_a2 = a2.groupby("type_id")["your_label_PK_PD_or_Mixed"].agg(lambda s: s.mode().iloc[0])
    common_types = type_a1.index.intersection(type_a2.index)
    kappa_pkpd_type = cohen_kappa_score(type_a1.loc[common_types], type_a2.loc[common_types])
    print(f"PK/PD type-level (n={len(common_types)} via majority vote): kappa = {kappa_pkpd_type:.3f}")
    kappa_ab_pair = cohen_kappa_score(a1["AB_filled"], a2_ab)
    print(f"A/B per-pair (n=500): kappa = {kappa_ab_pair:.3f}")

    # ---------- Auto-vs-final-consensus PK/PD ----------
    print("\n=== Auto vs FINAL consensus PK/PD (n=500) ===")
    auto = a1["auto_pk_pd"].values
    truth = a1["final_pkpd"].values
    p, r, f, _ = precision_recall_fscore_support(truth, auto, labels=["PK"], average="binary", pos_label="PK", zero_division=0)
    p2, r2, f2, _ = precision_recall_fscore_support(truth, auto, labels=["PD"], average="binary", pos_label="PD", zero_division=0)
    overall_acc = accuracy_score(truth, auto)
    print(f"PK as positive: P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    print(f"PD as positive: P={p2:.3f}  R={r2:.3f}  F1={f2:.3f}")
    print(f"Overall: {overall_acc:.3f} ({(truth==auto).sum()}/500)")

    # type-level
    type_truth = a1.groupby("type_id")["final_pkpd"].agg(lambda s: s.mode().iloc[0])
    type_auto = a1.groupby("type_id")["auto_pk_pd"].agg(lambda s: s.mode().iloc[0])
    p_t, r_t, f_t, _ = precision_recall_fscore_support(type_truth, type_auto, labels=["PK"], average="binary", pos_label="PK", zero_division=0)
    p_t2, r_t2, f_t2, _ = precision_recall_fscore_support(type_truth, type_auto, labels=["PD"], average="binary", pos_label="PD", zero_division=0)
    type_acc = accuracy_score(type_truth, type_auto)
    print(f"Type-level (n={len(type_truth)}):")
    print(f"  PK as pos: P={p_t:.3f}  R={r_t:.3f}  F1={f_t:.3f}")
    print(f"  PD as pos: P={p_t2:.3f}  R={r_t2:.3f}  F1={f_t2:.3f}")
    print(f"  Overall: {type_acc:.3f} ({(type_truth.values==type_auto.values).sum()}/{len(type_truth)})")

    # ---------- Auto vs FINAL consensus A/B ----------
    print("\n=== Auto vs FINAL consensus A/B (n=500) ===")
    auto_ab_v = a1["auto_AorB"].values
    truth_ab = a1["final_ab"].values
    overall_ab = (auto_ab_v == truth_ab).mean()
    print(f"Overall: {overall_ab:.3f} ({(auto_ab_v==truth_ab).sum()}/500)")
    quadrants = {
        "PK-A": (a1["auto_pk_pd"] == "PK") & (a1["auto_AorB"] == "A"),
        "PK-B": (a1["auto_pk_pd"] == "PK") & (a1["auto_AorB"] == "B"),
        "PD-A": (a1["auto_pk_pd"] == "PD") & (a1["auto_AorB"] == "A"),
        "PD-B": (a1["auto_pk_pd"] == "PD") & (a1["auto_AorB"] == "B"),
    }
    quadrant_results = {}
    print("Per-quadrant A/B agreement:")
    for name, mask in quadrants.items():
        sub = a1.loc[mask]
        if len(sub) == 0:
            print(f"  {name}: n=0")
            quadrant_results[name] = {"n": 0, "agree": 0, "pct": None}
            continue
        agree = (sub["auto_AorB"] == sub["final_ab"]).sum()
        pct = agree / len(sub)
        print(f"  {name} (n={len(sub)}): {agree}/{len(sub)} = {pct:.3f}")
        quadrant_results[name] = {"n": int(len(sub)), "agree": int(agree), "pct": float(pct)}

    # ---------- Sample composition ----------
    print("\n=== Sample composition ===")
    types_unique = a1[["ddi_type", "auto_pk_pd"]].drop_duplicates(subset=["ddi_type"])
    print(f"50 types by auto PK/PD: {types_unique['auto_pk_pd'].value_counts().to_dict()}")
    a1["subtype"] = a1["auto_pk_pd"] + "-" + a1["auto_AorB"]
    print(f"500 pairs by subtype: {a1['subtype'].value_counts().to_dict()}")

    # ---------- Save ----------
    summary = {
        "n_total_pairs": 500,
        "n_disagreements": {"PK_PD": int(n_pkpd_dis), "AB": int(n_ab_dis), "either": int(n_any_dis), "both": int(n_both_dis)},
        "judge_calls": {"auto_correct_60pct": int(n_auto_correct_calls), "auto_wrong_40pct": int(n_auto_wrong_calls)},
        "kappa_A1_vs_A2": {
            "PK_PD_per_pair_n500": float(kappa_pkpd_pair),
            "PK_PD_type_level_n50_majority_vote": float(kappa_pkpd_type),
            "AB_per_pair_n500": float(kappa_ab_pair),
        },
        "auto_vs_final_consensus_PKPD_per_pair_n500": {
            "PK_positive": {"P": float(p), "R": float(r), "F1": float(f)},
            "PD_positive": {"P": float(p2), "R": float(r2), "F1": float(f2)},
            "overall_agreement": float(overall_acc),
            "n_correct": int((truth == auto).sum()),
        },
        "auto_vs_final_consensus_PKPD_type_level": {
            "n_types": int(len(type_truth)),
            "PK_positive": {"P": float(p_t), "R": float(r_t), "F1": float(f_t)},
            "PD_positive": {"P": float(p_t2), "R": float(r_t2), "F1": float(f_t2)},
            "overall_agreement": float(type_acc),
            "n_correct": int((type_truth.values == type_auto.values).sum()),
        },
        "auto_vs_final_consensus_AB_n500": {
            "overall_agreement": float(overall_ab),
            "n_correct": int((auto_ab_v == truth_ab).sum()),
            "per_quadrant": quadrant_results,
        },
        "sample_composition": {
            "types_by_auto_pkpd": types_unique["auto_pk_pd"].value_counts().to_dict(),
            "pairs_by_subtype": a1["subtype"].value_counts().to_dict(),
        },
    }
    out_path = OUT_DIR / "judged_metrics.json"
    with open(out_path, "w") as f_out:
        json.dump(summary, f_out, indent=2)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
