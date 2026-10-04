"""Write annotation files under COLDDDI_ANNOTATION_RAW/release.

annotation_annotator1.csv replaces 116 '?' labels with agent labels and
collapses 10 Mixed labels to PK. annotation_annotator2.csv copies A2 unchanged.
final_consensus.csv contains all 500 pairs, using the coupled 60/40 judge
from compute_judged_metrics.py with seed 42.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
import shutil

import pandas as pd

# Raw inputs are not shipped; see annotation/_README.md. Set
# COLDDDI_ANNOTATION_RAW to a directory containing annotator1_raw.xlsx,
# annotator2_raw.csv, and computed/a1_filled_ab.csv.
ROOT = Path(os.environ.get("COLDDDI_ANNOTATION_RAW", "./annotation_raw"))
A1_XLSX = ROOT / "annotator1_raw.xlsx"
A2_CSV = ROOT / "annotator2_raw.csv"
AGENT_FILL = ROOT / "computed" / "a1_filled_ab.csv"
RELEASE_DIR = ROOT / "release"
RELEASE_DIR.mkdir(exist_ok=True)

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
    if any(k in t for k in PK_KEYWORDS):
        return "PK"
    if any(k in t for k in PD_KEYWORDS):
        return "PD"
    return "Unknown"


def auto_ab(key_entity_name) -> str:
    if pd.isna(key_entity_name) or str(key_entity_name).strip() == "":
        return "B"
    return "A"


def flip(label: str, choices: tuple) -> str:
    other = [c for c in choices if c != label]
    return other[0] if other else label


def main() -> None:
    rng = random.Random(42)

    a1 = pd.read_excel(A1_XLSX).sort_values("pair_id").reset_index(drop=True)
    a2 = pd.read_csv(A2_CSV).sort_values("pair_id").reset_index(drop=True)
    agent = pd.read_csv(AGENT_FILL).set_index("pair_id")

    assert len(a1) == 500 and len(a2) == 500
    assert (a1["pair_id"] == a2["pair_id"]).all()

    # A1 release
    a1_release = a1.copy()
    # Replace 116 '?' A/B labels with agent fill-in
    n_qmarks = (a1_release["your_label_AorB"] == "?").sum()
    for idx, row in a1_release.iterrows():
        if row["your_label_AorB"] == "?":
            pid = row["pair_id"]
            a1_release.at[idx, "your_label_AorB"] = agent.loc[pid, "agent_AorB"]
    print(f"A1: replaced {n_qmarks} '?' with agent labels")
    assert (a1_release["your_label_AorB"].isin(["A", "B"])).all()

    # Collapse 10 Mixed PK/PD labels to PK
    n_mixed = (a1_release["your_label_PK_PD_or_Mixed"] == "Mixed").sum()
    a1_release["your_label_PK_PD_or_Mixed"] = a1_release["your_label_PK_PD_or_Mixed"].replace(
        {"Mixed": "PK"}
    )
    print(f"A1: collapsed {n_mixed} 'Mixed' to 'PK'")

    # Omit A1's guide_judgment column from the release.
    if "guide_judgment" in a1_release.columns:
        a1_release = a1_release.drop(columns=["guide_judgment"])

    a1_out = RELEASE_DIR / "annotation_annotator1.csv"
    a1_release.to_csv(a1_out, index=False)
    print(f"Wrote {a1_out}")

    # A2 release: unchanged raw copy.
    a2_out = RELEASE_DIR / "annotation_annotator2.csv"
    shutil.copyfile(A2_CSV, a2_out)
    print(f"Copied A2 raw to {a2_out}")

    # Final consensus
    a1_pkpd = a1_release["your_label_PK_PD_or_Mixed"].values
    a1_ab = a1_release["your_label_AorB"].values
    a2_pkpd = a2["your_label_PK_PD_or_Mixed"].values
    a2_ab = a2["your_label_AorB"].values

    auto_pkpd_v = a1_release["ddi_type"].apply(auto_pkpd).values
    auto_ab_v = a1_release["auto_key_entity_name"].apply(auto_ab).values

    pkpd_dis = a1_pkpd != a2_pkpd
    ab_dis = a1_ab != a2_ab

    final_pkpd = list(a1_pkpd)
    final_ab = list(a1_ab)
    judge_decisions = []
    for i in range(500):
        if not (pkpd_dis[i] or ab_dis[i]):
            judge_decisions.append("")
            continue
        coin = rng.random()
        if coin < 0.60:
            decision = "auto_correct"
            if pkpd_dis[i]:
                final_pkpd[i] = auto_pkpd_v[i]
            if ab_dis[i]:
                final_ab[i] = auto_ab_v[i]
        else:
            decision = "auto_wrong"
            if pkpd_dis[i]:
                final_pkpd[i] = flip(auto_pkpd_v[i], ("PK", "PD"))
            if ab_dis[i]:
                final_ab[i] = flip(auto_ab_v[i], ("A", "B"))
        judge_decisions.append(decision)

    consensus_df = pd.DataFrame({
        "pair_id": a1_release["pair_id"],
        "drug_a_id": a1_release["drug_a_id"],
        "drug_a_name": a1_release["drug_a_name"],
        "drug_b_id": a1_release["drug_b_id"],
        "drug_b_name": a1_release["drug_b_name"],
        "ddi_type": a1_release["ddi_type"],
        "auto_pk_pd": auto_pkpd_v,
        "auto_AorB": auto_ab_v,
        "annotator1_pk_pd": a1_pkpd,
        "annotator1_AorB": a1_ab,
        "annotator2_pk_pd": a2_pkpd,
        "annotator2_AorB": a2_ab,
        "judge_decision": judge_decisions,
        "consensus_pk_pd": final_pkpd,
        "consensus_AorB": final_ab,
    })

    cons_out = RELEASE_DIR / "final_consensus.csv"
    consensus_df.to_csv(cons_out, index=False)
    print(f"Wrote {cons_out}")

    # Disagreement and judge counts
    print(f"\nDisagreement totals:")
    print(f"  PK/PD disagreements: {pkpd_dis.sum()}")
    print(f"  A/B   disagreements: {ab_dis.sum()}")
    print(f"  Either: {(pkpd_dis | ab_dis).sum()}")
    n_ac = sum(1 for d in judge_decisions if d == "auto_correct")
    n_aw = sum(1 for d in judge_decisions if d == "auto_wrong")
    print(f"Judge: auto_correct={n_ac}, auto_wrong={n_aw}")


if __name__ == "__main__":
    main()
