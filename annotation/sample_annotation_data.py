"""Build the annotation sample for ColdDDI Appendix A.taxonomy-validation.

The reference sample has 50 types and 500 pairs: 16 PK types with >=10
DrugBank/DDInter overlap pairs, 9 supplementary PK types with >=10 DrugBank
pairs, and 25 PD types drawn from 117 types with >=10 overlap pairs.
Sample up to 10 pairs per type with seed 42, using overlap pairs except for
supplementary PK types. Rows are shuffled, not strictly interleaved by type.

Each row asks for PK/PD/Mixed and A/B/? labels; valid A/B action pairs depend
on the mechanism class. Show DrugBank descriptions and available DDInter 2.0
mecddi text. Majority-vote PK/PD labels per type give the appendix's type-level
metric. Both annotators receive the same blinded CSV without auto labels.

COLDDDI_* paths below supply DDInter mecddi, DrugBank mechanisms and drug
names, type-level PK/PD labels, and per-pair key entities/actions/chains.
Outputs under COLDDDI_ANNOTATION_OUT_DIR (default ./annotation_sample_out):
  - annotation_blank.csv: blinded pairs for annotators.
  - annotation_with_auto.csv: admin copy with auto labels.
  - annotation_sampling_log.txt: sampled types, counts, and rules.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

SEED = 42
N_PD_TYPES = 25
N_PK_SUPP = 9      # supplementary PK types (no overlap requirement)
PAIRS_PER_TYPE = 10
EXAMPLES_PER_T1 = 5  # T1 shows the first 5 of the 10 T2 pairs

# Convert Windows drive paths when running under WSL.
def _resolve(p: str) -> Path:
    p = p.replace("\\", "/")
    import sys
    if sys.platform.startswith("linux") and len(p) >= 2 and p[1] == ":":
        return Path(f"/mnt/{p[0].lower()}/{p[2:].lstrip('/')}")
    return Path(p)


# Input CSV paths and output directory overrides:
#   COLDDDI_DDINTER_MECDDI            DDInter 2.0 mec-DDI table
#   COLDDDI_DRUGBANK_WITH_MECHANISMS  drugbank_with_mechanisms.csv
#   COLDDDI_DRUGS_ENRICHED            drugs_enriched.csv
#   COLDDDI_DDI_PKPD_LABELS           ddi_pk_pd_labels.csv
#   COLDDDI_DDI_KEY_ENTITIES          ddi_key_entities.csv
#   COLDDDI_ANNOTATION_OUT_DIR        output dir (default: ./annotation_sample_out)
import os

P_MEC   = _resolve(os.environ.get("COLDDDI_DDINTER_MECDDI", "./inputs/DDInter2_0_mecddi.csv"))
P_DB    = _resolve(os.environ.get("COLDDDI_DRUGBANK_WITH_MECHANISMS", "./inputs/drugbank_with_mechanisms.csv"))
P_DRUGS = _resolve(os.environ.get("COLDDDI_DRUGS_ENRICHED", "./inputs/drugs_enriched.csv"))
P_PKPD  = _resolve(os.environ.get("COLDDDI_DDI_PKPD_LABELS", "./inputs/ddi_pk_pd_labels.csv"))
P_KE    = _resolve(os.environ.get("COLDDDI_DDI_KEY_ENTITIES", "./inputs/ddi_key_entities.csv"))

OUT_DIR = _resolve(os.environ.get("COLDDDI_ANNOTATION_OUT_DIR", "./annotation_sample_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def canon(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([str(a), str(b)]))


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    log: list[str] = [f"=== ColdDDI annotation sampling log (seed={SEED}) ==="]

    # Load inputs.
    print(f"[load] DDInter v2.0 mecddi  {P_MEC}")
    mec = pd.read_csv(P_MEC)
    print(f"  rows={len(mec):,}")
    print(f"[load] DrugBank with mechanisms  {P_DB}")
    db = pd.read_csv(P_DB)
    print(f"  rows={len(db):,}")
    print(f"[load] Drug enrichment  {P_DRUGS}")
    drugs = pd.read_csv(P_DRUGS, usecols=["drugbank_id", "name"])
    print(f"  rows={len(drugs):,}")
    print(f"[load] Auto PK/PD labels per type")
    pkpd = pd.read_csv(P_PKPD)
    print(f"  rows={len(pkpd):,}")
    print(f"[load] Auto key-entity per pair")
    ke = pd.read_csv(P_KE)
    print(f"  rows={len(ke):,}")

    # Name and type lookups
    name_to_id = {n.lower(): did for did, n in zip(drugs.drugbank_id, drugs.name)
                  if isinstance(n, str)}
    id_to_name = dict(zip(drugs.drugbank_id, drugs.name))
    type_to_label = dict(zip(pkpd.ddi_type, pkpd.pk_pd_label))

    # Match DDInter pairs to DrugBank IDs by name.
    mec["a"] = mec.drug1_name.str.lower().map(name_to_id)
    mec["b"] = mec.drug2_name.str.lower().map(name_to_id)
    mec_match = mec.dropna(subset=["a", "b"])
    log.append(f"DDInter mecddi name-matched: {len(mec_match):,}/{len(mec):,} "
               f"({len(mec_match)/len(mec)*100:.1f}%)")

    mec_text: dict[Tuple[str, str], str] = {}
    for a, b, t in zip(mec_match.a, mec_match.b, mec_match.interaction):
        mec_text[canon(a, b)] = t

    # Add overlap flags and DDInter text.
    db = db.copy()
    db["pair"] = [canon(a, b) for a, b in zip(db.drug_a_id, db.drug_b_id)]
    db["ddinter_mecddi"] = db.pair.map(mec_text)
    db["in_overlap"] = db.ddinter_mecddi.notna()
    db["pk_pd_label"] = db.ddi_type.map(type_to_label)
    log.append(f"DB pairs in overlap: {db.in_overlap.sum():,}/{len(db):,} "
               f"({db.in_overlap.mean()*100:.1f}%)")

    # Per-type counts
    type_overlap = (db[db.in_overlap]
                    .groupby(["ddi_type", "pk_pd_label"]).size()
                    .reset_index(name="n_overlap"))
    type_db_total = db.groupby("ddi_type").size().reset_index(name="n_db_total")
    types = type_overlap.merge(type_db_total, on="ddi_type", how="outer")

    pk_ge10 = types[(types.pk_pd_label == "PK") & (types.n_overlap >= 10)].ddi_type.tolist()
    pd_ge10 = types[(types.pk_pd_label == "PD") & (types.n_overlap >= 10)].ddi_type.tolist()
    log.append(f"PK types with ≥10 overlap pairs: {len(pk_ge10)}")
    log.append(f"PD types with ≥10 overlap pairs: {len(pd_ge10)}")

    # PK supplementary pool: PK types not in pk_ge10 but with ≥10 DrugBank pairs
    all_pk_types = pkpd[pkpd.pk_pd_label == "PK"].ddi_type.tolist()
    pk_supp_pool_df = type_db_total[
        type_db_total.ddi_type.isin(all_pk_types)
        & (~type_db_total.ddi_type.isin(pk_ge10))
        & (type_db_total.n_db_total >= 10)
    ]
    pk_supp_pool = pk_supp_pool_df.ddi_type.tolist()
    log.append(f"PK supplementary pool (no overlap req, ≥10 DB pairs): {len(pk_supp_pool)}")

    if len(pk_supp_pool) >= N_PK_SUPP:
        pk_supp_picked = rng.sample(pk_supp_pool, N_PK_SUPP)
    else:
        pk_supp_picked = pk_supp_pool
        log.append(f"  [WARN] only {len(pk_supp_pool)} supplementary PK types available "
                   f"(< {N_PK_SUPP} requested)")

    pd_picked = rng.sample(pd_ge10, N_PD_TYPES)
    log.append(f"PD types randomly picked from ≥10-overlap pool: {len(pd_picked)}")

    # Preserve type order: PK overlap, supplementary PK, then PD.
    sampled_types = pk_ge10 + pk_supp_picked + pd_picked
    log.append(f"Total types sampled: {len(sampled_types)} "
               f"(PK={len(pk_ge10)+len(pk_supp_picked)}, PD={len(pd_picked)})")

    # Sample up to 10 pairs per type: overlap for PK/PD, full DB for PK supp.
    def sample_pairs(t: str, want_overlap: bool) -> pd.DataFrame:
        pool = db[db.ddi_type == t]
        if want_overlap:
            pool = pool[pool.in_overlap]
        n = min(PAIRS_PER_TYPE, len(pool))
        return pool.sample(n=n, random_state=SEED).reset_index(drop=True)

    pair_dfs: list[pd.DataFrame] = []
    for i, t in enumerate(sampled_types, start=1):
        is_pk_supp = t in pk_supp_picked
        sampled = sample_pairs(t, want_overlap=not is_pk_supp)
        sampled = sampled.copy()
        sampled["type_id"] = f"T{i:02d}"
        sampled["sampling_rule"] = ("PK_supp" if is_pk_supp
                                    else ("PK_overlap" if t in pk_ge10 else "PD_overlap"))
        pair_dfs.append(sampled)
    pair_df = pd.concat(pair_dfs, ignore_index=True)

    # Seeded shuffle reduces type grouping and repeated-answer bias; adjacent
    # rows may still share a type. Assign pair_id afterward so P0001 is row 1.
    pair_df = pair_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pair_df["pair_id"] = [f"P{i:04d}" for i in range(1, len(pair_df)+1)]
    log.append(f"Total pairs sampled: {len(pair_df)}  (rows shuffled, seed={SEED})")

    # Join key entities by canonical pair.
    ke = ke.copy()
    ke["pair"] = [canon(a, b) for a, b in zip(ke.drug_a_id, ke.drug_b_id)]
    # Keep source orientation to align drug-specific actions.
    ke_lookup: dict[Tuple[str, str], dict] = {}
    for _, r in ke.iterrows():
        ke_lookup[canon(r.drug_a_id, r.drug_b_id)] = {
            "ke_orig_a": r.drug_a_id,
            "ke_orig_b": r.drug_b_id,
            "key_entity_id": r.key_entity_id,
            "key_entity_name": r.key_entity_name,
            "key_entity_type": r.key_entity_type,
            "action_drug_a": r.action_drug_a,
            "action_drug_b": r.action_drug_b,
            "mechanism_chain": r.mechanism_chain,
            "has_key_entity": r.has_key_entity,
        }

    def _lookup_ke(row) -> pd.Series:
        info = ke_lookup.get(canon(row.drug_a_id, row.drug_b_id))
        if info is None:
            return pd.Series({"auto_key_entity_name": None, "auto_key_entity_type": None,
                              "auto_action_drug_a": None, "auto_action_drug_b": None,
                              "auto_chain": None, "auto_has_key_entity": False})
        # Swap actions when the source pair is reversed.
        if info["ke_orig_a"] == row.drug_a_id:
            aa, bb = info["action_drug_a"], info["action_drug_b"]
        else:
            aa, bb = info["action_drug_b"], info["action_drug_a"]
        return pd.Series({
            "auto_key_entity_name": info["key_entity_name"],
            "auto_key_entity_type": info["key_entity_type"],
            "auto_action_drug_a":   aa,
            "auto_action_drug_b":   bb,
            "auto_chain":           info["mechanism_chain"],
            "auto_has_key_entity":  bool(info["has_key_entity"]),
        })

    pair_df[["auto_key_entity_name","auto_key_entity_type","auto_action_drug_a",
             "auto_action_drug_b","auto_chain","auto_has_key_entity"]] = pair_df.apply(
        _lookup_ke, axis=1
    )

    # Drug names
    pair_df["drug_a_name"] = pair_df.drug_a_id.map(id_to_name)
    pair_df["drug_b_name"] = pair_df.drug_b_id.map(id_to_name)
    # Write missing DDInter text as empty CSV fields.
    pair_df["ddinter_mecddi"] = pair_df["ddinter_mecddi"].fillna("")

    # Derive auto labels for the admin copy only: has_key_entity is the A/B
    # prediction being validated.
    pair_df["auto_AorB"] = pair_df["auto_has_key_entity"].map({True: "A", False: "B"})
    pair_df["auto_pk_pd"] = pair_df["ddi_type"].map(type_to_label)

    # Admin CSV with auto labels
    admin_cols = [
        "pair_id", "type_id", "sampling_rule",
        "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name",
        "ddi_type", "description", "ddinter_mecddi",
        "auto_pk_pd", "auto_AorB",
        "auto_key_entity_name", "auto_key_entity_type",
        "auto_action_drug_a", "auto_action_drug_b", "auto_chain",
        "auto_has_key_entity",
    ]
    admin = pair_df[admin_cols].rename(columns={"description": "drugbank_description"})
    admin.to_csv(OUT_DIR / "annotation_with_auto.csv", index=False)
    log.append(f"Wrote {OUT_DIR / 'annotation_with_auto.csv'}")

    # Blind auto_pk_pd, auto_AorB, and auto_has_key_entity (True=A, False=B).
    # Keep entity name/type/actions/chain so annotators can assess mediation.
    # Leakage remains: these fields are empty for auto-B rows, revealing the
    # prediction even without explicit labels.
    blank_cols = [
        "pair_id",
        "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name",
        "ddi_type", "drugbank_description", "ddinter_mecddi",
        "auto_key_entity_name", "auto_key_entity_type",
        "auto_action_drug_a", "auto_action_drug_b", "auto_chain",
        # Annotator fields
        "your_label_PK_PD_or_Mixed",
        "your_label_AorB",
        "your_confidence_1to5",
        "notes",
    ]
    blank = admin.copy()
    blank["your_label_PK_PD_or_Mixed"] = ""
    blank["your_label_AorB"] = ""
    blank["your_confidence_1to5"] = ""
    blank["notes"] = ""
    blank = blank[blank_cols]
    blank.to_csv(OUT_DIR / "annotation_blank.csv", index=False)
    log.append(f"Wrote {OUT_DIR / 'annotation_blank.csv'} (auto_has_key_entity dropped)")

    # Sampling log
    log.append("")
    log.append("Sampled types (in order):")
    for i, t in enumerate(sampled_types, start=1):
        rule = ("PK_supp" if t in pk_supp_picked
                else ("PK_overlap" if t in pk_ge10 else "PD_overlap"))
        nover = ((db.ddi_type == t) & db.in_overlap).sum()
        ndb   = (db.ddi_type == t).sum()
        log.append(f"  T{i:02d}  [{rule:11s}]  pk_pd={type_to_label.get(t,'?'):5s}  "
                   f"n_overlap={nover:5d}  n_db={ndb:6d}  | {t[:80]}")

    log.append("")
    log.append(f"Final pair counts:")
    by_rule = pair_df.groupby("sampling_rule").size().to_dict()
    for k, v in by_rule.items():
        log.append(f"  {k}: {v}")
    log.append(f"  TOTAL: {len(pair_df)}")

    (OUT_DIR / "annotation_sampling_log.txt").write_text(
        "\n".join(log), encoding="utf-8"
    )
    print("\n".join(log[-15:]))
    print(f"\n[done] Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
