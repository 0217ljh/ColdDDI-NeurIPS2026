"""
Build the human-annotation dataset for ColdDDI Appendix A.taxonomy-validation.

Design (locked with user, 2026-04-30):
  • 50 DDI types: 25 PK (16 overlap-eligible + 9 supplementary from PK pool
    without ≥10 overlap pairs) + 25 PD (random sample from 117 ≥10-overlap PD
    types). Per-type pair pool: 10 pairs (or all if fewer available).
  • Single merged annotation file (per-pair). Each row asks the annotator for
    BOTH a PK/PD/Mixed label AND an A/B/? label on the same drug pair, since
    the two judgments inform each other (A/B's valid action-pair table depends
    on the mechanism class). 500 rows total.
  • Each pair shows DrugBank description AND DDInter v2.0 mecddi free-text
    mechanism description side-by-side. The 9 supplementary PK types have NO
    DDInter text (sampled from DrugBank only, no overlap requirement).
  • Two annotators receive blinded CSVs (no auto_pk_pd / auto_AorB columns).
    Rows shuffled (consecutive rows do not share the same DDI type).
    Sampling seed = 42 (matches training seeds).

Note on paper appendix: Appendix A.taxonomy-validation describes a two-task
protocol (50 types for PK/PD + 500 pairs for A/B). In practice we merge into
a single per-pair CSV; aggregating the per-pair PK/PD labels back to type
level (majority vote per type) reproduces the metric the paper reports.

Inputs (paths supplied via env vars — see the COLDDDI_* block near the
bottom of this file):
  - DDInter v2.0 mecddi (free-text mechanism per pair, 152K rows)
  - DrugBank with mechanisms (565K pairs, our DDI source)
  - DrugBank drug enrichment (id ↔ name + targets/enzymes/etc.)
  - Auto PK/PD labels per type (keyword-derived)
  - Auto key-entity table (per pair: entity, role, chain)

Outputs (default ``./annotation_sample_out``, overridable):
  • annotation_blank.csv         — 500 pairs, blinded (give to each annotator)
  • annotation_with_auto.csv     — admin-only: same rows + auto_pk_pd + auto_AorB
  • annotation_sampling_log.txt  — what was sampled, by which rule
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

# Paths --- tolerate both Win and WSL invocation (auto-detect drive prefix)
def _resolve(p: str) -> Path:
    p = p.replace("\\", "/")
    import sys
    if sys.platform.startswith("linux") and len(p) >= 2 and p[1] == ":":
        return Path(f"/mnt/{p[0].lower()}/{p[2:].lstrip('/')}")
    return Path(p)


# Required inputs (env-var driven; raw paths intentionally not hard-coded).
# Set each var to a real CSV before running this sampler:
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

    # ── Load ───────────────────────────────────────────────────────────────
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

    # ── Maps ────────────────────────────────────────────────────────────────
    name_to_id = {n.lower(): did for did, n in zip(drugs.drugbank_id, drugs.name)
                  if isinstance(n, str)}
    id_to_name = dict(zip(drugs.drugbank_id, drugs.name))
    type_to_label = dict(zip(pkpd.ddi_type, pkpd.pk_pd_label))

    # ── Map DDInter pairs to DrugBank IDs by drug name ─────────────────────
    mec["a"] = mec.drug1_name.str.lower().map(name_to_id)
    mec["b"] = mec.drug2_name.str.lower().map(name_to_id)
    mec_match = mec.dropna(subset=["a", "b"])
    log.append(f"DDInter mecddi name-matched: {len(mec_match):,}/{len(mec):,} "
               f"({len(mec_match)/len(mec)*100:.1f}%)")

    mec_text: dict[Tuple[str, str], str] = {}
    for a, b, t in zip(mec_match.a, mec_match.b, mec_match.interaction):
        mec_text[canon(a, b)] = t

    # Annotate DB with overlap flag + DDInter text
    db = db.copy()
    db["pair"] = [canon(a, b) for a, b in zip(db.drug_a_id, db.drug_b_id)]
    db["ddinter_mecddi"] = db.pair.map(mec_text)
    db["in_overlap"] = db.ddinter_mecddi.notna()
    db["pk_pd_label"] = db.ddi_type.map(type_to_label)
    log.append(f"DB pairs in overlap: {db.in_overlap.sum():,}/{len(db):,} "
               f"({db.in_overlap.mean()*100:.1f}%)")

    # ── Build per-type counts ──────────────────────────────────────────────
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

    # ── Compose final type list (preserve order: PK overlap, PK supp, PD) ──
    sampled_types = pk_ge10 + pk_supp_picked + pd_picked
    log.append(f"Total types sampled: {len(sampled_types)} "
               f"(PK={len(pk_ge10)+len(pk_supp_picked)}, PD={len(pd_picked)})")

    # ── For each sampled type, sample 10 pairs ─────────────────────────────
    # PK overlap and PD: from overlap subset; PK supp: from full DB pool.
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

    # ── Shuffle pair order so consecutive rows do NOT share the same DDI type.
    # Without this, an annotator seeing 10 rows of identical type_template will
    # quickly learn the "obvious" answer and copy-paste, which would inflate the
    # auto-vs-consensus agreement and depress Cohen's kappa power. Shuffle is
    # done with the same SEED so it is reproducible. pair_id is assigned AFTER
    # shuffling, so P0001 corresponds to the first row in the delivered file.
    pair_df = pair_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pair_df["pair_id"] = [f"P{i:04d}" for i in range(1, len(pair_df)+1)]
    log.append(f"Total pairs sampled: {len(pair_df)}  (rows shuffled, seed={SEED})")

    # ── Join key-entity columns (canonical-pair lookup) ─────────────────────
    ke = ke.copy()
    ke["pair"] = [canon(a, b) for a, b in zip(ke.drug_a_id, ke.drug_b_id)]
    # Track original drug_a_id ordering so we know whether to swap action labels.
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
        # If pair order in ke is reversed relative to current row, swap action_*
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
    # Fill ddinter_mecddi NaN as "" so CSV reads cleanly
    pair_df["ddinter_mecddi"] = pair_df["ddinter_mecddi"].fillna("")

    # ── Build T2 outputs ───────────────────────────────────────────────────
    # The "automated A/B" label is derived from has_key_entity (the very
    # quantity the annotator is asked to validate). Keep in admin copy.
    pair_df["auto_AorB"] = pair_df["auto_has_key_entity"].map({True: "A", False: "B"})
    pair_df["auto_pk_pd"] = pair_df["ddi_type"].map(type_to_label)

    # ── Build the merged per-pair admin CSV (with auto labels) ──────────────
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

    # ── Build the blinded per-pair CSV given to annotators ──────────────────
    # Drop auto_pk_pd and auto_AorB (the two answers we're verifying); also
    # drop auto_has_key_entity since it is the *exact* boolean form of
    # auto_AorB (True ↔ A, False ↔ B) and would let an annotator one-shot
    # copy auto's A/B prediction. Keep the auto entity *content* columns
    # (name / type / actions / chain) since those are inputs to the A/B
    # decision: the annotator must see WHICH entity auto flagged in order
    # to verify it. Note: there is still a structural information leak —
    # for auto-B rows these content columns are all empty — but that leak
    # is inherent to the task design (we are asking "does the auto-flagged
    # entity mediate the interaction?", which requires showing the entity
    # when one is flagged and showing nothing when none is flagged).
    blank_cols = [
        "pair_id",
        "drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name",
        "ddi_type", "drugbank_description", "ddinter_mecddi",
        "auto_key_entity_name", "auto_key_entity_type",
        "auto_action_drug_a", "auto_action_drug_b", "auto_chain",
        # ↓ annotator fills these
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

    # ── Sampling log ───────────────────────────────────────────────────────
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
