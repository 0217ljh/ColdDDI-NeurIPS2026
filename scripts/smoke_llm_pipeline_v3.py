"""End-to-end LLM pipeline smoke v3 — full chain in ONE script.

Chains every stage of the LLM stack (L1-L6) on the toy fixture, using
the tiny random Llama for mechanism verification:

  Stage 1  Load PairDataset (seed=42) + build retrieval artifacts
  Stage 2  Render P4 (OHS) prompts for FT samples
  Stage 3  Train LoRA with multi-split val (S0/S1/S2)
  Stage 4  Per-split best-ckpt selection (L5) — read yes/no token +
           prompt_cfg from fit_info.json so L5 cannot diverge from L3
  Stage 5  Build cold-start-preserving swap candidates (G2-restricted) —
           **must run before inference** so Stage 6 knows which extra
           swap-target pairs (qa_prime, qb) to score
  Stage 6  Use best-S2 LoRA → run R0/R1/R2/R3 inference on the union
           of test_s2 base pairs and swap-target pairs
  Stage 7  Compute L6 indicators + A-B gap, with coverage warnings on

Expected runtime: ~2 minutes on CPU.  Numerics are meaningless (random
weights); the point is to verify every cross-stage handoff works:

  L3.save_adapter → L5.parse → L5.score → L5.select → L2.predict → L6.indicators
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_RELEASE = REPO_ROOT / "data" / "public" / "intermediate"
AB_PARQUET = REPO_ROOT / "annotations" / "ab_sample.parquet"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def banner(s: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {s}")
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — Dataset, retrieval, lookup tables
# ══════════════════════════════════════════════════════════════════════════
def stage1_setup():
    banner("STAGE 1  Load PairDataset + retrieval artifacts")
    from coldddi.data.dataset import PairDataset
    from coldddi.diagnostics import build_bucket_lookup
    from coldddi.llm.retrieval import build_subgraph_map

    ds = PairDataset.from_release_dir(TOY_RELEASE, seed=42)
    sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
    bucket_lookup = build_bucket_lookup(AB_PARQUET)

    id2name = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["name"].astype(str),
    ))
    id2smi = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["smiles"].astype(str),
    ))

    # Key-entity map for R2/R3 entity-masking prompts.
    ab_df = pd.read_parquet(AB_PARQUET)
    ke_map = {
        (str(r.drug_a_id), str(r.drug_b_id)): {
            "key_entity_name": str(r.key_entity_name)
                if not pd.isna(r.key_entity_name) else "",
            "key_entity_type": str(r.key_entity_type)
                if not pd.isna(r.key_entity_type) else "",
            "has_key_entity": bool(r.has_key_entity),
        }
        for r in ab_df.itertuples(index=False)
    }

    print(f"  drugs        : {len(ds.drugs)}")
    print(f"  train pairs  : {len(ds.splits.train)}")
    print(f"  test_s2      : {len(ds.splits.test_s2)} positives + "
          f"{len(ds.get_negatives('test_s2'))} negatives")
    print(f"  subgraph_map : {len(sm.data)} drugs covered")
    print(f"  bucket_lookup: {len(bucket_lookup.pair_to_bucket)} PK/PD-labelled pairs")
    print(f"  ke_map       : {sum(1 for v in ke_map.values() if v['has_key_entity'])} "
          f"pairs with confirmed key entity")
    return ds, sm, id2name, id2smi, ke_map, bucket_lookup


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — Build FT samples (P4 baseline) with multi-split val
# ══════════════════════════════════════════════════════════════════════════
def stage2_build_ft_samples(ds, id2name, id2smi):
    banner("STAGE 2  Build P4 FT samples + multi-split val")
    from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
    from coldddi.llm.retrieval import to_llm_samples, build_subgraph_map

    sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
    cfg_p4 = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=TINY_MODEL,
    )

    def _samples(pos_df: pd.DataFrame, neg_df: pd.DataFrame, *, k_max=8):
        pos = pos_df.head(k_max)[["drug_a_id", "drug_b_id"]]
        neg = neg_df.head(k_max)[["drug_a_id", "drug_b_id"]]
        pairs = pd.concat([pos, neg], ignore_index=True)
        labels = [1] * len(pos) + [0] * len(neg)
        samples = to_llm_samples(pairs, labels, ds=ds, subgraph_map=sm)
        # Render full FT prompts with the answer token at end.
        feats = []
        for s in samples:
            text = build_binary_prompt(
                s, cfg_p4,
                drug_id2name=id2name, drug_id2smiles=id2smi,
            )
            feats.append({"text": text, "cls_labels": s["label"]})
        return feats

    train_feats = _samples(ds.splits.train, ds.get_train_negatives(0), k_max=8)
    val_dict: dict[str, list[dict]] = {}
    for short, name in (("S0", "val_s0"), ("S1", "val_s1"), ("S2", "val_s2")):
        pos = dict(ds.splits.items())[name]
        if len(pos) == 0:
            continue
        neg = ds.get_negatives(name)
        if len(neg) == 0:
            continue
        val_dict[short] = _samples(pos, neg, k_max=4)

    print(f"  train_samples: {len(train_feats)} "
          f"(8 pos + 8 neg from G1×G1 train)")
    for k, v in val_dict.items():
        print(f"  val[{k}]    : {len(v):>2} samples")
    return train_feats, val_dict, cfg_p4


# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — Run LoRA fit (multi-split eval) — L3
# ══════════════════════════════════════════════════════════════════════════
def stage3_fit(train_feats, val_dict, cfg_p4):
    banner("STAGE 3  LoRA fit with S0/S1/S2 multi-split eval")
    from coldddi.llm.trainer import LLMTrainerConfig, LoRATrainer

    out = Path(tempfile.mkdtemp(prefix="v3_ft_"))
    cfg = LLMTrainerConfig(
        model_name=TINY_MODEL,
        output_dir=str(out / "ckpts"),
        dtype="float32", device="cpu",
        num_epochs=1, micro_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=1e-4, max_length=512,
        logging_steps=1, save_steps=2, eval_steps=2,
        save_total_limit=5, eval_strategy="steps",
        primary_val_split="S2",
        disable_tqdm=True, report_to=(), seed=42,
    )
    cfg.lora.r = 4
    cfg.lora.alpha = 8
    cfg.lora.target_modules = ("q_proj", "v_proj")

    print(f"  output_dir       : {cfg.output_dir}")
    print(f"  primary_val_split: {cfg.primary_val_split}")
    print(f"  LoRA r/alpha     : {cfg.lora.r} / {cfg.lora.alpha}")
    print(f"  val splits       : {sorted(val_dict)}")

    # Pass cfg_p4 through so fit_info.json records the prompt method
    # the LoRA was trained on; L5 will then enforce/warn on mismatch.
    trainer = LoRATrainer(cfg)
    info = trainer.fit(
        train_samples=train_feats, val_samples=val_dict,
        prompt_cfg=cfg_p4,
    )

    eval_keys = sorted({
        k for row in info["log_history"]
        for k in row if k.startswith("eval_") and k.endswith("_loss")
    })
    print(f"  log_history eval keys: {eval_keys}")
    print(f"  best_ckpt (HF Trainer): {info['best_ckpt']}")
    print(f"  → fit OK, primary metric = eval_S2_loss")
    return cfg, info


# ══════════════════════════════════════════════════════════════════════════
# Stage 4 — Per-split best-ckpt selection (L5)
# ══════════════════════════════════════════════════════════════════════════
def stage4_select(cfg, ds, sm, cfg_p4, val_dict):
    banner("STAGE 4  Per-split best-ckpt selection (L5)")
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.select_best import (
        assert_prompt_cfg_matches_fit_info,
        parse_candidate_ckpts,
        read_fit_info,
        score_candidate_ckpts,
        select_best,
    )

    # Reconstruct training contract from fit_info.json so the runner
    # config used here cannot drift from the one the LoRA was trained
    # under — yes_token / no_token / prompt method are persisted by
    # LoRATrainer.fit() expressly for this handoff.
    fit_info = read_fit_info(cfg.output_dir)
    print(f"  fit_info.yes_token : {fit_info['yes_token']!r}")
    print(f"  fit_info.no_token  : {fit_info['no_token']!r}")
    print(f"  fit_info.prompt    : {fit_info['prompt_cfg']}")
    assert_prompt_cfg_matches_fit_info(fit_info, cfg_p4)

    splits = tuple(val_dict.keys())
    print(f"  splits requested: {splits}")
    cands = parse_candidate_ckpts(cfg.output_dir, splits=splits, topk=2)
    for s, lst in cands.items():
        steps = sorted({c.step for c in lst})
        print(f"  {s}: {len(lst):>2} candidates, steps={steps}")

    scored = score_candidate_ckpts(
        cands,
        base_model_name=fit_info["model_name"],
        dataset=ds,
        prompt_cfg=cfg_p4,
        subgraph_map=sm,
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
        device="cpu", dtype="float32",
        batch_size=4, max_length=fit_info["max_length"],
    )
    manifest = select_best(scored)
    print()
    print(f"  {'split':<6}{'best_step':>12}{'val_AUC':>12}  best_ckpt")
    for s, m in manifest.items():
        print(f"  {s:<6}{m['best_step']:>12}{m['best_val_auc']:>12.3f}  "
              f"{m['best_ckpt']}")
    print(f"  → best S2 LoRA selected: {manifest['S2']['best_ckpt']}")
    return manifest


# ══════════════════════════════════════════════════════════════════════════
# Stage 5 — Cold-start swap candidates (L6 swap) — MUST run before
# inference so Stage 6 knows which (qa_prime, qb) pairs to score.
# Without this, KPS-F silently undercounts every triple whose swap
# target wasn't already in the test_s2 base pair list.
# ══════════════════════════════════════════════════════════════════════════
def stage5_swap_candidates(ds):
    banner("STAGE 5  Build swap candidates (G2-restricted, both directions)")
    from coldddi.diagnostics import build_swap_candidates

    swap = build_swap_candidates(
        ds, source_split="test_s2",
        search_pool_splits=("test_s1", "test_s2"),
        # drug_pool=None → auto-restrict to same-partition (G2 for test_s2)
    )
    pos = sum(1 for t in swap if t.label_uv == 1)
    neg = sum(1 for t in swap if t.label_uv == 0)
    print(f"  total triples : {len(swap)}")
    print(f"  positive base : {pos}")
    print(f"  negative base : {neg}")
    # Sanity: no self-pairs and u' != u.
    assert all(t.qa_prime != t.qa and t.qa_prime != t.qb for t in swap)
    print(f"  → all triples pass `u' != u` and `u' != v` filters")
    return swap


# ══════════════════════════════════════════════════════════════════════════
# Stage 6 — R0/R1/R2/R3 inference using best-S2 LoRA (L2)
# ══════════════════════════════════════════════════════════════════════════
def stage6_inference(manifest, swap, ds, sm, id2name, id2smi, ke_map):
    banner("STAGE 6  R0/R1/R2/R3 inference (best-S2 LoRA, union pair set)")
    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.retrieval import to_llm_samples
    from coldddi.llm.select_best import read_fit_info

    best_s2 = manifest["S2"]["best_ckpt"]
    print(f"  adapter : {best_s2}")

    # Reconstruct yes/no token from the trainer's fit_info — the
    # checkpoint dir does not carry adapter_info.json under HF
    # Trainer's save_strategy="steps", so use the run-dir-level file.
    run_dir = Path(best_s2).parent
    fit_info = read_fit_info(run_dir)

    # Pair set = test_s2 base ∪ swap-target pairs (qa_prime, qb).
    # Without the swap-target union, every KPS-F triple referencing a
    # qa_prime that's not in test_s2 silently drops (codex blocker #1).
    test_s2_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    test_s2_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    base_pairs = pd.concat([test_s2_pos, test_s2_neg], ignore_index=True)
    swap_target_pairs = pd.DataFrame(
        [{"drug_a_id": t.qa_prime, "drug_b_id": t.qb} for t in swap]
    )
    all_pairs = pd.concat(
        [base_pairs, swap_target_pairs], ignore_index=True,
    ).drop_duplicates(subset=["drug_a_id", "drug_b_id"])
    print(f"  test_s2 base pairs    : {len(base_pairs)}")
    print(f"  swap-target pairs     : {len(swap_target_pairs)}")
    print(f"  union (unique) pairs  : {len(all_pairs)}")

    runner = LLMInferenceRunner(LLMRunnerConfig(
        model_name=fit_info["model_name"],
        adapter_path=best_s2,
        dtype="float32", device="cpu",
        batch_size=8, max_length=fit_info["max_length"],
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
    ))
    runner.load()
    samples = to_llm_samples(all_pairs, labels=None, ds=ds, subgraph_map=sm)

    method_by_cond = {
        "R0": "One_Hop_Subgraph_Sequence",
        "R1": "OHS_Mask_Name",
        "R2": "OHS_Mask_Entity",
        "R3": "OHS_Mask_Name_Entity",
    }
    predictions: dict[str, dict] = {}
    for cond, method in method_by_cond.items():
        # Use the trained model_name so the chat-template family
        # stays identical between train and inference.
        cfg = PromptBuildConfig(
            task_name="Binary_cls",
            method=method,
            model_name=fit_info["model_name"],
        )
        df = runner.score_samples(
            samples, cfg,
            drug_id2name=id2name, drug_id2smiles=id2smi,
            key_entity_map=ke_map,
        )
        predictions[cond] = {
            (str(r["drug_a_id"]), str(r["drug_b_id"])): float(r["p_yes"])
            for _, r in df.iterrows()
        }
        all_in_range = all(0 <= v <= 1 for v in predictions[cond].values())
        print(f"  {cond} ({method:<28}) → {len(predictions[cond])} preds  "
              f"in [0,1]={all_in_range}")

    # Hard assert: every swap target has an R0 prediction. If this
    # ever fails, the union above is wrong and KPS-F will undercount.
    miss = [
        (t.qa_prime, t.qb) for t in swap
        if (str(t.qa_prime), str(t.qb)) not in predictions["R0"]
        and (str(t.qb), str(t.qa_prime)) not in predictions["R0"]
    ]
    assert not miss, (
        f"{len(miss)} swap-target pairs missing from R0 predictions — "
        "Stage 6 pair union is incomplete."
    )
    print(f"  → all {len(swap)} swap targets covered by R0 preds")
    return predictions


# ══════════════════════════════════════════════════════════════════════════
# Stage 7 — Compute L6 indicators + A-B gap
# ══════════════════════════════════════════════════════════════════════════
def stage7_indicators(predictions, swap, bucket_lookup):
    banner("STAGE 7  Compute L6 indicators (LLM full 7-indicator panel)")
    from coldddi.diagnostics import compute_ab_gap, compute_indicators

    df = compute_indicators(
        predictions, swap, bucket_fn=bucket_lookup.bucket,
        coverage_warnings=True,
    )
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 100)
    print(df.to_string(index=False))

    print()
    print(f"  {'indicator':<22}{'ALL value':>12}{'n_pos':>10}{'A-B gap':>12}")
    print("  " + "-" * 56)
    for ind in (
        "KPS-F", "KPS-Name", "KPS-KG", "KPS-KG-Named",
        "KPS-KG-Masked", "KPS-Name-KGMasked", "KSAI",
    ):
        all_row = df.query(f"indicator == '{ind}' and bucket == 'ALL'")
        if len(all_row):
            v = all_row.iloc[0]["value"]
            n = all_row.iloc[0]["n"]
            gap = compute_ab_gap(df, ind)
            print(f"  {ind:<22}{v:>12.4f}{int(n):>10}{gap:>+12.4f}")
        else:
            print(f"  {ind:<22}{'NaN':>12}{'-':>10}{'NaN':>12}")
    return df


def main():
    print(f"\n{'#' * 78}")
    print(f"  v3 — full LLM pipeline smoke (single script chains L1→L6)")
    print(f"  Tiny random model: {TINY_MODEL}")
    print(f"  Toy fixture: {TOY_RELEASE}")
    print(f"{'#' * 78}")

    ds, sm, id2name, id2smi, ke_map, bucket_lookup = stage1_setup()
    train_feats, val_dict, cfg_p4 = stage2_build_ft_samples(
        ds, id2name, id2smi,
    )
    cfg_train, _info = stage3_fit(train_feats, val_dict, cfg_p4)
    manifest = stage4_select(cfg_train, ds, sm, cfg_p4, val_dict)
    swap = stage5_swap_candidates(ds)
    predictions = stage6_inference(
        manifest, swap, ds, sm, id2name, id2smi, ke_map,
    )
    _df = stage7_indicators(predictions, swap, bucket_lookup)

    print(f"\n{'#' * 78}")
    print(f"  v3 smoke COMPLETE — every cross-stage handoff passed.")
    print(f"{'#' * 78}\n")


if __name__ == "__main__":
    main()
