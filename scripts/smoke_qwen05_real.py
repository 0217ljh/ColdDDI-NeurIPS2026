"""Real-LLM smoke for Qwen2.5-0.5B on the 1900-drug release.

Goal: stress every L1-L6 component against a REAL (non-tiny-random)
LLM on the REAL (non-toy) dataset to surface bugs that the tiny-random
v3 smoke cannot catch:

  Stage 0  CLI + auto batch size from VRAM
  Stage 1  Dataset loading        — PairDataset.from_release_dir + sizes
  Stage 2  Yes/No token sanity    — Qwen2.5 tokenizer
  Stage 3  Prompt rendering check — dump 2 samples for visual inspection
  Stage 4  Loss-function sanity   — collator + compute_loss on 1 batch
  Stage 5  Mini FT + multi-eval   — small train subset, eval on val_s0/s1/s2
  Stage 6  L5 select_best         — per-split val AUC ranking
  Stage 7  Test inference         — best-S2 LoRA → test_s2 AUC

Defaults are tuned for a single 24-40 GB GPU; pass ``--micro-bs``
etc. to override. Skip a stage with ``--skip 4,5`` if iterating.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "private" / "intermediate"
DEFAULT_AB_PARQUET = REPO_ROOT / "data" / "private" / "outputs_full" / "annotations" / "ab.parquet"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ───────────────────────── helpers ─────────────────────────

def banner(s: str) -> None:
    print("\n" + "=" * 90)
    print(f"  {s}")
    print("=" * 90)


def auto_batch_size(
    model_name: str,
    *,
    max_length: int = 1024,
    lora_r: int = 16,
    cache_dir: str | None = None,
) -> tuple[int, int]:
    """Pick a safe ``(train_bs, eval_bs)`` from VRAM + the model's
    own AutoConfig.

    Empirical formula (bf16, RTX-class GPUs, LoRA-only trainable
    parameters; assumes ``trainer.py:compute_loss`` does NOT pass
    ``labels`` into ``model.forward`` — that's the post-fix path).
    Treats every term as ``B × per_sample_bytes`` so we can divide
    free VRAM by the per-sample budget.

    Per-sample memory budget at seq-length L, vocab V, hidden H,
    intermediate I (SwiGLU MLP width), layers N,
    bytes-per-param d (= 2 for bf16):

      mlp_acts       ≈ L · I · N · d · 6   ← SwiGLU keeps 3 (B,L,I) tensors
                                             per layer (gate, up, gate*up),
                                             x6 for fwd + bwd scratch +
                                             grad activations. Empirically
                                             calibrated: x3 OOMs on backward.
      attn_acts      ≈ L · H · N · d · 6   ← Q/K/V proj outputs;
                                             same x6 reasoning.
      logits         ≈ L · V · d           ← (B,L,V); fp32 CE skipped
                                             after the trainer fix.

      per_sample_train ≈ mlp_acts + attn_acts + logits
      per_sample_eval  ≈ (mlp_acts + attn_acts) / 6 + logits  ← no
                                                                bwd scratch

    Static budget (not B-scaled):

      model_weights  ≈ 2 · n_params · d / 1e9   ← x2 because gradients
                                                  live on the LoRA path
                                                  anyway; conservative.
      cuda headroom  = 4 GB                     ← kernel scratch,
                                                  allocator slack.

    Free VRAM ÷ per-sample-budget, multiplied by 0.7 safety factor so
    live usage stays well under the calculated ceiling.  Falls back to
    ``(4, 8)`` on CPU. The MLP-intermediate term dominates for SwiGLU
    models like Qwen2 / Llama where I is 4-6× H; ignoring it
    underestimates training memory by 5×.

    Parameters
    ----------
    lora_r
        Currently unused in the budget (LoRA optimizer state is
        sub-1% of everything else); kept in the signature so we can
        tighten the estimate later if full FT becomes a use case.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return (4, 8)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )
        V = int(getattr(cfg, "vocab_size", 50000))
        H = int(getattr(cfg, "hidden_size", 1024))
        # SwiGLU MLP width — falls back to 4·H if the config doesn't
        # expose it (older GPT2-style models).  This is the term that
        # actually dominates training memory for Qwen/Llama.
        I = int(getattr(cfg, "intermediate_size", 4 * H))
        N = int(getattr(cfg, "num_hidden_layers",
                        getattr(cfg, "num_layers", 24)))
        dtype_bytes = 2
        # Crude transformer scaling for n_params:
        # per layer ≈ 4·H² (attention QKVO) + 3·H·I (SwiGLU gate/up/down)
        # plus embedding + lm_head ≈ 2·V·H.
        n_params_M = (N * (4 * H * H + 3 * H * I) + 2 * V * H) / 1e6

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        model_gb = 2 * n_params_M * dtype_bytes / 1e3
        headroom_gb = 4.0
        free_gb = max(vram_gb - model_gb - headroom_gb, 1.0)

        # x6 multiplier on activations: 1x fwd stored + several backward
        # scratch tensors (the autograd graph for `down_proj(silu(gate)
        # * up)` materialises extras during bwd). x3 OOMs in practice.
        mlp_acts = max_length * I * N * dtype_bytes * 6
        attn_acts = max_length * H * N * dtype_bytes * 6
        logits = max_length * V * dtype_bytes
        per_train_gb = (mlp_acts + attn_acts + logits) / 1e9
        # Eval path has no bwd scratch — divide acts back down by 6:
        per_eval_gb = ((mlp_acts + attn_acts) / 6 + logits) / 1e9

        safety = 0.7
        train_bs = max(1, min(int(safety * free_gb / per_train_gb), 64))
        eval_bs = max(1, min(int(safety * free_gb / per_eval_gb), 128))
        return (train_bs, eval_bs)
    except Exception:
        return (4, 8)


def make_cli() -> argparse.Namespace:
    micro_default, eval_default = auto_batch_size(DEFAULT_MODEL)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (default: {DEFAULT_MODEL})")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="release-style dir w/ filtered/ + splits/seed{N}/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ab-parquet", type=Path, default=DEFAULT_AB_PARQUET,
                   help="A/B annotation parquet (built by release_parquet)")
    p.add_argument("--micro-bs", type=int, default=micro_default,
                   help=f"per-device train batch size "
                        f"(auto = {micro_default} from VRAM)")
    p.add_argument("--eval-bs", type=int, default=eval_default,
                   help=f"per-device eval batch size "
                        f"(auto = {eval_default} from VRAM)")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--train-subset", type=int, default=800,
                   help="how many train pairs (balanced pos+neg) to FT on")
    p.add_argument("--val-subset", type=int, default=200,
                   help="per-split val pair cap (pos+neg combined, "
                        "drawn equally)")
    p.add_argument("--test-subset", type=int, default=2000,
                   help="test_s2 pair cap for final AUC; pass 0 for full")
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--skip", default="",
                   help="comma-separated stage ids to skip, e.g. '5,6'")
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--legacy-pkl", type=Path, default=None,
                   help="Path to a legacy 800-drug / 1900-drug `*.pkl` "
                        "fold bundle. When set, overrides --data-root and "
                        "the dataset is loaded via PairDataset.from_pkl().")
    return p.parse_args()


# ───────────────────── Stage 1: dataset ─────────────────────

def stage1_dataset(args):
    banner("STAGE 1  Dataset loading")
    from coldddi.data.dataset import PairDataset
    from coldddi.diagnostics import build_bucket_lookup
    from coldddi.llm.retrieval import build_subgraph_map

    t = time.time()
    if args.legacy_pkl is not None:
        ds = PairDataset.from_pkl(args.legacy_pkl)
        src = f"legacy pkl: {args.legacy_pkl}"
    else:
        ds = PairDataset.from_release_dir(args.data_root, seed=args.seed)
        src = f"release dir: {args.data_root} (seed={args.seed})"
    print(f"  source               : {src}")
    print(f"  load time            : {time.time() - t:.1f}s")
    print(f"  drugs (lookup table) : {len(ds.drugs) if ds.drugs is not None else None}")
    print(f"  train positives      : {len(ds.splits.train):>8}")
    for split in ("val_s0", "val_s1", "val_s2", "test_s0", "test_s1", "test_s2"):
        pos = len(getattr(ds.splits, split))
        neg = len(ds.get_negatives(split))
        print(f"  {split:<20} : pos={pos:>7}  neg={neg:>7}")

    print(f"\n  G1 (seen) drugs      : {len(ds.splits.g1_drugs)}")
    print(f"  G2 (cold) drugs      : {len(ds.splits.g2_drugs)}")

    t = time.time()
    sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
    print(f"\n  subgraph_map         : {len(sm.data)} drugs, "
          f"{time.time() - t:.1f}s")

    t = time.time()
    bucket_lookup = build_bucket_lookup(args.ab_parquet)
    print(f"  bucket_lookup        : {len(bucket_lookup.pair_to_bucket)} "
          f"PK/PD-labelled pairs, {time.time() - t:.1f}s")

    id2name = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["name"].astype(str),
    ))
    id2smi = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["smiles"].astype(str),
    ))

    # Key-entity map for entity-mask prompts (R2/R3) — load FULL ab
    # parquet for the real run (was ab_sample.parquet in toy smoke).
    ab_df = pd.read_parquet(args.ab_parquet)
    ke_map = {
        (str(r.drug_a_id), str(r.drug_b_id)): {
            "key_entity_name": (
                str(r.key_entity_name) if not pd.isna(r.key_entity_name)
                else ""
            ),
            "key_entity_type": (
                str(r.key_entity_type) if not pd.isna(r.key_entity_type)
                else ""
            ),
            "has_key_entity": bool(r.has_key_entity),
        }
        for r in ab_df.itertuples(index=False)
    }
    print(f"  ke_map               : "
          f"{sum(1 for v in ke_map.values() if v['has_key_entity'])} "
          f"pairs with confirmed key entity")
    return ds, sm, id2name, id2smi, ke_map, bucket_lookup


# ─────────────────── Stage 2: tokens ───────────────────

def stage2_tokens(args):
    banner("STAGE 2  Yes/No token sanity (Qwen tokenizer)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left",
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    yes_str, no_str = " Yes", " No"
    yes_ids = tok.encode(yes_str, add_special_tokens=False)
    no_ids = tok.encode(no_str, add_special_tokens=False)
    print(f"  tokenizer class      : {type(tok).__name__}")
    print(f"  vocab size           : {tok.vocab_size}")
    print(f"  '{yes_str}' encodes to : {yes_ids}")
    print(f"  '{no_str}'  encodes to : {no_ids}")
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise AssertionError(
            f"YES/NO must be single tokens; got yes={yes_ids} no={no_ids}"
        )
    print(f"  → both are single tokens; ids stable for FT + inference")
    return tok, int(yes_ids[0]), int(no_ids[0])


# ─────────────────── Stage 3: prompt rendering ───────────────────

def stage3_prompts(args, ds, sm, id2name, id2smi, tok):
    banner("STAGE 3  Prompt rendering check (P4 OHS, one pos + one neg)")
    from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
    from coldddi.llm.retrieval import to_llm_samples

    cfg = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=args.model,
    )
    pos_row = ds.splits.train.head(1)[["drug_a_id", "drug_b_id"]]
    neg_row = ds.get_negatives("val_s2").head(1)[["drug_a_id", "drug_b_id"]]
    pairs = pd.concat([pos_row, neg_row], ignore_index=True)
    labels = [1, 0]
    samples = to_llm_samples(pairs, labels, ds=ds, subgraph_map=sm)
    for s, label in zip(samples, labels):
        # FT-time prompt (carries answer token at end)
        text = build_binary_prompt(
            s, cfg,
            drug_id2name=id2name, drug_id2smiles=id2smi,
        )
        # Inference-time prompt (no answer, ready for next-token scoring)
        text_inf = build_binary_prompt(
            s, cfg,
            drug_id2name=id2name, drug_id2smiles=id2smi,
            assistant_content="",
        )
        print(f"\n  ── sample label={label}  ({s['drug_a_id']} - {s['drug_b_id']}) ──")
        print(f"  FT prompt ({len(tok(text).input_ids)} tokens):")
        print("    " + text.replace("\n", "\n    "))
        print(f"\n  Inference prompt ({len(tok(text_inf).input_ids)} tokens, "
              "ends right before answer token):")
        print("    " + text_inf[-200:].replace("\n", "\n    "))
    print(f"\n  → 1 positive + 1 negative sample rendered cleanly")


# ─────────────────── Stage 4: loss-function sanity ───────────────────

def stage4_loss_sanity(args, ds, sm, id2name, id2smi, tok, yes_id, no_id):
    banner("STAGE 4  Loss-function sanity (collator + compute_loss on 1 batch)")
    import torch
    from coldddi.llm.collator import BinaryFTCollator
    from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
    from coldddi.llm.prompts.binary_cls import infer_model_family
    from coldddi.llm.retrieval import to_llm_samples
    from transformers import AutoModelForCausalLM

    cfg = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=args.model,
    )
    pos_df = ds.splits.train.head(2)[["drug_a_id", "drug_b_id"]]
    neg_df = ds.get_train_negatives(0).head(2)[["drug_a_id", "drug_b_id"]]
    pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    labels = [1, 1, 0, 0]
    raw_samples = to_llm_samples(pairs, labels, ds=ds, subgraph_map=sm)
    feats = []
    for s in raw_samples:
        feats.append({
            "text": build_binary_prompt(
                s, cfg,
                drug_id2name=id2name, drug_id2smiles=id2smi,
            ),
            "cls_labels": s["label"],
        })
    family = infer_model_family(args.model)
    collator = BinaryFTCollator(
        tokenizer=tok, model_family=family, max_length=args.max_length,
    )
    batch = collator(feats)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"  batch keys           : {sorted(batch.keys())}")
    print(f"  batch.input_ids      : {batch['input_ids'].shape}  dtype={batch['input_ids'].dtype}")
    print(f"  batch.labels (masked): {batch['labels'].shape}  "
          f"non-masked tokens={int((batch['labels'] != -100).sum())}")
    print(f"  batch.cls_labels     : {batch['cls_labels'].tolist()}")

    # Load model in bf16 (full grad path).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.train()  # gradients needed for compute_loss

    from coldddi.llm.trainer import _BinaryClsTrainer
    # Build a no-op trainer just to access compute_loss-style logic.
    # We call the inner subclass directly (it's the only thing we need).
    from coldddi.llm.trainer import _build_binary_trainer_cls
    Cls = _build_binary_trainer_cls()
    # Construct without going through HF Trainer.__init__ (which wants
    # TrainingArguments etc.). We only need compute_loss as a method —
    # bind self manually.
    class _Stub:
        yes_token_id = yes_id
        no_token_id = no_id

    loss = Cls.compute_loss(_Stub(), model, batch, return_outputs=False)
    print(f"\n  compute_loss output  : {loss.item():.4f}  "
          f"(grad_fn={loss.grad_fn.__class__.__name__ if loss.grad_fn else None})")
    if not torch.isfinite(loss):
        raise AssertionError(f"compute_loss returned non-finite: {loss}")

    # Backward sanity — verify gradients actually flow.
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  backward grad pieces : {len(grad_norms)}, "
          f"max_norm={max(grad_norms):.4f}")
    if max(grad_norms) == 0:
        raise AssertionError("All gradients are zero — loss did not flow.")
    print(f"  → loss is finite and gradients propagate")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ─────────────────── Stage 5: mini FT + multi-eval ───────────────────

def _build_ft_pairs(ds, args, sm, id2name, id2smi, cfg):
    """Balanced train subset + per-split val subset."""
    from coldddi.llm.prompts import build_binary_prompt
    from coldddi.llm.retrieval import to_llm_samples

    n_each = args.train_subset // 2
    pos_df = ds.splits.train.head(n_each)[["drug_a_id", "drug_b_id"]]
    neg_df = ds.get_train_negatives(0).head(n_each)[["drug_a_id", "drug_b_id"]]
    pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    labels = [1] * len(pos_df) + [0] * len(neg_df)
    raw = to_llm_samples(pairs, labels, ds=ds, subgraph_map=sm)
    train_feats = [
        {
            "text": build_binary_prompt(
                s, cfg, drug_id2name=id2name, drug_id2smiles=id2smi,
            ),
            "cls_labels": s["label"],
        }
        for s in raw
    ]

    val_dict: dict[str, list[dict]] = {}
    n_v_each = args.val_subset // 2
    for short, name in (("S0", "val_s0"), ("S1", "val_s1"), ("S2", "val_s2")):
        pos = dict(ds.splits.items())[name].head(n_v_each)[["drug_a_id", "drug_b_id"]]
        neg = ds.get_negatives(name).head(n_v_each)[["drug_a_id", "drug_b_id"]]
        pairs_v = pd.concat([pos, neg], ignore_index=True)
        labels_v = [1] * len(pos) + [0] * len(neg)
        raw_v = to_llm_samples(pairs_v, labels_v, ds=ds, subgraph_map=sm)
        val_dict[short] = [
            {
                "text": build_binary_prompt(
                    s, cfg, drug_id2name=id2name, drug_id2smiles=id2smi,
                ),
                "cls_labels": s["label"],
            }
            for s in raw_v
        ]
    return train_feats, val_dict


def stage5_ft(args, ds, sm, id2name, id2smi):
    banner("STAGE 5  Mini FT + multi-eval (S0/S1/S2)")
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.trainer import LLMTrainerConfig, LoRATrainer

    cfg_p4 = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=args.model,
    )
    train_feats, val_dict = _build_ft_pairs(ds, args, sm, id2name, id2smi, cfg_p4)
    print(f"  train samples        : {len(train_feats)}")
    for k, v in val_dict.items():
        print(f"  val[{k}]              : {len(v)}")

    out = Path(tempfile.mkdtemp(prefix="qwen05_ft_"))
    cfg = LLMTrainerConfig(
        model_name=args.model,
        output_dir=str(out / "ckpts"),
        dtype="bfloat16", device="auto",
        num_epochs=args.num_epochs,
        micro_batch_size=args.micro_bs,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=5,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=4,
        eval_strategy="steps",
        primary_val_split="S2",
        disable_tqdm=False, report_to=(), seed=args.seed,
    )
    cfg.lora.r = args.lora_r
    cfg.lora.alpha = args.lora_alpha
    cfg.lora.target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    print(f"\n  micro_bs / eval_bs   : {cfg.micro_batch_size} / {args.eval_bs}")
    print(f"  LoRA r / alpha       : {cfg.lora.r} / {cfg.lora.alpha}")
    print(f"  primary_val_split    : {cfg.primary_val_split}")
    print(f"  output_dir           : {cfg.output_dir}")

    t0 = time.time()
    trainer = LoRATrainer(cfg)
    info = trainer.fit(
        train_samples=train_feats, val_samples=val_dict,
        prompt_cfg=cfg_p4,
    )
    print(f"\n  total fit time       : {time.time() - t0:.1f}s")
    print(f"  best_ckpt (HF)       : {info['best_ckpt']}")
    eval_keys = sorted({
        k for row in info["log_history"]
        for k in row if k.startswith("eval_S") and k.endswith("_loss")
    })
    print(f"  eval keys logged     : {eval_keys}")
    assert eval_keys == ["eval_S0_loss", "eval_S1_loss", "eval_S2_loss"], (
        f"expected eval_S0/S1/S2_loss; got {eval_keys}"
    )
    return cfg, info, cfg_p4, val_dict


# ─────────────────── Stage 6: L5 select_best ───────────────────

def stage6_select(args, cfg, ds, sm, cfg_p4, val_dict):
    banner("STAGE 6  L5 select_best (per-split val AUC)")
    from coldddi.llm.select_best import (
        assert_prompt_cfg_matches_fit_info,
        parse_candidate_ckpts,
        read_fit_info,
        score_candidate_ckpts,
        select_best,
    )
    fit_info = read_fit_info(cfg.output_dir)
    print(f"  fit_info.yes_token   : {fit_info['yes_token']!r}")
    print(f"  fit_info.no_token    : {fit_info['no_token']!r}")
    print(f"  fit_info.prompt      : {fit_info['prompt_cfg']}")
    assert_prompt_cfg_matches_fit_info(fit_info, cfg_p4)

    splits = tuple(val_dict.keys())
    cands = parse_candidate_ckpts(cfg.output_dir, splits=splits, topk=1)
    for s, lst in cands.items():
        steps = sorted({c.step for c in lst})
        print(f"  {s}: {len(lst)} candidates, steps={steps}")

    t = time.time()
    scored = score_candidate_ckpts(
        cands,
        base_model_name=fit_info["model_name"],
        dataset=ds,
        prompt_cfg=cfg_p4,
        subgraph_map=sm,
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
        device="cuda", dtype="bfloat16",
        batch_size=args.eval_bs,
        max_length=fit_info["max_length"],
    )
    print(f"  scoring time         : {time.time() - t:.1f}s")
    manifest = select_best(scored)
    print()
    print(f"  {'split':<6}{'best_step':>12}{'val_AUC':>12}  best_ckpt")
    for s, m in manifest.items():
        print(f"  {s:<6}{m['best_step']:>12}{m['best_val_auc']:>12.3f}  "
              f"{m['best_ckpt']}")
    return manifest, fit_info


# ─────────────────── Stage 7: test inference ───────────────────

def stage7_test(args, manifest, fit_info, ds, sm, id2name, id2smi):
    banner("STAGE 7  Test inference on test_s2")
    import torch
    from sklearn.metrics import roc_auc_score
    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.retrieval import to_llm_samples

    best_s2 = manifest["S2"]["best_ckpt"]
    print(f"  adapter              : {best_s2}")

    pos_df = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg_df = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    if args.test_subset > 0:
        n_each = args.test_subset // 2
        pos_df = pos_df.head(n_each)
        neg_df = neg_df.head(n_each)
    pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    y_true = np.concatenate([
        np.ones(len(pos_df), dtype=np.int64),
        np.zeros(len(neg_df), dtype=np.int64),
    ])
    print(f"  test pairs           : pos={len(pos_df)}  neg={len(neg_df)}  "
          f"total={len(pairs)}")

    samples = to_llm_samples(pairs, labels=None, ds=ds, subgraph_map=sm)
    cfg_p4 = PromptBuildConfig(
        task_name="Binary_cls",
        method="One_Hop_Subgraph_Sequence",
        model_name=fit_info["model_name"],
    )
    runner = LLMInferenceRunner(LLMRunnerConfig(
        model_name=fit_info["model_name"],
        adapter_path=best_s2,
        dtype="bfloat16", device="cuda",
        batch_size=args.eval_bs,
        max_length=fit_info["max_length"],
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
    ))
    runner.load()
    t = time.time()
    df = runner.score_samples(
        samples, cfg_p4,
        drug_id2name=id2name, drug_id2smiles=id2smi,
        progress=True,
    )
    print(f"  inference time       : {time.time() - t:.1f}s")
    auc = float(roc_auc_score(y_true, df["p_yes"].to_numpy(dtype=np.float64)))
    print(f"  test_s2 AUROC        : {auc:.4f}  "
          f"(p_yes mean={df['p_yes'].mean():.3f})")


# ─────────────────── main ───────────────────

def main() -> int:
    args = make_cli()
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    print(f"\n{'#' * 90}")
    print(f"  Real-LLM smoke — Qwen2.5-0.5B on 1900-drug release")
    print(f"  model       : {args.model}")
    print(f"  data_root   : {args.data_root}")
    print(f"  seed        : {args.seed}")
    print(f"  batch sizes : micro_bs={args.micro_bs}  eval_bs={args.eval_bs}")
    print(f"  subsets     : train={args.train_subset}  val={args.val_subset} "
          f"per split  test={args.test_subset}")
    print(f"  skipping    : {sorted(skip) if skip else '(none)'}")
    print(f"{'#' * 90}")

    # Stage 1
    if 1 in skip:
        raise SystemExit("Cannot skip Stage 1 — every later stage depends on it.")
    ds, sm, id2name, id2smi, ke_map, bucket_lookup = stage1_dataset(args)

    # Stage 2
    if 2 not in skip:
        tok, yes_id, no_id = stage2_tokens(args)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, padding_side="left",
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
        no_id = tok.encode(" No", add_special_tokens=False)[0]

    # Stage 3
    if 3 not in skip:
        stage3_prompts(args, ds, sm, id2name, id2smi, tok)

    # Stage 4
    if 4 not in skip:
        stage4_loss_sanity(args, ds, sm, id2name, id2smi, tok, yes_id, no_id)

    # Stage 5
    cfg, info, cfg_p4, val_dict = None, None, None, None
    if 5 not in skip:
        cfg, info, cfg_p4, val_dict = stage5_ft(args, ds, sm, id2name, id2smi)

    # Stage 6
    manifest, fit_info = None, None
    if 6 not in skip and cfg is not None:
        manifest, fit_info = stage6_select(args, cfg, ds, sm, cfg_p4, val_dict)

    # Stage 7
    if 7 not in skip and manifest is not None:
        stage7_test(args, manifest, fit_info, ds, sm, id2name, id2smi)

    print(f"\n{'#' * 90}")
    print(f"  Qwen2.5-0.5B real smoke COMPLETE")
    print(f"{'#' * 90}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
