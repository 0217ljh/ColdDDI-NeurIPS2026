"""Paper-grade LLM runner — single entry point for one model × one prompt
× one dataset × one seed.

Chains L1 (prompt rendering) → L3 (LoRA FT, multi-split eval) → L5
(per-split best-ckpt by val AUC) → L2 (test_s2 inference) → L6 (KPS /
KSAI indicators, optional).

Designed to be the canonical "one-click" runner: every paper cell can
be reproduced by varying CLI flags only.  Each invocation is a single
(model, dataset, prompt, seed) cell — you intentionally CANNOT pass
multiple models at once, because we want each run to be a self-
contained, resumable, attributable artefact.  Sweep across cells by
calling the script multiple times (e.g. in a shell loop).

Usage examples
--------------

  # Smallest valid paper-grade cell: Qwen 0.5B on 800-drug seed 42, P4.
  python scripts/run_llm.py \\
      --model qwen-0.5b --dataset 800-drug --prompt P4 --seed 42

  # Llama-3.2-1B + P1 (zero-shot, no FT) on 1900-drug seed 43.
  python scripts/run_llm.py \\
      --model llama-1b --dataset 1900-drug --prompt P1 --seed 43 \\
      --skip-ft

  # Smoke-sized override (tiny train + capped test) for fast debug.
  python scripts/run_llm.py --model qwen-0.5b --dataset 800-drug \\
      --train-subset 400 --test-subset 1000

Output
------
Every run writes a fresh ``runs/<run_id>/`` directory containing:
  * ``ckpts/checkpoint-*/`` — LoRA adapter snapshots (HF Trainer)
  * ``fit_info.json``        — training contract for L5/L2 reuse
  * ``manifest.json``        — per-split best-ckpt + val AUC
  * ``test_predictions.parquet`` — test_s2 per-row p_yes / pred / prompt
  * ``run.log``              — combined stdout/stderr (you may also tee)

If the run dies mid-way and you re-invoke with the same ``--run-id``
(or the same default ``--output-dir`` and ``--model / --dataset /
--seed`` combination), L3 auto-resumes from the latest checkpoint
and L2's test inference resumes from any partially-saved predictions
parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Model + dataset shorthand resolvers ────────────────────────────

MODEL_SHORTCUTS: dict[str, str] = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-3b":   "Qwen/Qwen2.5-3B",
    "qwen-7b":   "Qwen/Qwen2.5-7B",
    "qwen-14b":  "Qwen/Qwen2.5-14B",
    "llama-1b":  "meta-llama/Llama-3.2-1B",
    "llama-3b":  "meta-llama/Llama-3.2-3B",
}

PROMPT_METHODS: dict[str, str] = {
    "P1": "Zero_Shot_Sequence",
    "P2": "Few_Shot_Similarity_SMILES",
    "P3": "One_Hop_Subgraph_Single",
    "P4": "One_Hop_Subgraph_Sequence",
    "P5": "Few_Shot_2hop",
    "P6": "One_Hop_Subgraph_Sequence_Desc",
    "P7": "Desc_Only",
}


def _resolve_model(name: str) -> str:
    """Accept either a shorthand (``qwen-0.5b``) or a full HF id."""
    if name in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[name]
    return name


def _resolve_dataset(name: str, seed: int) -> tuple[Path | None, Path | None]:
    """Return ``(data_root, legacy_pkl)``; exactly one is non-None.

    Shorthand mapping
    -----------------
    * ``toy``       → ``data/public/intermediate/``
    * ``800-drug``  → ``data/private/outputs_full/splits_legacy/800drug/<seed>.pkl``
    * ``1900-drug`` → ``data/private/intermediate/`` (new-format release)
    * any path     → interpreted as a literal path
    """
    if name == "toy":
        return REPO_ROOT / "data" / "public" / "intermediate", None
    if name == "1900-drug":
        return REPO_ROOT / "data" / "private" / "intermediate", None
    if name == "800-drug":
        release = REPO_ROOT / "data/private/subsets/800" / f"seed{seed}" / "intermediate"
        if release.is_dir():
            return release, None
        return None, (
            REPO_ROOT / "data" / "private" / "outputs_full"
            / "splits_legacy" / "800drug"
            / f"latest_drugbank_ddi-Binary_cls-{seed}"
              "+cold_start_split_fair_step-and-fair_negatives_step.pkl"
        )
    p = Path(name)
    if p.is_dir():
        return p, None
    if p.is_file() and p.suffix == ".pkl":
        return None, p
    raise ValueError(
        f"Cannot resolve --dataset {name!r}; try one of "
        f"{{toy, 800-drug, 1900-drug}} or a directory / .pkl path."
    )


# ─── Auto batch size (shared with smoke_qwen05_real.py heuristic) ──

def auto_batch_size(
    model_name: str, max_length: int = 1024, cache_dir: str | None = None,
) -> tuple[int, int]:
    """Empirical (train_bs, eval_bs) from VRAM + AutoConfig.

    Same formula as ``scripts/smoke_qwen05_real.py``:
    activations dominated by SwiGLU MLP (L · I · N · d · 6) + logits
    (L · V · d); safety factor 0.7.  Falls back to (4, 8) on CPU.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return (4, 8)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )
        V = int(getattr(cfg, "vocab_size", 50000))
        H = int(getattr(cfg, "hidden_size", 1024))
        I = int(getattr(cfg, "intermediate_size", 4 * H))
        N = int(getattr(cfg, "num_hidden_layers",
                        getattr(cfg, "num_layers", 24)))
        d = 2
        n_params_M = (N * (4 * H * H + 3 * H * I) + 2 * V * H) / 1e6
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        model_gb = 2 * n_params_M * d / 1e3
        free_gb = max(vram_gb - model_gb - 4.0, 1.0)
        mlp_acts = max_length * I * N * d * 6
        attn_acts = max_length * H * N * d * 6
        logits = max_length * V * d
        per_train = (mlp_acts + attn_acts + logits) / 1e9
        per_eval = ((mlp_acts + attn_acts) / 6 + logits) / 1e9
        train_bs = max(1, min(int(0.7 * free_gb / per_train), 64))
        eval_bs = max(1, min(int(0.7 * free_gb / per_eval), 128))
        return (train_bs, eval_bs)
    except Exception:
        return (4, 8)


# ─── CLI ────────────────────────────────────────────────────────────

def make_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help=f"Model shorthand or full HF id. Shortcuts: "
                        f"{', '.join(sorted(MODEL_SHORTCUTS))}")
    p.add_argument("--dataset", required=True,
                   help="Dataset shorthand: 'toy', '800-drug', "
                        "'1900-drug', or a directory / .pkl path.")
    p.add_argument("--prompt", default="P4",
                   choices=tuple(PROMPT_METHODS),
                   help="Prompt template family (P1-P7). Default P4 (OHS).")
    p.add_argument("--desc-json", type=Path, default=None,
                   help="P6/P7 descriptions: JSON object mapping drug IDs to text.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override the auto-derived runs/<run_id>/ path.")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-derived run id "
                        "(default: <model_slug>__<dataset>__<prompt>__seed<N>)")

    # Compute
    micro_default, eval_default = (8, 16)
    p.add_argument("--train-bs", type=int, default=None,
                   help="Per-device train batch size; default = auto from VRAM.")
    p.add_argument("--eval-bs", type=int, default=None,
                   help="Per-device eval batch size; default = auto from VRAM.")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=float, default=1.0)

    # Sample caps (full = pass huge number; subset = pass small)
    p.add_argument("--train-subset", type=int, default=999_999_999,
                   help="Cap on (pos + neg) train pairs. Pass small "
                        "value for fast smoke; default = full split.")
    p.add_argument("--val-subset", type=int, default=1000,
                   help="Cap on (pos + neg) per-split val pairs used "
                        "during HF Trainer's eval-during-FT. Stage 6 "
                        "(L5 best-ckpt selection) always uses the FULL "
                        "val splits regardless of this cap.")
    p.add_argument("--test-subset", type=int, default=0,
                   help="Cap on test_s2 pair count. 0 = use full.")

    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)

    # Flow control
    p.add_argument("--skip-ft", action="store_true",
                   help="Skip L3 FT — only run zero-shot test inference. "
                        "Useful for P1 baseline / no-train ablations.")
    p.add_argument("--skip-test", action="store_true",
                   help="Skip Stage 7 test inference (FT + select only).")
    return p.parse_args()


# ─── Run id + output dir ────────────────────────────────────────────

def _slug(s: str) -> str:
    return s.replace("/", "__").replace(".", "-").lower()


def _run_id(model_name: str, dataset_name: str, prompt: str, seed: int) -> str:
    return f"{_slug(model_name)}__{dataset_name}__{prompt}__seed{seed}"


# ─── Stage helpers (modular; one entry point each) ──────────────────

def _load_description_map(args: argparse.Namespace) -> dict[str, str]:
    """Load descriptions once for training, checkpoint scoring and inference."""
    if args.prompt not in ("P6", "P7"):
        if args.desc_json is not None:
            raise ValueError("--desc-json is only used with --prompt P6 or P7.")
        return {}
    if args.desc_json is None:
        raise ValueError(f"--prompt {args.prompt} requires --desc-json <path>.")
    descriptions = json.loads(args.desc_json.read_text(encoding="utf-8"))
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("--desc-json must contain a nonempty object: {drug_id: text}.")
    for drug_id, description in descriptions.items():
        if not drug_id.strip() or not isinstance(description, str) or not description.strip():
            raise ValueError(f"Invalid description for {drug_id!r}; expected nonempty text.")
    return descriptions


def _check_description_run(args: argparse.Namespace, output_dir: Path) -> None:
    """Prevent reusing adapters or prediction caches with different descriptions."""
    if args.prompt not in ("P6", "P7"):
        return
    digest = hashlib.sha256(json.dumps(
        args.description_map, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    contract = {"prompt": args.prompt, "descriptions_sha256": digest,
                "description_count": len(args.description_map)}
    path = output_dir / "description_input.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Description input changed; choose a new --output-dir.")
    elif any(output_dir.iterdir()):
        raise ValueError("Existing P6/P7 output has no description contract; choose a new --output-dir.")
    else:
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


def load_dataset(args):
    from coldddi.data.dataset import PairDataset
    from coldddi.llm.retrieval import build_subgraph_map

    data_root, legacy_pkl = _resolve_dataset(args.dataset, args.seed)
    print(f"[load_dataset] dataset={args.dataset!r}  seed={args.seed}", flush=True)
    t = time.time()
    if legacy_pkl is not None:
        ds = PairDataset.from_pkl(legacy_pkl)
        src_repr = f"legacy pkl: {legacy_pkl.name}"
    else:
        ds = PairDataset.from_release_dir(data_root, seed=args.seed)
        src_repr = f"release dir: {data_root}"
    print(f"  source            : {src_repr}", flush=True)
    print(f"  load time         : {time.time() - t:.1f}s", flush=True)
    print(f"  train positives   : {len(ds.splits.train)}", flush=True)
    print(f"  test_s2 positives : {len(ds.splits.test_s2)}", flush=True)

    sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
    id2name = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["name"].astype(str),
    )) if ds.drugs is not None else {}
    id2smi = dict(zip(
        ds.drugs["drugbank_id"].astype(str),
        ds.drugs["smiles"].astype(str),
    )) if ds.drugs is not None else {}
    return ds, sm, id2name, id2smi


def build_ft_samples(args, ds, sm, id2name, id2smi, prompt_method: str):
    from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
    from coldddi.llm.retrieval import to_llm_samples

    model_full = _resolve_model(args.model)
    cfg = PromptBuildConfig(
        task_name="Binary_cls", method=prompt_method, model_name=model_full,
        extra={"drug_id2description": args.description_map} if args.prompt in ("P6", "P7") else {},
    )

    def _samples(pos_df, neg_df):
        pairs = pd.concat([pos_df, neg_df], ignore_index=True)
        labels = [1] * len(pos_df) + [0] * len(neg_df)
        raw = to_llm_samples(pairs, labels, ds=ds, subgraph_map=sm)
        return [
            {"text": build_binary_prompt(
                s, cfg,
                drug_id2name=id2name, drug_id2smiles=id2smi,
             ),
             "cls_labels": s["label"]}
            for s in raw
        ]

    n_each = args.train_subset // 2
    pos_train = ds.splits.train.head(n_each)[["drug_a_id", "drug_b_id"]]
    neg_train = ds.get_train_negatives(0).head(n_each)[["drug_a_id", "drug_b_id"]]
    train_feats = _samples(pos_train, neg_train)

    val_dict: dict[str, list[dict]] = {}
    nv = args.val_subset // 2
    for short, name in (("S0", "val_s0"), ("S1", "val_s1"), ("S2", "val_s2")):
        pos = dict(ds.splits.items())[name].head(nv)[["drug_a_id", "drug_b_id"]]
        neg = ds.get_negatives(name).head(nv)[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        val_dict[short] = _samples(pos, neg)
    return train_feats, val_dict, cfg


def run_ft(args, train_feats, val_dict, cfg_prompt, output_dir: Path):
    from coldddi.llm.trainer import LLMTrainerConfig, LoRATrainer

    micro, evalbs = auto_batch_size(
        _resolve_model(args.model), max_length=args.max_length,
    )
    train_bs = args.train_bs if args.train_bs is not None else micro
    eval_bs = args.eval_bs if args.eval_bs is not None else evalbs

    cfg = LLMTrainerConfig(
        model_name=_resolve_model(args.model),
        output_dir=str(output_dir / "ckpts"),
        dtype="bfloat16", device="auto",
        num_epochs=args.num_epochs,
        micro_batch_size=train_bs,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=20,
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

    print(f"[run_ft] train_bs={train_bs}  eval_bs={eval_bs}  "
          f"LoRA r={cfg.lora.r}/α={cfg.lora.alpha}", flush=True)
    print(f"  output_dir        : {cfg.output_dir}", flush=True)

    trainer = LoRATrainer(cfg)
    info = trainer.fit(
        train_samples=train_feats, val_samples=val_dict, prompt_cfg=cfg_prompt,
    )
    return cfg, info, eval_bs


def run_select_best(args, cfg, ds, sm, cfg_prompt, val_dict, eval_bs):
    from coldddi.llm.select_best import (
        assert_prompt_cfg_matches_fit_info,
        parse_candidate_ckpts,
        read_fit_info,
        score_candidate_ckpts,
        select_best,
    )
    fit_info = read_fit_info(cfg.output_dir)
    assert_prompt_cfg_matches_fit_info(fit_info, cfg_prompt)

    splits = tuple(val_dict.keys())
    cands = parse_candidate_ckpts(cfg.output_dir, splits=splits, topk=1)
    for s, lst in cands.items():
        print(f"[run_select_best] {s}: {len(lst)} candidates, "
              f"steps={sorted({c.step for c in lst})}", flush=True)

    t = time.time()
    scored = score_candidate_ckpts(
        cands,
        base_model_name=fit_info["model_name"],
        dataset=ds,
        prompt_cfg=cfg_prompt,
        subgraph_map=sm,
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
        device="cuda", dtype="bfloat16",
        batch_size=eval_bs,
        max_length=fit_info["max_length"],
    )
    manifest = select_best(scored)
    print(f"[run_select_best] scoring time: {time.time() - t:.1f}s", flush=True)
    return manifest, fit_info


def run_test(args, manifest, fit_info, ds, sm, id2name, id2smi,
             prompt_method: str, eval_bs: int, output_dir: Path):
    from sklearn.metrics import roc_auc_score
    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
    from coldddi.llm.prompts import PromptBuildConfig
    from coldddi.llm.retrieval import to_llm_samples

    best_s2 = manifest["S2"]["best_ckpt"]
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
    samples = to_llm_samples(pairs, labels=None, ds=ds, subgraph_map=sm)

    cfg_p = PromptBuildConfig(
        task_name="Binary_cls", method=prompt_method,
        model_name=fit_info["model_name"],
        extra={"drug_id2description": args.description_map} if args.prompt in ("P6", "P7") else {},
    )
    runner = LLMInferenceRunner(LLMRunnerConfig(
        model_name=fit_info["model_name"],
        adapter_path=best_s2,
        dtype="bfloat16", device="cuda",
        batch_size=eval_bs,
        max_length=fit_info["max_length"],
        yes_token=fit_info["yes_token"],
        no_token=fit_info["no_token"],
    ))
    runner.load()

    resume_pq = output_dir / "test_predictions.parquet"
    print(f"[run_test] pairs={len(pairs)}  resume_path={resume_pq}", flush=True)
    t = time.time()
    df = runner.score_samples(
        samples, cfg_p,
        drug_id2name=id2name, drug_id2smiles=id2smi,
        progress=True,
        resume_path=resume_pq,
        save_every=50,
    )
    print(f"[run_test] inference time: {time.time() - t:.1f}s", flush=True)

    if len(df) == len(y_true):
        auc = float(roc_auc_score(y_true, df["p_yes"].to_numpy()))
        print(f"[run_test] test_s2 AUROC: {auc:.4f}", flush=True)
    else:
        # If resume read back a stale parquet with reordered rows, the
        # AUC calc above wouldn't be valid. Print row counts so the
        # caller can sanity-check.
        print(f"[run_test] WARNING: df rows ({len(df)}) != y_true "
              f"len ({len(y_true)}); test_predictions.parquet may have "
              "drifted from the pair list.", flush=True)
        auc = float("nan")
    return df, auc


# ─── Main ──────────────────────────────────────────────────────────

def main() -> int:
    args = make_cli()
    args.description_map = _load_description_map(args)
    prompt_method = PROMPT_METHODS[args.prompt]
    model_full = _resolve_model(args.model)

    run_id = args.run_id or _run_id(
        model_full, args.dataset, args.prompt, args.seed,
    )
    output_dir = args.output_dir or (REPO_ROOT / "runs" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print(f"  run_id       : {run_id}")
    print(f"  model        : {model_full}")
    print(f"  dataset      : {args.dataset}  (seed={args.seed})")
    print(f"  prompt       : {args.prompt} ({prompt_method})")
    print(f"  output_dir   : {output_dir}")
    print("#" * 78 + "\n", flush=True)

    ds, sm, id2name, id2smi = load_dataset(args)

    if args.prompt in ("P6", "P7"):
        required_ids: set[str] = set()
        for split_name, positives in ds.splits.items():
            negatives = (ds.get_train_negatives(0) if split_name == "train"
                         else ds.get_negatives(split_name))
            for frame in (positives, negatives):
                for column in ("drug_a_id", "drug_b_id"):
                    required_ids.update(frame[column].astype(str))
        missing = sorted(required_ids - args.description_map.keys())
        if missing:
            raise ValueError(f"Descriptions missing for {len(missing)} dataset drugs: {', '.join(missing[:10])}.")
    _check_description_run(args, output_dir)

    if args.skip_ft:
        print("[main] --skip-ft set → zero-shot test inference path",
              flush=True)
        # Skip L3/L5 entirely; build a stub fit_info from defaults.
        fit_info = {
            "model_name": model_full,
            "max_length": args.max_length,
            "yes_token": " Yes",
            "no_token": " No",
        }
        manifest = {
            "S2": {
                "best_ckpt": None,
                "best_step": 0,
                "best_val_auc": float("nan"),
            }
        }
        _, eval_bs = auto_batch_size(model_full, max_length=args.max_length)
        eval_bs = args.eval_bs if args.eval_bs is not None else eval_bs
        # Zero-shot inference uses the base model directly (no adapter).
        # Below: rewrite manifest's best_ckpt to None so the runner skips
        # adapter load.
    else:
        train_feats, val_dict, cfg_prompt = build_ft_samples(
            args, ds, sm, id2name, id2smi, prompt_method,
        )
        print(f"[main] train samples: {len(train_feats)}", flush=True)
        for k, v in val_dict.items():
            print(f"[main] val[{k}]      : {len(v)}", flush=True)

        cfg, info, eval_bs = run_ft(
            args, train_feats, val_dict, cfg_prompt, output_dir,
        )
        manifest, fit_info = run_select_best(
            args, cfg, ds, sm, cfg_prompt, val_dict, eval_bs,
        )
        # Persist manifest so later analysis can reload without rerunning.
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    if args.skip_test:
        print("[main] --skip-test set → done.", flush=True)
        return 0

    _, auc = run_test(
        args, manifest, fit_info, ds, sm, id2name, id2smi,
        prompt_method, eval_bs, output_dir,
    )
    # Persist final headline.
    with open(output_dir / "result.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "model": model_full,
            "dataset": args.dataset,
            "prompt": args.prompt,
            "seed": args.seed,
            "test_s2_auroc": auc,
        }, f, indent=2)
    print(f"\n[main] DONE  result.json → {output_dir / 'result.json'}\n",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
