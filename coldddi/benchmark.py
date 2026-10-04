"""Run P4 training, evaluation and knowledge-channel diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from coldddi.diagnostics.buckets import BucketLookup
    from coldddi.diagnostics.kps_swap import SwapTriple


def write_json(path: Path, value: dict) -> None:
    """Replace a JSON artifact atomically; never emit nonstandard NaN values."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Release directory containing filtered/ and splits/seedN/.")
    parser.add_argument("--ab-parquet", type=Path, required=True, help="Matching pair-level A/B annotations; never auto-falls back to toy annotations.")
    parser.add_argument("--model", default="qwen-0.5b", help="qwen-0.5b, tiny-random-qwen, a Hugging Face ID, or a local model directory.")
    parser.add_argument("--revision", default="main", help="HF revision; resolved to a local snapshot before training.")
    parser.add_argument("--prompt", choices=("P4",), default="P4", help="P4 is the existing R0-R3 factorial diagnostic recipe.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New run directory; default includes model, dataset fingerprint, seed and mode.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="Training micro-batch and inference batch size; full runs preserve effective training batch 16.")
    parser.add_argument("--epochs", type=int, default=None, help="Default: paper preset (4); smoke: 1.")
    parser.add_argument("--max-length", type=int, default=None, help="Default: paper Top-3 KG preset (1250); smoke: 512.")
    parser.add_argument("--smoke", action="store_true", help="Same pipeline, capped at 16 training, 8 validation and 16 test pairs per split. Not a paper result.")
    parser.add_argument("--check-only", action="store_true", help="Validate data without downloading a model, training, or writing outputs.")
    parser.add_argument("--resume", action="store_true", help="Continue a matching run; reject changed inputs/configuration or corrupted completed artifacts.")
    return parser


def resolve_model(model: str, revision: str) -> tuple[str, str, str]:
    """Resolve remote weights once; later stages use this exact local snapshot."""
    from huggingface_hub import snapshot_download
    from coldddi.benchmark_data import file_sha256

    shortcuts = {"qwen-0.5b": "Qwen/Qwen2.5-0.5B",
                 "tiny-random-qwen": "yujiepan/qwen2.5-tiny-random"}
    identifier = shortcuts.get(model, model)
    local = Path(identifier).is_dir()
    path = Path(identifier).resolve() if local else Path(snapshot_download(
        repo_id=identifier, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken"],
        # Serial downloads avoid Hub 0.x's first-use symlink-probe race on
        # Windows accounts without symlink privileges.
        max_workers=1,
    ))
    if not (path / "config.json").is_file():
        raise ValueError(f"Model directory has no config.json: {path}")
    # Keep the model name for chat-template selection.
    if not any(family in str(path).lower() for family in ("qwen", "llama", "gemma", "mistral")):
        raise ValueError("Model path must identify its supported prompt family (qwen/llama/gemma/mistral); "
                         "the existing prompt builder otherwise guesses. Use a family-named local directory.")
    model_files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.endswith(".lock"))
    digest = hashlib.sha256()
    for file in model_files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file_sha256(file).encode())
    return identifier, str(path.resolve()), digest.hexdigest()


def balanced_pairs(pos: pd.DataFrame, neg: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    import pandas as pd

    if cap is not None:
        pos, neg = pos.head(cap // 2), neg.head(cap // 2)
    left = pos[["drug_a_id", "drug_b_id"]].copy()
    right = neg[["drug_a_id", "drug_b_id"]].copy()
    left["label"], right["label"] = 1, 0
    pairs = pd.concat([left, right], ignore_index=True)
    pairs[["drug_a_id", "drug_b_id"]] = pairs[["drug_a_id", "drug_b_id"]].astype(str)
    return pairs


def validate_predictions(frame: pd.DataFrame, pairs: pd.DataFrame, *, complete: bool = True) -> None:
    """Validate exact directed pair coverage before passing predictions to L6."""
    import numpy as np
    from coldddi.benchmark_data import require_columns

    require_columns(frame, ("drug_a_id", "drug_b_id", "p_yes"), "predictions")
    keys = list(zip(frame.drug_a_id.astype(str), frame.drug_b_id.astype(str)))
    expected = set(zip(pairs.drug_a_id.astype(str), pairs.drug_b_id.astype(str)))
    if len(keys) != len(set(keys)) or not set(keys) <= expected or (complete and set(keys) != expected):
        raise ValueError("Prediction coverage mismatch: duplicate, missing or unexpected directed pairs")
    values = frame.p_yes.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")


def diagnostic_pairs(test_pairs: pd.DataFrame, swaps: list[SwapTriple]) -> pd.DataFrame:
    """Include both anchor orientations as well as swap targets."""
    import pandas as pd

    extra = [(t.qa, t.qb) for t in swaps] + [(t.qa_prime, t.qb) for t in swaps]
    return pd.concat([test_pairs[["drug_a_id", "drug_b_id"]],
                      pd.DataFrame(extra, columns=["drug_a_id", "drug_b_id"])],
                     ignore_index=True).drop_duplicates().reset_index(drop=True)


def save_diagnostics(predictions: dict, swaps: list[SwapTriple], bucket_lookup: BucketLookup, output: Path) -> dict:
    from coldddi.diagnostics import compute_ab_gap, compute_indicators

    needed = {(t.qa, t.qb) for t in swaps} | {(t.qa_prime, t.qb) for t in swaps}
    for condition in ("R0", "R1", "R2", "R3"):
        if not needed <= set(predictions[condition]):
            raise ValueError(f"{condition}: missing swap anchor/target predictions")
    indicators = compute_indicators(predictions, swaps, bucket_fn=bucket_lookup.bucket, coverage_warnings=True)
    indicators.to_csv(output / "indicators.csv", index=False)
    gaps = {}
    for indicator in indicators.indicator.unique():
        gap = compute_ab_gap(indicators, indicator)
        gaps[indicator] = {"value": gap if math.isfinite(gap) else None,
                           "reason": None if math.isfinite(gap) else "One or more PK-A/PK-B/PD-A/PD-B buckets has no usable pairs."}
    write_json(output / "ab_gaps.json", gaps)
    return {"swap_triples": len(swaps), "directed_anchor_pairs": len({(t.qa, t.qb) for t in swaps}),
            "missing_predictions": 0, "indicators": sorted(gaps),
            "undefined_ab_gaps": [name for name, item in gaps.items() if item["value"] is None]}


def run(args: argparse.Namespace) -> Path | None:
    import pandas as pd
    import torch
    from filelock import FileLock
    from sklearn.metrics import average_precision_score, roc_auc_score

    from coldddi.benchmark_data import file_sha256, pair_keys, validate_dataset
    from coldddi.diagnostics import build_bucket_lookup, build_swap_candidates
    from coldddi.llm.inference import LLMInferenceRunner, LLMRunnerConfig
    from coldddi.llm.prompts import PromptBuildConfig, build_binary_prompt
    from coldddi.llm.retrieval import build_subgraph_map, to_llm_samples
    from coldddi.llm.select_best import (assert_prompt_cfg_matches_fit_info,
                                        parse_candidate_ckpts, read_fit_info,
                                        score_candidate_ckpts, select_best)
    from coldddi.llm.trainer import LoRATrainer, paper_llm_trainer_config

    print("[1/6] Validate dataset", flush=True)
    ds, ab, report = validate_dataset(args.data, args.ab_parquet, args.seed)
    print(json.dumps({key: value for key, value in report.items() if key != "input_sha256"}, indent=2), flush=True)
    if args.check_only:
        print("Input validation passed. No model was downloaded and no output was written.")
        return None
    # Fail on missing LLM dependencies before model download or output creation.
    packages = ("torch", "transformers", "peft", "accelerate", "numpy", "pandas", "pyarrow",
                "rdkit", "scikit-learn", "huggingface-hub", "tokenizers", "safetensors", "sentencepiece")
    versions = {name: metadata.version(name) for name in packages}
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable; use --device cpu or fix the PyTorch installation")
    dtype = "float32" if device == "cpu" else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", args.model).strip("-")[-60:]
    output = (args.output or Path("runs") / f"{slug}_P4_seed{args.seed}_{report['fingerprint'][:12]}_{'smoke' if args.smoke else 'full'}").resolve()
    if output.is_dir() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output is not empty: {output}. Use --resume for the same configuration, or a new --output.")

    print("[2/6] Resolve model snapshot", flush=True)
    identifier, model_path, model_hash = resolve_model(args.model, args.revision)
    batch = args.batch_size or (2 if args.smoke else 1)
    cfg = paper_llm_trainer_config(
        model_name=model_path, output_dir=str(output / "ckpts"), billions=14,
        device=device, dtype=dtype, trust_remote_code=False,
        micro_batch_size=batch, gradient_accumulation_steps=1 if args.smoke else 16 // batch,
        num_epochs=args.epochs or (1 if args.smoke else 4),
        max_length=args.max_length or (512 if args.smoke else 1250),
        eval_strategy="epoch", logging_steps=1 if args.smoke else 20,
        disable_tqdm=args.smoke, seed=args.seed,
    )
    if args.smoke:
        cfg.lora.r, cfg.lora.alpha = 4, 8
    effective = {"format_version": 1, "input_fingerprint": report["fingerprint"],
                 "model": identifier, "model_path": model_path, "model_sha256": model_hash,
                 "prompt": args.prompt, "seed": args.seed, "smoke": args.smoke,
                 "caps": {"train": 16 if args.smoke else None, "val": 8 if args.smoke else None,
                          "test": 16 if args.smoke else None},
                 "trainer": asdict(cfg), "inference_batch_size": batch,
                 "prediction_save_every": 10 if args.smoke else 1000,
                 "python": sys.version.split()[0], "packages": versions}
    # Invalidate cached results when code or data changes.
    code_root = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for source in sorted(code_root.rglob("*.py")):
        code_digest.update(str(source.relative_to(code_root)).encode())
        code_digest.update(file_sha256(source).encode())
    effective["code_sha256"] = code_digest.hexdigest()
    # Compare serialized settings because JSON converts tuples to lists.
    effective = json.loads(json.dumps(effective))
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".benchmark.lock"), timeout=0):
        config_path = output / "config.json"
        state_path = output / "status.json"
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text(encoding="utf-8")) != effective:
                raise ValueError("Existing run configuration differs (data/model/code/environment/settings); choose a new --output")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for name, digest in state.get("artifacts", {}).items():
                path = output / name
                if not path.is_file() or file_sha256(path) != digest:
                    raise ValueError(f"Completed artifact missing or changed: {name}. Choose a new --output.")
            if state["status"] == "complete":
                print(f"Verified completed run: {output}")
                return output
        else:
            if any(p.name != ".benchmark.lock" for p in output.iterdir()):
                raise ValueError("Output contains unrelated files; refusing to overwrite them")
            write_json(config_path, effective)
            state = {"status": "running", "stage": "setup", "artifacts": {}}
            write_json(state_path, state)

        def record(paths: list[Path], stage: str) -> None:
            for path in paths:
                state["artifacts"][path.relative_to(output).as_posix()] = file_sha256(path)
            state.update(status="running", stage=stage)
            state.pop("error", None)
            write_json(state_path, state)

        try:
            write_json(output / "input_report.json", report)
            record([output / "input_report.json"], "setup")
            sm = build_subgraph_map(ds.kg, ds.drugs, topk=3)
            id2name = dict(zip(ds.drugs.drugbank_id.astype(str), ds.drugs.name.astype(str)))
            id2smi = dict(zip(ds.drugs.drugbank_id.astype(str), ds.drugs.smiles.fillna("").astype(str)))
            key_entities = {}
            for row in ab.itertuples(index=False):
                item = {"key_entity_name": "" if pd.isna(row.key_entity_name) else str(row.key_entity_name),
                        "key_entity_type": "" if pd.isna(row.key_entity_type) else str(row.key_entity_type),
                        "has_key_entity": bool(row.has_key_entity)}
                key_entities[(str(row.drug_a_id), str(row.drug_b_id))] = item
                key_entities[(str(row.drug_b_id), str(row.drug_a_id))] = item
            bucket_lookup = build_bucket_lookup(ab)
            prompt = PromptBuildConfig(method="One_Hop_Subgraph_Sequence", model_name=model_path)
            # Selection and training validation use the same explicit smoke cap.
            selection_ds = ds
            if args.smoke:
                folds = replace(ds.splits, **{f"val_s{i}": getattr(ds.splits, f"val_s{i}").head(4) for i in range(3)})
                negatives = dict(ds.negatives_by_split)
                negatives.update({f"val_s{i}": ds.get_negatives(f"val_s{i}").head(4) for i in range(3)})
                selection_ds = replace(ds, splits=folds, negatives_by_split=negatives)
            train_pairs = balanced_pairs(ds.splits.train, ds.get_train_negatives(0), effective["caps"]["train"])
            val_pairs = {f"S{i}": balanced_pairs(getattr(selection_ds.splits, f"val_s{i}"),
                                                selection_ds.get_negatives(f"val_s{i}"), None) for i in range(3)}
            test_pairs = {f"S{i}": balanced_pairs(getattr(ds.splits, f"test_s{i}"), ds.get_negatives(f"test_s{i}"),
                                                 effective["caps"]["test"]) for i in range(3)}
            swaps = build_swap_candidates(ds, source_split="test_s2", search_pool_splits=("test_s1", "test_s2"))
            if args.smoke:
                anchors = set(pair_keys(test_pairs["S2"], "smoke test pairs"))
                swaps = [t for t in swaps if tuple(sorted((t.qa, t.qb))) in anchors]
            if not swaps or not any(t.label_uv == 1 for t in swaps):
                raise ValueError("No eligible positive S2 drug-swap candidates: full L6 diagnostics are undefined for this dataset/sample")
            pd.DataFrame([asdict(t) for t in swaps]).to_parquet(output / "swap_candidates.parquet", index=False)
            write_json(output / "effective_counts.json", {"train": len(train_pairs),
                       "validation": {s: len(p) for s, p in val_pairs.items()},
                       "test": {s: len(p) for s, p in test_pairs.items()}, "swap_triples": len(swaps)})
            record([output / "swap_candidates.parquet", output / "effective_counts.json"], "training")

            def features(pairs: pd.DataFrame) -> list[dict]:
                samples = to_llm_samples(pairs, pairs.label.tolist(), ds=ds, subgraph_map=sm)
                return [{"text": build_binary_prompt(s, prompt, drug_id2name=id2name, drug_id2smiles=id2smi),
                         "cls_labels": s["label"]} for s in samples]

            print("[3/6] LoRA training", flush=True)
            if "training_complete.json" not in state["artifacts"]:
                trainer = LoRATrainer(cfg)
                info = trainer.fit(train_samples=features(train_pairs),
                                   val_samples={s: features(p) for s, p in val_pairs.items()}, prompt_cfg=prompt)
                write_json(output / "training_complete.json", {"completed": True, "best_ckpt": info["best_ckpt"]})
                del trainer
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                record([output / "training_complete.json"] + sorted(
                    p for p in (output / "ckpts").rglob("*") if p.is_file()), "selection")
            fit_info = read_fit_info(cfg.output_dir)
            assert_prompt_cfg_matches_fit_info(fit_info, prompt, strict=True)
            print("[4/6] Checkpoint selection on validation splits", flush=True)
            manifest_path = output / "manifest.json"
            if "manifest.json" in state["artifacts"]:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                candidates = parse_candidate_ckpts(cfg.output_dir, splits=("S0", "S1", "S2"), topk=2)
                scored = score_candidate_ckpts(candidates, base_model_name=model_path, dataset=selection_ds,
                    prompt_cfg=prompt, subgraph_map=sm, device=device, dtype=dtype, batch_size=batch,
                    max_length=cfg.max_length, yes_token=fit_info["yes_token"], no_token=fit_info["no_token"])
                if any(not math.isfinite(c.val_auc) for group in scored.values() for c in group):
                    raise ValueError("A candidate checkpoint failed validation scoring; refusing to silently discard it")
                manifest = select_best(scored)
                write_json(manifest_path, manifest)
                record([manifest_path], "inference")

            metrics = {}
            predictions = {}
            print("[5/6] Test prediction and R0-R3 inference", flush=True)
            for split in ("S0", "S1", "S2"):
                runner = LLMInferenceRunner(LLMRunnerConfig(model_name=model_path,
                    adapter_path=manifest[split]["best_ckpt"], device=device, dtype=dtype,
                    batch_size=batch, max_length=cfg.max_length, trust_remote_code=False,
                    yes_token=fit_info["yes_token"], no_token=fit_info["no_token"]))
                pairs = diagnostic_pairs(test_pairs[split], swaps) if split == "S2" else test_pairs[split]
                samples = to_llm_samples(pairs, labels=None, ds=ds, subgraph_map=sm)
                conditions = (("R0", "One_Hop_Subgraph_Sequence"),)
                if split == "S2":
                    conditions += (("R1", "OHS_Mask_Name"), ("R2", "OHS_Mask_Entity"), ("R3", "OHS_Mask_Name_Entity"))
                for condition, method in conditions:
                    target = output / f"predictions_{split}_{condition}.parquet"
                    partial = target.with_suffix(target.suffix + ".partial")
                    if target.is_file():
                        frame = pd.read_parquet(target)
                    else:
                        if partial.is_file():
                            validate_predictions(pd.read_parquet(partial), pairs, complete=False)
                        frame = runner.score_samples(samples, PromptBuildConfig(method=method, model_name=model_path),
                            drug_id2name=id2name, drug_id2smiles=id2smi, key_entity_map=key_entities,
                            progress=not args.smoke, resume_path=target,
                            save_every=effective["prediction_save_every"])
                    validate_predictions(frame, pairs)
                    if not target.is_file():
                        frame.to_parquet(target, index=False)
                    record([target], "inference")
                    if split == "S2":
                        predictions[condition] = {(str(r.drug_a_id), str(r.drug_b_id)): float(r.p_yes)
                                                  for r in frame.itertuples(index=False)}
                    if condition == "R0":
                        aligned = test_pairs[split].merge(frame[["drug_a_id", "drug_b_id", "p_yes"]],
                            on=["drug_a_id", "drug_b_id"], how="left", validate="one_to_one")
                        if aligned.p_yes.isna().any():
                            raise ValueError(f"{split}: missing labeled test predictions")
                        aligned.to_csv(output / f"test_{split}.csv", index=False)
                        metrics[split] = {"auroc": float(roc_auc_score(aligned.label, aligned.p_yes)),
                                          "average_precision": float(average_precision_score(aligned.label, aligned.p_yes)),
                                          "pairs": len(aligned)}
                        record([output / f"test_{split}.csv"], "inference")
                del runner
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            print("[6/6] KPS/KSAI and A-B gaps", flush=True)
            coverage = save_diagnostics(predictions, swaps, bucket_lookup, output)
            coverage["positive_anchor_buckets"] = bucket_lookup.coverage_report(
                sorted({(t.qa, t.qb) for t in swaps if t.label_uv == 1}))
            write_json(output / "coverage.json", coverage)
            write_json(output / "metrics.json", metrics)
            record([output / name for name in ("coverage.json", "metrics.json", "indicators.csv", "ab_gaps.json")], "done")
            state["status"] = "complete"
            write_json(state_path, state)
            print(f"Complete ({'SMOKE; not a paper result' if args.smoke else 'full dataset'}): {output}")
            return output
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(state_path, state)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("batch_size", "epochs", "max_length"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.smoke and args.batch_size is not None and (args.batch_size > 16 or 16 % args.batch_size):
        parser.error("full-run --batch-size must divide 16 to preserve the effective batch size")
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check the input specification and requirements-benchmark.txt; failed runs are not reported as complete.", file=sys.stderr)
        return 1
    return 0
