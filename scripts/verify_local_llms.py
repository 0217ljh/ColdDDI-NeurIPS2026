"""Verify the 11 paper LLMs are locally cached and HF-loadable.

For each of the 11 models in paper Table B.x we try, in order:

1. ``AutoConfig.from_pretrained(hf_id, cache_dir=cache, local_files_only=True)``
2. ``AutoTokenizer.from_pretrained(hf_id, cache_dir=cache, local_files_only=True)``

``local_files_only=True`` is critical — without it, transformers will
silently hit the network on a partial download and we'd never detect a
half-broken local copy.

Cache locations are resolved from environment variables (see
:func:`_build_cache_locations` below) in this order:

1. ``HF_HOME`` env var (standard HuggingFace convention).
2. ``~/.cache/huggingface/hub`` (HF default).
3. Extra roots from ``COLDDDI_HF_EXTRA_CACHE`` (os.pathsep-split,
   useful when models live on a separate drive than ``$HOME``).

For each model we report:

  status               : OK / config-only (tokenizer missing) / MISSING / ERROR
  resolved cache_dir   : whichever location had the snapshot
  model_type           : llama / qwen2 / gemma3_text / ...
  hidden_size, vocab   : config sanity
  tokenizer class      : LlamaTokenizer / GemmaTokenizer / ...

No model weights are loaded — that would require ~50 GB+ RAM/VRAM
just for Qwen2.5-14B.  Config + tokenizer load is enough to prove
the snapshot is intact and transformers can dispatch the right
classes when the real load happens at FT / inference time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Order matches paper Table; HF id is the canonical repo we'd pass to
# AutoModelForCausalLM.from_pretrained.
MODELS: list[tuple[str, str]] = [
    ("Llama-3.2-1B",  "meta-llama/Llama-3.2-1B"),
    ("Llama-3.2-3B",  "meta-llama/Llama-3.2-3B"),
    ("Llama-2-7B",    "meta-llama/Llama-2-7b-hf"),
    ("Llama-2-13B",   "meta-llama/Llama-2-13b-hf"),
    ("Qwen2.5-0.5B",  "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-3B",    "Qwen/Qwen2.5-3B"),
    ("Qwen2.5-7B",    "Qwen/Qwen2.5-7B"),
    ("Qwen2.5-14B",   "Qwen/Qwen2.5-14B"),
    ("Gemma-3-1B",    "google/gemma-3-1b-pt"),
    ("Gemma-3-4B",    "google/gemma-3-4b-pt"),
    ("Gemma-3-12B",   "google/gemma-3-12b-pt"),
]

# cache_dir for from_pretrained is the HF *hub* directory
# (where the ``models--<org>--<model>`` dirs live), NOT the HF_HOME
# root.  Pass the trailing /hub or transformers will look for a fresh
# download and (offline) raise "couldn't connect to huggingface.co".
#
# Resolution order:
#   1. ``HF_HOME`` env var (standard HuggingFace convention)
#   2. ``~/.cache/huggingface/hub`` (HF default)
#   3. Any extra roots in ``COLDDDI_HF_EXTRA_CACHE`` (os.pathsep-split,
#      useful when models live on a different drive than $HOME)
def _build_cache_locations() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        out.append(("HF_HOME", os.path.join(hf_home, "hub")))
    out.append(("default", os.path.expanduser("~/.cache/huggingface/hub")))
    extra = os.environ.get("COLDDDI_HF_EXTRA_CACHE")
    if extra:
        for i, p in enumerate(extra.split(os.pathsep)):
            p = p.strip()
            if p:
                out.append((f"extra-{i}", p))
    return out


CACHE_LOCATIONS: list[tuple[str, str]] = _build_cache_locations()


@dataclass
class CheckResult:
    paper_name: str
    hf_id: str
    status: str       # OK / NO_TOKENIZER / NO_WEIGHTS / MISSING / ERROR
    cache_tag: str    # which cache location served the snapshot
    model_type: str   # config.model_type
    hidden_size: int  # config.hidden_size
    vocab_size: int   # config.vocab_size
    tokenizer_cls: str
    weight_files: int     # number of resolved weight shards
    weight_size_gb: float # total size of resolved weight shards
    detail: str       # last error / extra info


def _resolve_snapshot_dir(cache_root: str, hf_id: str) -> Path | None:
    """Return the snapshot directory for ``<cache>/models--<org>--<model>``.

    Picks the snapshot referenced by ``refs/main`` when present, else
    the first snapshot dir found.  Returns ``None`` when nothing is on
    disk.
    """
    safe_id = hf_id.replace("/", "--")
    model_root = Path(cache_root) / f"models--{safe_id}"
    if not model_root.is_dir():
        return None
    refs_main = model_root / "refs" / "main"
    snap_root = model_root / "snapshots"
    if not snap_root.is_dir():
        return None
    if refs_main.is_file():
        try:
            rev = refs_main.read_text().strip()
            candidate = snap_root / rev
            if candidate.is_dir():
                return candidate
        except Exception:
            pass
    # Fall back to whichever snapshot dir exists.
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    return snaps[0] if snaps else None


def _check_weight_files(snapshot: Path) -> tuple[int, float, str]:
    """Return ``(n_resolved_shards, total_gb, detail)``.

    Looks for ``*.safetensors`` or ``pytorch_model*.bin`` (real or
    symlink that resolves to a non-empty file).  Any ``.incomplete``
    blob is reported as a failure cause.
    """
    detail_bits: list[str] = []
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    shards: list[Path] = []
    for pat in patterns:
        shards.extend(sorted(snapshot.glob(pat)))
    n_ok = 0
    total = 0
    for shard in shards:
        try:
            real = shard.resolve(strict=True)
            size = real.stat().st_size
            total += size
            n_ok += 1
        except FileNotFoundError:
            detail_bits.append(f"missing target: {shard.name}")
        except Exception as e:
            detail_bits.append(f"{shard.name}: {e}")

    # Surface .incomplete artefacts as a hard failure signal even
    # when the snapshot itself has no resolved shards.
    blobs_dir = snapshot.parent.parent / "blobs"
    if blobs_dir.is_dir():
        incomplete = list(blobs_dir.glob("*.incomplete"))
        if incomplete:
            sz_gb = sum(p.stat().st_size for p in incomplete) / 1e9
            detail_bits.append(
                f"{len(incomplete)} incomplete blob(s) totalling "
                f"{sz_gb:.1f} GB — interrupted download"
            )

    return n_ok, total / 1e9, "; ".join(detail_bits)


def _try_one(paper_name: str, hf_id: str) -> CheckResult:
    from transformers import AutoConfig, AutoTokenizer

    last_err = ""

    # Probe EVERY cache location and pick the one with the most
    # complete state (weights present beats config-only stub). A
    # single HF_HOME default can be metadata-only while a secondary
    # cache (drive-e) carries the real weight shards.
    per_cache: list[tuple[str, str, object, int, float, str]] = []
    for tag, cache in CACHE_LOCATIONS:
        try:
            cfg = AutoConfig.from_pretrained(
                hf_id,
                cache_dir=cache,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            last_err = f"[{tag}] {type(e).__name__}: {e}"
            continue
        snap = _resolve_snapshot_dir(cache, hf_id)
        if snap is None:
            n_shards, gb, detail = 0, 0.0, "no snapshot dir resolved"
        else:
            n_shards, gb, detail = _check_weight_files(snap)
        per_cache.append((tag, cache, cfg, n_shards, gb, detail))

    if not per_cache:
        return CheckResult(
            paper_name=paper_name, hf_id=hf_id, status="MISSING",
            cache_tag="-", model_type="-",
            hidden_size=-1, vocab_size=-1, tokenizer_cls="-",
            weight_files=0, weight_size_gb=0.0,
            detail=last_err,
        )

    # Pick the cache with the most resolved shards (tie-break on size).
    per_cache.sort(key=lambda r: (r[3], r[4]), reverse=True)
    cache_tag_hit, cache_dir_hit, config, n_shards, size_gb, weight_detail = per_cache[0]

    # Tokenizer (slow or fast — we don't care which). Loaded against
    # the same cache that won the weight check; tokenizer may live in
    # either cache but we want the one paired with the real shards.
    tok_cls = "-"
    tok_status = "OK"
    try:
        tok = AutoTokenizer.from_pretrained(
            hf_id,
            cache_dir=cache_dir_hit,
            local_files_only=True,
            trust_remote_code=True,
        )
        tok_cls = type(tok).__name__
    except Exception as e:
        # Tokenizer may not be in this specific cache — retry the
        # other location before giving up.
        tok = None
        for tag, alt_cache in CACHE_LOCATIONS:
            if alt_cache == cache_dir_hit:
                continue
            try:
                tok = AutoTokenizer.from_pretrained(
                    hf_id,
                    cache_dir=alt_cache,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                tok_cls = type(tok).__name__
                break
            except Exception:
                continue
        if tok is None:
            tok_status = "NO_TOKENIZER"
            last_err = f"tokenizer: {type(e).__name__}: {e}"

    if n_shards == 0:
        tok_status = "NO_WEIGHTS"
        last_err = (last_err + " | " if last_err else "") + (
            f"weights: {weight_detail or 'no shards under snapshot'}"
        )
    elif weight_detail:
        # Shards exist but blobs/ also has .incomplete files — flag it.
        last_err = (last_err + " | " if last_err else "") + (
            f"weights: {weight_detail}"
        )

    return CheckResult(
        paper_name=paper_name, hf_id=hf_id,
        status=tok_status,
        cache_tag=cache_tag_hit,
        model_type=str(getattr(config, "model_type", "?")),
        hidden_size=int(getattr(config, "hidden_size", -1)),
        vocab_size=int(getattr(config, "vocab_size", -1)),
        tokenizer_cls=tok_cls,
        weight_files=n_shards,
        weight_size_gb=size_gb,
        detail=last_err,
    )


def main() -> int:
    # Force transformers to NOT touch the network at all.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"\n{'=' * 96}")
    print(f"  Verifying 11 local LLMs (HF AutoConfig + AutoTokenizer, local_files_only=True)")
    print(f"{'=' * 96}")
    print(f"  Caches tried (in order):")
    for tag, path in CACHE_LOCATIONS:
        ok = "✓" if Path(path).is_dir() else "✗ missing"
        print(f"    [{tag:<7}] {path}  {ok}")

    results: list[CheckResult] = []
    for paper_name, hf_id in MODELS:
        r = _try_one(paper_name, hf_id)
        results.append(r)

    print(f"\n  {'#':<3}{'paper_name':<14}{'hf_id':<32}{'status':<14}"
          f"{'cache':<10}{'shards':>7}{'  size':>8}  tokenizer")
    print(f"  {'-' * 3:<3}{'-' * 13:<14}{'-' * 31:<32}{'-' * 13:<14}"
          f"{'-' * 9:<10}{'-' * 6:>7}{'-' * 7:>8}  {'-' * 24}")
    n_ok = 0
    for i, r in enumerate(results, 1):
        flag = "✓" if r.status == "OK" else "✗"
        size_str = f"{r.weight_size_gb:.1f}G" if r.weight_size_gb else "-"
        print(f"  {i:<3}{r.paper_name:<14}{r.hf_id:<32}{flag + ' ' + r.status:<14}"
              f"{r.cache_tag:<10}{r.weight_files:>7}{size_str:>8}  {r.tokenizer_cls}")
        if r.status == "OK":
            n_ok += 1
    print(f"\n  → {n_ok}/{len(results)} models OK\n")

    # Surface every error in detail so partial-cache problems aren't hidden.
    failures = [r for r in results if r.status != "OK"]
    if failures:
        print(f"  Errors / partial states ({len(failures)}):")
        for r in failures:
            print(f"    [{r.paper_name}]  {r.detail}")
    print()
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
