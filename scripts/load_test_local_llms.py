"""Load cached LLMs sequentially and test a forward pass, without downloads.

Unlike config and shard-presence checks, this loads weights with
AutoModelForCausalLM. Report load time, peak VRAM, parameter count,
and next-token ``p_yes / p_no`` on a test prompt; release each model
before loading the next. These scores do not measure task accuracy.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Cache order: HF_HOME/hub, ~/.cache/huggingface/hub, then
# COLDDDI_HF_EXTRA_CACHE entries split by os.pathsep.
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

MODELS: list[tuple[str, str]] = [
    ("Llama-3.2-1B",  "meta-llama/Llama-3.2-1B"),
    ("Llama-3.2-3B",  "meta-llama/Llama-3.2-3B"),
    ("Qwen2.5-0.5B",  "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-3B",    "Qwen/Qwen2.5-3B"),
]


@dataclass
class LoadResult:
    paper_name: str
    hf_id: str
    status: str       # OK / FAIL
    cache: str
    load_seconds: float
    peak_vram_gb: float
    n_params_million: float
    dtype: str
    p_yes: float
    p_no: float
    detail: str


def _resolve_cache_with_weights(hf_id: str) -> tuple[str, str] | None:
    """Return ``(cache_tag, cache_dir)`` with the most resolved weight shards."""
    safe_id = hf_id.replace("/", "--")
    best: tuple[str, str, int] | None = None
    for tag, cache in CACHE_LOCATIONS:
        model_root = Path(cache) / f"models--{safe_id}"
        if not model_root.is_dir():
            continue
        snap_root = model_root / "snapshots"
        if not snap_root.is_dir():
            continue
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            shards = list(snap.glob("*.safetensors")) + list(snap.glob("*.bin"))
            real = [s for s in shards if s.resolve().exists() and s.resolve().is_file()]
            if not real:
                continue
            if best is None or len(real) > best[2]:
                best = (tag, cache, len(real))
    return (best[0], best[1]) if best else None


def _load_one(paper_name: str, hf_id: str) -> LoadResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_hit = _resolve_cache_with_weights(hf_id)
    if cache_hit is None:
        return LoadResult(
            paper_name=paper_name, hf_id=hf_id, status="FAIL",
            cache="-", load_seconds=0.0, peak_vram_gb=0.0,
            n_params_million=0.0, dtype="-",
            p_yes=float("nan"), p_no=float("nan"),
            detail="no cache has weight shards",
        )
    cache_tag, cache_dir = cache_hit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(
            hf_id, cache_dir=cache_dir,
            local_files_only=True, trust_remote_code=True,
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, cache_dir=cache_dir,
            local_files_only=True, trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
    except Exception as e:
        return LoadResult(
            paper_name=paper_name, hf_id=hf_id, status="FAIL",
            cache=cache_tag, load_seconds=time.time() - t0,
            peak_vram_gb=0.0, n_params_million=0.0,
            dtype=str(dtype).rsplit(".", 1)[-1],
            p_yes=float("nan"), p_no=float("nan"),
            detail=f"{type(e).__name__}: {e}",
        )

    load_seconds = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Score " Yes" vs " No" to exercise the forward pass, not task accuracy.
    try:
        prompt = "Does drug A interact with drug B? Answer:"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        last = out.logits[0, -1, :].float()
        yes_ids = tok.encode(" Yes", add_special_tokens=False)
        no_ids = tok.encode(" No", add_special_tokens=False)
        if len(yes_ids) == 1 and len(no_ids) == 1:
            ly = last[yes_ids[0]].item()
            ln = last[no_ids[0]].item()
            import math
            mx = max(ly, ln)
            ey, en = math.exp(ly - mx), math.exp(ln - mx)
            p_yes = ey / (ey + en)
            p_no = en / (ey + en)
        else:
            p_yes = p_no = float("nan")
    except Exception as e:
        peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
        del model, tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return LoadResult(
            paper_name=paper_name, hf_id=hf_id, status="FAIL",
            cache=cache_tag, load_seconds=load_seconds,
            peak_vram_gb=peak_vram, n_params_million=n_params,
            dtype=str(dtype).rsplit(".", 1)[-1],
            p_yes=float("nan"), p_no=float("nan"),
            detail=f"forward: {type(e).__name__}: {e}",
        )

    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0

    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return LoadResult(
        paper_name=paper_name, hf_id=hf_id, status="OK",
        cache=cache_tag, load_seconds=load_seconds,
        peak_vram_gb=peak_vram, n_params_million=n_params,
        dtype=str(dtype).rsplit(".", 1)[-1],
        p_yes=p_yes, p_no=p_no, detail="",
    )


def main() -> int:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"\n{'=' * 100}")
    print(f"  Real-load test — AutoModelForCausalLM.from_pretrained + 1 forward pass")
    print(f"{'=' * 100}")

    import torch
    if torch.cuda.is_available():
        print(f"  Device       : cuda ({torch.cuda.get_device_name(0)})")
        print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print(f"  Device       : cpu (no CUDA)")

    results: list[LoadResult] = []
    for paper_name, hf_id in MODELS:
        print(f"\n  → loading {paper_name} ({hf_id}) ...", flush=True)
        r = _load_one(paper_name, hf_id)
        results.append(r)
        flag = "✓" if r.status == "OK" else "✗"
        if r.status == "OK":
            print(f"    {flag} OK  cache={r.cache}  "
                  f"load={r.load_seconds:.1f}s  "
                  f"VRAM={r.peak_vram_gb:.2f}G  "
                  f"params={r.n_params_million:,.1f}M  "
                  f"p_yes={r.p_yes:.4f}  p_no={r.p_no:.4f}")
        else:
            print(f"    {flag} FAIL  detail: {r.detail}")

    print(f"\n  {'#':<3}{'paper_name':<14}{'hf_id':<32}{'status':<8}"
          f"{'cache':<10}{'dtype':<10}{'load(s)':>8}{'VRAM(G)':>9}"
          f"{'params(M)':>11}{'p_yes':>8}{'p_no':>8}")
    print("  " + "-" * 121)
    n_ok = 0
    for i, r in enumerate(results, 1):
        flag = "✓" if r.status == "OK" else "✗"
        py = f"{r.p_yes:.3f}" if r.p_yes == r.p_yes else "-"
        pn = f"{r.p_no:.3f}" if r.p_no == r.p_no else "-"
        print(f"  {i:<3}{r.paper_name:<14}{r.hf_id:<32}{flag + ' ' + r.status:<8}"
              f"{r.cache:<10}{r.dtype:<10}{r.load_seconds:>8.1f}"
              f"{r.peak_vram_gb:>9.2f}{r.n_params_million:>11,.1f}"
              f"{py:>8}{pn:>8}")
        if r.status == "OK":
            n_ok += 1
    print(f"\n  → {n_ok}/{len(results)} models loaded + forward-passed successfully\n")

    if n_ok < len(results):
        print("  Failures:")
        for r in results:
            if r.status != "OK":
                print(f"    [{r.paper_name}]  {r.detail}")
        print()
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
