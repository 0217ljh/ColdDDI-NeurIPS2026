#!/usr/bin/env bash
# One-command reproduction of paper §5.5 — LLM-FT masking factorial
# (R0 / R1 / R2 / R3) yielding KPS-Name / KPS-KG / KPS-KG-Masked /
# KPS-Name-KGMasked / KSAI for the LLM stack.
#
# This wraps `scripts/run_llm.py` for one model × one prompt across
# 3 seeds.  Default cell is Llama-3.2-1B + P4 + 800-drug, which is
# the paper-headline configuration (Appendix D.2).  Override via
# the positional arguments to sweep other cells.
#
# Usage:
#   bash exps/sec5_masking.sh [MODEL] [PROMPT] [SUBSET]
#       MODEL   = llama-1b | qwen-0.5b | ...       default: llama-1b
#       PROMPT  = P1 | P2 | P3 | P4 | P5           default: P4
#       SUBSET  = 800 | 1900 | toy                 default: 800
#
# Output:
#   runs/<model>/<prompt>/seed<N>/                  (per-seed cells)
#   runs/sec5_masking.csv                           (3-seed mean ± std)
#
# Note: the LLM stack has its own end-to-end pipeline; this script
# is the LLM equivalent of sec5_kps.sh and produces the same shape
# of long-form KPS / KSAI table.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-llama-1b}"
PROMPT="${2:-P4}"
SUBSET="${3:-800}"
SEEDS=(42 43 44)

if [[ ! -f "scripts/run_llm.py" ]]; then
    echo "[sec5_masking] scripts/run_llm.py not found — LLM stack missing?" >&2
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    echo "[sec5_masking] ${MODEL} ${PROMPT} seed=${SEED} subset=${SUBSET}"
    python scripts/run_llm.py \
        --model "${MODEL}" \
        --prompt "${PROMPT}" \
        --dataset "${SUBSET}-drug" \
        --seed "${SEED}" \
        --output-dir "runs/${MODEL}/${PROMPT}/seed${SEED}"
done

# The LLM output tree is runs/<model>/<prompt>/seed<N>/ (the model+
# prompt cell IS the method identity), one level deeper than the
# baseline runs/<method>/seed<N>/ layout.  The aggregator's
# `--method NAME` flag puts it in flat mode so it walks the cell
# dir's seed<N>/ children directly and labels them with the
# composite "model_prompt" method name.
#
# CAVEAT: this aggregation step depends on scripts/run_llm.py
# emitting `indicators_test_s2_seed<N>.csv` in the standard
# diagnostics-CSV schema.  The LLM-L6 write-out is a separate
# work item (smoke_llm_pipeline_v3.py wires the math but doesn't
# yet persist the canonical CSV from run_llm.py).  Until that
# lands, this script's per-seed runs are reproducible but the
# final aggregate KPS / KSAI table will be empty.
echo "[sec5_masking] aggregating LLM KPS / KSAI across seeds ..."
python -m coldddi.eval.aggregate kps \
    --runs "runs/${MODEL}/${PROMPT}" \
    --method "${MODEL}_${PROMPT}" \
    --out "runs/sec5_masking_${MODEL}_${PROMPT}.csv"

echo "[sec5_masking] DONE — see runs/sec5_masking_${MODEL}_${PROMPT}.csv"
