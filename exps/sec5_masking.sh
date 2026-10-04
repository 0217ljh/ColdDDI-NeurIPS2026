#!/usr/bin/env bash
# Run a model/prompt cell for paper §5.5 LLM masking analysis across 3 seeds.
#
# Default: Llama-3.2-1B + P4 + 800-drug (Appendix D.2).
#
# Usage:
#   bash exps/sec5_masking.sh [MODEL] [PROMPT] [SUBSET]
#       MODEL   = llama-1b | qwen-0.5b | ...       default: llama-1b
#       PROMPT  = P1 | P2 | P3 | P4 | P5           default: P4
#       SUBSET  = 800 | 1900 | toy                 default: 800
#
# Output:
#   runs/<model>/<prompt>/seed<N>/                  (per-seed cells)
#   runs/sec5_masking_<model>_<prompt>.csv  (mean ± std; requires indicators)
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

# --method reads seed<N>/ directly under the model/prompt cell and labels
# the combined method as model_prompt.
#
# WARNING: final aggregation requires indicators_test_s2_seed<N>.csv in
# the standard diagnostics schema. scripts/run_llm.py does not write it;
# without those CSVs, the aggregate KPS / KSAI table is empty.
echo "[sec5_masking] aggregating LLM KPS / KSAI across seeds ..."
python -m coldddi.eval.aggregate kps \
    --runs "runs/${MODEL}/${PROMPT}" \
    --method "${MODEL}_${PROMPT}" \
    --out "runs/sec5_masking_${MODEL}_${PROMPT}.csv"

echo "[sec5_masking] DONE — see runs/sec5_masking_${MODEL}_${PROMPT}.csv"
