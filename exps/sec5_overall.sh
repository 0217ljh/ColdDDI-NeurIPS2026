#!/usr/bin/env bash
# Paper §5.2: overall S2 cold-start results for 8 baselines across 3 seeds.
#
# Write per-pair prediction CSVs and metrics JSON, then aggregate
# AUROC / AUPRC mean ± std per method.
#
# Usage:
#   bash exps/sec5_overall.sh [SUBSET]
#       SUBSET  = "800" (legacy pkl) | "1900" (release dir) | "toy"
#                 default: 1900
#
# Output:
#   runs/<method>/seed<N>/                       (per-(method,seed) artefacts)
#   runs/sec5_overall.csv                        (final paper-table CSV)
#
# --no-indicators skips L6; use sec5_kps.sh for the indicator sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SUBSET="${1:-1900}"
SEEDS=(42 43 44)
METHODS=(deepddi ssi_ddi dsn_ddi hdn_ddi emergnn textddi mkg_fenn tiger)

if [[ "${SUBSET}" == "toy" ]]; then
    DATA_FLAG=(--data data/public/intermediate)
else
    DATA_FLAG=(--subset "${SUBSET}")
fi

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "[sec5_overall] ${METHOD} seed=${SEED} subset=${SUBSET}"
        python evaluate.py \
            --method "${METHOD}" \
            "${DATA_FLAG[@]}" \
            --seed "${SEED}" \
            --setting S2 \
            --out "runs/${METHOD}/seed${SEED}" \
            --no-indicators
    done
done

echo "[sec5_overall] aggregating across (method, seed) ..."
python -m coldddi.eval.aggregate overall \
    --runs runs/ \
    --setting test_s2 \
    --out runs/sec5_overall.csv

echo "[sec5_overall] DONE — see runs/sec5_overall.csv"
