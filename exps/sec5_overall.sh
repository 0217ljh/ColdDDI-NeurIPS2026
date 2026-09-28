#!/usr/bin/env bash
# One-command reproduction of paper §5.2 — overall S2 cold-start results.
#
# Trains all 8 baselines on 3 seeds against the chosen subset,
# writes per-pair predictions CSVs (Step 1) and the aggregate
# metrics JSON, then rolls up AUROC / AUPRC mean ± std per method
# via `coldddi.eval.aggregate overall`.
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
# Indicators step (L6) is intentionally skipped here via
# --no-indicators — sec5_kps.sh re-runs evaluate with indicators
# enabled.  Splitting the two pipelines keeps the wall-clock
# predictable: overall metrics are cheap; L6 is a separate sweep.
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
