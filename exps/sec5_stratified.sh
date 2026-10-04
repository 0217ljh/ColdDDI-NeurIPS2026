#!/usr/bin/env bash
# Paper §5.3: per-bucket AUC for PK-A / PK-B / PD-A / PD-B.
#
# Reuse sec5_overall.sh predictions; train only method/seed runs whose
# prediction CSV is missing.
#
# Usage:
#   bash exps/sec5_stratified.sh [SUBSET] [AB_PARQUET]
#       SUBSET     = "800" | "1900" | "toy"            default: 1900
#       AB_PARQUET = path/to/ab.parquet                default: auto-discover
#
# Output:
#   runs/<method>/seed<N>/predictions_test_s2_seed<N>.csv  (reused)
#   runs/sec5_stratified.csv  (per-bucket mean ± std across seeds)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SUBSET="${1:-1900}"
AB_PARQUET="${2:-}"
SEEDS=(42 43 44)
METHODS=(deepddi ssi_ddi dsn_ddi hdn_ddi emergnn textddi mkg_fenn tiger)

if [[ "${SUBSET}" == "toy" ]]; then
    DATA_FLAG=(--data data/public/intermediate)
else
    DATA_FLAG=(--subset "${SUBSET}")
fi

# Auto-discover the AB parquet when omitted.
if [[ -z "${AB_PARQUET}" ]]; then
    if [[ -f "annotations/ab.parquet" ]]; then
        AB_PARQUET="annotations/ab.parquet"
    elif [[ -f "annotations/ab_sample.parquet" ]]; then
        AB_PARQUET="annotations/ab_sample.parquet"
    else
        echo "[sec5_stratified] no annotations/{ab,ab_sample}.parquet — pass one explicitly" >&2
        exit 1
    fi
fi
echo "[sec5_stratified] using AB parquet: ${AB_PARQUET}"

# Backfill missing method/seed predictions.
for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CSV="runs/${METHOD}/seed${SEED}/predictions_test_s2_seed${SEED}.csv"
        if [[ ! -f "${CSV}" ]]; then
            echo "[sec5_stratified] backfilling ${METHOD} seed=${SEED}"
            python evaluate.py \
                --method "${METHOD}" \
                "${DATA_FLAG[@]}" \
                --seed "${SEED}" \
                --setting S2 \
                --out "runs/${METHOD}/seed${SEED}" \
                --no-indicators
        fi
    done
done

echo "[sec5_stratified] aggregating per-bucket AUC ..."
python -m coldddi.eval.aggregate stratified \
    --runs runs/ \
    --ab-parquet "${AB_PARQUET}" \
    --setting test_s2 \
    --out runs/sec5_stratified.csv

echo "[sec5_stratified] DONE — see runs/sec5_stratified.csv"
