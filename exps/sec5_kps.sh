#!/usr/bin/env bash
# Paper §5.4: KPS / KSAI for 8 baselines across 3 seeds.
# KPS-F applies to all; KPS-mol / KPS-KG require separable mol+KG inputs.
# Unsupported single-modality indicators remain NaN.
#
# evaluate.py writes L6 indicators by default; aggregate kps combines seeds.
#
# Usage:
#   bash exps/sec5_kps.sh [SUBSET] [AB_PARQUET]
#       SUBSET     = "800" | "1900" | "toy"          default: 1900
#       AB_PARQUET = path/to/ab.parquet              default: auto-discover
#
# Output:
#   runs/<method>/seed<N>/indicators_test_s2_seed<N>.csv
#   runs/<method>/seed<N>/predictions_test_s2_seed<N>.csv
#   runs/<method>/seed<N>/predictions_test_s2_mask_{mol,kg}_seed<N>.csv
#     (masked predictions: mol+KG baselines only)
#   runs/sec5_kps.csv  (mean ± std across seeds)
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

AB_ARGS=()
if [[ -n "${AB_PARQUET}" ]]; then
    AB_ARGS=(--ab-parquet "${AB_PARQUET}")
fi

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "[sec5_kps] ${METHOD} seed=${SEED} subset=${SUBSET}"
        python evaluate.py \
            --method "${METHOD}" \
            "${DATA_FLAG[@]}" \
            --seed "${SEED}" \
            --setting S2 \
            --out "runs/${METHOD}/seed${SEED}" \
            "${AB_ARGS[@]}"
    done
done

echo "[sec5_kps] aggregating KPS / KSAI across (method, seed) ..."
python -m coldddi.eval.aggregate kps \
    --runs runs/ \
    --out runs/sec5_kps.csv

echo "[sec5_kps] DONE — see runs/sec5_kps.csv"
