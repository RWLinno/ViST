#!/bin/bash
# run_longhorizon_ecl.sh — Long-horizon TS benchmark on Electricity
# Lookback=96, horizons={96,192,336,720}, metrics: MSE/MAE
# Seeds: 2024, 2025, 2026

set -e
cd "$(dirname "$0")/.."

DATASET="Electricity"
CONFIG_DIR="ViST/longhorizon"
SEEDS=(2024 2025 2026)
HORIZONS=(96 192 336 720)

for HORIZON in "${HORIZONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "[$(date)] Running ${DATASET} horizon=${HORIZON} seed=${SEED}"
        CUDA_VISIBLE_DEVICES=1 python experiments/train.py \
            -c ${CONFIG_DIR}/ViST_${DATASET}_H${HORIZON}.py \
            --seed ${SEED} \
            -g 0 2>&1 | tee -a logs/longhorizon_${DATASET}_H${HORIZON}_s${SEED}.log
        echo "[$(date)] Done ${DATASET} horizon=${HORIZON} seed=${SEED}"
    done
done

echo "[$(date)] All ECL long-horizon experiments complete."
