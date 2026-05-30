#!/bin/bash
# run_multiseed_sd.sh — Multi-seed robustness on SD dataset
# 3 seeds for mean±std reporting

set -e
cd "$(dirname "$0")/.."

DATASET="SD"
SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] Running ViST on ${DATASET} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=6 python experiments/train.py \
        -c ViST/ViST_SD.py \
        --seed ${SEED} \
        -g 0 2>&1 | tee -a logs/multiseed_${DATASET}_s${SEED}.log
    echo "[$(date)] Done ${DATASET} seed=${SEED}"
done

echo "[$(date)] All multi-seed SD experiments complete."
