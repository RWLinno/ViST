#!/bin/bash
# run_multiseed_gba.sh — Multi-seed robustness on GBA dataset

set -e
cd "$(dirname "$0")/.."

DATASET="GBA"
SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] Running ViST on ${DATASET} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=7 python experiments/train.py \
        -c ViST/ViST_GBA.py \
        --seed ${SEED} \
        -g 0 2>&1 | tee -a logs/multiseed_${DATASET}_s${SEED}.log
    echo "[$(date)] Done ${DATASET} seed=${SEED}"
done

echo "[$(date)] All multi-seed GBA experiments complete."
