#!/bin/bash
# run_adaptive_alpha_ecl.sh — Adaptive alpha ablation on ECL dataset

set -e
cd "$(dirname "$0")/.."

DATASET="Electricity"
SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] Running adaptive alpha on ${DATASET} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=2 python experiments/train.py \
        -c ViST/ablation/ViST_${DATASET}_adaptive_alpha.py \
        --seed ${SEED} \
        -g 0 2>&1 | tee -a logs/adaptive_alpha_${DATASET}_s${SEED}.log
    echo "[$(date)] Done adaptive alpha ${DATASET} seed=${SEED}"
done

echo "[$(date)] All adaptive alpha ECL experiments complete."
