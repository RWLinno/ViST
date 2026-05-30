#!/bin/bash
# run_adaptive_alpha_sd.sh — Adaptive alpha ablation on SD dataset
# Replaces fixed alpha in SCG with learnable alpha_t = sigmoid(MLP([h_t; t/T]))

set -e
cd "$(dirname "$0")/.."

DATASET="SD"
SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] Running adaptive alpha on ${DATASET} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=0 python experiments/train.py \
        -c ViST/ablation/ViST_${DATASET}_adaptive_alpha.py \
        --seed ${SEED} \
        -g 0 2>&1 | tee -a logs/adaptive_alpha_${DATASET}_s${SEED}.log
    echo "[$(date)] Done adaptive alpha ${DATASET} seed=${SEED}"
done

echo "[$(date)] All adaptive alpha SD experiments complete."
