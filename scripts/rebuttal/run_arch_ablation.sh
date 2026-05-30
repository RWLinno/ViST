#!/bin/bash
# run_arch_ablation.sh — Architecture ablation on SD
# ViST(CNN encoder) vs ViST-MLP vs ViST-Transformer

set -e
cd "$(dirname "$0")/.."

DATASET="SD"
SEEDS=(2024 2025 2026)
VARIANTS=("cnn" "mlp" "transformer")

for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "[$(date)] Running arch ablation ${VARIANT} on ${DATASET} seed=${SEED}"
        CUDA_VISIBLE_DEVICES=7 python experiments/train.py \
            -c ViST/ablation/ViST_${DATASET}_arch_${VARIANT}.py \
            --seed ${SEED} \
            -g 0 2>&1 | tee -a logs/arch_ablation_${VARIANT}_${DATASET}_s${SEED}.log
        echo "[$(date)] Done arch ablation ${VARIANT} ${DATASET} seed=${SEED}"
    done
done

echo "[$(date)] All architecture ablation experiments complete."
