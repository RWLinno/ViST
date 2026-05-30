#!/bin/bash
# run_llm_tfg_etth2.sh — LLM-augmented TFG (Qwen2.5-1.5B) on ETTh2 dataset

set -e
cd "$(dirname "$0")/.."

DATASET="ETTh2"
SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] Running LLM TFG on ${DATASET} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=5 python experiments/train.py \
        -c ViST/ablation/ViST_${DATASET}_llm_tfg.py \
        --seed ${SEED} \
        -g 0 2>&1 | tee -a logs/llm_tfg_${DATASET}_s${SEED}.log
    echo "[$(date)] Done LLM TFG ${DATASET} seed=${SEED}"
done

echo "[$(date)] All LLM TFG ETTh2 experiments complete."
