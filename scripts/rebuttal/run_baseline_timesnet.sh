#!/bin/bash
# run_baseline_timesnet.sh — Run TimesNet baseline
# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

set -e
cd "$(dirname "$0")/.."

BASELINE="TimesNet"
SEEDS=(2024 2025 2026)
DATASETS=("ETTh2" "Electricity" "Weather")
HORIZONS=(96 192 336 720)

if [ ! -d "baselines/${BASELINE}" ]; then
    echo "Cloning TimesNet repository..."
    mkdir -p baselines
    git clone https://github.com/thuml/Time-Series-Library.git baselines/${BASELINE} 2>/dev/null || \
        echo "Warning: Could not clone TimesNet. Please manually place code in baselines/${BASELINE}/"
fi

for DATASET in "${DATASETS[@]}"; do
    for HORIZON in "${HORIZONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "[$(date)] Running ${BASELINE} on ${DATASET} H=${HORIZON} seed=${SEED}"
            CUDA_VISIBLE_DEVICES=6 python baselines/${BASELINE}/run.py \
                --model TimesNet \
                --data ${DATASET} \
                --pred_len ${HORIZON} \
                --seq_len 96 \
                --random_seed ${SEED} \
                2>&1 | tee -a logs/baseline_${BASELINE}_${DATASET}_H${HORIZON}_s${SEED}.log
        done
    done
done

echo "[$(date)] All TimesNet baseline experiments complete."
