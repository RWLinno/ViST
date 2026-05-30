#!/bin/bash
# run_baseline_timexl.sh — Run TimeXL baseline on ETTh2/ECL/Weather
# Requires TimeXL code cloned to baselines/TimeXL/

set -e
cd "$(dirname "$0")/.."

BASELINE="TimeXL"
SEEDS=(2024 2025 2026)
DATASETS=("ETTh2" "Electricity" "Weather")
HORIZONS=(96 192 336 720)

if [ ! -d "baselines/${BASELINE}" ]; then
    echo "Cloning TimeXL repository..."
    mkdir -p baselines
    git clone https://github.com/AdityaLab/TimeXL.git baselines/${BASELINE} 2>/dev/null || \
        echo "Warning: Could not clone TimeXL. Please manually place code in baselines/${BASELINE}/"
fi

for DATASET in "${DATASETS[@]}"; do
    for HORIZON in "${HORIZONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "[$(date)] Running ${BASELINE} on ${DATASET} H=${HORIZON} seed=${SEED}"
            CUDA_VISIBLE_DEVICES=3 python baselines/${BASELINE}/run.py \
                --dataset ${DATASET} \
                --pred_len ${HORIZON} \
                --seq_len 96 \
                --seed ${SEED} \
                2>&1 | tee -a logs/baseline_${BASELINE}_${DATASET}_H${HORIZON}_s${SEED}.log
        done
    done
done

echo "[$(date)] All TimeXL baseline experiments complete."
