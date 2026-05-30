#!/bin/bash
# run_baseline_lvmmts.sh — Run LVM-MTS baseline on ETTh2/ECL/Weather
# Multi-Modal View Enhanced Large Vision Models for LTTSF (NeurIPS'25)

set -e
cd "$(dirname "$0")/.."

BASELINE="LVM-MTS"
SEEDS=(2024 2025 2026)
DATASETS=("ETTh2" "Electricity" "Weather")
HORIZONS=(96 192 336 720)

if [ ! -d "baselines/${BASELINE}" ]; then
    echo "Cloning LVM-MTS repository..."
    mkdir -p baselines
    git clone https://github.com/LVM-MTS/LVM-MTS.git baselines/${BASELINE} 2>/dev/null || \
        echo "Warning: Could not clone LVM-MTS. Please manually place code in baselines/${BASELINE}/"
fi

for DATASET in "${DATASETS[@]}"; do
    for HORIZON in "${HORIZONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "[$(date)] Running ${BASELINE} on ${DATASET} H=${HORIZON} seed=${SEED}"
            CUDA_VISIBLE_DEVICES=4 python baselines/${BASELINE}/run.py \
                --dataset ${DATASET} \
                --pred_len ${HORIZON} \
                --seq_len 96 \
                --seed ${SEED} \
                2>&1 | tee -a logs/baseline_${BASELINE}_${DATASET}_H${HORIZON}_s${SEED}.log
        done
    done
done

echo "[$(date)] All LVM-MTS baseline experiments complete."
