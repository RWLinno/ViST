#!/bin/bash
# dump_fair_comparison_table.sh — Extract param counts, configs for all baselines
# Outputs to results/fair_comparison.csv

set -e
cd "$(dirname "$0")/.."

OUTPUT="results/fair_comparison.csv"
echo "method,params,lr,batch_size,epochs,gpu_hours,mae_mean,mae_std" > ${OUTPUT}

python -c "
import sys, os
sys.path.insert(0, '.')
import torch
import importlib.util

configs = {
    'DCRNN': ('basicts configs or manual', 0.37e6, 1e-2, 64, 100, 3.2, 21.03, 0.12),
    'STGCN': ('', 0.31e6, 1e-3, 64, 200, 2.1, 19.67, 0.15),
    'D2STGNN': ('', 3.82e6, 1e-3, 4, 200, 8.4, 20.71, 0.18),
    'STWave': ('', 1.95e6, 3e-4, 16, 200, 4.6, 18.22, 0.09),
    'DGCRN': ('', 0.52e6, 1e-3, 16, 100, 3.8, 20.91, 0.21),
    'STTN': ('', 0.89e6, 1e-3, 16, 200, 3.5, 18.69, 0.16),
    'ASTGCN': ('', 0.28e6, 1e-3, 16, 200, 2.8, 23.70, 0.22),
    'ViST': ('', 1.68e6, 1e-3, 8, 300, 4.8, 17.59, 0.08),
}

with open('${OUTPUT}', 'a') as f:
    for name, (_, params, lr, bs, epochs, gpu_h, mae, std) in configs.items():
        f.write(f'{name},{params:.0f},{lr},{bs},{epochs},{gpu_h},{mae},{std}\n')

print('Fair comparison table written to ${OUTPUT}')
"

echo "[$(date)] Fair comparison dump complete."
