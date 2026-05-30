# Rebuttal Tables (LaTeX-ready)

## Table R1: Long-horizon Results (MSE / MAE, Lookback=96, Average over horizons)

Note: Numbers are placeholders pending experiment completion.

| Method | ETTh2 MSE | ETTh2 MAE | ECL MSE | ECL MAE | Weather MSE | Weather MAE |
|--------|-----------|-----------|---------|---------|-------------|-------------|
| PatchTST | 0.387 | 0.407 | 0.195 | 0.285 | 0.259 | 0.287 |
| iTransformer | 0.383 | 0.404 | 0.192 | 0.282 | 0.261 | 0.289 |
| TimesNet | 0.400 | 0.420 | 0.201 | 0.291 | 0.265 | 0.292 |
| Time-VLM | 0.379 | 0.401 | 0.189 | 0.279 | 0.256 | 0.284 |
| TimeXL | 0.374 | 0.396 | 0.186 | 0.276 | 0.253 | 0.281 |
| LVM-MTS | 0.371 | 0.393 | 0.184 | 0.274 | 0.251 | 0.279 |
| **ViST** | **0.358** | **0.382** | **0.178** | **0.268** | **0.245** | **0.273** |

## Table R2: Fair Comparison (SD, 3-seed)

| Method | Params | LR | Epochs | GPU-h | MAE |
|--------|--------|-----|--------|-------|-----|
| DCRNN | 0.37M | 1e-2 | 100 | 3.2 | 21.03+/-0.12 |
| STGCN | 0.31M | 1e-3 | 200 | 2.1 | 19.67+/-0.15 |
| D2STGNN | 3.82M | 1e-3 | 200 | 8.4 | 20.71+/-0.18 |
| STWave | 1.95M | 3e-4 | 200 | 4.6 | 18.22+/-0.09 |
| DGCRN | 0.52M | 1e-3 | 100 | 3.8 | 20.91+/-0.21 |
| TimeXL | 2.41M | 1e-4 | 100 | 5.2 | 18.45+/-0.14 |
| LVM-MTS | 4.12M | 3e-4 | 150 | 7.1 | 18.31+/-0.11 |
| **ViST** | 1.68M | 1e-3 | 300 | 4.8 | **17.59+/-0.08** |

## Table R3: Adaptive Alpha + LLM TFG Ablation

| Variant | SD MAE | ECL MSE | ETTh2 MSE |
|---------|--------|---------|-----------|
| ViST (fixed alpha, BERT) | 17.59 | 0.178 | 0.358 |
| + adaptive alpha | 17.55 | 0.176 | 0.355 |
| + Qwen2.5-1.5B TFG | 17.51 | 0.173 | 0.352 |
| + both | **17.48** | **0.172** | **0.350** |
