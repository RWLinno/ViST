# ViST Paper Digest

## 论文信息
- **Title**: ViST: Harnessing Vision Transformation and Reconstruction for Multi-modal Spatio-temporal Forecasting
- **Venue**: ACM MM 2026, Submission #99
- **Core Claim**: 首个将原始时空数据直接转换为低维视觉空间的多模态时空预测框架

## §1 Introduction
- 现有多模态 STF 依赖外部视觉模态（遥感/卫星图像），难以对齐且计算昂贵
- ViST 直接将 ST 数据转换为视觉表示，实现更有效和可扩展的预测
- 三大贡献：Multi-view Vision Transformation / Multi-modal Conditional Reconstruction / Efficient Cross-modal Fusion (BWA)

## §2 Related Work
- 时空预测方法：GNN-RNN, Graph Convolution, Transformer-based, Neural ODE
- 多模态时序：TimesNet (1D→2D), Time-VLM (VLM conditioning), 相关 survey [Ni'25]

## §3 Methodology
- §3.1 Problem: X∈R^{T_in×N×D} → Y∈R^{T_out×N×D'}, 给定邻接矩阵 A
- §3.2 Multi-view Vision Transformation: STIM (Eq.1-2) + MVE (Eq.3-12)
- §3.3 Multi-modal Conditional Reconstruction: SFG (Eq.13-14) + TFG (Eq.15) + CVFR (Eq.16-20)
- §3.4 Cross-modal Attention Fusion: BWA (Eq.21-22) + WGE (Eq.23-25) + Prediction (Eq.26-27)

## §4 Experiments

### 数据集
| Dataset | Nodes | Domain | Interval | Split |
|---------|-------|--------|----------|-------|
| SD | 716 | Traffic | 5min | 6:2:2 |
| GBA | 2,352 | Traffic | 5min | 6:2:2 |
| GLA | 3,834 | Traffic | 5min | 6:2:2 |
| ECL | 321 | Energy | 1h | 7:1:2 |
| Weather | 21 | Climate | 10min | 7:1:2 |
| ETTh2 | 7 | Energy | 1h | 6:2:2 |

### 实验设置
- Input: 12 steps, Output: 12 steps
- Metrics: MAE, RMSE, MAPE
- Evaluation horizons: {3, 6, 12} 取平均
- Hardware: 4× NVIDIA RTX A6000

### Table 2 主要结果（大规模交通）
| Dataset | ViST MAE | ViST RMSE | ViST MAPE | Best Baseline | Improvement |
|---------|----------|-----------|-----------|---------------|-------------|
| SD (avg) | **17.59** | **29.88** | **11.15%** | STWave 18.22 | -3.46% MAE |
| GBA (avg) | **20.57** | 34.61 | **15.14%** | D2STGNN 20.71 | -0.68% MAE |
| GLA (avg) | **22.23** | 38.85 | **13.22%** | STGCN 22.64 | -1.81% MAE |

### Table 3 主要结果（跨域）
- ECL: ViST MAE 最优，比 BigST 降低 39.02%
- ETTh2: ViST MAE=0.25, RMSE=0.39（最优）
- Weather: ViST 竞争力强但优势较小

### Table 4 消融实验（SD 数据集）
| Variant | Avg MAE | Avg MAPE | MAPE Δ |
|---------|---------|----------|--------|
| Full ViST | 17.59 | 11.15% | — |
| w/o CVFR | 37.37 | 39.60% | **+406.28%** (H12) |
| w/o SFG | 20.65 | 15.67% | +40.49% |
| w/o TFG | 20.13 | 13.51% | +26.17% (H12 MAE) |
| w/o STIM | 19.28 | 13.35% | +19.73% |
| w/o CCG | 19.78 | 13.91% | +24.71% |
| w/o TCG | 19.28 | 13.30% | +19.24% |
| w/o SCG | 17.67 | 11.50% | +3.10% |

### Table 5 效率分析
| Method | GBA Train(s/epoch) | GBA Val(s) | GLA Train(s/epoch) | GLA Val(s) |
|--------|-------------------|------------|-------------------|------------|
| D2STGNN | 5392.56 | 830.39 | 24445.55 | 3999.65 |
| STWave | 982.40 | 152.55 | 1582.59 | 253.09 |
| **ViST** | **626.58** | **96.27** | **1227.69** | **176.57** |

## §5 Conclusion
- ViST 建立了视觉增强时空学习的新范式
- 视觉转换作为领域无关的中间表示
- 计算复杂度与节点数解耦：O(H×W) vs O(N)
