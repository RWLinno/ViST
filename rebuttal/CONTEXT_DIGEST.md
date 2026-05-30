# ViST Context Digest

## 代码模块与论文章节对应关系

| 模块 | 代码文件 | 论文章节 | 公式编号 | 功能 |
|------|----------|----------|----------|------|
| STIM | `ViST/arch/vist_arch.py` (class STIM) | §3.2.1 | Eq.(1)-(2) | 时空身份映射：将原始输入通过 Conv2d + 节点/时间嵌入融合为隐藏表示 H_t |
| MVE | `ViST/arch/blocks/MultiPerspectiveVisualEncoder.py` | §3.2.2 | Eq.(3) | 多视角视觉编码器：生成 V=[V_s; V_t; V_c] ∈ R^{B×T×3×H×W} |
| SCG | 同上 `_generate_spatial_channel_vectorized()` | §3.2.2 ¶1 | Eq.(4)-(6) | 空间通道生成：特征重要性加权 + 邻接矩阵拓扑融合，alpha∈[0.15,0.3] |
| TCG | 同上 `_generate_temporal_channel_vectorized()` | §3.2.2 ¶2 | Eq.(7)-(8) | 时间通道生成：指数加权滚动统计 + 时间标记 |
| CCG | 同上 `_generate_correlation_channel_vectorized()` | §3.2.2 ¶3 | Eq.(9)-(12) | 相关性通道生成：节点间相关矩阵（短序列直接计算/长序列滑动窗口） |
| SFG | `ViST/arch/vist_arch.py` (CVFR.forward, adj_mx branch) | §3.3.1 ¶1 | Eq.(13)-(14) | 结构特征生成器：多尺度图卷积 C_g = Σ α_s * A^s * G * W_s |
| TFG | `ViST/arch/blocks/Text_Encoder.py` (class TextEncoder) | §3.3.1 ¶2 | Eq.(15) | 文本特征生成器：frozen BERT-base 编码数据统计描述 → C_t |
| CVFR | `ViST/arch/vist_arch.py` (class CVFR) | §3.3.2 | Eq.(16)-(20) | 条件视觉特征重构器：CNN编码视觉输入 + GRU时序 + 文本/图条件注入 → H_v |
| BWA | `ViST/arch/vist_arch.py` (class BlockWiseCrossAttention) | §3.4 ¶1 | Eq.(21)-(22) | 块级交叉注意力：将序列分块后双向交叉注意力融合 |
| WGE | `ViST/arch/vist_arch.py` (class CrossModalFusionLayer) | §3.4 ¶2 | Eq.(23)-(25) | 加权门控增强：importance_estimator 估计模态权重 → 加权融合 |
| Prediction | `ViST/arch/vist_arch.py` (ViST.forward, conv_sampler+mlp_preditor) | §3.4.1 | Eq.(26)-(27) | 下采样卷积 + MLP 投影 → Y_hat; masked MAE loss |

## 仓库结构

```
/mnt/users/rwl/ViST/
├── basicts/              # BasicTS 框架（数据加载、训练循环、指标、scaler）
│   ├── data/             # TimeSeriesForecastingDataset
│   ├── metrics/          # MAE, MAPE, RMSE, MSE, WAPE
│   ├── runners/          # SimpleTimeSeriesForecastingRunner
│   ├── scaler/           # ZScoreScaler, MinMaxScaler
│   └── utils/            # get_regular_settings, load_adj
├── datasets/             # 数据目录（SD/GBA/GLA/ECL/Weather/ETTh2 等）
├── experiments/          # train.py, evaluate.py
├── scripts/              # data_preparation/ 各数据集预处理脚本
├── ViST/                 # 模型代码
│   ├── arch/
│   │   ├── vist_arch.py  # 主模型（STIM, CVFR, BWA, WGE, ViST）
│   │   └── blocks/       # MVE, TextEncoder, MultimodalFusion, MLP, VE, utils
│   └── ViST_GLA.py      # GLA 数据集配置示例
├── MM26_ViST/            # 论文 LaTeX 源码
├── cv_rebuttal_template/ # Rebuttal 模板（已 clone）
├── requirements.txt      # 依赖（torch 2.6, transformers 4.49, timm 1.0.15 等）
└── README.md             # 快速开始指南
```

## 训练配置（以 GLA 为例）

- INPUT_LEN = 12, OUTPUT_LEN = 12
- Optimizer: Adam, lr=0.001, weight_decay=1e-4
- Scheduler: MultiStepLR, milestones=[1,30,38,46,54,62,70,80], gamma=0.5
- Batch size: 8, Epochs: 300
- Gradient clipping: max_norm=5.0
- Curriculum learning: warm_epochs=30, cl_epochs=3
- Evaluation horizons: [3, 6, 12]
- Model: hidden_dim=256, d_temp=64, image_size=128, encoder_layers=3, n_heads=8
- LLM: bert-base-uncased, llm_dim=768, d_ff=32
