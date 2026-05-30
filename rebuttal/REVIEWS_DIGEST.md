# ViST Reviews Digest (ACM MM 2026 Submission #99)

## 总览

| Reviewer | Rating | Confidence | 倾向 | P0 必答 | P1 次要 |
|----------|--------|------------|------|---------|---------|
| XfSt | 3 (Borderline) | 3 (Knowledgeable) | 偏负 | 4 条 | 4 条 |
| oLMK | 5 (Accept) | 3 (Knowledgeable) | 正面 | 2 条 | 0 条 |
| fbbb | 4 (Weak Accept) | 2 (Familiar) | 偏正 | 3 条 | 0 条 |
| ghvf | 3 (Borderline) | 4 (Expert) | 偏负 | 1 条 | 1 条 |

**当前分数**: 3/5/4/3 → 均值 3.75，需要翻转 XfSt 或 ghvf 至少一位。

---

## Reviewer XfSt (Rating 3, Conf 3)

### Strengths
1. 大规模实证评估展示了强可扩展性（多个 baseline OOM 而 ViST 成功训练）
2. 效率声明有具体测量支撑（训练时间、推理时间、内存）
3. 消融实验有意义，清晰展示关键组件重要性（特别是 CVFR）
4. 多视角框架概念连贯，通道角色定义清晰，有定性可视化支撑
5. 良好的可复现性意图（标准化 benchmark 配置 + 匿名代码）

### Weaknesses (P0)
- **(a) Novelty 过强**: "first attempt" 声明难以成立，因为已有 TimesNet (1D→2D), Time-VLM (VLM conditioning), [Ni'25] survey 等
- **(b) 比较不对称**: ViST 使用多模态条件（text+graph），而许多 baseline (PatchTST/iTransformer/DLinear) 是单模态；Table 4 显示去掉 TFG/SFG 性能大幅下降，说明增益可能来自条件通道而非视觉转换
- **(c) 公平比较未证实**: 未报告 per-baseline 的 epochs, lr, param counts, search protocol, seeds, early-stopping, compute budgets
- **(d) 评估协议不符长序列惯例**: 12→12 步，horizons {3,6,12}；标准长序列 TS benchmark 使用 lookback 96/336/512, horizons 96/192/336/720, MSE/MAE

### Weaknesses (P1 - Minor)
- Header/copyright 年份不一致（"Conference '17" vs 2025 引用）
- §3.4 prose 说 "H_g 是跨维度平均"，但 Eq.(25) 定义为 softmax-gated linear blend → 矛盾
- 引言 "[72]" 缺左括号
- Eq.(8) μ 未定义其具体值或可学习性

---

## Reviewer oLMK (Rating 5, Conf 3)

### Strengths
1. 高度创新：直接将原始 ST 数值数据转换为低维视觉空间，优雅解决语义鸿沟
2. 计算复杂度与节点数解耦：O(H×W) vs O(N)，大规模数据集上显著更快
3. 6 个真实数据集上达到 SOTA，比 DSTAGNN 平均 MAE 提升 19.4%

### Weaknesses (P0)
- **(a) SCG 中 α∈[0.15,0.3] 非自适应**: 手动设置的经验规则可能限制模型在不同拓扑结构数据集上的灵活性和泛化能力
- **(b) TFG 仅用 frozen BERT**: 未充分利用现代 LLM 的深层语义推理能力进行跨模态对齐

---

## Reviewer fbbb (Rating 4, Conf 2)

### Strengths
1. 将时空预测转化为视觉表示学习问题，视角新颖
2. 框架结构良好，集成了视觉转换、条件重构和跨模态融合
3. 实验评估全面（大规模 benchmark、消融、效率分析）
4. 大规模数据集上实现强预测性能和计算效率

### Weaknesses (P0)
- **(a) 模块必要性论证不足**: 框架较复杂，某些模块的必要性需要更好的论证
- **(b) "Multi-modal" 偏弱**: 文本模态主要是统计描述，不是真正的语义丰富的多模态
- **(c) 缺乏理论分析**: 为什么视觉转换比传统图建模更有效？缺少理论支撑

---

## Reviewer ghvf (Rating 3, Conf 4)

### Strengths
1. 论文结构良好，易于理解
2. 想法合理且有前景，视觉增强 ST 预测可进行多视角信号融合
3. 门控机制有效选择正确通道，贡献非平凡
4. 大量实验验证有效性

### Weaknesses (P0)
- **(a) 缺少重要 baseline**: 必须补充 **TimeXL** (NeurIPS'25) 和 **Multi-Modal View Enhanced LVM for LTTSF** (NeurIPS'25)

### Weaknesses (P1)
- 建议讨论的文献：
  - Parametric Augmentation for Time Series Contrastive Learning (ICLR'24)
  - Time Series Contrastive Learning with Information-Aware Augmentations (AAAI'23)
  - Tensorized LSTM with Adaptive Shared Memory (AAAI'20)
  - A Dual-Stage Attention-Based RNN for Time Series Prediction (IJCAI'17)

---

## 优先级排序

### P0 实验（必须用数据回应）
1. 长序列 96/192/336/720 评测 (R-XfSt-d) — 消解 SOTA 范围质疑
2. 补 TimeXL + LVM-MTS baseline (R-ghvf-a, R-XfSt-b) — 消解 baseline 缺失
3. Adaptive α 消融 (R-oLMK-a) — 证明鲁棒性
4. LLM 替换 BERT (R-oLMK-b) — 展示可扩展性

### P1 澄清（文字回应 + 承诺修订）
5. 参数量/训练配置表 (R-XfSt-c)
6. Novelty 重述 (R-XfSt-a)
7. 模块必要性 → 指向 Table 4 (R-fbbb-a)
8. 理论直觉 (R-fbbb-c)

### P2 文字修订（承诺 camera-ready）
9. Eq.(25) 描述修正、μ 定义、[72] 括号、header 年份
10. 文献综述补全 (R-ghvf-b)
