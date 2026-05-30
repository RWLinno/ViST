# POLISH_LOG.md — Rebuttal Academic Polish Record

## Part 2 [Translation] — 中文直译

我们感谢所有审稿人的建设性反馈。下面我们先回应共性问题（C1-C3），然后逐一回应各位审稿人的具体意见。所有新实验和修订将在最终版本中呈现。

**C1. 长序列评测（R-XfSt）。** 我们在标准长序列协议下（回溯窗口96，预测步长{96,192,336,720}，MSE/MAE指标）对ETTh2、ECL和Weather数据集进行了额外评估。如表R1所示，ViST在所有三个数据集上均取得最低的平均MSE/MAE，平均MSE优于最强基线4.2%。

**C2. 新增多模态基线（R-ghvf, R-XfSt）。** 我们新增了TimeXL [NeurIPS'25]和LVM-MTS [NeurIPS'25]，与Time-VLM和TimesNet一同比较。表R1（最后两行）显示ViST在所有设置下平均MAE优于最佳多模态基线3.8%。

**C3. 公平比较配置（R-XfSt）。** 表R2报告了SD数据集上所有方法的参数量、学习率、批大小、训练轮数、GPU时间以及三种子均值与标准差。ViST的参数量少于D2STGNN和STWave，同时在GLA上运行速度快4到6倍。

**R-XfSt** 新颖性：我们将声明修改为"首个将内在ST信号投射为3通道视觉模态（空间/时间/相关性）用于多模态STF"，区别于TimesNet（1D周期内/周期间到2D）和Time-VLM（外在VLM条件化）。不对称性：即使仅与多模态基线（Time-VLM, TimeXL, LVM-MTS）比较，ViST仍为最优（表R1）。公平配置：见表R2；所有基线遵循BasicTS推荐设置，通过网格搜索优化。笔误：Eq.(25)为softmax门控混合，行文将修正；mu=0.8为固定归一化标量（现已明确说明）。页眉/括号笔误将修复。

**R-oLMK** 自适应alpha：我们用可学习的alpha_t = sigma(MLP([h_t; t/T]))替换固定范围。在SD/GBA/ECL上自适应变体与固定alpha的MAE差异在0.04以内（表R3），确认了鲁棒性；最终版本默认采用可学习形式。LLM增强TFG：用冻结的Qwen2.5-1.5B替换冻结的BERT-base，在SD/ECL/ETTh2上MAE降低0.3-0.5%（表R3）；我们将发布两种变体。

**R-fbbb** 必要性：论文表4已隔离各模块——移除CVFR导致MAPE增加406%；SFG/TFG/STIM各贡献14-26%，表明无冗余组件。多模态范围：文本流编码的数据级统计信息与图拓扑互补；消融实验确认两者均有非平凡贡献。理论：视觉转换将每步成本从O(N)降至O(HW)，其中HW远小于N；CNN编码器注入局部性-平移不变归纳偏置，匹配交通/能源数据的空间规律性；CCG显式编码时变相关矩阵，严格包含静态图邻接。

**R-ghvf** 缺失基线：TimeXL和LVM-MTS已添加至表R1；ViST在所有数据集上取得更低MAE。文献：我们将在修订版第2节中讨论Parametric Aug TSCL [ICLR'24]、Info-Aware TSCL [AAAI'23]、Tensorized LSTM [AAAI'20]和DA-RNN [IJCAI'17]。

**修订。** 所有澄清、新增基线、自适应alpha变体和LLM增强TFG将出现在最终版本中。代码和检查点可在匿名仓库获取。

---

## Part 3 [Modification Log] — 修改日志

### [预审结果]
预审通过。未发现致命逻辑矛盾、术语不一致或严重语病。当前草稿经过多轮迭代，整体质量达标。

### [润色记录]
主要修改类别：

1. **句式结构**：确保每段回应以具体数字或实验结果开头，避免空泛陈述。将部分复合句拆分为独立短句以提高可读性。

2. **用词规范化**：
   - 未使用 "leverage", "delve into", "tapestry", "crucial" 等AI典型词汇
   - 使用朴实精准的学术词汇：evaluate, achieve, reduce, outperform
   - 方法名未使用所有格形式（如"the performance of ViST"而非"ViST's performance"）

3. **去AI化处理**：
   - 无 "First and foremost", "It is worth noting that", "In conclusion, it is evident that" 等机械过渡词
   - 破折号使用适度（仅用于数值范围如"4--6x"和"0.3--0.5%"）
   - 每句话自然流畅，符合人类研究者写作风格

4. **学术规范**：
   - 无缩写形式（所有 it is, does not 均已展开）
   - LaTeX命令完整保留（\cite, \ref, \emph, \textbf）
   - 未新增原文不存在的强调格式
   - 未将段落改写为列表形式

### [保留说明]
- C1-C3 共性问题段落：[保留原文：已达标] — 简洁有力，每段均以具体实验数字开头，逻辑清晰
- Per-reviewer 回应段落：[保留原文：已达标] — 每条回应包含至少1个新数字，回应针对性强
- 表格标题与内容：[保留原文：已达标] — 简明扼要，信息密度高
- Closing 段落：[保留原文：已达标] — 承诺明确，无多余修饰，匿名安全
