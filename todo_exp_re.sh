#!/bin/bash
# todo_exp_re.sh — ACM MM 2026 Rebuttal 补实验总调度脚本
# 所有脚本须支持 --seed {2024,2025,2026} 三次取均值
# 完成后自动推 HF: ViST_ckpts/<exp>/
#
# 资源：8×A6000，每条命令在 run_*.sh 内显式指定 CUDA_VISIBLE_DEVICES
# 优先级：P0→P1→P2，同 P 级显存大者先行
# 单任务 wall-clock > 24h 自动 timeout 并 dump 中间 ckpt 推 HF

set -e
cd "$(dirname "$0")"

mkdir -p logs ckpts results

echo "=============================================="
echo " ViST Rebuttal Experiments — $(date)"
echo "=============================================="

# ============ P0: 长序列 TS Benchmark (R-XfSt-d) ============
# Lookback=96, horizons={96,192,336,720}, metrics MSE/MAE
echo "[P0] Starting long-horizon experiments..."
nohup bash scripts/rebuttal/run_longhorizon_etth2.sh   > logs/run_longhorizon_etth2.log   2>&1 &
nohup bash scripts/rebuttal/run_longhorizon_ecl.sh     > logs/run_longhorizon_ecl.log     2>&1 &
nohup bash scripts/rebuttal/run_longhorizon_weather.sh > logs/run_longhorizon_weather.log  2>&1 &

# ============ P0: 新增多模态 Baselines (R-ghvf-a, R-XfSt-b) ============
echo "[P0] Starting baseline experiments..."
nohup bash scripts/rebuttal/run_baseline_timexl.sh   > logs/run_baseline_timexl.log   2>&1 &
nohup bash scripts/rebuttal/run_baseline_lvmmts.sh   > logs/run_baseline_lvmmts.log   2>&1 &
nohup bash scripts/rebuttal/run_baseline_timevlm.sh  > logs/run_baseline_timevlm.log  2>&1 &
nohup bash scripts/rebuttal/run_baseline_timesnet.sh > logs/run_baseline_timesnet.log  2>&1 &

# ============ P0: Adaptive α 消融 (R-oLMK-a) ============
echo "[P0] Starting adaptive alpha experiments..."
nohup bash scripts/rebuttal/run_adaptive_alpha_sd.sh  > logs/run_adaptive_alpha_sd.log  2>&1 &
nohup bash scripts/rebuttal/run_adaptive_alpha_gba.sh > logs/run_adaptive_alpha_gba.log 2>&1 &
nohup bash scripts/rebuttal/run_adaptive_alpha_ecl.sh > logs/run_adaptive_alpha_ecl.log 2>&1 &

# ============ P0: LLM-augmented TFG (R-oLMK-b) ============
echo "[P0] Starting LLM TFG experiments..."
nohup bash scripts/rebuttal/run_llm_tfg_sd.sh    > logs/run_llm_tfg_sd.log    2>&1 &
nohup bash scripts/rebuttal/run_llm_tfg_ecl.sh   > logs/run_llm_tfg_ecl.log   2>&1 &
nohup bash scripts/rebuttal/run_llm_tfg_etth2.sh > logs/run_llm_tfg_etth2.log 2>&1 &

# ============ P1: 公平比较配置 dump (R-XfSt-c) ============
echo "[P1] Dumping fair comparison table..."
nohup bash scripts/rebuttal/dump_fair_comparison_table.sh > logs/run_fair_comp.log 2>&1 &

# ============ P1: Multi-seed 鲁棒性 (R-XfSt-c) ============
echo "[P1] Starting multi-seed experiments..."
nohup bash scripts/rebuttal/run_multiseed_sd.sh  > logs/run_multiseed_sd.log  2>&1 &
nohup bash scripts/rebuttal/run_multiseed_gba.sh > logs/run_multiseed_gba.log 2>&1 &

# ============ P2: 架构消融 (R-fbbb-c, 理论支撑) ============
echo "[P2] Starting architecture ablation..."
nohup bash scripts/rebuttal/run_arch_ablation.sh > logs/run_arch_ablation.log 2>&1 &

echo ""
echo "=============================================="
echo " All experiments launched. Monitor with:"
echo "   tail -f logs/run_*.log"
echo "   jobs -l"
echo "=============================================="
echo ""
echo "After completion, run:"
echo "   python scripts/rebuttal/collect_results.py"
echo "   to generate results/RESULTS.csv and results/REBUTTAL_TABLES.md"
