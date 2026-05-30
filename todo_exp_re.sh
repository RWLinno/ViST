#!/bin/bash
# todo_exp_re.sh — ACM MM 2026 Rebuttal 一键并行实验脚本
# GPU: 仅使用 GPU 0,1 (2x H800 140GB)
# 策略: 尽可能塞满每张卡，所有实验并行启动
#
# 显存估算 (训练模式, full model with BERT):
#   ETTh2/Weather (N<=21, bs=32): ~8GB
#   ECL (N=321, bs=16): ~7GB
#   SD (N=716, bs=8): ~5GB
#   GBA (N=2352, bs=4): ~25GB
# 每张卡 140GB，可同时跑 5-10 个实验

cd /mnt/users/rwl/ViST

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate vist

mkdir -p logs ckpts results

echo "=============================================="
echo " ViST Rebuttal — Parallel Launch on GPU 0,1"
echo " $(date)"
echo "=============================================="

# ============================================================
# GPU 0: ETTh2 长序列 x4 + Weather 长序列 x4 + adaptive_alpha SD
# 预计总占用: 8*4 + 8*4 + 5 = ~69GB
# ============================================================

CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_ETTh2_H96.py -g 0 > logs/lh_ETTh2_H96.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_ETTh2_H192.py -g 0 > logs/lh_ETTh2_H192.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_ETTh2_H336.py -g 0 > logs/lh_ETTh2_H336.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_ETTh2_H720.py -g 0 > logs/lh_ETTh2_H720.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_Weather_H96.py -g 0 > logs/lh_Weather_H96.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_Weather_H192.py -g 0 > logs/lh_Weather_H192.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_Weather_H336.py -g 0 > logs/lh_Weather_H336.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/longhorizon/ViST_Weather_H720.py -g 0 > logs/lh_Weather_H720.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python experiments/train.py -c ViST/ablation/ViST_SD_adaptive_alpha.py -g 0 > logs/ada_alpha_SD.log 2>&1 &

# ============================================================
# GPU 1: ECL 长序列 x4 + adaptive_alpha GBA + adaptive_alpha ECL
# 预计总占用: 7*4 + 25 + 7 = ~60GB
# ============================================================

CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/longhorizon/ViST_Electricity_H96.py -g 0 > logs/lh_ECL_H96.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/longhorizon/ViST_Electricity_H192.py -g 0 > logs/lh_ECL_H192.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/longhorizon/ViST_Electricity_H336.py -g 0 > logs/lh_ECL_H336.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/longhorizon/ViST_Electricity_H720.py -g 0 > logs/lh_ECL_H720.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/ablation/ViST_GBA_adaptive_alpha.py -g 0 > logs/ada_alpha_GBA.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python experiments/train.py -c ViST/ablation/ViST_Electricity_adaptive_alpha.py -g 0 > logs/ada_alpha_ECL.log 2>&1 &

echo ""
echo "=============================================="
echo " 15 experiments launched"
echo " GPU 0: 9 jobs (ETTh2x4 + Weatherx4 + ada_alpha_SD)"
echo " GPU 1: 6 jobs (ECLx4 + ada_alpha_GBA + ada_alpha_ECL)"
echo "=============================================="
echo ""
echo " NOTE: LLM TFG experiments (Qwen2.5-1.5B) skipped — model not available locally."
echo "       Download with: huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir /mnt/users/rwl/models/Qwen2.5-1.5B"
echo "       Then run: bash scripts/rebuttal/run_llm_tfg_all.sh"
echo ""
echo " Monitor:"
echo "   watch -n 30 nvidia-smi"
echo "   tail -f logs/lh_ETTh2_H96.log"
echo "   grep -c 'Epoch' logs/*.log  # check epoch progress"
echo ""
echo " Check completion:"
echo "   grep -l 'best\|Best' logs/*.log | wc -l"
echo ""
