#!/bin/bash
# ============================================================
# run_ablation.sh — 消融实验：一键训练所有变体
#
# 用法：bash scripts/run_ablation.sh
#
# 会依次训练：
#   1. U-Net (baseline)
#   2. Attention U-Net
#   3. STAG-UNet (full model)
# ============================================================

echo "=============================="
echo "Ablation Study"
echo "=============================="

EPOCHS=50
BATCH_SIZE=16
LR=0.0001

mkdir -p logs

# ---- Baseline 1: U-Net ----
echo ""
echo ">>> [1/3] Training U-Net..."
python train.py --model unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
    2>&1 | tee logs/ablation_unet.log

# ---- Baseline 2: Attention U-Net ----
echo ""
echo ">>> [2/3] Training Attention U-Net..."
python train.py --model attention_unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
    2>&1 | tee logs/ablation_attention_unet.log

# ---- Full Model: STAG-UNet ----
echo ""
echo ">>> [3/3] Training STAG-UNet..."
python train.py --model stag_unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
    2>&1 | tee logs/ablation_stag_unet.log

echo ""
echo "=============================="
echo "Ablation study complete!"
echo "Check logs/ for detailed training logs"
echo "Check outputs/ for model checkpoints and results"
echo "=============================="
