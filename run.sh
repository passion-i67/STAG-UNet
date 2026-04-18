#!/bin/bash
# ============================================================
# run.sh — STAG-UNet one-click train + evaluate
# Usage: bash run.sh
# ============================================================

set -e

PROCESSED_DIR="/scratch/zwangot/processed_v2"
STAIN_DIR="/scratch/zwangot/stain_descriptors"
EPOCHS=80
BATCH_SIZE=32
LR=3e-4

echo "=============================="
echo "STAG-UNet Training Pipeline"
echo "=============================="

mkdir -p logs

echo ""
echo ">>> [1/3] Training U-Net baseline..."
python train.py --model unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  2>&1 | tee logs/run_unet.log

echo ""
echo ">>> [2/3] Training Attention U-Net baseline..."
python train.py --model attention_unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  2>&1 | tee logs/run_attention_unet.log

echo ""
echo ">>> [3/3] Training STAG-UNet (proposed)..."
python train.py --model stag_unet --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR \
  2>&1 | tee logs/run_stag_unet.log

echo ""
echo "=============================="
echo "Training complete! Check outputs/ for checkpoints."
echo "=============================="
