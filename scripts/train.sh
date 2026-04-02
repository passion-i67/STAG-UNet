#!/bin/bash
# ============================================================
# train.sh — 训练脚本
#
# 本地运行：  bash scripts/train.sh
# HPC4 提交： sbatch scripts/train.sh
# ============================================================

# ---- HPC4 Slurm 配置（本地运行时会被忽略）----
#SBATCH --job-name=stag_unet
#SBATCH --partition=gpu-rtx5880
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# 创建日志目录
mkdir -p logs

# HPC4 环境加载
if command -v module &> /dev/null; then
    module load anaconda3 2>/dev/null || true
    source activate stag 2>/dev/null || conda activate stag 2>/dev/null || true
fi

echo "=============================="
echo "STAG-UNet Training"
echo "=============================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# 显示 GPU 信息
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

# ============================================================
# 训练配置（在这里修改参数）
# ============================================================
MODEL="stag_unet"      # 可选: unet, attention_unet, stag_unet
EPOCHS=50
BATCH_SIZE=16
LR=0.0001

echo ""
echo "Model:      $MODEL"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LR"
echo ""

# ============================================================
# 运行训练
# ============================================================
python train.py \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo ""
echo "Training finished at $(date)"
