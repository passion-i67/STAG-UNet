#!/bin/bash
#SBATCH --job-name=stag_v9
#SBATCH --partition=gpu-rtx5880
#SBATCH --account=mscbehi5011hpc4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v9_%j.log
#SBATCH --error=logs/v9_%j.err

# ============================================================
# STAG-UNet v9: 三项改进 (Cosine Warm Restarts + 强数据增强 + CBAM)
# Baseline: Round 7 (processed_v2, 256px, batch=32), Dice=0.6051
# Target: 突破 val loss 平台, Dice > 0.62
# ============================================================

echo "==========================================="
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "==========================================="

cd $HOME/stag_unet

# 激活环境
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate stag

# 打印 GPU 信息
nvidia-smi

# 使用 Round 7 的成功配置: processed_v2 (256px), batch=32
# 改动只在模型架构 / 增强 / scheduler, 数据完全一致
python train.py \
    --model stag_unet \
    --epochs 80 \
    --batch_size 32 \
    --lr 3e-4

echo ""
echo "==========================================="
echo "Finished at $(date)"
echo "==========================================="
