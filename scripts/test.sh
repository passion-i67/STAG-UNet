#!/bin/bash
# ============================================================
# test.sh — 测试脚本
#
# 用法：bash scripts/test.sh <checkpoint_path>
# 例如：bash scripts/test.sh outputs/stag_unet_20260401/checkpoints/best_model.pth
# ============================================================

#SBATCH --job-name=stag_test
#SBATCH --partition=gpu-rtx5880
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_%j.log

# HPC4 环境
if command -v module &> /dev/null; then
    module load anaconda3 2>/dev/null || true
    source activate stag 2>/dev/null || conda activate stag 2>/dev/null || true
fi

# 参数
CHECKPOINT="${1:-outputs/latest/checkpoints/best_model.pth}"
OUTPUT_DIR="${2:-test_results}"

echo "=============================="
echo "STAG-UNet Testing"
echo "=============================="
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "Usage: bash scripts/test.sh <path_to_checkpoint.pth>"
    exit 1
fi

python test.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Testing complete! Results in: $OUTPUT_DIR"
