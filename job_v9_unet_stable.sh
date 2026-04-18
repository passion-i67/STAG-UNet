#!/bin/bash
#SBATCH --job-name=unet_v9
#SBATCH --account=mscbehi5011hpc4
#SBATCH --partition=gpu-rtx5880
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zwangot/stag_unet/logs/unet_v9_%j.log
#SBATCH --error=/home/zwangot/stag_unet/logs/unet_v9_%j.err

set -euo pipefail
module purge
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate stag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$HOME/stag_unet}"
DATA_DIR="${DATA_DIR:-/scratch/zwangot/processed_v2}"
STAIN_DIR="${STAIN_DIR:-/scratch/zwangot/stain_descriptors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/stag_unet/outputs}"
LAUNCHER="${LAUNCHER:-$REPO_ROOT/run_v9_baseline_stable.py}"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

echo "=============================="
echo "U-Net v9 baseline (stable)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_DIR"
echo "Stain: $STAIN_DIR"
echo "Out:  $OUTPUT_ROOT"
echo "=============================="
python -V
nvidia-smi

python "$LAUNCHER" \
  --model unet \
  --repo_root "$REPO_ROOT" \
  --data_dir "$DATA_DIR" \
  --stain_dir "$STAIN_DIR" \
  --output_root "$OUTPUT_ROOT" \
  --epochs 80 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_workers 8 \
  --patience 20

echo "U-Net v9 baseline finished at $(date)"
