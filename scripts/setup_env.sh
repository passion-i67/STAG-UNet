#!/bin/bash
# ============================================================
# setup_env.sh — 环境安装脚本
# 
# 用法：bash scripts/setup_env.sh
# ============================================================

echo "=============================="
echo "STAG-UNet Environment Setup"
echo "=============================="

# 检测运行环境
if [ -d "/content" ]; then
    echo "Detected: Google Colab"
    ENV="colab"
elif command -v sbatch &> /dev/null; then
    echo "Detected: HPC (Slurm)"
    ENV="hpc"
else
    echo "Detected: Local machine"
    ENV="local"
fi

# ============================================================
# HPC4 专用设置
# ============================================================
if [ "$ENV" = "hpc" ]; then
    echo "Loading modules for HPC4..."
    module load anaconda3 2>/dev/null || true

    # 创建 conda 环境（如果不存在）
    if ! conda env list | grep -q "stag"; then
        echo "Creating conda environment 'stag'..."
        conda create -n stag python=3.10 -y
    fi
    
    echo "Activating conda environment..."
    source activate stag 2>/dev/null || conda activate stag

    # 安装 PyTorch（CUDA 12.4 for RTX 5880）
    echo "Installing PyTorch for CUDA 12.4..."
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

# ============================================================
# Colab 专用设置
# ============================================================
if [ "$ENV" = "colab" ]; then
    echo "Colab environment detected, PyTorch is pre-installed."
fi

# ============================================================
# 本地专用设置
# ============================================================
if [ "$ENV" = "local" ]; then
    # 创建 conda 环境（如果不存在）
    if ! conda env list 2>/dev/null | grep -q "stag"; then
        echo "Creating conda environment 'stag'..."
        conda create -n stag python=3.10 -y
    fi
    echo "Please activate the environment: conda activate stag"
    echo "Then install PyTorch matching your CUDA version from https://pytorch.org"
fi

# ============================================================
# 安装通用依赖
# ============================================================
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ============================================================
# 验证安装
# ============================================================
echo ""
echo "=============================="
echo "Verifying installation..."
echo "=============================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:    {torch.version.cuda}')

import timm
print(f'timm version:    {timm.__version__}')

import albumentations
print(f'albumentations:  {albumentations.__version__}')

import cv2
print(f'OpenCV:          {cv2.__version__}')

print()
print('All dependencies installed successfully!')
"

echo ""
echo "Setup complete!"
echo "Next step: prepare your data and run preprocessing"
echo "  python preprocess_main.py --raw_dir <path_to_BEETLE_data>"
