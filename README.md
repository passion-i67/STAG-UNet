# STAG-UNet: Stain-invariant Attention-Gated U-Net

Multicentric breast histopathology semantic segmentation on the BEETLE dataset.

**BEHI 5011 Final Project — HKUST, Spring 2026**

## Overview

STAG-UNet introduces stain-conditioned attention gating into the U-Net decoder. A slide-level stain descriptor is extracted from each WSI, encoded via a small MLP, and injected into every Attention Gate so that attention weights adapt to the staining style of each slide. This improves cross-center generalization on multi-institutional H&E data.

**Four-class segmentation:** Invasive Epithelium · Non-invasive Epithelium · Necrosis · Other

## Project Structure

```
stag_unet/
├── configs/config.py              # All hyperparameters
├── data/
│   ├── dataset.py                 # PyTorch Dataset & DataLoader
│   ├── preprocess_wsi_v2.py       # WSI → patch preprocessing (OpenSlide + tifffile memmap)
│   └── stain_utils.py             # Macenko normalization & stain descriptor extraction
├── models/
│   ├── unet.py                    # Baseline U-Net
│   ├── attention_unet.py          # Attention U-Net
│   └── stag_unet.py               # STAG-UNet (proposed model)
├── utils/
│   ├── losses.py                  # Dice + Focal combined loss
│   ├── metrics.py                 # Dice, IoU, HD95
│   └── visualize.py               # Prediction & attention map visualization
├── scripts/
│   ├── setup_env.sh               # Environment setup
│   ├── preprocess.sh              # Data preprocessing
│   ├── train.sh                   # Training (local & HPC4 Slurm)
│   ├── test.sh                    # Evaluation
│   └── run_ablation.sh            # Run all ablation variants
├── train.py                       # Training entry point
├── test.py                        # Evaluation entry point
├── eda_check.py                   # Dataset EDA script
└── requirements.txt               # Python dependencies
```

## Environment Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support
- Anaconda/Miniconda
- OpenSlide ([download binaries](https://openslide.org/download/))

### Installation

```bash
# Create environment
conda create -n stag python=3.10 -y
conda activate stag

# Install PyTorch (choose your CUDA version)
# CUDA 12.8 (RTX 50-series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.4 (HPC4 RTX 5880):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Windows only: add OpenSlide to PATH
set PATH=C:\openslide\bin;%PATH%
```

### Verify

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import openslide; print('OpenSlide OK')"
```

## Data Preparation

### 1. Download BEETLE Dataset

Place the BEETLE dataset so that the structure looks like:

```
data/raw/
├── data_overview.csv
├── images/
│   ├── development/wsis/*.tif    # WSI images
│   └── evaluation/rois/*.png     # ROI images
└── annotations/
    └── masks/*.tif               # Annotation masks
```

### 2. Preprocess: WSI → Patches

```bash
python data/preprocess_wsi_v2.py \
    --csv data/raw/data_overview.csv \
    --data_root data/raw \
    --output_dir data/processed \
    --stain_dir data/stain_descriptors \
    --val_fold 0 \
    --max_patches 500 \
    --no_stain_norm
```

### 3. Filter Imbalanced Data

The raw patches are ~96% Other class. Filter to keep foreground-containing patches:

```bash
python -c "
import glob, cv2, numpy as np, os, shutil
for split in ['train', 'val']:
    imgs = sorted(glob.glob(f'data/processed/{split}/images/*.png'))
    out_img = f'data/processed_filtered/{split}/images'
    out_msk = f'data/processed_filtered/{split}/masks'
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_msk, exist_ok=True)
    kept, other_kept = 0, 0
    for p in imgs:
        name = os.path.basename(p)
        m = cv2.imread(f'data/processed/{split}/masks/{name}', 0)
        if any(v in np.unique(m) for v in [0,1,2]):
            shutil.copy2(p, f'{out_img}/{name}'); shutil.copy2(f'data/processed/{split}/masks/{name}', f'{out_msk}/{name}'); kept += 1
        elif other_kept < 2000:
            shutil.copy2(p, f'{out_img}/{name}'); shutil.copy2(f'data/processed/{split}/masks/{name}', f'{out_msk}/{name}'); kept += 1; other_kept += 1
    print(f'{split}: {kept}')
"
```

## Training

```bash
# Single model
python train.py --model stag_unet --epochs 50 --batch_size 8 --lr 0.0001

# Ablation study (all three models)
python train.py --model unet --epochs 50 --batch_size 8 --lr 0.0001
python train.py --model attention_unet --epochs 50 --batch_size 8 --lr 0.0001
python train.py --model stag_unet --epochs 50 --batch_size 8 --lr 0.0001
```

### HPC4 (Slurm)

```bash
sbatch scripts/train.sh
```

### Monitor

```bash
tensorboard --logdir outputs/
```

## Evaluation

```bash
python test.py --checkpoint outputs/<experiment>/checkpoints/best_model.pth --output_dir test_results
```

## Results

### Ablation Study

| Model | Mean Dice | Mean IoU |
|-------|-----------|----------|
| U-Net (Baseline) | 0.4357 | — |
| Attention U-Net | 0.4459 | — |
| **STAG-UNet (Ours)** | **0.5638** | **0.3931** |

### STAG-UNet Per-class Results

| Class | Dice | IoU | HD95 |
|-------|------|-----|------|
| Invasive Epithelium | 0.5573 | 0.3863 | 62.44 |
| Non-invasive Epithelium | 0.5471 | 0.3765 | 11.78 |
| Necrosis | 0.5348 | 0.3650 | 23.84 |
| Other | 0.6157 | 0.4447 | 91.79 |

## Configuration

All hyperparameters are in `configs/config.py`. Key settings:

| Parameter | Value |
|-----------|-------|
| Patch size | 256×256 |
| Encoder | EfficientNet-B0 (ImageNet pretrained) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | Cosine Annealing |
| Loss | Dice + Focal (class weights: [1, 1, 2, 0.5]) |
| Augmentation | HFlip, VFlip, Rotate90, ColorJitter |



## License

This project is for academic purposes (BEHI 5011 course project).
