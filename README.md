# STAG-UNet: Stain-Invariant Attention-Gated U-Net for Breast Histopathology Segmentation

**Course:** BEHI 5011 – Artificial Intelligence and Medical Imaging, HKUST Spring 2026  
**Student:** Wang Ziyu (ITSC: zwangot)  
**Task:** Four-class semantic segmentation of breast H&E whole-slide images on the BEETLE dataset  
**Classes:** Invasive Epithelium (IE) · Non-invasive Epithelium (NE) · Necrosis (NC) · Other (OT)  
**Best Result:** Mean Dice = **0.6115** (STAG-UNet v9, BEETLE val set)

---

## Overview

STAG-UNet introduces a **Stain-Aware Gating (STAG)** module that conditions attention gate computations on slide-level stain descriptors derived from Macenko stain decomposition. For each WSI, a 6-dimensional stain descriptor (H&E stain matrix unit vectors) is extracted and mapped through a shared MLP to produce FiLM modulation parameters (γ, β), which modulate skip-connection attention across all four encoder levels of an EfficientNet-B0 backbone. This makes the model robust to stain appearance variation arising from multi-center H&E slides.

In v9, STAG-UNet is further enhanced with:
- **CBAM** (Channel + Spatial attention) applied after each decoder block
- **Cosine Annealing Warm Restarts** (T₀=10, T_mult=2) for better optimization
- **Strong data augmentation** including elastic transforms, random resized crop, and HSV jitter

---

## Project Structure

```
stag_unet/
├── configs/
│   └── config.py                    # All hyperparameters and data paths
├── data/
│   ├── dataset.py                   # BreastPathologyDataset + augmentation pipeline
│   └── preprocess_wsi.py            # WSI patch extraction + stain descriptor computation
├── models/
│   ├── stag_unet.py                 # STAG-UNet (proposed method, with CBAM)
│   ├── attention_unet.py            # Attention U-Net baseline
│   ├── unet.py                      # U-Net baseline
│   └── cbam.py                      # CBAM module (Woo et al., ECCV 2018)
├── utils/
│   └── losses.py                    # CombinedLoss = DiceLoss + FocalLoss
├── train.py                         # Training entry point
├── evaluate.py                      # Evaluation script (Dice, IoU per class)
├── run.sh                           # One-click: train all 3 models sequentially
├── job_v9_stag.sh                   # HPC4 Slurm job — STAG-UNet v9
├── job_v9_unet_stable.sh            # HPC4 Slurm job — U-Net baseline (v9 strategy)
├── job_v9_attention_unet_stable.sh  # HPC4 Slurm job — Attention U-Net baseline (v9 strategy)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Environment Setup

### Option 1: HPC4 (HKUST, recommended)

```bash
# Activate existing environment
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate stag
```

### Option 2: Create fresh environment

```bash
conda create -n stag python=3.11 -y
conda activate stag

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Segmentation models
pip install segmentation-models-pytorch efficientnet-pytorch timm

# Image processing
pip install opencv-python-headless albumentations scikit-image openslide-python tifffile

# Scientific computing
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm
```

### Hardware requirements

| Setting | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 24 GB+ |
| CPU RAM | 16 GB | 32 GB |
| Disk (data) | 50 GB (scratch) | 100 GB |

---

## Data Preparation

### Step 1: Obtain the BEETLE dataset

Request access from the official BEETLE challenge page. Place the downloaded files as:

```
/path/to/BEETLE/
├── data_overview.csv        # WSI metadata with fold assignments
├── images/                  # H&E whole-slide images (.tiff)
└── masks/                   # Annotation masks (.tiff)
```

Class encoding in masks: `0=IE, 1=NE, 2=NC, 3=OT`

### Step 2: Edit paths in `configs/config.py`

```python
# HPC4 example
DATA_ROOT    = "/scratch/zwangot/beetle_data"
PROCESSED_DIR = "/scratch/zwangot/processed_v2"
STAIN_DESC_DIR = "/scratch/zwangot/stain_descriptors"
OUTPUT_DIR   = "/home/zwangot/stag_unet/outputs"
CSV_PATH     = "/scratch/zwangot/beetle_data/data_overview.csv"
```

### Step 3: Run preprocessing

```bash
python data/preprocess_wsi.py \
  --csv /scratch/zwangot/beetle_data/data_overview.csv \
  --data_root /scratch/zwangot/beetle_data \
  --output_dir /scratch/zwangot/processed_v2 \
  --stain_dir /scratch/zwangot/stain_descriptors \
  --val_fold 0 \
  --max_patches 500
```

Expected output (fold 0 split):
- Training patches: ~17,327
- Validation patches: ~5,723
- Stain descriptors: `stain_descriptors/stain_descriptors.npy`

---

## Training

### Quick start (local, all 3 models sequentially)

```bash
bash run.sh
```

### Train individual models

```bash
# STAG-UNet (proposed method)
python train.py --model stag_unet --epochs 80 --batch_size 32 --lr 3e-4

# U-Net baseline
python train.py --model unet --epochs 80 --batch_size 32 --lr 3e-4

# Attention U-Net baseline
python train.py --model attention_unet --epochs 80 --batch_size 32 --lr 3e-4
```

### On HPC4 (Slurm)

```bash
sbatch job_v9_stag.sh                   # STAG-UNet
sbatch job_v9_unet_stable.sh            # U-Net baseline
sbatch job_v9_attention_unet_stable.sh  # Attention U-Net baseline
```

Monitor progress:
```bash
squeue -u zwangot
tail -f logs/v9_<JOBID>.err   # tqdm progress (stderr)
grep "New best\|Early stop" logs/v9_<JOBID>.log
```

### Training hyperparameters (v9)

| Parameter | Value |
|---|---|
| Optimizer | AdamW, weight_decay=1e-4 |
| Learning rate | 3e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2, η_min=1e-6) |
| Loss | DiceLoss + FocalLoss (λ=1.0 each) |
| Class weights | [1.0, 1.0, 2.0, 0.5] for IE/NE/NC/OT |
| Batch size | 32 |
| Max epochs | 80 |
| Early stopping patience | 20 |
| Patch size | 256 × 256 px |

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint outputs/stag_unet_20260416_142314/checkpoints/best_model.pth \
  --model stag_unet \
  --data_dir /scratch/zwangot/processed_v2 \
  --stain_dir /scratch/zwangot/stain_descriptors
```

Output: per-class Dice and IoU, saved to `outputs/eval_results.json`.

### Best checkpoint

```
outputs/stag_unet_20260416_142314/checkpoints/best_model.pth
```

(70 MB, includes CBAM parameters)

---

## Results

### Main comparison (BEETLE val set, fold 0)

| Model | Mean Dice | IE | NE | NC | OT | Mean IoU |
|---|---|---|---|---|---|---|
| U-Net | 0.5507 | 0.5910 | 0.5101 | 0.4399 | 0.6618 | 0.3846 |
| Attention U-Net | 0.5433 | 0.5737 | 0.5157 | 0.4588 | 0.6251 | 0.3755 |
| **STAG-UNet v9 (Ours)** | **0.6115** | **0.6089** | **0.6302** | **0.5036** | **0.7033** | **0.4442** |

STAG-UNet v9 outperforms U-Net by **+11.1% mDice** and Attention U-Net by **+12.5% mDice**.

### Ablation (256px patch experiments)

| Round | Config | Mean Dice | Notes |
|---|---|---|---|
| Round 7 | Baseline STAG-UNet, batch=32 | 0.6051 | Without CBAM |
| Round 8a | 512px patches, batch=8 | 0.5973 | Resolution ablation |
| **Round 9 (v9)** | **+CBAM +WarmRestart +StrongAug** | **0.6115** | **Best** |

---

## Key Design Notes

**Why STAG?** Multi-center breast H&E slides exhibit significant stain variation. Standard normalization discards stain-specific signal. STAG instead encodes stain appearance as a 6D descriptor and uses it to modulate *where* the model attends, not *what* features it sees.

**Why CBAM?** FiLM (STAG) conditions on slide-level stain information; CBAM provides local channel/spatial self-attention within decoder features. They are complementary and operate on different feature spaces.

**`torch.compile` is disabled** — causes CUDAGraph tensor overwriting errors during backprop with this architecture. The relevant lines are commented out in `train.py`.

---

## Repository

GitHub: https://github.com/passion-i67/STAG-UNet

---

## AI Usage Statement

Claude (Anthropic) was used to assist with code debugging, HPC job script generation, report writing/proofreading, and architecture diagram generation. All experimental design, model implementation decisions, and analysis are the author's own work.
