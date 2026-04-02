# STAG-UNet 本地 Windows 操作手册
## 你的环境：RTX 5060 笔记本 + Windows + Anaconda

---

## 第1步：环境搭建

打开 **Anaconda Prompt**（不是普通 CMD），逐行运行：

```cmd
:: 创建环境
conda create -n stag python=3.10 -y
conda activate stag

:: 安装 PyTorch（RTX 5060 用 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

:: 安装依赖
pip install timm albumentations scikit-image tensorboard tqdm tifffile opencv-python-headless scipy scikit-learn pandas matplotlib seaborn pyyaml

:: 安装 OpenSlide Python 绑定
pip install openslide-python

:: 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NO GPU')"
```

### OpenSlide Windows 安装（必须做）

1. 下载：https://github.com/openslide/openslide-bin/releases
   - 找到最新版本，下载 `openslide-bin-*-windows-x64.zip`
2. 解压到 `C:\openslide\`
3. 添加到系统 PATH（二选一）：
   - **临时**（每次开终端都要运行）：`set PATH=C:\openslide\bin;%PATH%`
   - **永久**：系统设置 → 环境变量 → Path → 新建 → 添加 `C:\openslide\bin`
4. 验证：
```cmd
python -c "import openslide; print('OK')"
```

---

## 第2步：项目代码部署

```cmd
:: 进入项目根目录
cd /d D:\BEHI5011_STAG_UNet

:: 解压项目代码（把 stag_unet_project_v2.zip 放到这个目录下）
:: 解压后应该有 D:\BEHI5011_STAG_UNet\stag_unet\ 文件夹

:: 进入项目
cd stag_unet
```

---

## 第3步：修改配置文件中的路径

用 VSCode 打开 `configs\config.py`，修改最上面的路径：

```python
DATA_ROOT = "D:/BEHI5011_STAG_UNet/beetle/data"
PROCESSED_DIR = "D:/BEHI5011_STAG_UNet/stag_unet/data/processed"
STAIN_DESC_DIR = "D:/BEHI5011_STAG_UNet/stag_unet/data/stain_descriptors"
OUTPUT_DIR = "D:/BEHI5011_STAG_UNet/stag_unet/outputs"
```

注意：Windows 路径用 `/` 不用 `\`（Python 里 `\` 是转义符）

---

## 第4步：EDA — 检查 mask 像素值

```cmd
conda activate stag
cd /d D:\BEHI5011_STAG_UNet\stag_unet
python eda_check.py
```

（eda_check.py 在下面会一起创建）

---

## 第5步：从 WSI 切 patch

```cmd
python data/preprocess_wsi.py ^
    --csv "D:/BEHI5011_STAG_UNet/beetle/data/data_overview.csv" ^
    --data_root "D:/BEHI5011_STAG_UNet/beetle/data" ^
    --output_dir "data/processed" ^
    --stain_dir "data/stain_descriptors" ^
    --val_fold 0 ^
    --max_patches 200 ^
    --no_stain_norm
```

---

## 第6步：训练

```cmd
:: 快速验证（5 epochs）
python train.py --model stag_unet --epochs 5 --batch_size 8 --lr 0.0001

:: 正式训练
python train.py --model unet --epochs 50 --batch_size 8 --lr 0.0001
python train.py --model attention_unet --epochs 50 --batch_size 8 --lr 0.0001
python train.py --model stag_unet --epochs 50 --batch_size 8 --lr 0.0001
```

---

## 第7步：测试

```cmd
python test.py --checkpoint "outputs\stag_unet_XXXXXXXX\checkpoints\best_model.pth" --output_dir "test_results"
```
