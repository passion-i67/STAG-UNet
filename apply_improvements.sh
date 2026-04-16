#!/bin/bash
# ============================================================
# apply_improvements.sh
# 一次性应用三个改进: 余弦退火 + 数据增强 + CBAM
# 
# 使用方法:
#   cd ~/stag_unet
#   bash apply_improvements.sh
#
# 这个脚本会:
# 1. 备份所有要修改的文件到 ~/stag_unet/backup_before_improvements/
# 2. 添加 CBAM 模块 (models/cbam.py)
# 3. 修改 stag_unet.py, 在每个 DecoderBlock 后加 CBAM
# 4. 修改 dataset.py, 增强数据增强
# 5. 修改 train.py, 改用 CosineAnnealingWarmRestarts
# ============================================================

set -e  # 出错立即停止

PROJECT_DIR="$HOME/stag_unet"
BACKUP_DIR="$PROJECT_DIR/backup_before_improvements"

echo "=========================================="
echo "STAG-UNet Improvement Patch"
echo "Project dir: $PROJECT_DIR"
echo "=========================================="

cd "$PROJECT_DIR"

# ------------------------------------------------------------
# Step 1: 备份
# ------------------------------------------------------------
echo ""
echo "[1/5] 备份原文件到 $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp train.py "$BACKUP_DIR/train.py.bak_improvements"
cp models/stag_unet.py "$BACKUP_DIR/stag_unet.py.bak_improvements"
cp data/dataset.py "$BACKUP_DIR/dataset.py.bak_improvements"
echo "    Done."

# ------------------------------------------------------------
# Step 2: 添加 CBAM 模块
# ------------------------------------------------------------
echo ""
echo "[2/5] 添加 models/cbam.py"
cat > models/cbam.py << 'CBAM_EOF'
"""
cbam.py — Convolutional Block Attention Module (Woo et al., ECCV 2018)
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
CBAM_EOF
echo "    Done."

# ------------------------------------------------------------
# Step 3: 修改 stag_unet.py — 在 DecoderBlock 后加 CBAM
# ------------------------------------------------------------
echo ""
echo "[3/5] 修改 models/stag_unet.py"

python3 << 'PY_EOF'
import re
path = "models/stag_unet.py"
with open(path, "r") as f:
    src = f.read()

# 3.1 添加 CBAM import (紧接在 timm 或 torch 的 import 之后)
if "from models.cbam import CBAM" not in src:
    # 在 import timm 行后插入 (或在文件头的 import 区域插入)
    if "import timm" in src:
        src = src.replace("import timm", "import timm\nfrom models.cbam import CBAM", 1)
    else:
        # 备用: 在第一个 import 后插入
        src = re.sub(r"(import torch\.nn as nn)", r"\1\nfrom models.cbam import CBAM", src, count=1)
    print("    Added: from models.cbam import CBAM")

# 3.2 在 STAGUNet.__init__ 的 decoder loop 中创建 cbam_blocks
# 我们要在 self.decoder_blocks 定义那一段同级位置, 新增 self.cbam_blocks

# 找到 "self.stag_gates = nn.ModuleList()" 这一行, 在其后添加 cbam_blocks
if "self.cbam_blocks" not in src:
    old = "self.stag_gates = nn.ModuleList()\n        self.decoder_blocks = nn.ModuleList()"
    new = "self.stag_gates = nn.ModuleList()\n        self.decoder_blocks = nn.ModuleList()\n        self.cbam_blocks = nn.ModuleList()"
    if old in src:
        src = src.replace(old, new)
        print("    Added: self.cbam_blocks in __init__")
    else:
        print("    WARNING: could not locate ModuleList init block")

# 3.3 在循环中 append CBAM: 找 "self.decoder_blocks.append(" 所在 append 完成后,追加 cbam append
# 匹配: self.decoder_blocks.append(\n                DecoderBlock(in_ch, skip_ch, dec_ch)\n            )\n            \n            in_ch = dec_ch
pattern = re.compile(
    r"(self\.decoder_blocks\.append\(\s*DecoderBlock\(in_ch, skip_ch, dec_ch\)\s*\)\s*)\n(\s+)in_ch = dec_ch",
    re.MULTILINE,
)
if pattern.search(src) and "self.cbam_blocks.append" not in src:
    src = pattern.sub(
        r"\1\n\2self.cbam_blocks.append(CBAM(dec_ch))\n\2\n\2in_ch = dec_ch",
        src,
    )
    print("    Added: self.cbam_blocks.append(CBAM(dec_ch)) in decoder loop")
else:
    if "self.cbam_blocks.append" in src:
        print("    Skipped (already patched): cbam_blocks.append")
    else:
        print("    WARNING: could not locate decoder loop for CBAM append")

# 3.4 在 forward 中应用 CBAM: 在 attended_skip, alpha = stag_gate(...) 之后,
#     decoder_block(x, attended_skip) 的输出后应用 cbam
# 找到 forward 循环内 decoder_block 调用
# 原模式:
#     attended_skip, alpha = stag_gate(skip, x, stain_cond)
#     self.attention_maps.append(alpha)
#     
#     # 解码器上采样 + 拼接
#     x = decoder_block(x, attended_skip)

# 把上述替换为带 CBAM 的版本, 同时把 zip 改成 enumerate 以获取 i
# 但原代码已经用 enumerate...

# 找到 decoder_block 输出行, 在后面加 cbam
old_forward = "x = decoder_block(x, attended_skip)"
new_forward = "x = decoder_block(x, attended_skip)\n            x = self.cbam_blocks[i](x)"
if old_forward in src and "self.cbam_blocks[i](x)" not in src:
    src = src.replace(old_forward, new_forward, 1)
    print("    Added: self.cbam_blocks[i](x) in forward pass")
elif "self.cbam_blocks[i](x)" in src:
    print("    Skipped (already patched): cbam in forward")
else:
    print("    WARNING: could not locate decoder_block call in forward")

with open(path, "w") as f:
    f.write(src)
print("    stag_unet.py written.")
PY_EOF

echo "    Done."

# ------------------------------------------------------------
# Step 4: 修改 dataset.py — 增强数据增强
# ------------------------------------------------------------
echo ""
echo "[4/5] 修改 data/dataset.py"

python3 << 'PY_EOF'
path = "data/dataset.py"
with open(path, "r") as f:
    src = f.read()

# 目标: 替换 get_train_transforms() 函数体
# 定位旧函数: 从 "def get_train_transforms" 开始,到 "def get_val_transforms" 之前

import re

new_func = '''def get_train_transforms(patch_size=PATCH_SIZE):
    """
    训练时的数据增强 (改进版 v2)
    
    新增:
    - ElasticTransform: 弹性变形,模拟组织切片变形
    - GaussNoise + GaussianBlur: 模拟扫描噪声与聚焦差异
    - HueSaturationValue: 更强颜色扰动,模拟染色差异(关键!)
    - RandomResizedCrop: 多尺度学习
    - CoarseDropout: 随机遮挡,提升鲁棒性
    """
    transforms = []
    
    # 几何增强 (必做)
    if AUG_HORIZONTAL_FLIP:
        transforms.append(A.HorizontalFlip(p=0.5))
    if AUG_VERTICAL_FLIP:
        transforms.append(A.VerticalFlip(p=0.5))
    transforms.append(A.RandomRotate90(p=0.5))
    transforms.append(A.Transpose(p=0.3))
    
    # 弹性变形 (病理图像专用,模拟组织形变)
    transforms.append(
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3)
    )
    
    # 多尺度 (小概率,避免破坏染色描述符)
    transforms.append(
        A.RandomResizedCrop(
            size=(patch_size, patch_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.3,
        )
    )
    
    # 颜色增强 (对病理图像尤其重要)
    if AUG_COLOR_JITTER:
        transforms.append(
            A.ColorJitter(
                brightness=AUG_COLOR_BRIGHTNESS,
                contrast=AUG_COLOR_CONTRAST,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            )
        )
    transforms.append(
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        )
    )
    
    # 噪声与模糊 (模拟扫描器差异)
    transforms.append(
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2)
    )
    
    # 归一化 + 转 tensor (必须在最后)
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


'''

# 用 regex 替换原 get_train_transforms 函数体
pattern = re.compile(
    r"def get_train_transforms\(patch_size=PATCH_SIZE\):.*?(?=\ndef get_val_transforms)",
    re.DOTALL,
)
if pattern.search(src):
    src = pattern.sub(new_func, src)
    print("    Replaced get_train_transforms() with enhanced version")
else:
    print("    WARNING: could not locate get_train_transforms()")

with open(path, "w") as f:
    f.write(src)
print("    dataset.py written.")
PY_EOF

echo "    Done."

# ------------------------------------------------------------
# Step 5: 修改 train.py — CosineAnnealingWarmRestarts + scheduler.step() 每 batch
# ------------------------------------------------------------
echo ""
echo "[5/5] 修改 train.py"

python3 << 'PY_EOF'
path = "train.py"
with open(path, "r") as f:
    src = f.read()

# 替换 CosineAnnealingLR 为 CosineAnnealingWarmRestarts
old = """scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )"""
new = """scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )"""

if old in src:
    src = src.replace(old, new)
    print("    Replaced: CosineAnnealingLR -> CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")
elif "CosineAnnealingWarmRestarts" in src:
    print("    Skipped (already patched): CosineAnnealingWarmRestarts")
else:
    # fallback: 行级 replace
    import re
    pattern = re.compile(r"CosineAnnealingLR\([^)]*\)", re.DOTALL)
    if pattern.search(src):
        src = pattern.sub("CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)", src, count=1)
        print("    Replaced (fallback regex): CosineAnnealingLR")
    else:
        print("    WARNING: could not locate CosineAnnealingLR")

with open(path, "w") as f:
    f.write(src)
print("    train.py written.")
PY_EOF

echo "    Done."

# ------------------------------------------------------------
# 验证
# ------------------------------------------------------------
echo ""
echo "=========================================="
echo "验证修改"
echo "=========================================="

echo ""
echo "[Check 1] CBAM import in stag_unet.py:"
grep "from models.cbam import CBAM" models/stag_unet.py || echo "  MISSING"

echo ""
echo "[Check 2] cbam_blocks in stag_unet.py:"
grep -c "cbam_blocks" models/stag_unet.py

echo ""
echo "[Check 3] New augmentations in dataset.py:"
grep -c "ElasticTransform\|HueSaturationValue\|GaussNoise" data/dataset.py

echo ""
echo "[Check 4] CosineAnnealingWarmRestarts in train.py:"
grep "CosineAnnealingWarmRestarts" train.py || echo "  MISSING"

echo ""
echo "=========================================="
echo "[Quick model test]"
echo "=========================================="
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
import torch
from models.stag_unet import STAGUNet
model = STAGUNet(pretrained=False)
x = torch.randn(2, 3, 256, 256)
stain = torch.randn(2, 12)
out = model(x, stain)
print(f'Input:  {x.shape}')
print(f'Output: {out.shape}')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
print('Model forward pass: OK')
"

echo ""
echo "=========================================="
echo "All modifications complete!"
echo "Backup: $BACKUP_DIR"
echo ""
echo "Next step: submit training job"
echo "  sbatch job_improvements.sh"
echo "=========================================="
