"""
preprocess_wsi.py — BEETLE 数据集专用预处理

BEETLE 数据结构：
  images/development/wsis/patient1_wsi1.tif   (WSI, ~135000x147000 像素)
  annotations/masks/patient1_wsi1.tif          (Mask, 同等大小)

处理流程：
  1. 用 OpenSlide 打开 WSI 和对应 mask
  2. 在低倍率下检测组织区域
  3. 在高倍率下从组织区域切 256x256 patch
  4. 对每个 patch 做染色归一化
  5. 按官方的 validation_fold 划分 train/val
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image

# 尝试导入 OpenSlide
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("[Warning] OpenSlide not installed. Install with: pip install openslide-python")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from data.stain_utils import MacenkoNormalizer, extract_stain_descriptor


def get_tissue_mask(slide, level=-1, threshold=220):
    """
    在低倍率下检测组织区域（快速定位哪里有组织）

    原理：WSI 背景是白色的，组织区域颜色更深。
    在最低倍率读取缩略图，转灰度，低于阈值的就是组织。

    Args:
        slide: OpenSlide 对象
        level: 使用哪个金字塔层级（-1=最低倍率，最快）
        threshold: 灰度阈值

    Returns:
        tissue_mask: 二值 mask（1=组织，0=背景）
        scale_factors: (scale_x, scale_y) 从该层级到 level 0 的缩放比
    """
    # 使用最低倍率（最小的图）
    if level == -1:
        level = slide.level_count - 1

    dims = slide.level_dimensions[level]
    # 读取整个低倍率图像
    thumbnail = slide.read_region((0, 0), level, dims).convert("RGB")
    thumbnail = np.array(thumbnail)

    # 转灰度 → 二值化
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray < threshold).astype(np.uint8)

    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

    # 计算缩放比
    level0_dims = slide.level_dimensions[0]
    scale_x = level0_dims[0] / dims[0]
    scale_y = level0_dims[1] / dims[1]

    return tissue_mask, (scale_x, scale_y)


def extract_patches_from_wsi(
    wsi_path,
    mask_path,
    output_img_dir,
    output_mask_dir,
    patch_size=PATCH_SIZE,
    stride=None,
    max_patches_per_wsi=200,
    tissue_threshold=0.3,
    normalizer=None,
    wsi_name=None,
):
    """
    从一对 WSI + Mask 中提取 patch

    Args:
        wsi_path: WSI 文件路径
        mask_path: 对应 mask 文件路径
        output_img_dir: patch 图像保存目录
        output_mask_dir: patch mask 保存目录
        patch_size: patch 大小
        stride: 滑窗步长（None = patch_size，即不重叠）
        max_patches_per_wsi: 每张 WSI 最多提取多少个 patch
        tissue_threshold: 组织区域最低占比
        normalizer: MacenkoNormalizer 实例
        wsi_name: WSI 名称（用于命名 patch 文件）

    Returns:
        num_patches: 提取了多少个 patch
        stain_desc: 该 WSI 的染色描述符
    """
    if stride is None:
        stride = patch_size

    if wsi_name is None:
        wsi_name = Path(wsi_path).stem

    # 打开 WSI
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        print(f"  [Error] Cannot open WSI: {wsi_path}: {e}")
        return 0, np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    # 打开 Mask（mask 也是大 tif，用 OpenSlide 或 tifffile 读）
    try:
        mask_slide = openslide.OpenSlide(str(mask_path))
        mask_is_openslide = True
    except Exception:
        # 如果 OpenSlide 打不开 mask，尝试用 tifffile
        try:
            import tifffile
            mask_full = tifffile.imread(str(mask_path))
            mask_is_openslide = False
        except Exception:
            # 最后尝试 PIL（可能内存不够）
            try:
                mask_img = Image.open(str(mask_path))
                mask_full = np.array(mask_img)
                mask_is_openslide = False
            except Exception as e:
                print(f"  [Error] Cannot open mask: {mask_path}: {e}")
                slide.close()
                return 0, np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    # 1. 在低倍率检测组织区域
    tissue_mask, (sx, sy) = get_tissue_mask(slide)

    # 2. 在组织区域内采样 patch 坐标
    w0, h0 = slide.level_dimensions[0]
    th, tw = tissue_mask.shape

    # 找到组织区域的坐标（在低倍率下）
    tissue_coords = np.argwhere(tissue_mask > 0)  # (row, col) in low-res
    if len(tissue_coords) == 0:
        print(f"  [Warning] No tissue found in {wsi_name}")
        slide.close()
        if mask_is_openslide:
            mask_slide.close()
        return 0, np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    # 随机采样一部分组织坐标
    np.random.seed(hash(wsi_name) % 2**32)
    if len(tissue_coords) > max_patches_per_wsi * 3:
        indices = np.random.choice(len(tissue_coords), max_patches_per_wsi * 3, replace=False)
        tissue_coords = tissue_coords[indices]

    # 3. 逐个提取 patch
    num_patches = 0
    stain_descs = []

    for row, col in tissue_coords:
        if num_patches >= max_patches_per_wsi:
            break

        # 转换到 level 0 坐标
        x0 = int(col * sx)
        y0 = int(row * sy)

        # 边界检查
        if x0 + patch_size > w0 or y0 + patch_size > h0:
            continue

        # 读取图像 patch
        try:
            patch_img = slide.read_region((x0, y0), 0, (patch_size, patch_size)).convert("RGB")
            patch_img = np.array(patch_img)
        except Exception:
            continue

        # 检查组织占比
        gray = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
        tissue_ratio = np.sum(gray < BACKGROUND_THRESHOLD) / gray.size
        if tissue_ratio < tissue_threshold:
            continue

        # 读取 mask patch
        try:
            if mask_is_openslide:
                patch_mask = mask_slide.read_region((x0, y0), 0, (patch_size, patch_size))
                patch_mask = np.array(patch_mask)
                # OpenSlide 返回 RGBA，取第一个通道或转灰度
                if patch_mask.ndim == 3:
                    if patch_mask.shape[2] == 4:
                        patch_mask = patch_mask[:, :, 0]  # 取 R 通道
                    elif patch_mask.shape[2] == 3:
                        patch_mask = cv2.cvtColor(patch_mask, cv2.COLOR_RGB2GRAY)
            else:
                # 从完整 mask 中切取
                patch_mask = mask_full[y0:y0+patch_size, x0:x0+patch_size]
                if patch_mask.ndim == 3:
                    patch_mask = patch_mask[:, :, 0]
        except Exception:
            continue

        # 确保 mask 尺寸正确
        if patch_mask.shape[0] != patch_size or patch_mask.shape[1] != patch_size:
            continue

        # 染色归一化
        if normalizer is not None:
            try:
                patch_img = normalizer.normalize(patch_img)
            except Exception:
                pass

        # 提取染色描述符
        desc = extract_stain_descriptor(patch_img, normalizer)
        if np.any(desc != 0):
            stain_descs.append(desc)

        # 转换 mask 类别值
        class_mask = convert_beetle_mask(patch_mask)

        # 保存
        save_name = f"{wsi_name}_patch{num_patches:04d}"
        cv2.imwrite(
            str(Path(output_img_dir) / f"{save_name}.png"),
            cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(Path(output_mask_dir) / f"{save_name}.png"),
            class_mask
        )
        num_patches += 1

    # 清理
    slide.close()
    if mask_is_openslide:
        mask_slide.close()

    # 计算 slide-level 染色描述符
    if len(stain_descs) > 0:
        slide_desc = np.mean(stain_descs, axis=0).astype(np.float32)
    else:
        slide_desc = np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    return num_patches, slide_desc


def convert_beetle_mask(mask):
    """
    将 BEETLE mask 的像素值转换为类别索引 (0-3)

    📌 BEETLE 数据集的 mask 编码：
       通常：0=background, 1=invasive, 2=non-invasive, 3=necrosis
       但你需要根据实际 EDA 结果确认！

    如果你的 mask unique values 是 [0, 1, 2, 3]，那这个函数不需要改。
    如果是其他值（比如 [0, 85, 170, 255]），就需要修改映射。

    Args:
        mask: 原始 mask (H, W)

    Returns:
        class_mask: 类别索引 mask (H, W)，值为 0-3
    """
    class_mask = np.zeros_like(mask, dtype=np.uint8)

    for class_idx, pixel_val in CLASS_PIXEL_VALUES.items():
        class_mask[mask == pixel_val] = class_idx

    return class_mask


def preprocess_beetle(
    csv_path,
    data_root,
    output_dir,
    stain_desc_dir,
    val_fold=0,
    max_patches_per_wsi=200,
    use_stain_norm=USE_STAIN_NORM,
):
    """
    BEETLE 数据集完整预处理流程

    Args:
        csv_path: data_overview.csv 路径
        data_root: 数据根目录（包含 images/ 和 annotations/）
        output_dir: 输出目录
        stain_desc_dir: 染色描述符保存目录
        val_fold: 使用哪个 fold 作为验证集 (0-4)
        max_patches_per_wsi: 每张 WSI 最多切多少 patch
    """
    output_dir = Path(output_dir)
    stain_desc_dir = Path(stain_desc_dir)
    data_root = Path(data_root)

    # 创建输出目录
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    stain_desc_dir.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Split distribution:\n{df['split'].value_counts()}")

    if 'validation_fold' in df.columns:
        print(f"\nFold distribution:\n{df['validation_fold'].value_counts()}")
    if 'source' in df.columns:
        print(f"\nSource distribution:\n{df['source'].value_counts()}")

    # 初始化
    normalizer = MacenkoNormalizer() if use_stain_norm else None
    stain_descriptors = {}
    stats = {"train": 0, "val": 0, "test": 0}

    # 只处理 development 集（有 mask 的）
    dev_df = df[df['split'] == 'development'].copy()
    print(f"\nDevelopment set: {len(dev_df)} WSIs")

    # 按 validation_fold 划分
    if 'validation_fold' in dev_df.columns:
        val_fold_name = f"fold{val_fold}"
        val_mask = dev_df['validation_fold'] == val_fold_name
        train_df = dev_df[~val_mask]
        val_df = dev_df[val_mask]
        print(f"Using fold {val_fold} as validation")
        print(f"Train: {len(train_df)} WSIs, Val: {len(val_df)} WSIs")
    else:
        # 手动划分
        np.random.seed(SEED)
        indices = np.random.permutation(len(dev_df))
        n_val = max(1, int(len(dev_df) * 0.2))
        val_df = dev_df.iloc[indices[:n_val]]
        train_df = dev_df.iloc[indices[n_val:]]

    # 处理 evaluation 集的 ROI（如果有的话）
    eval_df = df[df['split'] == 'evaluation']
    print(f"Evaluation set: {len(eval_df)} entries")

    # ============================================================
    # 逐张处理 development WSIs
    # ============================================================
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} set ({len(split_df)} WSIs)")
        print(f"{'='*50}")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            wsi_name = row['name']

            # 跳过缺失路径的行
            if pd.isna(row['wsi_path']) or pd.isna(row['annotation_mask_path']):
                continue

            # 构建文件路径
            wsi_path = data_root / str(row['wsi_path'])
            mask_path = data_root / str(row['annotation_mask_path'])

            if not wsi_path.exists():
                print(f"  [Skip] WSI not found: {wsi_path}")
                continue
            if not mask_path.exists():
                print(f"  [Skip] Mask not found: {mask_path}")
                continue

            print(f"\n  Processing: {wsi_name}")

            n_patches, slide_desc = extract_patches_from_wsi(
                wsi_path=wsi_path,
                mask_path=mask_path,
                output_img_dir=output_dir / split_name / "images",
                output_mask_dir=output_dir / split_name / "masks",
                patch_size=PATCH_SIZE,
                max_patches_per_wsi=max_patches_per_wsi,
                normalizer=normalizer,
                wsi_name=wsi_name,
            )

            stats[split_name] += n_patches
            stain_descriptors[wsi_name] = slide_desc.tolist()
            print(f"  → {n_patches} patches extracted")

    # 保存染色描述符
    np.save(stain_desc_dir / "stain_descriptors.npy", stain_descriptors)

    # 保存 split 信息
    split_info = {
        "train_wsis": train_df['name'].tolist(),
        "val_wsis": val_df['name'].tolist(),
        "val_fold": val_fold,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # 统计
    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print(f"  Train patches: {stats['train']}")
    print(f"  Val patches:   {stats['val']}")
    print(f"  Total patches: {sum(stats.values())}")
    print(f"  Stain descriptors: {len(stain_descriptors)}")
    print("=" * 50)

    # 分析类别分布
    analyze_patches(output_dir)


def analyze_patches(processed_dir):
    """分析处理后 patch 的类别分布"""
    import glob

    processed_dir = Path(processed_dir)

    for split in ["train", "val"]:
        mask_dir = processed_dir / split / "masks"
        if not mask_dir.exists():
            continue

        mask_files = sorted(mask_dir.glob("*.png"))
        if len(mask_files) == 0:
            continue

        print(f"\n{split}: {len(mask_files)} patches")

        total = 0
        counts = np.zeros(NUM_CLASSES)
        for mf in mask_files[:500]:  # 采样分析，不需要全部
            m = cv2.imread(str(mf), 0)
            total += m.size
            for c in range(NUM_CLASSES):
                counts[c] += np.sum(m == c)

        for c in range(NUM_CLASSES):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
            pct = counts[c] / total * 100 if total > 0 else 0
            print(f"  {name:30s}: {pct:5.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to data_overview.csv")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir containing images/ and annotations/")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--stain_dir", type=str, default=STAIN_DESC_DIR)
    parser.add_argument("--val_fold", type=int, default=0, help="Which fold for validation (0-4)")
    parser.add_argument("--max_patches", type=int, default=200, help="Max patches per WSI")
    parser.add_argument("--no_stain_norm", action="store_true")
    args = parser.parse_args()

    preprocess_beetle(
        csv_path=args.csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        stain_desc_dir=args.stain_dir,
        val_fold=args.val_fold,
        max_patches_per_wsi=args.max_patches,
        use_stain_norm=not args.no_stain_norm,
    )
