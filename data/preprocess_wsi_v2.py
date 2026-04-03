"""
preprocess_wsi_v2.py — 修复版：用 tifffile 读 mask，大幅增加数据量

改动：
  - mask 用 tifffile 读取（OpenSlide 对很多 BEETLE mask 报错）
  - 支持分块读取大 mask（不炸内存）
  - 增加每张 WSI 的 patch 数量上限
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import tifffile
import openslide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from data.stain_utils import MacenkoNormalizer, extract_stain_descriptor


def get_tissue_mask(slide, threshold=220):
    """在最低倍率下检测组织区域"""
    level = slide.level_count - 1
    dims = slide.level_dimensions[level]
    thumbnail = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    tissue = (gray < threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel)

    sx = slide.level_dimensions[0][0] / dims[0]
    sy = slide.level_dimensions[0][1] / dims[1]
    return tissue, (sx, sy)

# 全局缓存：避免重复读取同一个大 mask
_mask_cache = {}

def read_mask_patch_cached(tif_path, x, y, w, h):
    """用 memmap 读取 mask 的一小块，不炸内存"""
    global _mask_cache
    tif_str = str(tif_path)

    if tif_str not in _mask_cache:
        try:
            tif = tifffile.TiffFile(tif_str)
            page = tif.pages[0]
            data = page.asarray(out='memmap')
            _mask_cache[tif_str] = (tif, data)

            # 只保留 2 个缓存
            if len(_mask_cache) > 2:
                oldest_key = next(iter(_mask_cache))
                old_tif, old_data = _mask_cache.pop(oldest_key)
                del old_data
                old_tif.close()

        except Exception as e:
            print(f"    [Error] Cannot read mask: {e}")
            return None

    entry = _mask_cache.get(tif_str)
    if entry is None:
        return None

    _, data = entry
    full_h, full_w = data.shape[:2]
    if x >= full_w or y >= full_h:
        return None

    x2 = min(x + w, full_w)
    y2 = min(y + h, full_h)
    patch = data[y:y2, x:x2].copy()

    if patch.ndim == 3:
        patch = patch[:, :, 0]

    if patch.shape[0] != h or patch.shape[1] != w:
        full_patch = np.zeros((h, w), dtype=patch.dtype)
        full_patch[:patch.shape[0], :patch.shape[1]] = patch
        patch = full_patch

    return patch



def convert_beetle_mask(mask):
    """BEETLE mask: 0=bg, 1=invasive, 2=non-invasive, 3=necrosis, 4→other"""
    class_mask = np.full_like(mask, 3, dtype=np.uint8)  # 默认 Other
    class_mask[mask == 1] = 0  # Invasive
    class_mask[mask == 2] = 1  # Non-invasive
    class_mask[mask == 3] = 2  # Necrosis
    # mask==0 和 mask==4 都归为 Other (class 3)
    return class_mask


def extract_patches_from_wsi(
    wsi_path, mask_path, output_img_dir, output_mask_dir,
    patch_size=256, max_patches=500, tissue_threshold=0.3,
    normalizer=None, wsi_name=None,
):
    """从一对 WSI + mask 中提取 patch"""
    if wsi_name is None:
        wsi_name = Path(wsi_path).stem

    # 打开 WSI
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        print(f"    [Skip] Cannot open WSI: {e}")
        return 0, np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    w0, h0 = slide.level_dimensions[0]

    # 检测组织区域
    tissue_mask, (sx, sy) = get_tissue_mask(slide)
    tissue_coords = np.argwhere(tissue_mask > 0)

    if len(tissue_coords) == 0:
        slide.close()
        return 0, np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    # 随机采样坐标
    np.random.seed(hash(wsi_name) % 2**32)
    n_candidates = min(len(tissue_coords), max_patches * 5)
    indices = np.random.choice(len(tissue_coords), n_candidates, replace=False)
    sampled_coords = tissue_coords[indices]

    num_patches = 0
    stain_descs = []

    for row, col in sampled_coords:
        if num_patches >= max_patches:
            break

        x0 = int(col * sx)
        y0 = int(row * sy)

        if x0 + patch_size > w0 or y0 + patch_size > h0:
            continue

        # 读 WSI patch
        try:
            patch_img = np.array(
                slide.read_region((x0, y0), 0, (patch_size, patch_size)).convert("RGB")
            )
        except:
            continue

        # 组织检查
        gray = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
        if np.sum(gray < BACKGROUND_THRESHOLD) / gray.size < tissue_threshold:
            continue

        # 读 mask patch（用 tifffile！）
        patch_mask = read_mask_patch_cached(mask_path, x0, y0, patch_size, patch_size)
        if patch_mask is None:
            continue

        # 转换类别
        class_mask = convert_beetle_mask(patch_mask)

        # 染色归一化
        if normalizer is not None:
            try:
                patch_img = normalizer.normalize(patch_img)
            except:
                pass

        # 染色描述符
        desc = extract_stain_descriptor(patch_img, normalizer)
        if np.any(desc != 0):
            stain_descs.append(desc)

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

    slide.close()

    if len(stain_descs) > 0:
        slide_desc = np.mean(stain_descs, axis=0).astype(np.float32)
    else:
        slide_desc = np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)

    return num_patches, slide_desc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--stain_dir", type=str, default="data/stain_descriptors")
    parser.add_argument("--val_fold", type=int, default=0)
    parser.add_argument("--max_patches", type=int, default=500)
    parser.add_argument("--no_stain_norm", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    stain_desc_dir = Path(args.stain_dir)

    # 清除旧数据
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    stain_desc_dir.mkdir(parents=True, exist_ok=True)

    # 读 CSV
    df = pd.read_csv(args.csv)
    dev_df = df[df['split'] == 'development'].copy()

    # 清理：去掉路径为空的行
    dev_df = dev_df.dropna(subset=['wsi_path', 'annotation_mask_path'])

    val_fold_name = f"fold{args.val_fold}"
    val_mask = dev_df['validation_fold'] == val_fold_name
    train_df = dev_df[~val_mask]
    val_df = dev_df[val_mask]

    print(f"Development: {len(dev_df)} WSIs")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    normalizer = MacenkoNormalizer() if not args.no_stain_norm else None
    stain_descriptors = {}
    stats = {"train": 0, "val": 0}

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\n{'='*50}")
        print(f"{split_name}: {len(split_df)} WSIs")
        print(f"{'='*50}")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            wsi_name = row['name']

            wsi_path = data_root / str(row['wsi_path'])
            mask_path = data_root / str(row['annotation_mask_path'])

            if not wsi_path.exists() or not mask_path.exists():
                continue

            n, desc = extract_patches_from_wsi(
                wsi_path, mask_path,
                output_dir / split_name / "images",
                output_dir / split_name / "masks",
                patch_size=PATCH_SIZE,
                max_patches=args.max_patches,
                normalizer=normalizer,
                wsi_name=wsi_name,
            )

            stats[split_name] += n
            stain_descriptors[wsi_name] = desc.tolist()

            if n > 0:
                tqdm.write(f"  {wsi_name}: {n} patches")

    # 清理 mask 缓存
    global _mask_cache
    _mask_cache.clear()

    np.save(stain_desc_dir / "stain_descriptors.npy", stain_descriptors)

    print(f"\n{'='*50}")
    print(f"Done! Train: {stats['train']}, Val: {stats['val']}, Total: {sum(stats.values())}")
    print(f"{'='*50}")

    # 类别分布
    import glob as g
    for split in ["train", "val"]:
        masks = sorted(g.glob(str(output_dir / split / "masks" / "*.png")))
        if not masks:
            continue
        total = 0
        counts = np.zeros(NUM_CLASSES)
        for mf in masks[:500]:
            m = cv2.imread(mf, 0)
            total += m.size
            for c in range(NUM_CLASSES):
                counts[c] += np.sum(m == c)
        print(f"\n{split} ({len(masks)} patches):")
        for c in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[c]:30s}: {counts[c]/total*100:5.1f}%")


if __name__ == "__main__":
    main()
