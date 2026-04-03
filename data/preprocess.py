"""
preprocess.py — 数据预处理流程

负责：
1. 组织区域检测（去除纯白背景）
2. 切 patch（固定大小 + overlap）
3. Macenko 染色归一化
4. 按 WSI 级别划分 train/val/test
5. 提取并保存每张 WSI 的 stain descriptor
"""
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
import json
import shutil

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.stain_utils import MacenkoNormalizer, extract_stain_descriptor


def detect_tissue(image, threshold=BACKGROUND_THRESHOLD, min_ratio=TISSUE_THRESHOLD):
    """
    检测图像中是否有足够的组织区域
    
    原理：将图像转灰度，背景是接近白色的（灰度值 > threshold），
    组织区域灰度值较低。如果组织占比太低，说明这个 patch 大部分是背景，丢弃。
    
    Args:
        image: RGB 图像 (H, W, 3), uint8
        threshold: 背景灰度阈值
        min_ratio: 组织最低占比
        
    Returns:
        bool: True 表示有足够组织，保留这个 patch
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < threshold
    tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
    return tissue_ratio >= min_ratio


def extract_patches(image, mask, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """
    从一张大图中切出固定大小的 patch
    
    Args:
        image: RGB 图像 (H, W, 3)
        mask: 标注 mask (H, W)，像素值为类别标签
        patch_size: patch 边长
        overlap: 重叠像素数
        
    Returns:
        patches: list of (image_patch, mask_patch) tuples
    """
    h, w = image.shape[:2]
    stride = patch_size - overlap
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # 检查是否有足够组织
            if detect_tissue(img_patch):
                patches.append((img_patch, mask_patch))
    
    return patches


def convert_mask_to_classes(mask):
    """
    将原始 mask 的像素值转换为类别索引 (0, 1, 2, 3)
    
    注意：你需要根据 BEETLE 数据集的实际标注格式修改这个函数！
    不同数据集的 mask 编码方式不同。
    
    Args:
        mask: 原始 mask
        
    Returns:
        class_mask: 类别索引 mask，值为 0-3
    """
    class_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # 根据 BEETLE 数据集的实际像素值映射
    # ⚠️ 你需要先检查数据集的 mask 值，然后修改这里
    for class_idx, pixel_val in CLASS_PIXEL_VALUES.items():
        class_mask[mask == pixel_val] = class_idx
    
    return class_mask


def preprocess_dataset(
    raw_dir,
    output_dir,
    stain_desc_dir,
    use_stain_norm=USE_STAIN_NORM,
    patch_size=PATCH_SIZE,
    overlap=OVERLAP,
):
    """
    完整的预处理流程
    
    Args:
        raw_dir: 原始数据目录（包含 images/ 和 masks/ 子目录）
        output_dir: 输出目录
        stain_desc_dir: 染色描述符保存目录
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    stain_desc_dir = Path(stain_desc_dir)
    
    # 创建输出目录
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    stain_desc_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化染色归一化器
    normalizer = MacenkoNormalizer() if use_stain_norm else None
    
    # 收集所有图像文件
    image_dir = raw_dir / "images"
    mask_dir = raw_dir / "masks"
    
    if not image_dir.exists():
        print(f"[Error] Image directory not found: {image_dir}")
        print("Please organize your data as:")
        print("  data/raw/images/  (H&E images)")
        print("  data/raw/masks/   (annotation masks)")
        return
    
    # 获取所有图像（支持常见格式）
    image_files = sorted(
        list(image_dir.glob("*.png")) + 
        list(image_dir.glob("*.jpg")) + 
        list(image_dir.glob("*.tif")) +
        list(image_dir.glob("*.tiff"))
    )
    
    if len(image_files) == 0:
        print(f"[Error] No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # ============================================================
    # WSI-level split（按图像名划分，避免数据泄漏）
    # ============================================================
    # 提取 WSI ID（假设文件名格式为 WSIID_patchXX.png 或类似格式）
    # 如果是单张大图，每张图就是一个 WSI
    wsi_ids = []
    file_to_wsi = {}
    for f in image_files:
        # 尝试从文件名提取 WSI ID
        # 方式1：下划线分割取第一部分
        wsi_id = f.stem.split("_")[0]
        # 方式2：如果每个文件就是独立的，直接用文件名
        # wsi_id = f.stem
        file_to_wsi[f.name] = wsi_id
        if wsi_id not in wsi_ids:
            wsi_ids.append(wsi_id)
    
    print(f"Found {len(wsi_ids)} unique WSIs/images")
    
    # 按 WSI 划分：70% train, 15% val, 15% test
    np.random.seed(SEED)
    wsi_ids_arr = np.array(wsi_ids)
    np.random.shuffle(wsi_ids_arr)
    
    n = len(wsi_ids_arr)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_wsis = set(wsi_ids_arr[:n_train])
    val_wsis = set(wsi_ids_arr[n_train:n_train+n_val])
    test_wsis = set(wsi_ids_arr[n_train+n_val:])
    
    print(f"Split: {len(train_wsis)} train, {len(val_wsis)} val, {len(test_wsis)} test WSIs")
    
    # 保存划分信息
    split_info = {
        "train": list(train_wsis),
        "val": list(val_wsis),
        "test": list(test_wsis),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    # ============================================================
    # 逐图处理
    # ============================================================
    stats = {"train": 0, "val": 0, "test": 0}
    stain_descriptors = {}
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Cannot read: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取对应 mask
        mask_path = mask_dir / img_path.name
        # 尝试不同的扩展名
        if not mask_path.exists():
            for ext in [".png", ".tif", ".tiff", ".jpg"]:
                alt_path = mask_dir / (img_path.stem + ext)
                if alt_path.exists():
                    mask_path = alt_path
                    break
        
        if not mask_path.exists():
            print(f"[Warning] Mask not found for: {img_path.name}")
            continue
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[Warning] Cannot read mask: {mask_path}")
            continue
        
        # 确定这张图属于哪个 split
        wsi_id = file_to_wsi[img_path.name]
        if wsi_id in train_wsis:
            split = "train"
        elif wsi_id in val_wsis:
            split = "val"
        else:
            split = "test"
        
        # 提取染色描述符
        desc = extract_stain_descriptor(image, normalizer)
        stain_descriptors[img_path.stem] = desc.tolist()
        
        # 染色归一化
        if normalizer is not None:
            image = normalizer.normalize(image)
        
        # 转换 mask 到类别索引
        class_mask = convert_mask_to_classes(mask)
        
        # 如果图像已经是 patch 大小，直接保存
        if image.shape[0] <= patch_size * 1.5 and image.shape[1] <= patch_size * 1.5:
            # 直接 resize 到目标大小
            image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            class_mask = cv2.resize(class_mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
            
            if detect_tissue(image):
                save_name = img_path.stem
                cv2.imwrite(
                    str(output_dir / split / "images" / f"{save_name}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(output_dir / split / "masks" / f"{save_name}.png"),
                    class_mask
                )
                stats[split] += 1
        else:
            # 大图切 patch
            patches = extract_patches(image, class_mask, patch_size, overlap)
            
            for idx, (img_patch, mask_patch) in enumerate(patches):
                save_name = f"{img_path.stem}_patch{idx:04d}"
                cv2.imwrite(
                    str(output_dir / split / "images" / f"{save_name}.png"),
                    cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(output_dir / split / "masks" / f"{save_name}.png"),
                    mask_patch
                )
                stats[split] += 1
    
    # 保存染色描述符
    np.save(stain_desc_dir / "stain_descriptors.npy", stain_descriptors)
    
    # 打印统计
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"Train patches: {stats['train']}")
    print(f"Val patches:   {stats['val']}")
    print(f"Test patches:  {stats['test']}")
    print(f"Total patches: {sum(stats.values())}")
    print(f"Stain descriptors saved: {len(stain_descriptors)}")
    print("=" * 50)


# ============================================================
# 数据统计分析（EDA）
# ============================================================
def analyze_dataset(processed_dir):
    """
    分析处理后的数据集，打印类别分布等统计信息
    
    Args:
        processed_dir: 处理后的数据目录
    """
    processed_dir = Path(processed_dir)
    
    for split in ["train", "val", "test"]:
        mask_dir = processed_dir / split / "masks"
        if not mask_dir.exists():
            continue
        
        mask_files = sorted(mask_dir.glob("*.png"))
        print(f"\n{'='*40}")
        print(f"Split: {split} ({len(mask_files)} patches)")
        print(f"{'='*40}")
        
        total_pixels = 0
        class_pixels = np.zeros(NUM_CLASSES)
        
        for mf in tqdm(mask_files, desc=f"Analyzing {split}"):
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            total_pixels += mask.size
            for c in range(NUM_CLASSES):
                class_pixels[c] += np.sum(mask == c)
        
        print("\nClass distribution:")
        for c in range(NUM_CLASSES):
            ratio = class_pixels[c] / total_pixels * 100
            print(f"  {CLASS_NAMES[c]:30s}: {class_pixels[c]:>10.0f} pixels ({ratio:5.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=DATA_ROOT)
    parser.add_argument("--output_dir", default=PROCESSED_DIR)
    parser.add_argument("--stain_dir", default=STAIN_DESC_DIR)
    parser.add_argument("--analyze", action="store_true", help="只分析，不预处理")
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.output_dir)
    else:
        preprocess_dataset(args.raw_dir, args.output_dir, args.stain_dir)
        analyze_dataset(args.output_dir)
