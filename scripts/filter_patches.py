"""
filter_patches.py — 过滤 patch 实现类别平衡

用法:
    python scripts/filter_patches.py \
        --src /scratch/zwangot/processed \
        --dst /scratch/zwangot/processed_v2 \
        --max_other 1500

原理:
- 保留所有含前景类 (class 0/1/2) 的 patch
- 额外随机保留最多 max_other 个纯 Other (class 3) patch
- 在 train 和 val 两个 split 上都做过滤

Mask 编码约定 (经过 convert_beetle_mask 转换后):
  0 = Invasive Epithelium (IE)
  1 = Non-invasive Epithelium (NE)
  2 = Necrosis (NC)
  3 = Other (background/stroma/fat)
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def filter_patches(src, dst, max_other=1500, seed=42):
    src = Path(src)
    dst = Path(dst)
    random.seed(seed)

    for split in ["train", "val"]:
        img_dir = src / split / "images"
        msk_dir = src / split / "masks"
        out_img = dst / split / "images"
        out_msk = dst / split / "masks"

        if not msk_dir.exists():
            print(f"[Skip] {msk_dir} does not exist")
            continue

        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        masks = sorted(msk_dir.glob("*.png"))
        fg_list = []
        other_list = []

        for m in masks:
            img = cv2.imread(str(m), 0)
            if img is None:
                continue
            # Foreground = any pixel is class 0/1/2
            if np.any(img < 3):
                fg_list.append(m.stem)
            else:
                other_list.append(m.stem)

        # Cap the number of pure-Other patches
        if len(other_list) > max_other:
            random.shuffle(other_list)
            other_list = other_list[:max_other]

        # Copy selected patches
        keep = set(fg_list + other_list)
        for name in keep:
            shutil.copy2(str(img_dir / f"{name}.png"),
                         str(out_img / f"{name}.png"))
            shutil.copy2(str(msk_dir / f"{name}.png"),
                         str(out_msk / f"{name}.png"))

        print(f"{split}: foreground={len(fg_list)}, "
              f"other_kept={len(other_list)}, total={len(keep)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
                        help="Source directory with train/ and val/ subdirs")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination directory")
    parser.add_argument("--max_other", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    filter_patches(args.src, args.dst, args.max_other, args.seed)
