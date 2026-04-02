"""
eda_check.py — 数据探索脚本

运行这个脚本，它会告诉你：
1. 有多少 WSI 和 mask 文件
2. mask 里的像素值是什么（决定类别映射怎么设）
3. 显示 WSI 缩略图和 mask 的样例
4. CSV 的数据分布
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

# 路径配置
DATA_ROOT = "D:/BEHI5011_STAG_UNet/beetle/data"
CSV_PATH = os.path.join(DATA_ROOT, "data_overview.csv")

# 如果 data_overview.csv 不在 data 目录下，尝试上一级
if not os.path.exists(CSV_PATH):
    alt = "D:/BEHI5011_STAG_UNet/beetle/data_overview.csv"
    if os.path.exists(alt):
        CSV_PATH = alt

print("=" * 60)
print("BEETLE Dataset EDA")
print("=" * 60)

# ============================================================
# 1. CSV 分析
# ============================================================
print("\n--- 1. data_overview.csv ---")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts().to_string())
    if 'source' in df.columns:
        print(f"\nSource (hospital):")
        print(df['source'].value_counts().to_string())
    if 'validation_fold' in df.columns:
        print(f"\nValidation folds:")
        print(df['validation_fold'].value_counts().to_string())
    print(f"\nFirst 3 rows:")
    print(df[['patient_id', 'name', 'source', 'split', 'validation_fold']].head(3).to_string())
else:
    print(f"NOT FOUND: {CSV_PATH}")
    print("Please check the path!")
    # 尝试找到 CSV
    for root, dirs, files in os.walk("D:/BEHI5011_STAG_UNet"):
        for f in files:
            if f == "data_overview.csv":
                print(f"  Found at: {os.path.join(root, f)}")

# ============================================================
# 2. 文件统计
# ============================================================
print("\n--- 2. File counts ---")
wsi_dev = glob.glob(os.path.join(DATA_ROOT, "images/development/wsis/*.tif"))
wsi_eval = glob.glob(os.path.join(DATA_ROOT, "images/evaluation/wsis/*.tif"))
roi_eval = glob.glob(os.path.join(DATA_ROOT, "images/evaluation/rois/*.png"))
masks = glob.glob(os.path.join(DATA_ROOT, "annotations/masks/*.tif"))

print(f"Development WSIs:  {len(wsi_dev)}")
print(f"Evaluation WSIs:   {len(wsi_eval)}")
print(f"Evaluation ROIs:   {len(roi_eval)}")
print(f"Annotation masks:  {len(masks)}")

if wsi_dev:
    sizes = [os.path.getsize(f) / 1024**3 for f in wsi_dev]
    print(f"\nDev WSI sizes: min={min(sizes):.1f} GB, max={max(sizes):.1f} GB, total={sum(sizes):.1f} GB")

# ============================================================
# 3. Mask 像素值检查（最关键！）
# ============================================================
print("\n--- 3. Mask pixel values (CRITICAL!) ---")

if masks:
    # 尝试用 tifffile 读 mask
    try:
        import tifffile
        reader = "tifffile"
    except ImportError:
        reader = None

    if reader is None:
        try:
            import openslide
            reader = "openslide"
        except ImportError:
            reader = None

    if reader is None:
        print("ERROR: Neither tifffile nor openslide available!")
        print("Run: pip install tifffile")
        sys.exit(1)

    # 检查前3个 mask
    # 只取 development 的 mask（名字能对上的）
    dev_names = set()
    if os.path.exists(CSV_PATH):
        dev_df = df[df['split'] == 'development']
        dev_names = set(dev_df['name'].tolist())

    checked = 0
    for mf in sorted(masks):
        mask_name = os.path.splitext(os.path.basename(mf))[0]

        # 优先检查 development 的 mask
        if dev_names and mask_name not in dev_names and checked < 3:
            continue

        if checked >= 3:
            break

        print(f"\n  File: {os.path.basename(mf)}")
        print(f"  Size: {os.path.getsize(mf) / 1024**2:.1f} MB")

        try:
            if reader == "tifffile":
                mask = tifffile.imread(mf)
                print(f"  Shape: {mask.shape}")
                print(f"  Dtype: {mask.dtype}")

                if mask.ndim == 3:
                    print(f"  Channels: {mask.shape[2]}")
                    # 通常取第一个通道
                    mask_1ch = mask[:, :, 0]
                else:
                    mask_1ch = mask

                unique = np.unique(mask_1ch)
                print(f"  Unique values: {unique}")
                for v in unique:
                    pct = np.sum(mask_1ch == v) / mask_1ch.size * 100
                    print(f"    Value {v}: {pct:.1f}%")

            elif reader == "openslide":
                slide = openslide.OpenSlide(mf)
                print(f"  Dimensions: {slide.dimensions}")
                print(f"  Levels: {slide.level_count}")
                # 读缩略图
                thumb = np.array(slide.get_thumbnail((512, 512)))
                if thumb.ndim == 3:
                    thumb_gray = thumb[:, :, 0]
                else:
                    thumb_gray = thumb
                unique = np.unique(thumb_gray)
                print(f"  Thumbnail unique values: {unique}")
                slide.close()

        except Exception as e:
            print(f"  Error reading: {e}")

        checked += 1

    # 可视化（如果有 matplotlib）
    try:
        import matplotlib.pyplot as plt
        print("\n  Generating visualization...")

        # 读取一张 WSI 缩略图 + 对应 mask
        if wsi_dev and masks and os.path.exists(CSV_PATH):
            dev_df = df[df['split'] == 'development'].iloc[0]
            wsi_path = os.path.join(DATA_ROOT, dev_df['wsi_path'])
            mask_path = os.path.join(DATA_ROOT, dev_df['annotation_mask_path'])

            if os.path.exists(wsi_path) and os.path.exists(mask_path):
                try:
                    import openslide
                    wsi = openslide.OpenSlide(wsi_path)
                    wsi_thumb = np.array(wsi.get_thumbnail((800, 800)))

                    mask_slide = openslide.OpenSlide(mask_path)
                    mask_thumb = np.array(mask_slide.get_thumbnail((800, 800)))

                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    axes[0].imshow(wsi_thumb[:, :, :3])
                    axes[0].set_title(f"WSI: {dev_df['name']}\n{wsi.dimensions[0]}×{wsi.dimensions[1]}")
                    axes[0].axis("off")

                    if mask_thumb.ndim == 3:
                        mask_show = mask_thumb[:, :, 0]
                    else:
                        mask_show = mask_thumb
                    im = axes[1].imshow(mask_show, cmap="tab10")
                    axes[1].set_title(f"Mask\nUnique: {np.unique(mask_show)}")
                    axes[1].axis("off")
                    plt.colorbar(im, ax=axes[1])

                    plt.tight_layout()
                    save_path = "eda_visualization.png"
                    plt.savefig(save_path, dpi=150)
                    print(f"  Saved: {save_path}")
                    plt.show()

                    wsi.close()
                    mask_slide.close()
                except Exception as e:
                    print(f"  Visualization error: {e}")

    except ImportError:
        print("  (matplotlib not available, skipping visualization)")

# ============================================================
# 4. 总结
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Data root:    {DATA_ROOT}")
print(f"Dev WSIs:     {len(wsi_dev)}")
print(f"Masks:        {len(masks)}")
print(f"\n📌 NEXT STEP:")
print(f"   Look at the 'Unique values' above.")
print(f"   If they are [0, 1, 2, 3] → config is OK, proceed to preprocessing.")
print(f"   If different → tell me the values, I'll fix the config.")
print("=" * 60)
