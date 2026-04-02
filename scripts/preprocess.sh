#!/bin/bash
# ============================================================
# preprocess.sh — 数据预处理脚本
#
# 用法：bash scripts/preprocess.sh
# ============================================================

echo "=============================="
echo "Data Preprocessing"
echo "=============================="

# 默认路径（根据你的实际路径修改）
RAW_DIR="${1:-./data/raw}"
OUTPUT_DIR="${2:-./data/processed}"
STAIN_DIR="${3:-./data/stain_descriptors}"

echo "Raw data dir:    $RAW_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "Stain desc dir:  $STAIN_DIR"

# 检查原始数据是否存在
if [ ! -d "$RAW_DIR/images" ]; then
    echo ""
    echo "[ERROR] Cannot find $RAW_DIR/images/"
    echo ""
    echo "Please organize your BEETLE dataset as follows:"
    echo "  $RAW_DIR/"
    echo "  ├── images/     (H&E stained images)"
    echo "  │   ├── image001.png"
    echo "  │   ├── image002.png"
    echo "  │   └── ..."
    echo "  └── masks/      (annotation masks)"
    echo "      ├── image001.png"
    echo "      ├── image002.png"
    echo "      └── ..."
    echo ""
    echo "Then run this script again."
    exit 1
fi

# 运行预处理
python preprocess_main.py \
    --raw_dir "$RAW_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --stain_dir "$STAIN_DIR"

echo ""
echo "Preprocessing complete!"
echo "Processed data saved to: $OUTPUT_DIR"
