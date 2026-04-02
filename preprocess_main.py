"""
preprocess_main.py — 预处理入口脚本

直接运行：python preprocess_main.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocess import preprocess_dataset, analyze_dataset
from configs.config import DATA_ROOT, PROCESSED_DIR, STAIN_DESC_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess BEETLE dataset")
    parser.add_argument("--raw_dir", type=str, default=DATA_ROOT,
                        help="Path to raw data (containing images/ and masks/)")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--stain_dir", type=str, default=STAIN_DESC_DIR)
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing processed data")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_dataset(args.output_dir)
    else:
        preprocess_dataset(args.raw_dir, args.output_dir, args.stain_dir)
        analyze_dataset(args.output_dir)
