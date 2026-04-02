"""
test.py — 测试主程序

加载训练好的模型，在测试集上评估并生成可视化结果
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from configs.config import *
from data.dataset import create_dataloaders
from models.stag_unet import create_model
from utils.losses import CombinedLoss
from utils.metrics import MetricsTracker, compute_hd95
from utils.visualize import (
    plot_prediction, plot_attention_maps, 
    denormalize_image, mask_to_colormap,
)


@torch.no_grad()
def test(args):
    """测试主函数"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "attention_maps").mkdir(exist_ok=True)
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    _, test_loader, _ = create_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=1,  # 逐张测试便于可视化
        num_workers=NUM_WORKERS,
    )
    print(f"Test samples: {len(test_loader)}")
    
    # ============================================================
    # 2. 加载模型
    # ============================================================
    print(f"Loading model from: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("config", {}).get("model", args.model)
    
    model = create_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {model_name}")
    print(f"Best Dice from training: {checkpoint.get('best_dice', 'N/A')}")
    
    # ============================================================
    # 3. 评估
    # ============================================================
    tracker = MetricsTracker()
    criterion = CombinedLoss()
    total_loss = 0.0
    hd95_all = {c: [] for c in range(NUM_CLASSES)}
    
    for idx, (images, masks, stain_descs) in enumerate(tqdm(test_loader, desc="Testing")):
        images = images.to(device)
        masks = masks.to(device)
        stain_descs = stain_descs.to(device)
        
        # 推理
        outputs = model(images, stain_descs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        
        # 累积指标
        tracker.update(outputs, masks)
        
        # HD95（逐张计算）
        pred_mask = torch.argmax(outputs, dim=1)[0]  # (H, W)
        gt_mask = masks[0]
        hd95 = compute_hd95(pred_mask, gt_mask)
        for c in range(NUM_CLASSES):
            if hd95[c] != float('inf'):
                hd95_all[c].append(hd95[c])
        
        # 保存前 20 张的可视化
        if idx < 20:
            img_np = denormalize_image(images[0].cpu())
            gt_np = masks[0].cpu().numpy()
            pred_np = pred_mask.cpu().numpy()
            
            plot_prediction(
                img_np, gt_np, pred_np,
                save_path=output_dir / "predictions" / f"pred_{idx:04d}.png",
                title=f"Sample {idx}",
            )
            
            # 如果模型有 attention maps（STAG-UNet）
            if hasattr(model, "attention_maps") and len(model.attention_maps) > 0:
                plot_attention_maps(
                    model.attention_maps,
                    save_path=output_dir / "attention_maps" / f"attn_{idx:04d}.png",
                )
    
    # ============================================================
    # 4. 打印结果
    # ============================================================
    avg_loss = total_loss / len(test_loader)
    results = tracker.compute()
    
    print(f"\nTest Loss: {avg_loss:.4f}")
    tracker.print_results(results)
    
    # HD95
    print("\nHD95 per class:")
    for c in range(NUM_CLASSES):
        if len(hd95_all[c]) > 0:
            mean_hd = np.mean(hd95_all[c])
            print(f"  {CLASS_NAMES[c]:<30s}: {mean_hd:.2f}")
        else:
            print(f"  {CLASS_NAMES[c]:<30s}: N/A")
    
    # 保存结果到文件
    with open(output_dir / "test_results.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\nHD95 per class:\n")
        for c in range(NUM_CLASSES):
            if len(hd95_all[c]) > 0:
                f.write(f"  {CLASS_NAMES[c]}: {np.mean(hd95_all[c]):.2f}\n")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="./test_results")
    
    args = parser.parse_args()
    test(args)
