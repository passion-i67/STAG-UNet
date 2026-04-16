"""
train.py — 训练主程序

这是整个项目的核心训练脚本。
运行方式：python train.py --model stag_unet --epochs 50
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

# 项目模块
from configs.config import *
from data.dataset import create_dataloaders
from models.stag_unet import create_model
from utils.losses import CombinedLoss
from utils.metrics import MetricsTracker
from utils.visualize import plot_training_curves


def set_seed(seed=SEED):
    """设置随机种子，保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, accum_steps=1):
    """
    训练一个 epoch（支持梯度累积）
    
    Args:
        accum_steps: 梯度累积步数，等效 batch = batch_size * accum_steps
    
    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for step, (images, masks, stain_descs) in enumerate(pbar):
        # 移到 GPU
        images = images.to(device)
        masks = masks.to(device)
        stain_descs = stain_descs.to(device)
        
        # 前向传播
        outputs = model(images, stain_descs)
        loss = criterion(outputs, masks) / accum_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        # 每 accum_steps 步更新一次参数
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """
    验证
    
    Returns:
        avg_loss: 平均验证损失
        metrics: dict，包含 Dice、IoU 等指标
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for images, masks, stain_descs in pbar:
        images = images.to(device)
        masks = masks.to(device)
        stain_descs = stain_descs.to(device)
        
        outputs = model(images, stain_descs)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        # 累积指标
        tracker.update(outputs, masks)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    metrics = tracker.compute()
    avg_loss = total_loss / num_batches
    
    return avg_loss, metrics


def train(args):
    """主训练函数"""
    
    # ============================================================
    # 0. 准备
    # ============================================================
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    exp_name = f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(OUTPUT_DIR) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    # TensorBoard 日志
    writer = SummaryWriter(str(exp_dir / "tensorboard"))
    
    print(f"Experiment: {exp_name}")
    print(f"Output dir: {exp_dir}")
    
    # ============================================================
    # 1. 数据加载
    # ============================================================
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # ============================================================
    # 2. 模型
    # ============================================================
    print(f"\nCreating model: {args.model}")
    model = create_model(args.model)
    model = model.to(device)
    
    #     # torch.compile for faster training
    #     if hasattr(torch, "compile"):
    #         print("Compiling model with torch.compile...")
    #         model = torch.compile(model, mode="reduce-overhead")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # ============================================================
    # 3. 损失函数 + 优化器 + 调度器
    # ============================================================
    criterion = CombinedLoss()
    
    if OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
        )
    
    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    
    # ============================================================
    # 4. 训练循环
    # ============================================================
    best_dice = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dices = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            accum_steps=getattr(args, 'accum_steps', 1)
        )
        train_losses.append(train_loss)
        
        # 验证
        if epoch % VAL_INTERVAL == 0:
            val_loss, metrics = validate(
                model, val_loader, criterion, device, epoch
            )
            val_losses.append(val_loss)
            val_dices.append(metrics["mean_dice"])
            
            # 打印结果
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Mean Dice:  {metrics['mean_dice']:.4f}")
            print(f"  Mean IoU:   {metrics['mean_iou']:.4f}")
            
            for c in range(NUM_CLASSES):
                name = CLASS_NAMES[c]
                print(f"  Dice {name}: {metrics[f'dice_{name}']:.4f}")
            
            # TensorBoard 日志
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/mean_dice", metrics["mean_dice"], epoch)
            writer.add_scalar("Metrics/mean_iou", metrics["mean_iou"], epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            
            for c in range(NUM_CLASSES):
                name = CLASS_NAMES[c]
                writer.add_scalar(f"Dice/{name}", metrics[f"dice_{name}"], epoch)
            
            # 保存最优模型
            if metrics["mean_dice"] > best_dice:
                best_dice = metrics["mean_dice"]
                patience_counter = 0
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "metrics": metrics,
                    "config": {
                        "model": args.model,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "epochs": args.epochs,
                    },
                }, exp_dir / "checkpoints" / "best_model.pth")
                
                print(f"  ★ New best model! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOPPING} epochs)")
                    break
        
        # 更新学习率
        if SCHEDULER == "plateau":
            if epoch % VAL_INTERVAL == 0:
                scheduler.step(metrics["mean_dice"])
        else:
            scheduler.step()
    
    # ============================================================
    # 5. 保存最终结果
    # ============================================================
    # 保存最后一个模型
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_dice": best_dice,
    }, exp_dir / "checkpoints" / "last_model.pth")
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses, val_dices,
        save_path=exp_dir / "visualizations" / "training_curves.png"
    )
    
    writer.close()
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best Mean Dice: {best_dice:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 60)
    
    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STAG-UNet")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        choices=["unet", "attention_unet", "stag_unet"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    
    args = parser.parse_args()
    train(args)
