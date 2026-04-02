"""
visualize.py — 可视化工具

用于生成：
1. 预测结果对比图（原图 / 真值 / 预测）
2. Attention Map 可视化
3. 类别分布统计图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import cv2
from pathlib import Path

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import NUM_CLASSES, CLASS_NAMES


# 类别对应的颜色（用于可视化）
CLASS_COLORS = np.array([
    [255, 0, 0],       # Invasive Epithelium — 红色
    [0, 255, 0],       # Non-invasive Epithelium — 绿色
    [255, 255, 0],     # Necrosis — 黄色
    [128, 128, 128],   # Other — 灰色
], dtype=np.uint8)


def mask_to_colormap(mask, class_colors=CLASS_COLORS):
    """
    将类别 mask 转为彩色图
    
    Args:
        mask: (H, W) 类别索引
    Returns:
        color_mask: (H, W, 3) RGB 彩色图
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(len(class_colors)):
        color_mask[mask == c] = class_colors[c]
    
    return color_mask


def plot_prediction(image, gt_mask, pred_mask, save_path=None, title=None):
    """
    绘制预测对比图：原图 | 真值 | 预测
    
    Args:
        image: (H, W, 3) 原始图像，RGB，uint8
        gt_mask: (H, W) 真值 mask
        pred_mask: (H, W) 预测 mask
        save_path: 保存路径（None 则显示）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # 真值
    gt_color = mask_to_colormap(gt_mask)
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # 预测
    pred_color = mask_to_colormap(pred_mask)
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    # 图例
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i])/255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attention_maps(attention_maps, save_path=None):
    """
    可视化 STAG attention maps
    
    Args:
        attention_maps: list of (1, 1, H, W) tensors
    """
    n = len(attention_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    if n == 1:
        axes = [axes]
    
    for i, attn in enumerate(attention_maps):
        if isinstance(attn, torch.Tensor):
            attn = attn[0, 0].cpu().numpy()  # 取第一张图，第一个通道
        
        im = axes[i].imshow(attn, cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(f"STAG Layer {i+1}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.suptitle("STAG Attention Maps", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, val_dices, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_losses: list，每个 epoch 的训练 loss
        val_losses: list，每个 epoch 的验证 loss
        val_dices: list，每个 epoch 的验证 mean Dice
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss 曲线
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice 曲线
    ax2.plot(epochs, val_dices, "g-", label="Val Mean Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将归一化的 tensor 还原为可显示的图像
    
    Args:
        tensor: (3, H, W) 归一化后的 tensor
    Returns:
        image: (H, W, 3) uint8 RGB 图像
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    img = tensor.transpose(1, 2, 0)  # (H, W, 3)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
