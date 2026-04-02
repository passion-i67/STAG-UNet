"""
losses.py — 损失函数

Dice Loss + Focal Loss 的组合：
- Dice Loss：衡量预测和真值的区域重叠，对类别不平衡更鲁棒
- Focal Loss：增强对困难样本的关注，减少简单样本的权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    NUM_CLASSES, LOSS_DICE_WEIGHT, LOSS_FOCAL_WEIGHT,
    FOCAL_ALPHA, FOCAL_GAMMA, CLASS_WEIGHTS,
)


class DiceLoss(nn.Module):
    """
    Dice Loss — 衡量预测和真值的重叠程度
    
    公式：Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    特点：对类别不平衡问题比交叉熵更友好
    """
    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-5, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) 模型输出 logits（未经 softmax）
            target: (B, H, W) 真值标签，值为 0 ~ C-1
            
        Returns:
            loss: 标量
        """
        # Softmax 转概率
        pred_soft = F.softmax(pred, dim=1)  # (B, C, H, W)
        
        # One-hot 编码 target
        target_onehot = F.one_hot(target, self.num_classes)  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # 逐类别计算 Dice
        dice_per_class = []
        for c in range(self.num_classes):
            pred_c = pred_soft[:, c]     # (B, H, W)
            target_c = target_onehot[:, c]  # (B, H, W)
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)
        
        dice_per_class = torch.stack(dice_per_class)
        
        # 加权平均
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            loss = 1.0 - (dice_per_class * weights).sum() / weights.sum()
        else:
            loss = 1.0 - dice_per_class.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss — 对困难样本加大权重
    
    公式：FL = -α * (1-p)^γ * log(p)
    
    当 γ=0 时退化为标准交叉熵
    γ 越大，对简单样本的惩罚越重（即更关注困难样本）
    """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, 
                 class_weights=None, num_classes=NUM_CLASSES):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) 类别标签
        """
        # 计算交叉熵（逐像素）
        ce_loss = F.cross_entropy(
            pred, target, 
            weight=self.class_weights.to(pred.device) if self.class_weights is not None else None,
            reduction="none"
        )  # (B, H, W)
        
        # 计算每个像素的预测概率
        pred_soft = F.softmax(pred, dim=1)  # (B, C, H, W)
        # 取出每个像素真实类别对应的概率
        target_expanded = target.unsqueeze(1)  # (B, 1, H, W)
        p_t = pred_soft.gather(1, target_expanded).squeeze(1)  # (B, H, W)
        
        # Focal 权重
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        
        # 加权损失
        loss = (focal_weight * ce_loss).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：Dice Loss + Focal Loss
    
    L_total = λ1 * L_dice + λ2 * L_focal
    """
    def __init__(
        self,
        dice_weight=LOSS_DICE_WEIGHT,
        focal_weight=LOSS_FOCAL_WEIGHT,
        class_weights=CLASS_WEIGHTS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.focal_loss = FocalLoss(
            num_classes=num_classes,
            class_weights=class_weights,
        )
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


if __name__ == "__main__":
    # 快速测试
    pred = torch.randn(2, 4, 64, 64)   # (B, C, H, W)
    target = torch.randint(0, 4, (2, 64, 64))  # (B, H, W)
    
    dice = DiceLoss()
    focal = FocalLoss()
    combined = CombinedLoss()
    
    print(f"Dice Loss:     {dice(pred, target).item():.4f}")
    print(f"Focal Loss:    {focal(pred, target).item():.4f}")
    print(f"Combined Loss: {combined(pred, target).item():.4f}")
