"""
metrics.py — 评价指标

包含分割任务常用的评价指标：
- Dice Coefficient（Dice 系数）
- IoU / Jaccard Index
- HD95（95th percentile Hausdorff Distance）
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import NUM_CLASSES, CLASS_NAMES


def compute_dice(pred, target, num_classes=NUM_CLASSES, smooth=1e-5):
    """
    计算每个类别的 Dice 系数
    
    Dice = 2|A∩B| / (|A| + |B|)
    
    Args:
        pred: (B, C, H, W) logits 或 softmax 概率
        target: (B, H, W) 类别标签
        
    Returns:
        dice_per_class: (C,) 每个类别的 Dice
        mean_dice: 标量，所有类别的平均 Dice
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.argmax(pred, dim=1).cpu().numpy()  # (B, H, W)
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    dice_per_class = np.zeros(num_classes)
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice_per_class[c] = 1.0  # 如果该类别在 pred 和 target 中都不存在
        else:
            dice_per_class[c] = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice_per_class, np.mean(dice_per_class)


def compute_iou(pred, target, num_classes=NUM_CLASSES, smooth=1e-5):
    """
    计算每个类别的 IoU (Intersection over Union)
    
    IoU = |A∩B| / |A∪B|
    
    Args:
        pred: (B, C, H, W) logits 或 (B, H, W) 预测
        target: (B, H, W) 标签
        
    Returns:
        iou_per_class: (C,) 每个类别的 IoU
        mean_iou: 标量
    """
    if isinstance(pred, torch.Tensor) and pred.dim() == 4:
        pred = torch.argmax(pred, dim=1).cpu().numpy()
    elif isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    iou_per_class = np.zeros(num_classes)
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        if union == 0:
            iou_per_class[c] = 1.0
        else:
            iou_per_class[c] = (intersection + smooth) / (union + smooth)
    
    return iou_per_class, np.mean(iou_per_class)


def compute_hd95(pred, target, num_classes=NUM_CLASSES):
    """
    计算 95th percentile Hausdorff Distance (HD95)
    
    HD95 衡量预测边界和真值边界之间的最大距离（取 95 百分位以减少异常值影响）
    
    Args:
        pred: (H, W) 单张图的预测（类别索引）
        target: (H, W) 单张图的真值
        
    Returns:
        hd95_per_class: dict, 每个类别的 HD95
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 如果是 batch，取第一张
    if pred.ndim == 3:
        pred = pred[0]
        target = target[0]
    
    hd95_per_class = {}
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        
        # 如果该类在 pred 或 target 中不存在
        if pred_c.sum() == 0 and target_c.sum() == 0:
            hd95_per_class[c] = 0.0
            continue
        if pred_c.sum() == 0 or target_c.sum() == 0:
            hd95_per_class[c] = float('inf')
            continue
        
        # 提取边界点
        pred_boundary = _get_boundary_points(pred_c)
        target_boundary = _get_boundary_points(target_c)
        
        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            hd95_per_class[c] = float('inf')
            continue
        
        # 计算距离
        distances_pred_to_target = _compute_surface_distances(pred_boundary, target_boundary)
        distances_target_to_pred = _compute_surface_distances(target_boundary, pred_boundary)
        
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        hd95_per_class[c] = np.percentile(all_distances, 95)
    
    return hd95_per_class


def _get_boundary_points(binary_mask):
    """获取二值 mask 的边界点坐标"""
    # 使用形态学操作提取边界
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_mask)
    boundary = binary_mask.astype(np.int32) - eroded.astype(np.int32)
    points = np.argwhere(boundary > 0)
    return points


def _compute_surface_distances(points_a, points_b):
    """计算点集 A 中每个点到点集 B 的最近距离"""
    from scipy.spatial import cKDTree
    tree_b = cKDTree(points_b)
    distances, _ = tree_b.query(points_a)
    return distances


class MetricsTracker:
    """
    指标追踪器 — 在训练过程中累积计算指标
    
    用法：
        tracker = MetricsTracker()
        for batch in dataloader:
            tracker.update(pred, target)
        results = tracker.compute()
    """
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有累积值"""
        self.total_intersection = np.zeros(self.num_classes)
        self.total_pred = np.zeros(self.num_classes)
        self.total_target = np.zeros(self.num_classes)
        self.count = 0
    
    def update(self, pred, target):
        """
        更新一个 batch 的统计量
        
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) 标签
        """
        if isinstance(pred, torch.Tensor) and pred.dim() == 4:
            pred = torch.argmax(pred, dim=1).cpu().numpy()
        elif isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        for c in range(self.num_classes):
            pred_c = (pred == c).astype(np.float32)
            target_c = (target == c).astype(np.float32)
            
            self.total_intersection[c] += (pred_c * target_c).sum()
            self.total_pred[c] += pred_c.sum()
            self.total_target[c] += target_c.sum()
        
        self.count += pred.shape[0]
    
    def compute(self, smooth=1e-5):
        """
        计算累积的指标
        
        Returns:
            dict: 包含所有指标的字典
        """
        # Dice per class
        dice = (2.0 * self.total_intersection + smooth) / \
               (self.total_pred + self.total_target + smooth)
        
        # IoU per class
        union = self.total_pred + self.total_target - self.total_intersection
        iou = (self.total_intersection + smooth) / (union + smooth)
        
        results = {
            "mean_dice": np.mean(dice),
            "mean_iou": np.mean(iou),
        }
        
        for c in range(self.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
            results[f"dice_{name}"] = dice[c]
            results[f"iou_{name}"] = iou[c]
        
        return results
    
    def print_results(self, results=None):
        """打印格式化的结果"""
        if results is None:
            results = self.compute()
        
        print("\n" + "=" * 60)
        print(f"{'Metric':<35s} {'Value':>10s}")
        print("-" * 60)
        print(f"{'Mean Dice':<35s} {results['mean_dice']:>10.4f}")
        print(f"{'Mean IoU':<35s} {results['mean_iou']:>10.4f}")
        print("-" * 60)
        
        for c in range(self.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
            print(f"{'Dice - ' + name:<35s} {results[f'dice_{name}']:>10.4f}")
            print(f"{'IoU  - ' + name:<35s} {results[f'iou_{name}']:>10.4f}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试
    pred = torch.randn(4, 4, 64, 64)
    target = torch.randint(0, 4, (4, 64, 64))
    
    dice_per_class, mean_dice = compute_dice(pred, target)
    iou_per_class, mean_iou = compute_iou(pred, target)
    
    print(f"Dice per class: {dice_per_class}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"IoU per class: {iou_per_class}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # 测试 tracker
    tracker = MetricsTracker()
    tracker.update(pred, target)
    tracker.print_results()
