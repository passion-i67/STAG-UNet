"""
dataset.py — PyTorch Dataset 类

负责：
1. 加载 patch 图像和 mask
2. 数据增强（训练时）
3. 加载对应的 stain descriptor
4. 前景优先采样
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加项目根目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *


def get_train_transforms(patch_size=PATCH_SIZE):
    """
    训练时的数据增强 (改进版 v2)
    
    新增:
    - ElasticTransform: 弹性变形,模拟组织切片变形
    - GaussNoise + GaussianBlur: 模拟扫描噪声与聚焦差异
    - HueSaturationValue: 更强颜色扰动,模拟染色差异(关键!)
    - RandomResizedCrop: 多尺度学习
    - CoarseDropout: 随机遮挡,提升鲁棒性
    """
    transforms = []
    
    # 几何增强 (必做)
    if AUG_HORIZONTAL_FLIP:
        transforms.append(A.HorizontalFlip(p=0.5))
    if AUG_VERTICAL_FLIP:
        transforms.append(A.VerticalFlip(p=0.5))
    transforms.append(A.RandomRotate90(p=0.5))
    transforms.append(A.Transpose(p=0.3))
    
    # 弹性变形 (病理图像专用,模拟组织形变)
    transforms.append(
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3)
    )
    
    # 多尺度 (小概率,避免破坏染色描述符)
    transforms.append(
        A.RandomResizedCrop(
            size=(patch_size, patch_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.3,
        )
    )
    
    # 颜色增强 (对病理图像尤其重要)
    if AUG_COLOR_JITTER:
        transforms.append(
            A.ColorJitter(
                brightness=AUG_COLOR_BRIGHTNESS,
                contrast=AUG_COLOR_CONTRAST,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            )
        )
    transforms.append(
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        )
    )
    
    # 噪声与模糊 (模拟扫描器差异)
    transforms.append(
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2)
    )
    
    # 归一化 + 转 tensor (必须在最后)
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)



def get_val_transforms():
    """
    验证/测试时的变换（只做归一化，不做增强）
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class BreastPathologyDataset(Dataset):
    """
    乳腺病理图像分割数据集
    
    使用方法：
        dataset = BreastPathologyDataset(
            image_dir="data/processed/train/images",
            mask_dir="data/processed/train/masks",
            stain_desc_path="data/stain_descriptors/stain_descriptors.npy",
            transform=get_train_transforms(),
        )
        image, mask, stain_desc = dataset[0]
    """
    
    def __init__(self, image_dir, mask_dir, stain_desc_path=None, transform=None):
        """
        Args:
            image_dir: patch 图像目录
            mask_dir: mask 目录
            stain_desc_path: 染色描述符文件路径（.npy）
            transform: albumentations 变换
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # 获取所有图像文件
        self.image_files = sorted(
            list(self.image_dir.glob("*.png")) +
            list(self.image_dir.glob("*.jpg")) +
            list(self.image_dir.glob("*.tif"))
        )
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # 加载染色描述符
        self.stain_descriptors = {}
        if stain_desc_path and os.path.exists(stain_desc_path):
            desc_data = np.load(stain_desc_path, allow_pickle=True).item()
            self.stain_descriptors = desc_data
        
        print(f"Dataset loaded: {len(self.image_files)} images from {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            image: (3, H, W) float tensor，已归一化
            mask: (H, W) long tensor，值为 0-3（类别索引）
            stain_desc: (6,) float tensor，染色描述符
        """
        # 读取图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取 mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 处理 mask 读取失败的情况
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 确保 mask 的值在 [0, NUM_CLASSES-1] 范围内
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        
        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]        # (3, H, W) float tensor
            mask = augmented["mask"]           # (H, W) tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
        
        mask = mask.long()
        
        # 获取染色描述符
        # 从文件名推断 WSI ID（去掉 _patchXXXX 后缀）
        stem = img_path.stem
        # 尝试找到匹配的描述符
        if stem in self.stain_descriptors:
            stain_desc = np.array(self.stain_descriptors[stem], dtype=np.float32)
        else:
            # 尝试去掉 patch 后缀
            base_name = "_".join(stem.split("_")[:-1]) if "_patch" in stem else stem
            if base_name in self.stain_descriptors:
                stain_desc = np.array(self.stain_descriptors[base_name], dtype=np.float32)
            else:
                # 使用零向量作为默认值
                stain_desc = np.zeros(STAIN_DESCRIPTOR_DIM, dtype=np.float32)
        
        stain_desc = torch.from_numpy(stain_desc)
        
        return image, mask, stain_desc
    
    def get_foreground_weights(self):
        """
        计算每个 patch 的采样权重（前景优先采样）
        
        含有 Invasive Epithelium 或 Necrosis 的 patch 权重更高
        
        Returns:
            weights: (N,) 采样权重
        """
        weights = []
        
        for img_path in self.image_files:
            mask_path = self.mask_dir / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                weights.append(1.0)
                continue
            
            # 检查是否包含重要类别
            has_invasive = np.any(mask == 0)    # class 0: Invasive Epithelium
            has_necrosis = np.any(mask == 2)    # class 2: Necrosis
            has_noninvasive = np.any(mask == 1)  # class 1: Non-invasive
            
            if has_invasive or has_necrosis:
                weights.append(3.0)  # 高权重
            elif has_noninvasive:
                weights.append(2.0)  # 中等权重
            else:
                weights.append(1.0)  # 普通权重（主要是 Other）
        
        return weights


def create_dataloaders(
    processed_dir=PROCESSED_DIR,
    stain_desc_path=None,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_foreground_sampling=USE_FOREGROUND_SAMPLING,
):
    """
    创建 train/val/test 的 DataLoader（一键搞定）
    
    Args:
        processed_dir: 处理后的数据目录
        stain_desc_path: 染色描述符路径
        batch_size: 批大小
        num_workers: 工作线程数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    processed_dir = Path(processed_dir)
    
    if stain_desc_path is None:
        stain_desc_path = Path(STAIN_DESC_DIR) / "stain_descriptors.npy"
    
    # 创建 Dataset
    train_dataset = BreastPathologyDataset(
        image_dir=processed_dir / "train" / "images",
        mask_dir=processed_dir / "train" / "masks",
        stain_desc_path=stain_desc_path,
        transform=get_train_transforms(),
    )
    
    val_dataset = BreastPathologyDataset(
        image_dir=processed_dir / "val" / "images",
        mask_dir=processed_dir / "val" / "masks",
        stain_desc_path=stain_desc_path,
        transform=get_val_transforms(),
    )
    
    # test 集可能不存在（BEETLE 的 evaluation 集没有公开 mask）
    test_dir = processed_dir / "test" / "images"
    if test_dir.exists() and len(list(test_dir.glob("*"))) > 0:
        test_dataset = BreastPathologyDataset(
            image_dir=processed_dir / "test" / "images",
            mask_dir=processed_dir / "test" / "masks",
            stain_desc_path=stain_desc_path,
            transform=get_val_transforms(),
        )
    else:
        test_dataset = None
    
    # 前景优先采样器
    train_sampler = None
    train_shuffle = True
    if use_foreground_sampling:
        weights = train_dataset.get_foreground_weights()
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        train_shuffle = False  # 使用 sampler 时不能 shuffle
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
    


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("Testing dataset...")
    
    # 测试变换
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    
    # 创建假数据测试
    fake_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    fake_mask = np.random.randint(0, 4, (256, 256), dtype=np.uint8)
    
    result = train_tf(image=fake_img, mask=fake_mask)
    print(f"Augmented image shape: {result['image'].shape}")
    print(f"Augmented mask shape:  {result['mask'].shape}")
    print(f"Image dtype: {result['image'].dtype}")
    print(f"Mask unique values: {torch.unique(result['mask'])}")
    
    print("\nDone!")
