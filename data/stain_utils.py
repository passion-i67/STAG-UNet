"""
stain_utils.py — Macenko 染色归一化 + 染色描述符提取

这个文件做两件事：
1. Macenko 归一化：统一不同中心图像的染色风格
2. 提取 stain descriptor：作为 STAG 模块的条件输入
"""
import numpy as np
import cv2
from pathlib import Path


class MacenkoNormalizer:
    """
    Macenko 染色归一化器
    
    原理简述：
    - H&E 染色的图像由两种染料组成：Hematoxylin（蓝紫色，染细胞核）和 Eosin（粉红色，染细胞质）
    - 不同医院的染色浓度不同，导致图像颜色差异
    - Macenko 方法通过 SVD 分解提取染色矩阵，然后归一化到参考标准
    """
    
    def __init__(self, target_stain_matrix=None, target_concentrations=None):
        """
        Args:
            target_stain_matrix: 目标染色矩阵 (2x3)，None 则使用标准参考值
            target_concentrations: 目标浓度统计量，None 则使用标准参考值
        """
        # 标准 H&E 参考染色矩阵（来自文献常用值）
        if target_stain_matrix is None:
            self.target_stain_matrix = np.array([
                [0.5626, 0.2159, 0.7122],   # Hematoxylin
                [0.2461, 0.8012, 0.5425],   # Eosin
            ])
        else:
            self.target_stain_matrix = target_stain_matrix
            
        # 目标浓度的 99th percentile
        if target_concentrations is None:
            self.target_max_conc = np.array([1.9705, 1.0308])
        else:
            self.target_max_conc = target_concentrations
    
    def _get_stain_matrix(self, image, luminosity_threshold=0.8, angular_percentile=99):
        """
        从图像中提取染色矩阵（Macenko 方法核心）
        
        Args:
            image: RGB 图像 (H, W, 3), uint8
            luminosity_threshold: 亮度阈值，过滤背景像素
            angular_percentile: 角度百分位数
            
        Returns:
            stain_matrix: (2, 3) 染色矩阵
            concentrations: (N, 2) 染色浓度
            max_conc: (2,) 浓度最大值
        """
        # Step 1: 转换到光密度空间 (Optical Density)
        image = image.astype(np.float64)
        image = np.clip(image, 1, 255)  # 避免 log(0)
        od = -np.log(image / 255.0)
        
        # Step 2: 去除背景（亮度太高的像素）
        od_flat = od.reshape(-1, 3)
        od_threshold = -np.log(luminosity_threshold)
        mask = np.all(od_flat > od_threshold, axis=1)
        od_hat = od_flat[mask]
        
        if od_hat.shape[0] < 10:
            # 组织区域太少，返回默认值
            return self.target_stain_matrix, None, self.target_max_conc
        
        # Step 3: SVD 分解
        _, _, V = np.linalg.svd(od_hat, full_matrices=False)
        
        # 取前两个主成分方向
        plane = V[:2, :]
        
        # Step 4: 将数据投影到这个平面上
        projected = od_hat @ plane.T
        
        # 计算角度
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        
        # 取极端角度对应的方向
        min_angle = np.percentile(angles, 100 - angular_percentile)
        max_angle = np.percentile(angles, angular_percentile)
        
        # 构造染色向量
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane
        
        # 确保 Hematoxylin 在前（蓝色分量更大的那个）
        if v1[0] < v2[0]:
            stain_matrix = np.array([v1, v2])
        else:
            stain_matrix = np.array([v2, v1])
        
        # 归一化
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        
        # Step 5: 计算浓度
        concentrations = od_flat @ np.linalg.pinv(stain_matrix).T
        max_conc = np.percentile(concentrations, 99, axis=0)
        
        return stain_matrix, concentrations, max_conc
    
    def normalize(self, image):
        """
        对一张图像进行 Macenko 归一化
        
        Args:
            image: RGB 图像 (H, W, 3), uint8
            
        Returns:
            normalized: 归一化后的图像 (H, W, 3), uint8
        """
        h, w, c = image.shape
        
        try:
            # 提取源图像的染色矩阵
            src_stain, src_conc, src_max = self._get_stain_matrix(image)
            
            if src_conc is None:
                return image  # 无法提取，返回原图
            
            # 重新计算浓度
            od = -np.log(np.clip(image.astype(np.float64), 1, 255) / 255.0)
            od_flat = od.reshape(-1, 3)
            concentrations = od_flat @ np.linalg.pinv(src_stain).T
            
            # 归一化浓度到目标范围
            max_conc = np.percentile(concentrations, 99, axis=0)
            max_conc = np.clip(max_conc, 1e-6, None)  # 避免除零
            concentrations = concentrations * (self.target_max_conc / max_conc)
            
            # 用目标染色矩阵重建图像
            od_normalized = concentrations @ self.target_stain_matrix
            normalized = 255.0 * np.exp(-od_normalized)
            normalized = normalized.reshape(h, w, 3)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            return normalized
            
        except Exception as e:
            # 归一化失败时返回原图（不要因为一张图崩掉整个流程）
            print(f"[Warning] Stain normalization failed: {e}")
            return image


def extract_stain_descriptor(image, normalizer=None):
    """
    从一张图像中提取染色描述符（用于 STAG 模块的条件输入）
    
    描述符内容：H 和 E 两个通道各 3 个颜色均值 = 6 维向量
    
    Args:
        image: RGB 图像 (H, W, 3), uint8
        normalizer: MacenkoNormalizer 实例
        
    Returns:
        descriptor: (6,) numpy array，染色描述符
    """
    if normalizer is None:
        normalizer = MacenkoNormalizer()
    
    try:
        stain_matrix, _, _ = normalizer._get_stain_matrix(image)
        # 染色描述符 = 两个染色向量拼接 = 6 维
        descriptor = stain_matrix.flatten().astype(np.float32)
    except Exception:
        # 失败时返回默认描述符
        descriptor = np.zeros(6, dtype=np.float32)
    
    return descriptor


def extract_slide_descriptor(image_paths, normalizer=None):
    """
    从一张 WSI 的多个 patch 中提取 slide-level 染色描述符
    （取所有 patch 描述符的均值，更稳定）
    
    Args:
        image_paths: 该 WSI 对应的所有 patch 路径列表
        normalizer: MacenkoNormalizer 实例
        
    Returns:
        descriptor: (6,) numpy array
    """
    if normalizer is None:
        normalizer = MacenkoNormalizer()
    
    descriptors = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        desc = extract_stain_descriptor(img, normalizer)
        if np.any(desc != 0):
            descriptors.append(desc)
    
    if len(descriptors) == 0:
        return np.zeros(6, dtype=np.float32)
    
    return np.mean(descriptors, axis=0).astype(np.float32)


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    # 测试染色归一化
    print("Testing Macenko normalizer...")
    
    # 创建一个假图像测试
    fake_image = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
    
    normalizer = MacenkoNormalizer()
    normalized = normalizer.normalize(fake_image)
    print(f"Input shape: {fake_image.shape}, Output shape: {normalized.shape}")
    
    descriptor = extract_stain_descriptor(fake_image, normalizer)
    print(f"Stain descriptor: {descriptor}, shape: {descriptor.shape}")
    
    print("Done!")
