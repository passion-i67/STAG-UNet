"""
unet.py — Baseline U-Net

这是最基础的分割模型，作为实验的第一个 baseline。
U-Net 结构：编码器逐步下采样，解码器逐步上采样，通过 skip connection 融合。

结构图：
    输入图像 (3, 256, 256)
        ↓
    [Encoder Block 1] → 64 channels  ─────────────────→ [Decoder Block 4] → 64 → 输出 (4, 256, 256)
        ↓ MaxPool                                         ↑ UpConv
    [Encoder Block 2] → 128 channels ────────────→ [Decoder Block 3] → 128
        ↓ MaxPool                                         ↑ UpConv
    [Encoder Block 3] → 256 channels ──────→ [Decoder Block 2] → 256
        ↓ MaxPool                                   ↑ UpConv
    [Encoder Block 4] → 512 channels → [Decoder Block 1] → 512
        ↓ MaxPool                        ↑ UpConv
    [Bottleneck] → 1024 channels ─────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import NUM_CLASSES, IN_CHANNELS


class ConvBlock(nn.Module):
    """
    卷积块：Conv → BN → ReLU → Conv → BN → ReLU
    这是 U-Net 的基本构建单元
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net
    
    Args:
        in_channels: 输入通道数（RGB = 3）
        num_classes: 分割类别数（4类）
        features: 每层的通道数列表
    """
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, 
                 features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 编码器（下采样路径）
        for feature in features:
            self.encoders.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # 解码器（上采样路径）
        for feature in reversed(features):
            # 上采样卷积
            self.decoders.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, stride=2)
            )
            # 拼接后的卷积块
            self.decoders.append(ConvBlock(feature * 2, feature))
        
        # 最终分类头
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
    
    def forward(self, x, stain_desc=None):
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            stain_desc: 染色描述符（U-Net 不使用，保留接口一致性）
            
        Returns:
            output: (B, num_classes, H, W) 分割预测
        """
        # 编码器：保存每一层的特征用于 skip connection
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码器：上采样 + 拼接 skip connection
        skip_connections = skip_connections[::-1]  # 翻转顺序
        
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)           # 上采样
            skip = skip_connections[idx // 2]     # 对应的 skip feature
            
            # 处理尺寸不匹配（可能因为奇数尺寸导致）
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            
            x = torch.cat([skip, x], dim=1)     # 拼接
            x = self.decoders[idx + 1](x)        # 卷积处理
        
        return self.final_conv(x)


if __name__ == "__main__":
    # 快速测试
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
