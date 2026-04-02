"""
attention_unet.py — Attention U-Net

在标准 U-Net 基础上，在 skip connection 处加入 Attention Gate。
Attention Gate 可以自动学习"哪些区域的特征更重要"，抑制无关背景。

Attention Gate 原理：
    编码器特征 x ──→ [Wx] ──┐
                              ├──→ ReLU → ψ → Sigmoid → α (attention map)
    解码器信号 g ──→ [Wg] ──┘
    
    最终输出 = x * α （用 attention map 对编码器特征加权）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import NUM_CLASSES, IN_CHANNELS


class ConvBlock(nn.Module):
    """卷积块：Conv → BN → ReLU → Conv → BN → ReLU"""
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


class AttentionGate(nn.Module):
    """
    标准 Attention Gate
    
    公式：α = σ(ψ(ReLU(Wx·x + Wg·g)))
    
    Args:
        gate_channels: 门控信号通道数（来自解码器）
        skip_channels: 跳跃连接通道数（来自编码器）
        inter_channels: 中间特征通道数
    """
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        
        # Wx: 将 skip feature 映射到中间维度
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        # Wg: 将 gate signal 映射到中间维度
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        # ψ: 将加和后的特征映射到单通道 attention map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        """
        Args:
            x: skip connection feature (B, skip_channels, H, W) — 来自编码器
            g: gating signal (B, gate_channels, H', W') — 来自解码器（空间尺寸可能不同）
            
        Returns:
            attended: x 经过 attention 加权后的特征
            alpha: attention map（用于可视化）
        """
        # 将 g 上采样到和 x 相同的空间尺寸
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        
        # 计算 attention
        x_mapped = self.W_x(x)
        g_mapped = self.W_g(g_up)
        
        combined = self.relu(x_mapped + g_mapped)
        alpha = self.psi(combined)  # (B, 1, H, W)
        
        # 用 attention map 加权 skip feature
        attended = x * alpha
        
        return attended, alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net
    
    与标准 U-Net 的区别：在每个 skip connection 处加了 Attention Gate
    
    Args:
        in_channels: 输入通道数
        num_classes: 类别数
        features: 每层通道数
    """
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                 features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 编码器
        ch = in_channels
        for feature in features:
            self.encoders.append(ConvBlock(ch, feature))
            ch = feature
        
        # 瓶颈
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # 解码器 + Attention Gates
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        in_ch = features[-1] * 2  # bottleneck 输出通道数
        
        for i, feature in enumerate(reversed_features):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, feature, 2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(
                    gate_channels=in_ch,
                    skip_channels=feature,
                    inter_channels=feature // 2,
                )
            )
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))
            in_ch = feature  # 下一层的输入 = 这一层的输出
        
        # 最终分类头
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
        
        # 存储 attention maps（用于后续可视化）
        self.attention_maps = []
    
    def forward(self, x, stain_desc=None):
        """
        Args:
            x: (B, 3, H, W) 输入图像
            stain_desc: 染色描述符（Attention U-Net 不使用，保留接口一致性）
            
        Returns:
            output: (B, num_classes, H, W)
        """
        # 编码器
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码器
        skip_connections = skip_connections[::-1]
        self.attention_maps = []
        
        for i in range(len(self.upconvs)):
            # 用当前解码器特征作为 gate signal
            skip = skip_connections[i]
            
            # Attention Gate
            attended_skip, alpha = self.attention_gates[i](skip, x)
            self.attention_maps.append(alpha)
            
            # 上采样
            x = self.upconvs[i](x)
            
            # 尺寸对齐
            if x.shape != attended_skip.shape:
                x = F.interpolate(x, size=attended_skip.shape[2:], 
                                  mode="bilinear", align_corners=True)
            
            # 拼接 + 卷积
            x = torch.cat([attended_skip, x], dim=1)
            x = self.decoder_blocks[i](x)
        
        return self.final_conv(x)


if __name__ == "__main__":
    model = AttentionUNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Attention maps: {len(model.attention_maps)} layers")
    for i, am in enumerate(model.attention_maps):
        print(f"  Layer {i}: {am.shape}")
