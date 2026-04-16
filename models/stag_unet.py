"""
stag_unet.py — STAG-UNet: Stain-invariant Attention-Gated U-Net

★ 这是本项目的核心模型 ★

与 Attention U-Net 的关键区别：
1. 编码器换成 EfficientNet-B0（轻量 + ImageNet 预训练）
2. Attention Gate 中注入染色条件向量（STAG 模块）

STAG 模块原理：
    
    标准 Attention Gate:
        α = σ(ψ(ReLU(Wx·x + Wg·g)))
    
    STAG 改进版:
        s = MLP(stain_descriptor)          ← 染色信息编码
        α = σ(ψ(ReLU(Wx·x + Wg·g + Ws·s)))  ← 染色信息参与 attention 计算
    
    这样做的好处：
    - 不同染色风格的图像，attention 权重会自适应调整
    - 模型在面对"没见过的医院的图像"时更稳定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.cbam import CBAM  # 用来加载 EfficientNet

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    NUM_CLASSES, IN_CHANNELS, ENCODER_NAME, ENCODER_WEIGHTS,
    STAIN_DESCRIPTOR_DIM, STAG_CONDITION_DIM, STAG_MLP_HIDDEN,
)


class StainConditionMLP(nn.Module):
    """
    将染色描述符映射为条件向量
    
    输入：6 维的 stain descriptor（H 和 E 各 3 个通道的统计量）
    输出：32 维的 condition vector（用于注入 Attention Gate）
    
    结构简单：Linear → ReLU → Linear
    """
    def __init__(self, input_dim=STAIN_DESCRIPTOR_DIM, 
                 hidden_dim=STAG_MLP_HIDDEN, 
                 output_dim=STAG_CONDITION_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 6) 染色描述符
        Returns:
            (B, output_dim) 条件向量
        """
        return self.mlp(x)


class StainAttentionGate(nn.Module):
    """
    ★ STAG 模块 — 染色条件注意力门控 ★
    
    在标准 Attention Gate 基础上，加入染色条件向量 s：
        α = σ(ψ(ReLU(Wx·x + Wg·g + Ws·s)))
    
    其中 s 是通过 MLP 编码的全局染色描述符，
    通过 1x1 卷积 Ws 映射到与空间特征相同的维度，
    然后 broadcast 到每个空间位置。
    
    Args:
        gate_channels: 门控信号通道数
        skip_channels: skip connection 通道数
        inter_channels: 中间特征维度
        condition_dim: 染色条件向量维度
    """
    def __init__(self, gate_channels, skip_channels, inter_channels, 
                 condition_dim=STAG_CONDITION_DIM):
        super().__init__()
        
        # 标准 Attention Gate 组件
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        # ★ 新增：染色条件映射
        # 将染色条件向量（1D）映射到与空间特征相同的通道维度
        self.W_s = nn.Sequential(
            nn.Linear(condition_dim, inter_channels),
            nn.BatchNorm1d(inter_channels),
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g, stain_cond):
        """
        Args:
            x: (B, skip_ch, H, W) 编码器 skip feature
            g: (B, gate_ch, H', W') 解码器 gating signal
            stain_cond: (B, condition_dim) 染色条件向量
            
        Returns:
            attended: 加权后的 skip feature
            alpha: attention map
        """
        # 上采样 gate signal
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        
        # 标准部分
        x_mapped = self.W_x(x)       # (B, inter, H, W)
        g_mapped = self.W_g(g_up)     # (B, inter, H, W)
        
        # ★ 染色条件注入
        # stain_cond: (B, condition_dim) → (B, inter_channels)
        s_mapped = self.W_s(stain_cond)  # (B, inter)
        # 扩展空间维度：(B, inter) → (B, inter, 1, 1) → broadcast 到 (B, inter, H, W)
        s_mapped = s_mapped.unsqueeze(-1).unsqueeze(-1)
        
        # 三项加和
        combined = self.relu(x_mapped + g_mapped + s_mapped)
        alpha = self.psi(combined)
        
        attended = x * alpha
        return attended, alpha


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 拼接 + 卷积"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class STAGUNet(nn.Module):
    """
    ★ STAG-UNet: Stain-invariant Attention-Gated U-Net ★
    
    完整架构：
    1. EfficientNet-B0 编码器（ImageNet 预训练）
    2. 4 层解码器，每层配备 STAG 模块
    3. 染色条件 MLP 将 stain descriptor 编码为条件向量
    
    Args:
        in_channels: 输入通道数
        num_classes: 类别数
        encoder_name: 编码器名称
        pretrained: 是否使用预训练权重
        stain_desc_dim: 染色描述符维度
        condition_dim: 条件向量维度
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        encoder_name=ENCODER_NAME,
        pretrained=True,
        stain_desc_dim=STAIN_DESCRIPTOR_DIM,
        condition_dim=STAG_CONDITION_DIM,
    ):
        super().__init__()
        
        # ============================================================
        # 1. 编码器：EfficientNet-B0
        # ============================================================
        # 使用 timm 库加载，features_only=True 提取中间层特征
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,       # 只要特征，不要分类头
            in_chans=in_channels,
        )
        
        # 获取编码器每一层的输出通道数
        # EfficientNet-B0 典型输出：[16, 24, 40, 112, 320]（5 个层级）
        encoder_channels = self.encoder.feature_info.channels()
        print(f"Encoder channels: {encoder_channels}")
        
        # 我们使用后 4 层作为 skip connections
        # 通常 encoder_channels = [16, 24, 40, 112, 320]
        self.skip_channels = encoder_channels[:-1]  # [16, 24, 40, 112]
        self.bottleneck_channels = encoder_channels[-1]  # 320
        
        # ============================================================
        # 2. 染色条件 MLP
        # ============================================================
        self.stain_mlp = StainConditionMLP(
            input_dim=stain_desc_dim,
            hidden_dim=STAG_MLP_HIDDEN,
            output_dim=condition_dim,
        )
        
        # ============================================================
        # 3. 解码器 + STAG 模块
        # ============================================================
        decoder_channels = [256, 128, 64, 32]  # 解码器通道数
        
        self.stag_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()
        
        in_ch = self.bottleneck_channels  # 320
        
        for i, (dec_ch, skip_ch) in enumerate(
            zip(decoder_channels, reversed(self.skip_channels))
        ):
            # STAG 注意力门
            self.stag_gates.append(
                StainAttentionGate(
                    gate_channels=in_ch,
                    skip_channels=skip_ch,
                    inter_channels=skip_ch // 2 if skip_ch >= 4 else skip_ch,
                    condition_dim=condition_dim,
                )
            )
            
            # 解码器块
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, dec_ch)
            )
            
            self.cbam_blocks.append(CBAM(dec_ch))
            
            in_ch = dec_ch
        
        # ============================================================
        # 4. 最终分类头
        # ============================================================
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, 1),
        )
        
        # 存储 attention maps
        self.attention_maps = []
    
    def forward(self, x, stain_desc=None):
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
            stain_desc: (B, 6) 染色描述符。如果为 None，使用零向量
            
        Returns:
            output: (B, num_classes, H, W) 分割预测
        """
        input_size = x.shape[2:]  # 保存原始尺寸用于最终上采样
        
        # ============================================================
        # 编码器
        # ============================================================
        features = self.encoder(x)
        # features: list of 5 tensors，从浅到深
        # 例如 [f0: (B,16,128,128), f1: (B,24,64,64), f2: (B,40,32,32), 
        #        f3: (B,112,16,16), f4: (B,320,8,8)]
        
        skip_features = features[:-1]   # 前 4 层作为 skip
        bottleneck = features[-1]         # 最后一层作为 bottleneck
        
        # ============================================================
        # 染色条件编码
        # ============================================================
        if stain_desc is None:
            stain_desc = torch.zeros(x.shape[0], STAIN_DESCRIPTOR_DIM, device=x.device)
        
        stain_cond = self.stain_mlp(stain_desc)  # (B, condition_dim)
        
        # ============================================================
        # 解码器 + STAG
        # ============================================================
        x = bottleneck
        self.attention_maps = []
        
        for i, (stag_gate, decoder_block) in enumerate(
            zip(self.stag_gates, self.decoder_blocks)
        ):
            # 取对应层的 skip feature（从深到浅）
            skip = skip_features[-(i + 1)]
            
            # STAG 注意力门控
            attended_skip, alpha = stag_gate(skip, x, stain_cond)
            self.attention_maps.append(alpha)
            
            # 解码器上采样 + 拼接
            x = decoder_block(x, attended_skip)
            x = self.cbam_blocks[i](x)
        
        # ============================================================
        # 最终输出
        # ============================================================
        x = self.final_conv(x)
        
        # 上采样到原始输入尺寸
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        
        return x


# ============================================================
# 模型工厂函数（根据配置创建模型）
# ============================================================
def create_model(model_name, **kwargs):
    """
    根据名称创建模型
    
    Args:
        model_name: "unet", "attention_unet", "stag_unet"
        
    Returns:
        model: nn.Module
    """
    from models.unet import UNet
    from models.attention_unet import AttentionUNet
    
    models = {
        "unet": UNet,
        "attention_unet": AttentionUNet,
        "stag_unet": STAGUNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # 测试 STAG-UNet
    print("Testing STAG-UNet...")
    model = STAGUNet(pretrained=False)  # 测试时不下载预训练权重
    
    x = torch.randn(2, 3, 256, 256)
    stain_desc = torch.randn(2, 6)
    
    out = model(x, stain_desc)
    print(f"Input:       {x.shape}")
    print(f"Stain desc:  {stain_desc.shape}")
    print(f"Output:      {out.shape}")
    print(f"Params:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"Attention maps: {len(model.attention_maps)} layers")
    for i, am in enumerate(model.attention_maps):
        print(f"  Layer {i}: {am.shape}")
    
    # 测试不传 stain_desc
    out2 = model(x)
    print(f"\nWithout stain desc: {out2.shape}")
    
    print("\nAll tests passed!")
