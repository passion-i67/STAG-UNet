"""
config.py — 所有超参数集中管理
修改实验设置只需要改这个文件，不需要到处找参数
"""
import os

# ============================================================
# 路径配置（根据你的环境修改这里）
# ============================================================
# --- 本地 VSCode ---
# DATA_ROOT = "/path/to/your/BEETLE/dataset"
# OUTPUT_DIR = "./outputs"

# --- Google Colab ---
# DATA_ROOT = "/content/drive/MyDrive/BEETLE"
# OUTPUT_DIR = "/content/outputs"

# --- HPC4 ---
# DATA_ROOT = "/home/YOUR_ITSC/data/BEETLE"
# OUTPUT_DIR = "/home/YOUR_ITSC/outputs"

# 你的本地 Windows 路径
DATA_ROOT = "./data"
PROCESSED_DIR = r"/scratch/zwangot/processed_v2"
STAIN_DESC_DIR = r"/scratch/zwangot/stain_descriptors"
OUTPUT_DIR = r"./outputs"
CSV_PATH = r"./data/data_overview.csv"


# ============================================================
# 数据预处理参数
# ============================================================
PATCH_SIZE = 256          # patch 大小（像素）
OVERLAP = 64              # patch 之间的重叠像素数
MAGNIFICATION = "20x"     # 使用的放大倍率
TISSUE_THRESHOLD = 0.3    # 组织区域占比阈值（低于此值的 patch 丢弃）
BACKGROUND_THRESHOLD = 220  # 灰度高于此值视为背景

# ============================================================
# 类别定义
# ============================================================
NUM_CLASSES = 4
CLASS_NAMES = [
    "Invasive Epithelium",   # 0
    "Non-invasive Epithelium",  # 1
    "Necrosis",              # 2
    "Other",                 # 3
]
# BEETLE mask 编码：0=Other, 1=Invasive, 2=Non-invasive, 3=Necrosis, 4→归入Other
CLASS_PIXEL_VALUES = {
    0: 1,   # Invasive Epithelium  ← mask 像素值 1
    1: 2,   # Non-invasive Epithelium ← mask 像素值 2
    2: 3,   # Necrosis ← mask 像素值 3
    3: 0,   # Other ← mask 像素值 0（背景）
}

# ============================================================
# 数据增强
# ============================================================
USE_AUGMENTATION = True
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP = True
AUG_ROTATION_LIMIT = 90
AUG_COLOR_JITTER = True
AUG_COLOR_BRIGHTNESS = 0.2
AUG_COLOR_CONTRAST = 0.2

# ============================================================
# 染色归一化参数
# ============================================================
USE_STAIN_NORM = True                # 是否使用 Macenko 染色归一化
STAIN_NORM_METHOD = "macenko"        # 归一化方法
# Macenko 参考图像的 stain matrix（标准 H&E 参考值）
REFERENCE_STAIN_MATRIX = None        # None = 使用默认参考
STAIN_DESCRIPTOR_DIM = 6             # 染色描述符维度（H和E各3个通道的均值）

# ============================================================
# 模型配置
# ============================================================
MODEL_NAME = "stag_unet"  # 可选: "unet", "attention_unet", "stag_unet"
ENCODER_NAME = "efficientnet_b0"     # 编码器骨干网络（timm 格式用下划线）
ENCODER_WEIGHTS = "imagenet"         # 预训练权重
IN_CHANNELS = 3                      # 输入通道数（RGB）

# STAG 模块参数
STAG_CONDITION_DIM = 32   # stain condition vector 的维度
STAG_MLP_HIDDEN = 64      # STAG 中 MLP 的隐藏层维度

# ============================================================
# 训练参数
# ============================================================
SEED = 42                 # 随机种子（保证可复现）
BATCH_SIZE = 16           # 批大小
NUM_EPOCHS = 50           # 训练轮数
NUM_WORKERS = 8           # DataLoader 工作线程数, Windows=0, Linux/HPC=4

# 优化器
OPTIMIZER = "adamw"       # 可选: "adam", "adamw"
LEARNING_RATE = 1e-4      # 初始学习率
WEIGHT_DECAY = 1e-4       # 权重衰减

# 学习率调度器
SCHEDULER = "cosine"      # 可选: "cosine", "step", "plateau"
SCHEDULER_T_MAX = 50      # cosine scheduler 的 T_max
SCHEDULER_STEP_SIZE = 10  # step scheduler 的 step_size
SCHEDULER_GAMMA = 0.5     # step scheduler 的 gamma

# ============================================================
# 损失函数
# ============================================================
LOSS_DICE_WEIGHT = 1.0    # Dice Loss 权重
LOSS_FOCAL_WEIGHT = 1.0   # Focal Loss 权重
FOCAL_ALPHA = 0.25        # Focal Loss alpha
FOCAL_GAMMA = 2.0         # Focal Loss gamma
# 类别权重（Necrosis 通常最少，给更高权重）
CLASS_WEIGHTS = [1.0, 1.0, 2.0, 0.5]  # [Invasive, Non-invasive, Necrosis, Other]

# ============================================================
# 验证与评估
# ============================================================
VAL_INTERVAL = 1          # 每隔几个 epoch 做一次验证
SAVE_BEST_ONLY = True     # 只保存最优模型
EARLY_STOPPING = 20        # 早停 patience（验证集无提升多少轮后停止）

# ============================================================
# 采样策略
# ============================================================
USE_FOREGROUND_SAMPLING = True   # 是否使用前景优先采样
FOREGROUND_RATIO = 0.7           # mini-batch 中前景 patch 的比例

# ============================================================
# 推理
# ============================================================
USE_SLIDING_WINDOW = True        # 是否使用滑窗推理
SLIDING_WINDOW_SIZE = 256        # 滑窗大小
SLIDING_WINDOW_OVERLAP = 0.5     # 滑窗重叠率
USE_TTA = False                  # 是否使用 test-time augmentation
