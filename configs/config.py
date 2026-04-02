"""
config.py — 所有超参数集中管理
"""
import os

# ============================================================
# 路径配置（根据你的环境修改这里）
# ============================================================
DATA_ROOT = "./data/raw"                    # BEETLE 原始数据
PROCESSED_DIR = "./data/processed_filtered" # 预处理后的 patch
STAIN_DESC_DIR = "./data/stain_descriptors" # 染色描述符
OUTPUT_DIR = "./outputs"                    # 训练输出
CSV_PATH = "./data/raw/data_overview.csv"   # BEETLE CSV

# ============================================================
# 数据预处理
# ============================================================
PATCH_SIZE = 256
OVERLAP = 64
MAGNIFICATION = "20x"
TISSUE_THRESHOLD = 0.3
BACKGROUND_THRESHOLD = 220

# ============================================================
# 类别定义 (BEETLE dataset)
# ============================================================
NUM_CLASSES = 4
CLASS_NAMES = [
    "Invasive Epithelium",      # class 0
    "Non-invasive Epithelium",  # class 1
    "Necrosis",                 # class 2
    "Other",                    # class 3
]
# BEETLE mask pixel values → class index
# mask=0→Other, mask=1→Invasive, mask=2→Non-invasive, mask=3→Necrosis, mask=4→Other
CLASS_PIXEL_VALUES = {
    0: 1,   # Invasive Epithelium  ← mask pixel value 1
    1: 2,   # Non-invasive Epithelium ← mask pixel value 2
    2: 3,   # Necrosis ← mask pixel value 3
    3: 0,   # Other ← mask pixel value 0
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
# 染色归一化
# ============================================================
USE_STAIN_NORM = True
STAIN_NORM_METHOD = "macenko"
REFERENCE_STAIN_MATRIX = None
STAIN_DESCRIPTOR_DIM = 6

# ============================================================
# 模型
# ============================================================
MODEL_NAME = "stag_unet"
ENCODER_NAME = "efficientnet_b0"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
STAG_CONDITION_DIM = 32
STAG_MLP_HIDDEN = 64

# ============================================================
# 训练
# ============================================================
SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 0           # Windows=0, Linux/HPC=4

OPTIMIZER = "adamw"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SCHEDULER = "cosine"
SCHEDULER_T_MAX = 50
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5

# ============================================================
# 损失函数
# ============================================================
LOSS_DICE_WEIGHT = 1.0
LOSS_FOCAL_WEIGHT = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [1.0, 1.0, 2.0, 0.5]

# ============================================================
# 验证
# ============================================================
VAL_INTERVAL = 1
SAVE_BEST_ONLY = True
EARLY_STOPPING = 5

# ============================================================
# 采样
# ============================================================
USE_FOREGROUND_SAMPLING = True
FOREGROUND_RATIO = 0.7

# ============================================================
# 推理
# ============================================================
USE_SLIDING_WINDOW = True
SLIDING_WINDOW_SIZE = 256
SLIDING_WINDOW_OVERLAP = 0.5
USE_TTA = False
