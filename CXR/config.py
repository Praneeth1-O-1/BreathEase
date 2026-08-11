"""
Configuration for the Chest X-ray classification pipeline.

Auto-detects whether running locally (sample dataset) or on Kaggle
(full NIH ChestX-ray14). All paths, class names, and hyperparameters
are defined here so that no other file contains hardcoded values.

Kaggle usage:
    - Attach the NIH ChestX-ray14 dataset as an input.
    - Update KAGGLE_DATASET_NAME below to match the dataset slug.
    - All other settings are inherited automatically.

Local usage:
    - Place your sample CSV and images under the data/ directory.
"""

import os
import torch


# ============================================================
# Environment Detection
# ============================================================

IS_KAGGLE = os.path.exists("/kaggle/working")


# ============================================================
# Paths
# ============================================================

if IS_KAGGLE:
    # Kaggle: full NIH ChestX-ray14
    KAGGLE_DATASET_NAME = "nih-chest-xrays/data"
    DATA_DIR   = f"/kaggle/input/{KAGGLE_DATASET_NAME}"
    CSV_FILE   = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
    IMAGE_DIRS = [
        os.path.join(DATA_DIR, f"images_{i:03d}", "images")
        for i in range(1, 13)
    ]
    SPLIT_DIR  = "/kaggle/working/splits"
    OUTPUT_DIR = "/kaggle/working/outputs"

else:
    # Local: small sample dataset for debugging
    DATA_DIR   = "data"
    CSV_FILE   = os.path.join(DATA_DIR, "sample_labels.csv")
    IMAGE_DIRS = [os.path.join(DATA_DIR, "images")]
    SPLIT_DIR  = "splits"
    OUTPUT_DIR = "outputs"


# ============================================================
# Official NIH Split Files (optional but recommended on Kaggle)
# ============================================================

TRAIN_VAL_LIST = os.path.join(DATA_DIR, "train_val_list.txt")
TEST_LIST      = os.path.join(DATA_DIR, "test_list.txt")


# ============================================================
# NIH ChestX-ray14 Disease Classes
# ============================================================

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(CLASSES)   # 14


# ============================================================
# Backbone Architecture Configuration (Step 7)
# ============================================================
# Configurable choices: "densenet121", "efficientnet_b4"
# Default is "densenet121" to establish baseline on Kaggle first.

BACKBONE = "densenet121"


# ============================================================
# Image Resolution (Step 4 & Step 7)
# ============================================================
# RESIZE_SIZE: initial resize target
# IMAGE_SIZE: final crop size fed into network

if BACKBONE == "efficientnet_b4":
    IMAGE_SIZE  = 320    # EfficientNet-B4 high-res input
    RESIZE_SIZE = 352
else:
    IMAGE_SIZE  = 224    # DenseNet121 native input
    RESIZE_SIZE = 256


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


# ============================================================
# Training Hyperparameters
# ============================================================

# Epochs: 5 locally for quick testing. Raise to 40 on Kaggle (5 frozen + 35 unfrozen).
EPOCHS     = 5
BATCH_SIZE = 16

# Early stopping: stop if val AUROC does not improve for PATIENCE epochs.
PATIENCE   = 7

# DataLoader workers
NUM_WORKERS = 0 if not IS_KAGGLE else 4
PIN_MEMORY  = IS_KAGGLE


# ============================================================
# Step 5 — Optimizer & Two-Stage Fine-Tuning
# ============================================================

LR_BACKBONE  = 1e-4
LR_HEAD      = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0

# Stage 1: Backbone frozen for FREEZE_EPOCHS epochs (head training only).
# Stage 2: Backbone unfrozen for remaining epochs.
FREEZE_EPOCHS = 2 if not IS_KAGGLE else 5


# ============================================================
# Step 6 — OneCycleLR Scheduler
# ============================================================

PCT_START = 0.1


# ============================================================
# Step 8 — Automatic Mixed Precision (AMP)
# ============================================================
# Enables FP16 autocast training on CUDA GPUs to save VRAM and boost throughput.

USE_AMP = torch.cuda.is_available()


# ============================================================
# Step 9 — Logging Configuration
# ============================================================

CSV_LOG_FILE = os.path.join(OUTPUT_DIR, "history.csv")
TB_LOG_DIR   = os.path.join(OUTPUT_DIR, "tensorboard")
