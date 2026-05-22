import os
import random
import numpy as np

# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

# =========================================================
# DATASET PATHS
# =========================================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

RAW_DIR = os.path.join(DATASET_DIR, "raw")
BALANCED_DIR = os.path.join(DATASET_DIR, "balanced")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
SPLIT_DIR = os.path.join(DATASET_DIR, "split")

# =========================================================
# OUTPUT PATHS
# =========================================================

CNN_OUTPUT = os.path.join(PROCESSED_DIR, "cnn_224")
TRANSFORMER_OUTPUT = os.path.join(PROCESSED_DIR, "transformer_384")

# =========================================================
# IMAGE SETTINGS
# =========================================================

CNN_SIZE = (224, 224)
TRANSFORMER_SIZE = (384, 384)

MIN_IMAGE_SIZE = 50

VALID_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

# =========================================================
# DATA SPLIT
# =========================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6, "Split ratio must equal 1.0"

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# TRAINING CONFIG
# =========================================================

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# =========================================================
# MODEL CONFIG
# =========================================================

NUM_CLASSES = 10

# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda"

# =========================================================
# DATASET TYPES
# =========================================================

DATASET_TYPES = [
    "cnn_224",
    "transformer_384"
]

SPLITS = [
    "train",
    "val",
    "test"
]