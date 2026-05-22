import os
import shutil
import random
from tqdm import tqdm
import sys

# =========================================================
# CONFIG IMPORT
# =========================================================
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(ROOT_DIR)

from config import *

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(SEED)

# =========================================================
# RESET SPLIT FOLDER
# =========================================================
print("=" * 60)
print("SPLIT PIPELINE")
print("=" * 60)

if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)

for s in SPLITS:
    os.makedirs(os.path.join(PROCESSED_DIR, s), exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
data = []

for cls in os.listdir(PROCESSED_DIR):

    class_path = os.path.join(PROCESSED_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    for f in os.listdir(class_path):

        if f.lower().endswith(VALID_EXT):

            data.append((os.path.join(class_path, f), cls))

print(f"[INFO] Total images: {len(data)}")

random.shuffle(data)

# =========================================================
# SPLIT
# =========================================================
n = len(data)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

# =========================================================
# COPY FUNCTION
# =========================================================
def copy(dataset, split):

    count = 0

    for path, cls in tqdm(dataset, desc=split):

        dst_dir = os.path.join(PROCESSED_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy2(path, os.path.join(dst_dir, os.path.basename(path)))
        count += 1

    return count

# =========================================================
# RUN
# =========================================================
train_c = copy(train, "train")
val_c   = copy(val, "val")
test_c  = copy(test, "test")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Train: {train_c}")
print(f"Val  : {val_c}")
print(f"Test : {test_c}")
print(f"Total: {train_c + val_c + test_c}")