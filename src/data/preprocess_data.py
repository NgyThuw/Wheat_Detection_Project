import os
import cv2
import shutil
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

# =========================================================
# RESET PROCESSED FOLDER
# =========================================================
print("=" * 60)
print("PREPROCESS PIPELINE")
print("=" * 60)

if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =========================================================
# PROCESS
# =========================================================
stats = {}

for class_name in os.listdir(BALANCED_DIR):

    class_path = os.path.join(BALANCED_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(VALID_EXT)
    ]

    stats[class_name] = {"ok": 0, "skip": 0}

    save_dir = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    for f in tqdm(images, desc=f"Preprocess {class_name}"):

        path = os.path.join(class_path, f)

        try:
            img = cv2.imread(path)

            if img is None:
                stats[class_name]["skip"] += 1
                continue

            h, w = img.shape[:2]

            if h < 50 or w < 50:
                stats[class_name]["skip"] += 1
                continue

            # save cleaned image
            dst = os.path.join(save_dir, f)
            shutil.copy2(path, dst)

            stats[class_name]["ok"] += 1

        except:
            stats[class_name]["skip"] += 1

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("PREPROCESS DONE")
print("=" * 60)

total = 0

for cls, st in stats.items():
    print(f"{cls}: OK={st['ok']} | SKIP={st['skip']}")
    total += st["ok"]

print(f"\nTOTAL CLEAN IMAGES: {total}")
print(f"SAVED TO: {PROCESSED_DIR}")