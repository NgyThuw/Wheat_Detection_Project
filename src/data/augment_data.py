import os
import cv2
import random
from tqdm import tqdm
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(ROOT_DIR)

from config import *

# =========================================================
# AUGMENT FUNCTIONS
# =========================================================

def flip(img):
    return cv2.flip(img, 1)

def rotate(img):
    angle = random.randint(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def brightness(img):
    factor = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

AUG_LIST = [flip, rotate, brightness, blur]

# =========================================================
# PROCESS EACH CLASS
# =========================================================

print("=" * 60)
print("START OFFLINE AUGMENTATION (TRAIN ONLY)")
print("=" * 60)

for class_name in os.listdir(TRAIN_DIR):

    class_path = os.path.join(TRAIN_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(AUG_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(VALID_EXT)
    ]

    print(f"\nClass: {class_name} | Images: {len(images)}")

    for img_name in tqdm(images, desc=f"AUG-{class_name}"):

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # save original image first
        cv2.imwrite(
            os.path.join(output_class_path, img_name),
            img
        )

        # create augmented images
        for i in range(AUG_PER_IMAGE):

            aug_func = random.choice(AUG_LIST)
            aug_img = aug_func(img)

            new_name = img_name.replace(".jpg", f"_aug{i}.jpg")

            save_path = os.path.join(output_class_path, new_name)

            cv2.imwrite(save_path, aug_img)

print("\n" + "=" * 60)
print("AUGMENTATION COMPLETED")
print("=" * 60)
print("Saved to:", AUG_DIR)