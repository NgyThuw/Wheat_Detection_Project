import os
import random
import shutil

# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "split")

# =========================================================
# SPLIT RATIO
# =========================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# =========================================================
# RANDOM SEED
# =========================================================

random.seed(42)

# =========================================================
# DATASET TYPES
# =========================================================

DATASET_TYPES = [
    "cnn_224",
    "transformer_384"
]

# =========================================================
# CREATE SPLIT FOLDERS
# =========================================================

for dataset_type in DATASET_TYPES:

    for split_name in ["train", "val", "test"]:

        split_path = os.path.join(
            SPLIT_DIR,
            dataset_type,
            split_name
        )

        os.makedirs(split_path, exist_ok=True)

# =========================================================
# START SPLITTING
# =========================================================

print("=" * 60)
print("START SPLITTING DATASET")
print("=" * 60)

for dataset_type in DATASET_TYPES:

    print(f"\nProcessing: {dataset_type}")

    dataset_path = os.path.join(
        PROCESSED_DIR,
        dataset_type
    )

    # =====================================================
    # LOOP CLASSES
    # =====================================================

    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        print(f"  -> Class: {class_name}")

        # =================================================
        # GET IMAGE LIST
        # =================================================

        image_files = []

        for file_name in os.listdir(class_path):

            file_path = os.path.join(
                class_path,
                file_name
            )

            if os.path.isfile(file_path):
                image_files.append(file_name)

        # shuffle
        random.shuffle(image_files)

        total_images = len(image_files)

        # =================================================
        # SPLIT INDEX
        # =================================================

        train_end = int(total_images * TRAIN_RATIO)
        val_end = train_end + int(total_images * VAL_RATIO)

        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        # =================================================
        # CREATE CLASS FOLDERS
        # =================================================

        train_class_dir = os.path.join(
            SPLIT_DIR,
            dataset_type,
            "train",
            class_name
        )

        val_class_dir = os.path.join(
            SPLIT_DIR,
            dataset_type,
            "val",
            class_name
        )

        test_class_dir = os.path.join(
            SPLIT_DIR,
            dataset_type,
            "test",
            class_name
        )

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # =================================================
        # COPY TRAIN
        # =================================================

        for file_name in train_files:

            src_path = os.path.join(
                class_path,
                file_name
            )

            dst_path = os.path.join(
                train_class_dir,
                file_name
            )

            shutil.copy2(src_path, dst_path)

        # =================================================
        # COPY VAL
        # =================================================

        for file_name in val_files:

            src_path = os.path.join(
                class_path,
                file_name
            )

            dst_path = os.path.join(
                val_class_dir,
                file_name
            )

            shutil.copy2(src_path, dst_path)

        # =================================================
        # COPY TEST
        # =================================================

        for file_name in test_files:

            src_path = os.path.join(
                class_path,
                file_name
            )

            dst_path = os.path.join(
                test_class_dir,
                file_name
            )

            shutil.copy2(src_path, dst_path)

        # =================================================
        # SUMMARY
        # =================================================

        print(
            f"     Total: {total_images} | "
            f"Train: {len(train_files)} | "
            f"Val: {len(val_files)} | "
            f"Test: {len(test_files)}"
        )

# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 60)
print("DATASET SPLITTING COMPLETED")
print("=" * 60)

print("\nSaved to:")
print(SPLIT_DIR)