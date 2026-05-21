import os
import cv2

# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CLEANED_DIR = os.path.join(BASE_DIR, "dataset", "cleaned")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# =========================================================
# OUTPUT SIZE
# =========================================================

CNN_SIZE = (224, 224)
TRANSFORMER_SIZE = (384, 384)

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

CNN_OUTPUT = os.path.join(PROCESSED_DIR, "cnn_224")
TRANSFORMER_OUTPUT = os.path.join(PROCESSED_DIR, "transformer_384")

os.makedirs(CNN_OUTPUT, exist_ok=True)
os.makedirs(TRANSFORMER_OUTPUT, exist_ok=True)

# =========================================================
# PROCESS DATASET
# =========================================================

print("=" * 60)
print("START PREPROCESSING")
print("=" * 60)

for class_name in os.listdir(CLEANED_DIR):

    class_path = os.path.join(CLEANED_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")

    # tạo class folder
    cnn_class_dir = os.path.join(CNN_OUTPUT, class_name)
    transformer_class_dir = os.path.join(TRANSFORMER_OUTPUT, class_name)

    os.makedirs(cnn_class_dir, exist_ok=True)
    os.makedirs(transformer_class_dir, exist_ok=True)

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        image = cv2.imread(file_path)

        if image is None:
            continue

        # =================================================
        # CNN IMAGE
        # =================================================

        cnn_image = cv2.resize(image, CNN_SIZE)

        cnn_output_path = os.path.join(
            cnn_class_dir,
            file_name
        )

        cv2.imwrite(cnn_output_path, cnn_image)

        # =================================================
        # TRANSFORMER IMAGE
        # =================================================

        transformer_image = cv2.resize(
            image,
            TRANSFORMER_SIZE
        )

        transformer_output_path = os.path.join(
            transformer_class_dir,
            file_name
        )

        cv2.imwrite(
            transformer_output_path,
            transformer_image
        )

print("\nPREPROCESSING COMPLETED")
print("=" * 60)

print("\nSaved to:")
print(CNN_OUTPUT)
print(TRANSFORMER_OUTPUT)