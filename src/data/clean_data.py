import os
import cv2
import hashlib

# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "dataset", "cleaned")

# =========================================================
# CLASS MAPPING
# Thêm alias mới ở đây khi có dataset mới
# =========================================================

CLASS_MAPPING = {

    # =====================================================
    # HEALTHY
    # =====================================================

    "healthy": "healthy",
    "normal": "healthy",

    # =====================================================
    # RICE BLAST
    # =====================================================

    "blast": "rice_blast",
    "rice blast": "rice_blast",
    "riceblast": "rice_blast",
    "rice_blast": "rice_blast",

    # =====================================================
    # BROWN SPOT
    # =====================================================

    "brownspot": "brown_spot",
    "brown spot": "brown_spot",
    "brown_spot": "brown_spot",

    # =====================================================
    # BACTERIAL LEAF BLIGHT
    # =====================================================

    "bacterialblight": "bacterial_leaf_blight",
    "bacterial blight": "bacterial_leaf_blight",
    "bacterial_leaf_blight": "bacterial_leaf_blight",
    "blight": "bacterial_leaf_blight",

    # =====================================================
    # LEAF SMUT
    # =====================================================

    "leaf smut": "leaf_smut",
    "leaf_smut": "leaf_smut",
    "smut": "leaf_smut",

    # =====================================================
    # TUNGRO
    # =====================================================

    "tungro": "tungro"
}

# =========================================================
# IMAGE CONFIG
# =========================================================

ALLOWED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

MIN_WIDTH = 100
MIN_HEIGHT = 100

# =========================================================
# CREATE CLEANED FOLDERS
# =========================================================

for class_name in set(CLASS_MAPPING.values()):

    class_dir = os.path.join(CLEANED_DIR, class_name)

    os.makedirs(class_dir, exist_ok=True)

# =========================================================
# IMAGE COUNTER
# =========================================================

image_counter = {}

for class_name in set(CLASS_MAPPING.values()):
    image_counter[class_name] = 0

# =========================================================
# DUPLICATE HASH
# =========================================================

image_hashes = set()

# =========================================================
# STATISTICS
# =========================================================

stats = {
    "total_images": 0,
    "valid_images": 0,
    "invalid_images": 0,
    "duplicate_images": 0,
    "small_images": 0
}

# =========================================================
# CHECK EXTENSION
# =========================================================

def is_valid_extension(filename):

    extension = os.path.splitext(filename)[1].lower()

    return extension in ALLOWED_EXTENSIONS

# =========================================================
# GENERATE IMAGE HASH
# =========================================================

def generate_image_hash(image):

    resized = cv2.resize(image, (8, 8))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return hashlib.md5(gray.tobytes()).hexdigest()

# =========================================================
# VALIDATE IMAGE
# =========================================================

def validate_image(image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return None

        height, width = image.shape[:2]

        # ảnh quá nhỏ
        if width < MIN_WIDTH or height < MIN_HEIGHT:

            stats["small_images"] += 1

            return None

        return image

    except:
        return None

# =========================================================
# START CLEANING
# =========================================================

print("=" * 60)
print("START CLEANING DATASET")
print("=" * 60)

print(f"\nRAW DATASET PATH:")
print(RAW_DIR)

# =========================================================
# RECURSIVE SCAN
# =========================================================

for root, dirs, files in os.walk(RAW_DIR):

    # lấy tên thư mục hiện tại
    current_folder = os.path.basename(root)

    # normalize tên class
    class_folder = (
        current_folder
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )

    # không phải class
    if class_folder not in CLASS_MAPPING:
        continue

    target_class = CLASS_MAPPING[class_folder]

    print(f"\n[CLASS FOUND]")
    print(f"{current_folder} -> {target_class}")

    # =====================================================
    # PROCESS FILES
    # =====================================================

    for file_name in files:

        stats["total_images"] += 1

        file_path = os.path.join(root, file_name)

        # extension không hợp lệ
        if not is_valid_extension(file_name):
            continue

        # validate ảnh
        image = validate_image(file_path)

        if image is None:

            stats["invalid_images"] += 1

            print(f"[INVALID] {file_name}")

            continue

        # duplicate detection
        image_hash = generate_image_hash(image)

        if image_hash in image_hashes:

            stats["duplicate_images"] += 1

            print(f"[DUPLICATE] {file_name}")

            continue

        image_hashes.add(image_hash)

        # tăng counter
        image_counter[target_class] += 1

        # tên file mới
        new_file_name = (
            f"{target_class}_"
            f"{image_counter[target_class]:05d}.jpg"
        )

        destination_path = os.path.join(
            CLEANED_DIR,
            target_class,
            new_file_name
        )

        try:

            # convert toàn bộ sang JPG
            cv2.imwrite(destination_path, image)

            stats["valid_images"] += 1

        except Exception as e:

            print(f"[ERROR] {file_name}")
            print(e)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"Total Images      : {stats['total_images']}")
print(f"Valid Images      : {stats['valid_images']}")
print(f"Invalid Images    : {stats['invalid_images']}")
print(f"Duplicate Images  : {stats['duplicate_images']}")
print(f"Small Images      : {stats['small_images']}")

print("\nImages Per Class:")

for class_name, count in image_counter.items():

    print(f"{class_name}: {count}")

print("\nCleaned dataset saved at:")
print(CLEANED_DIR)