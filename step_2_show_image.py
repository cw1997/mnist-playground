import os
import struct
import sys

from PIL import Image

# MNIST 原始 IDX 檔所在目錄
MNIST_DIR = "mnist"
# PNG 輸出根目錄
OUTPUT_DIR = "images"

# 執行前必須存在的四個 IDX 檔
REQUIRED_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

# (資料集分割名稱, 圖像檔名, 標籤檔名)
SPLITS = [
    ("train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    ("test", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
]


def read_images(path: str) -> tuple[bytes, int, int, int]:
    """讀取 IDX3 圖像檔，回傳像素位元組、張數、列數、行數。"""
    with open(path, "rb") as f:
        # 大端序：magic(2051)、張數、列數、行數
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        return f.read(count * rows * cols), count, rows, cols


def read_labels(path: str) -> tuple[bytes, int]:
    """讀取 IDX1 標籤檔，回傳標籤位元組與筆數。"""
    with open(path, "rb") as f:
        # 大端序：magic(2049)、筆數
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        return f.read(count), count


def export_split(split: str, images_file: str, labels_file: str) -> None:
    """將單一分割（train 或 test）的全部圖片匯出為 PNG。"""
    pixels, count, rows, cols = read_images(f"{MNIST_DIR}/{images_file}")
    labels, label_count = read_labels(f"{MNIST_DIR}/{labels_file}")
    if count != label_count:
        raise ValueError(
            f"mismatch in {split}: {count} images vs {label_count} labels"
        )

    pixel_size = rows * cols
    for i in range(count):
        label = labels[i]
        # 依分割與數字標籤建立子目錄，例如 images/train/3/
        out_dir = f"{OUTPUT_DIR}/{split}/{label}"
        os.makedirs(out_dir, exist_ok=True)
        img_bytes = pixels[i * pixel_size : (i + 1) * pixel_size]
        # "L" 表示 8 位元灰階；檔名使用原始索引，與 MNIST 順序一致
        Image.frombytes("L", (cols, rows), img_bytes).save(f"{out_dir}/{i:05d}.png")

    print("export complete: ", split, f"({count} images)")


# 確認 step 1 已下載所需檔案
missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]
if missing:
    print("missing MNIST files:", ", ".join(missing))
    print("run step_1_download_mnist.py first")
    sys.exit(1)

for split, images_file, labels_file in SPLITS:
    export_split(split, images_file, labels_file)
