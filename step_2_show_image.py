"""
步驟 2：讀取 mnist/ 下的 IDX 原始檔，將每張 28×28 灰階圖匯出為 PNG。

依 train/test 分割與數字標籤（0–9）分類存放至 images/。
執行前請先跑 step_1_download_mnist.py 下載 mnist/ 下的 IDX 原始檔。
"""

import os
import struct
import sys

from PIL import Image

# === 設定常數 ===
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

# (資料集分割名稱, 圖像檔名, 標籤檔名, 主步驟編號)
SPLITS = [
    ("train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 2),
    ("test", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 3),
]


def read_images(path: str) -> tuple[bytes, int, int, int]:
    """讀取 MNIST IDX3 圖像檔，回傳原始像素位元組與維度資訊。

    參數
    ----
    path : str
        IDX3 格式圖像檔的本機路徑（magic number 須為 2051）。

    回傳
    ----
    tuple[bytes, int, int, int]
        四元組，依序為：
        - pixel_bytes：bytes，長度 ``count * rows * cols``，每 byte 為 0～255 灰階值
        - count：int，圖像張數
        - rows：int，每張圖列數（MNIST 為 28）
        - cols：int，每張圖行數（MNIST 為 28）
    """
    with open(path, "rb") as f:  # "rb"：以二進位唯讀模式開檔
        # 大端序：magic(2051)、張數、列數、行數
        header = f.read(16)  # read(16)：讀取前 16 bytes 的檔頭
        magic, count, rows, cols = struct.unpack(">IIII", header)  # unpack：解讀 4 個大端序整數
        if magic != 2051:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        pixel_bytes = f.read(count * rows * cols)  # read：讀取全部像素的 bytes（每張 rows×cols）
        return pixel_bytes, count, rows, cols


def read_labels(path: str) -> tuple[bytes, int]:
    """讀取 MNIST IDX1 標籤檔，回傳原始標籤位元組與筆數。

    參數
    ----
    path : str
        IDX1 格式標籤檔的本機路徑（magic number 須為 2049）。

    回傳
    ----
    tuple[bytes, int]
        二元組，依序為：
        - label_bytes：bytes，長度 ``count``，每 byte 為 0～9 的數字標籤
        - count：int，標籤筆數（須與對應圖像檔張數一致）
    """
    with open(path, "rb") as f:  # "rb"：以二進位唯讀模式開檔
        # 大端序：magic(2049)、筆數
        header = f.read(8)  # read(8)：讀取前 8 bytes 的檔頭
        magic, count = struct.unpack(">II", header)  # unpack：解讀 magic 與標籤筆數
        if magic != 2049:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        label_bytes = f.read(count)  # read：讀取 count 個標籤（每筆 1 byte，值 0~9）
        return label_bytes, count


def export_split(split: str, images_file: str, labels_file: str, step: int) -> None:
    """將單一分割（train 或 test）的全部圖片匯出為 PNG。

    依標籤建立 ``images/{split}/{0-9}/`` 子目錄，檔名使用原始索引
    （例如 ``images/train/3/00042.png``）。

    參數
    ----
    split : str
        資料集分割名稱，``"train"`` 或 ``"test"``。
    images_file : str
        IDX3 圖像檔名（不含目錄），例如 ``"train-images-idx3-ubyte"``。
    labels_file : str
        IDX1 標籤檔名（不含目錄），例如 ``"train-labels-idx1-ubyte"``。
    step : int
        主步驟編號，用於終端進度輸出（例如 ``2`` 表示 ``[2/3]``）。

    回傳
    ----
    None
        無回傳值；PNG 寫入 ``images/{split}/{label}/`` 目錄。
    """
    print(f"[{step}/3] Exporting {split} split ...")

    pixels, count, rows, cols = read_images(f"{MNIST_DIR}/{images_file}")  # 讀取圖像像素與張數、高、寬
    labels, label_count = read_labels(f"{MNIST_DIR}/{labels_file}")  # 讀取每張圖對應的數字標籤 0~9
    if count != label_count:
        raise ValueError(
            f"mismatch in {split}: {count} images vs {label_count} labels"
        )

    print(f"      Images: {count} samples, {rows}×{cols}")
    print(f"      Labels: {label_count} labels matched")
    print(f"      Output → {OUTPUT_DIR}/{split}/{{0-9}}/")

    pixel_size = rows * cols  # 每張圖 28×28 = 784 個像素
    for i in range(count):  # range(count)：依序處理第 0 到 count-1 張圖
        label = labels[i]  # labels[i]：第 i 張圖的數字標籤（0~9）
        # 依分割與數字標籤建立子目錄，例如 images/train/3/
        out_dir = f"{OUTPUT_DIR}/{split}/{label}"
        os.makedirs(out_dir, exist_ok=True)  # makedirs：建立輸出子目錄（已存在不報錯）
        start = i * pixel_size  # 這張圖在 pixels 中的起始位置
        end = (i + 1) * pixel_size  # 這張圖的結束位置（不含）
        img_bytes = pixels[start:end]  # 切片：取出第 i 張圖的 784 bytes
        # "L" 表示 8 位元灰階；從 bytes 建立 PIL 圖片物件
        img = Image.frombytes("L", (cols, rows), img_bytes)  # frombytes：bytes → 28×28 灰階圖
        img.save(f"{out_dir}/{i:05d}.png")  # save：寫入 PNG，檔名使用原始索引

    print(f"      Export complete: {count} images")


def run_export() -> None:
    """檢查 MNIST 前置檔案並匯出 train/test PNG。

    依 ``SPLITS`` 常數依序匯出訓練集與測試集；若缺少 IDX 檔則印出
    錯誤訊息並以 ``sys.exit(1)`` 結束（由 ``main`` 區塊負責檢查）。

    參數
    ----
    無。

    回傳
    ----
    None
        無回傳值；PNG 寫入 ``images/`` 目錄，進度以 ``print()`` 輸出至終端。
    """
    print("=== MNIST PNG Export ===")

    print("[1/3] Checking MNIST files ...")
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]  # 列表推導：找出缺少的 IDX 檔
    if missing:
        print("Missing MNIST files:", ", ".join(missing))
        print("Run step_1_download_mnist.py first.")
        sys.exit(1)
    print("      All 4 IDX files found")

    for split, images_file, labels_file, step in SPLITS:
        export_split(split, images_file, labels_file, step)  # 依 train/test 匯出 PNG 至 images/


# === 主程式 ===

if __name__ == "__main__":
    run_export()
