"""
步驟 1：從 Google CVDF 鏡像下載 MNIST 四個 .gz 壓縮檔，解壓至 mnist/ 目錄。

產出 train/test 的圖像與標籤 IDX 原始檔，供 step 2 與 step 3 讀取。
僅使用 Python 標準函式庫，無需額外安裝套件。
"""

import gzip
import os
import shutil
import urllib.request

# === 設定常數 ===
# MNIST 官方鏡像網址
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# 輸出目錄
MNIST_DIR = "mnist"
# 需下載的壓縮檔清單（圖像與標籤，訓練集與測試集各一組）
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def decompress_gz(gz_path: str, out_path: str) -> None:
    """將 .gz 檔解壓縮為原始 IDX 二進位檔。"""
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def run_download() -> None:
    """下載並解壓全部 MNIST 檔案，印出逐步進度。"""
    print("=== MNIST Download ===")

    os.makedirs(MNIST_DIR, exist_ok=True)
    total = len(FILES)

    for idx, file in enumerate(FILES, start=1):
        print(f"[{idx}/{total}] {file}")
        gz_path = f"{MNIST_DIR}/{file}"

        print("      Downloading ...")
        urllib.request.urlretrieve(MNIST_URL + file, gz_path)
        print(f"      Saved to {gz_path}")

        out_path = gz_path.removesuffix(".gz")
        print("      Decompressing ...")
        decompress_gz(gz_path, out_path)
        print(f"      Output → {out_path}")

    print(f"      All {total} IDX files ready in {MNIST_DIR}/")


# === 主程式 ===

if __name__ == "__main__":
    run_download()
