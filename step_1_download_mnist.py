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
    """將 .gz 壓縮檔解壓縮為原始 IDX 二進位檔。

    參數
    ----
    gz_path : str
        本機 .gz 壓縮檔路徑（例如 ``mnist/train-images-idx3-ubyte.gz``）。
    out_path : str
        解壓後 IDX 原始檔的輸出路徑（例如 ``mnist/train-images-idx3-ubyte``）。

    回傳
    ----
    None
        無回傳值；解壓結果直接寫入 ``out_path``。
    """
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:  # gzip.open：讀 .gz；"wb"：以二進位寫入
        shutil.copyfileobj(f_in, f_out)  # copyfileobj：把解壓後的 bytes 串流複製到輸出檔


def run_download() -> None:
    """從 Google CVDF 鏡像下載並解壓全部 MNIST 檔案，印出逐步進度。

    依 ``FILES`` 常數依序下載四個 .gz 檔至 ``mnist/``，解壓後刪除 .gz 副檔名
    得到 IDX 原始檔，供 step 2 與 step 3 讀取。

    參數
    ----
    無。

    回傳
    ----
    None
        無回傳值；檔案寫入 ``mnist/`` 目錄，進度以 ``print()`` 輸出至終端。
    """
    print("=== MNIST Download ===")

    os.makedirs(MNIST_DIR, exist_ok=True)  # makedirs：建立 mnist/ 目錄；exist_ok=True 表示已存在也不報錯
    total = len(FILES)  # len：清單長度，這裡是 4 個檔案

    for idx, file in enumerate(FILES, start=1):  # enumerate：同時取得編號 idx（從 1 起）與檔名
        print(f"[{idx}/{total}] {file}")
        gz_path = f"{MNIST_DIR}/{file}"  # f-string：組合本機 .gz 路徑

        print("      Downloading ...")
        urllib.request.urlretrieve(MNIST_URL + file, gz_path)  # urlretrieve：從網址下載並存到 gz_path
        print(f"      Saved to {gz_path}")

        out_path = gz_path.removesuffix(".gz")  # removesuffix：去掉 .gz 副檔名，得到 IDX 原始檔名
        print("      Decompressing ...")
        decompress_gz(gz_path, out_path)
        print(f"      Output → {out_path}")

    print(f"      All {total} IDX files ready in {MNIST_DIR}/")


# === 主程式 ===

if __name__ == "__main__":
    run_download()
