import gzip
import os
import shutil
import urllib.request

# MNIST 官方鏡像網址
url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# 需下載的壓縮檔清單（圖像與標籤，訓練集與測試集各一組）
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def decompress_gz(gz_path: str, out_path: str) -> None:
    """將 .gz 檔解壓縮為原始 IDX 二進位檔。"""
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


os.makedirs("mnist", exist_ok=True)

for file in files:
    gz_path = f"mnist/{file}"
    # 下載壓縮檔
    urllib.request.urlretrieve(url + file, gz_path)
    print("download complete: ", file)

    # 解壓縮，去掉 .gz 副檔名即為輸出路徑
    out_path = gz_path.removesuffix(".gz")
    decompress_gz(gz_path, out_path)
    print("decompress complete: ", out_path)
