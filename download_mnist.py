import gzip
import os
import shutil
import urllib.request

url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def decompress_gz(gz_path: str, out_path: str) -> None:
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


os.makedirs("mnist", exist_ok=True)

for file in files:
    gz_path = f"mnist/{file}"
    urllib.request.urlretrieve(url + file, gz_path)
    print("download complete: ", file)

    out_path = gz_path.removesuffix(".gz")
    decompress_gz(gz_path, out_path)
    print("decompress complete: ", out_path)
