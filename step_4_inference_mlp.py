"""
步驟 4（MLP 推理）：載入 step 3 訓練的 MLP 權重，對輸入圖片做數字辨識。

架構：展平 784 → FC(128) → ReLU → FC(10) → Softmax。
輸出逐層推理進度、10 個數字的機率，以及最高置信度預測。

執行前請先跑 step_3_train_mlp.py 產生 models/mlp.npz。
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# === 設定常數 ===
DEFAULT_IMAGE = "test.png"  # 預設待推理的圖片路徑
MODELS_DIR = "models"  # step 3 訓練產出的權重目錄
DEFAULT_WEIGHTS = f"{MODELS_DIR}/mlp.npz"  # 預設 MLP 權重檔

NUM_CLASSES = 10  # MNIST 有 10 個類別（數字 0～9）
INPUT_SIZE = 28 * 28  # 28×28 灰階圖展平後的輸入維度（784）
HIDDEN_SIZE = 128  # 全連接隱藏層維度（與 step 3 一致）
IMAGE_SIZE = (28, 28)  # 推理前將圖片縮放為 28×28


# === 圖片前處理 ===


def load_image_as_tensor(path: str) -> tuple[np.ndarray, tuple[int, int], str]:
    """讀取 PNG 等圖片，轉成與 step 3 MLP 訓練相同的前處理張量。

    流程：灰階化 → 縮放 28×28 → 正規化至 0～1。

    參數
    ----
    path : str
        待推理圖片的本機路徑（例如 ``test.png``）。

    回傳
    ----
    tuple[np.ndarray, tuple[int, int], str]
        三元組，依序為：
        - tensor：np.ndarray，形狀 ``(1, 1, 28, 28)``，dtype float64，像素 0～1
        - original_size：tuple[int, int]，原始 PIL 尺寸 ``(width, height)``
        - mode：str，原始色彩模式（如 ``"RGB"``、``"L"``）
    """
    with Image.open(path) as img:  # Image.open：讀取 PNG 等圖片檔
        original_size = img.size  # size：原始 (寬, 高)，供終端顯示
        mode = img.mode  # mode：色彩模式，如 RGB、L（灰階）
        gray = img.convert("L")  # convert("L")：轉成灰階（L = luminance 亮度）
        resized = gray.resize(IMAGE_SIZE, Image.LANCZOS)  # resize：縮放至 28×28；LANCZOS 高品質插值
        raw_pixels = np.asarray(resized, dtype=np.float64)  # asarray：PIL 圖轉 NumPy 陣列
        pixels = raw_pixels / 255.0  # 除以 255，正規化到 0~1（與 step 3 訓練一致）
    tensor = pixels.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # reshape：→ (1, 1, 28, 28)
    return tensor, original_size, mode


# === 權重載入 ===


def load_params(path: str) -> dict:
    """從 .npz 還原 MLP 的 params 字典（僅 W、b，不含梯度）。

    參數
    ----
    path : str
        權重 .npz 檔路徑（例如 ``models/mlp.npz``）。

    回傳
    ----
    dict
        模型參數字典，含：
        - ``"fc1"``：dict，``"W"`` 形狀 ``(784, 128)``、``"b"`` 形狀 ``(128,)``
        - ``"fc2"``：dict，``"W"`` 形狀 ``(128, 10)``、``"b"`` 形狀 ``(10,)``
    """
    data = np.load(path)  # load：讀取 .npz 壓縮檔，回傳類似字典的物件
    return {
        "fc1": {"W": data["fc1_W"], "b": data["fc1_b"]},  # fc1：784→128 全連接層權重
        "fc2": {"W": data["fc2_W"], "b": data["fc2_b"]},  # fc2：128→10 全連接層權重
    }


# === 激活函式 ===


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活函式：max(0, x)，逐元素將負值歸零。

    參數
    ----
    x : np.ndarray
        任意 shape 的輸入陣列，dtype float64。

    回傳
    ----
    np.ndarray
        與 ``x`` 同 shape，負值為 0、正值保留。
    """
    return np.maximum(0, x)  # maximum：逐元素取較大值，向量化實作 ReLU


def softmax(x: np.ndarray) -> np.ndarray:
    """對每個樣本的類別分數做 softmax，轉成機率分布。

    參數
    ----
    x : np.ndarray
        類別分數（logits），形狀 ``(batch, num_classes)``，dtype float64。

    回傳
    ----
    np.ndarray
        機率分布，形狀 ``(batch, num_classes)``，每列加總為 1。
    """
    shifted = x - np.max(x, axis=1, keepdims=True)  # max：axis=1 每列取最大；keepdims 保留維度方便相減
    exp_x = np.exp(shifted)  # exp：對每個分數取 e 的次方
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)  # sum：axis=1 每列加總，keepdims 保留維度方便相除
    return exp_x / sum_exp  # 逐元素相除，每列機率加總為 1


# === 全連接層 ===


def dense_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """全連接層前向傳播：y = x @ W + b。

    參數
    ----
    x : np.ndarray
        輸入特徵，形狀 ``(batch, in_features)``，dtype float64。
    params : dict
        全連接層參數字典，須含 ``"W"`` 與 ``"b"``。

    回傳
    ----
    np.ndarray
        線性輸出，形狀 ``(batch, out_features)``。
    """
    out = x @ params["W"]  # @：矩陣乘法 (batch, in) × (in, out)
    return out + params["b"]  # 加上偏置 b，shape (out,)


# === MLP 前向推理（含進度輸出）===


def model_forward_verbose(x: np.ndarray, params: dict) -> np.ndarray:
    """MLP 前向推理並印出各層 shape：展平 → FC(128) → ReLU → FC(10) → Softmax。

    參數
    ----
    x : np.ndarray
        輸入影像，形狀 ``(batch, 1, 28, 28)``，像素 0～1。
    params : dict
        模型參數字典，含 ``"fc1"`` 與 ``"fc2"``。

    回傳
    ----
    np.ndarray
        各類別預測機率，形狀 ``(batch, 10)``。
        副作用：以 ``print()`` 輸出各層 shape 至終端。
    """
    flat = x.reshape(x.shape[0], -1)  # reshape：-1 表示其餘維度自動計算 → (batch, 784)
    print(f"      Flatten     → shape {flat.shape}")

    f1 = dense_forward(flat, params["fc1"])  # fc1：784→128 線性變換
    r1 = relu(f1)  # ReLU：負值歸零，保留正特徵
    print(f"      FC128+ReLU  → shape {r1.shape}")

    logits = dense_forward(r1, params["fc2"])  # fc2：128→10，輸出 10 個類別分數
    print(f"      FC10 logits → shape {logits.shape}")

    probs = softmax(logits)  # Softmax：10 個分數轉成機率（加總為 1）
    print("      Softmax     → done")
    return probs


# === 結果格式化 ===


def print_probs(probs: np.ndarray) -> tuple[int, float]:
    """印出 0～9 各類別機率（百分比），並回傳最高置信度預測。

    參數
    ----
    probs : np.ndarray
        softmax 機率，形狀 ``(batch, 10)``；通常 batch=1。

    回傳
    ----
    tuple[int, float]
        二元組，依序為：
        - pred：int，預測數字 0～9
        - confidence：float，該類別機率 0～1
        副作用：以 ``print()`` 輸出各 digit 機率與預測結果。
    """
    row = probs[0]  # 取第一筆樣本（batch=1）的 10 類機率
    for digit in range(NUM_CLASSES):
        print(f"      Digit {digit}: {row[digit] * 100:6.2f}%")

    pred = int(np.argmax(row))  # argmax：10 個機率中取最大值 index → 預測數字
    confidence = float(row[pred])  # 取出預測類別的機率值
    print(f"      Predicted digit: {pred}")
    print(f"      Confidence:      {confidence * 100:.2f}%")
    return pred, confidence


def run_inference(image_path: str, weights_path: str) -> None:
    """對單張圖片執行完整 MLP 推理流程並印出五步進度。

    參數
    ----
    image_path : str
        待推理圖片路徑。
    weights_path : str
        MLP 權重 .npz 檔路徑（例如 ``models/mlp.npz``）。

    回傳
    ----
    None
        無回傳值；推理結果以 ``print()`` 輸出至終端。
    """
    print(f"=== Inference: {image_path} ===")

    print("[1/5] Loading image ...")
    tensor, original_size, mode = load_image_as_tensor(image_path)  # 讀圖並轉成 (1,1,28,28) 張量
    print(f"      Original size: {original_size[0]}×{original_size[1]}, mode: {mode}")

    print("[2/5] Preprocessing ...")
    print(
        f"      Grayscale → resize 28×28 → normalize [0,1]"
        f"  shape={tensor.shape}, "
        f"pixel range [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    print(f"[3/5] Loading weights {weights_path} ...")
    params = load_params(weights_path)  # 從 .npz 載入 fc1、fc2 權重
    print(
        f"      fc1 W: {params['fc1']['W'].shape}  "
        f"fc2 W: {params['fc2']['W'].shape}"
    )

    print("[4/5] Forward pass ...")
    probs = model_forward_verbose(tensor, params)  # 前向推理並印出各層 shape

    print("[5/5] Inference result")
    print_probs(probs)  # 印出 0~9 各類機率與最高置信度預測


# === 主程式 ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="以 MLP 對手寫數字圖片做推理（需先執行 step_3_train_mlp.py）"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"待推理圖片路徑（預設: {DEFAULT_IMAGE}）",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"權重檔路徑（預設: {DEFAULT_WEIGHTS}）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    if not os.path.isfile(args.weights):
        print(f"Weights not found: {args.weights}")
        print("Run step_3_train_mlp.py first.")
        sys.exit(1)

    run_inference(args.image, args.weights)
