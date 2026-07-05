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
DEFAULT_IMAGE = "test.png"
MODELS_DIR = "models"
DEFAULT_WEIGHTS = f"{MODELS_DIR}/mlp.npz"

NUM_CLASSES = 10
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
IMAGE_SIZE = (28, 28)


# === 圖片前處理 ===


def load_image_as_tensor(path: str) -> tuple[np.ndarray, tuple[int, int], str]:
    """
    讀取 PNG 等圖片，轉成與 step 3 訓練相同的張量格式。
    回傳：(1, 1, 28, 28) 張量、原始尺寸、PIL 模式。
    """
    with Image.open(path) as img:
        original_size = img.size
        mode = img.mode
        gray = img.convert("L")
        resized = gray.resize(IMAGE_SIZE, Image.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float64) / 255.0
    tensor = pixels.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])
    return tensor, original_size, mode


# === 權重載入 ===


def load_params(path: str) -> dict:
    """從 .npz 還原 MLP 的 params 字典（僅 W、b，不含梯度）。"""
    data = np.load(path)
    return {
        "fc1": {"W": data["fc1_W"], "b": data["fc1_b"]},
        "fc2": {"W": data["fc2_W"], "b": data["fc2_b"]},
    }


# === 激活函式 ===


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活：max(0, x)。"""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """對每個樣本的 10 個類別分數做 softmax，回傳機率。"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# === 全連接層 ===


def dense_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """全連接前向傳播：y = x @ W + b"""
    return x @ params["W"] + params["b"]


# === MLP 前向推理（含進度輸出）===


def model_forward_verbose(x: np.ndarray, params: dict) -> np.ndarray:
    """
    MLP 前向傳播並印出各層 shape。
    架構：展平 → FC(128) → ReLU → FC(10) → Softmax
    """
    flat = x.reshape(x.shape[0], -1)
    print(f"      Flatten     → shape {flat.shape}")

    f1 = dense_forward(flat, params["fc1"])
    r1 = relu(f1)
    print(f"      FC128+ReLU  → shape {r1.shape}")

    logits = dense_forward(r1, params["fc2"])
    print(f"      FC10 logits → shape {logits.shape}")

    probs = softmax(logits)
    print("      Softmax     → done")
    return probs


# === 結果格式化 ===


def print_probs(probs: np.ndarray) -> tuple[int, float]:
    """印出 0～9 各類機率（百分比），回傳預測數字與置信度。"""
    row = probs[0]
    for digit in range(NUM_CLASSES):
        print(f"      Digit {digit}: {row[digit] * 100:6.2f}%")

    pred = int(np.argmax(row))
    confidence = float(row[pred])
    print(f"      Predicted digit: {pred}")
    print(f"      Confidence:      {confidence * 100:.2f}%")
    return pred, confidence


def run_inference(image_path: str, weights_path: str) -> None:
    """對單張圖片執行完整推理流程並印出進度。"""
    print(f"=== Inference: {image_path} ===")

    print("[1/5] Loading image ...")
    tensor, original_size, mode = load_image_as_tensor(image_path)
    print(f"      Original size: {original_size[0]}×{original_size[1]}, mode: {mode}")

    print("[2/5] Preprocessing ...")
    print(
        f"      Grayscale → resize 28×28 → normalize [0,1]"
        f"  shape={tensor.shape}, "
        f"pixel range [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    print(f"[3/5] Loading weights {weights_path} ...")
    params = load_params(weights_path)
    print(
        f"      fc1 W: {params['fc1']['W'].shape}  "
        f"fc2 W: {params['fc2']['W'].shape}"
    )

    print("[4/5] Forward pass ...")
    probs = model_forward_verbose(tensor, params)

    print("[5/5] Inference result")
    print_probs(probs)


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
