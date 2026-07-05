"""
步驟 4（CNN 推理）：載入 step 3 訓練的 CNN 權重，對輸入圖片做數字辨識。

架構：Conv(16, pad=1) → ReLU → MaxPool → FC(128) → ReLU → FC(10) → Softmax。
輸出逐層推理進度、10 個數字的機率，以及最高置信度預測。

執行前請先跑 step_3_train_cnn.py 產生 models/cnn.npz。
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# === 設定常數 ===
DEFAULT_IMAGE = "test.png"
MODELS_DIR = "models"
DEFAULT_WEIGHTS = f"{MODELS_DIR}/cnn.npz"

NUM_CLASSES = 10
IMAGE_SIZE = (28, 28)

# 與 step_3_train_cnn.py 一致的前處理與卷積超參數
MNIST_MEAN = 0.1307
CONV_IN_CHANNELS = 1
CONV_OUT_CHANNELS = 16
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1


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
        pixels = np.asarray(resized, dtype=np.float64) / 255.0 - MNIST_MEAN
    tensor = pixels.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])
    return tensor, original_size, mode


# === 權重載入 ===


def load_params(path: str) -> dict:
    """從 .npz 還原 CNN 的 params 字典（僅 W、b 與卷積 metadata）。"""
    data = np.load(path)
    return {
        "conv1": {
            "W": data["conv1_W"],
            "b": data["conv1_b"],
            "in_channels": CONV_IN_CHANNELS,
            "out_channels": CONV_OUT_CHANNELS,
            "kernel_size": CONV_KERNEL_SIZE,
            "stride": CONV_STRIDE,
            "padding": CONV_PADDING,
        },
        "fc1": {"W": data["fc1_W"], "b": data["fc1_b"]},
        "fc2": {"W": data["fc2_W"], "b": data["fc2_b"]},
    }


# === im2col 輔助 ===


def _calc_output_size(size: int, kernel: int, stride: int) -> int:
    """依輸入邊長、卷積核大小、步幅，計算輸出邊長。"""
    return (size - kernel) // stride + 1


def im2col(
    x: np.ndarray, kernel_size: int, stride: int, padding: int
) -> tuple[np.ndarray, int, int]:
    """把輸入 x 轉成 im2col 矩陣，供卷積矩陣乘法使用。"""
    batch, channel, height, width = x.shape
    if padding > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
        height += 2 * padding
        width += 2 * padding

    out_h = _calc_output_size(height, kernel_size, stride)
    out_w = _calc_output_size(width, kernel_size, stride)

    col = np.zeros((batch, channel, kernel_size, kernel_size, out_h, out_w), dtype=x.dtype)
    for ky in range(kernel_size):
        y_max = ky + stride * out_h
        for kx in range(kernel_size):
            x_max = kx + stride * out_w
            col[:, :, ky, kx, :, :] = x[:, :, ky:y_max:stride, kx:x_max:stride]

    col = col.transpose(0, 1, 2, 3, 4, 5).reshape(batch, channel * kernel_size * kernel_size, -1)
    return col, out_h, out_w


# === 激活函式 ===


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活：max(0, x)。"""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """對每個樣本的 10 個類別分數做 softmax，回傳機率。"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# === 卷積、池化、全連接 ===


def conv_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """卷積前向傳播，輸入 (batch, in_ch, H, W)。"""
    W = params["W"]
    b = params["b"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = params["padding"]
    out_channels = params["out_channels"]
    batch = x.shape[0]

    col, out_h, out_w = im2col(x, kernel_size, stride, padding)
    W_col = W.reshape(out_channels, -1)

    out = np.zeros((batch, out_channels, col.shape[2]), dtype=np.float64)
    for i in range(batch):
        out[i] = W_col @ col[i] + b.reshape(-1, 1)

    return out.reshape(batch, out_channels, out_h, out_w)


def maxpool_forward(
    x: np.ndarray, pool_size: int = 2, stride: int = 2
) -> np.ndarray:
    """最大池化前向傳播。"""
    batch, channel, height, width = x.shape
    out_h = _calc_output_size(height, pool_size, stride)
    out_w = _calc_output_size(width, pool_size, stride)

    out = np.zeros((batch, channel, out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            y0 = i * stride
            x0 = j * stride
            window = x[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size]
            out[:, :, i, j] = np.max(window, axis=(2, 3))

    return out


def dense_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """全連接前向傳播：y = x @ W + b"""
    return x @ params["W"] + params["b"]


# === CNN 前向推理（含進度輸出）===


def model_forward_verbose(x: np.ndarray, params: dict) -> np.ndarray:
    """
    CNN 前向傳播並印出各層 shape。
    架構：Conv → ReLU → MaxPool → 展平 → FC → ReLU → FC → Softmax
    """
    c1 = conv_forward(x, params["conv1"])
    r1 = relu(c1)
    print(f"      Conv1+ReLU  → shape {r1.shape}")

    p1 = maxpool_forward(r1)
    print(f"      MaxPool     → shape {p1.shape}")

    flat = p1.reshape(p1.shape[0], -1)
    f1 = dense_forward(flat, params["fc1"])
    r3 = relu(f1)
    print(f"      FC128+ReLU  → shape {r3.shape}")

    logits = dense_forward(r3, params["fc2"])
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
        f"      conv1 W: {params['conv1']['W'].shape}  "
        f"fc1 W: {params['fc1']['W'].shape}  "
        f"fc2 W: {params['fc2']['W'].shape}"
    )

    print("[4/5] Forward pass ...")
    probs = model_forward_verbose(tensor, params)

    print("[5/5] Inference result")
    print_probs(probs)


# === 主程式 ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="以 CNN 對手寫數字圖片做推理（需先執行 step_3_train_cnn.py）"
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
        print("Run step_3_train_cnn.py first.")
        sys.exit(1)

    run_inference(args.image, args.weights)
