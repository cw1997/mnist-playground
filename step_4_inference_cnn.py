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
DEFAULT_IMAGE = "test.png"  # 預設待推理的圖片路徑
MODELS_DIR = "models"  # step 3 訓練產出的權重目錄
DEFAULT_WEIGHTS = f"{MODELS_DIR}/cnn.npz"  # 預設 CNN 權重檔

NUM_CLASSES = 10  # MNIST 有 10 個類別（數字 0～9）
IMAGE_SIZE = (28, 28)  # 推理前將圖片縮放為 28×28

# 與 step_3_train_cnn.py 一致的前處理與卷積超參數
MNIST_MEAN = 0.1307  # 訓練集像素均值（÷255 後）；推理時也要減去
CONV_IN_CHANNELS = 1  # 輸入 1 個通道（灰階）
CONV_OUT_CHANNELS = 16  # 第一層卷積輸出 16 個特徵通道
CONV_KERNEL_SIZE = 3  # 卷積核 3×3
CONV_STRIDE = 1  # 步幅 1：每次移動 1 格
CONV_PADDING = 1  # 四周補 1 圈 0


# === 圖片前處理 ===


def load_image_as_tensor(path: str) -> tuple[np.ndarray, tuple[int, int], str]:
    """讀取 PNG 等圖片，轉成與 step 3 CNN 訓練相同的前處理張量。

    流程：灰階化 → 縮放 28×28 → ÷255 → 減 ``MNIST_MEAN`` 零均值化。

    參數
    ----
    path : str
        待推理圖片的本機路徑（例如 ``test.png``）。

    回傳
    ----
    tuple[np.ndarray, tuple[int, int], str]
        三元組，依序為：
        - tensor：np.ndarray，形狀 ``(1, 1, 28, 28)``，dtype float64，已零均值化
        - original_size：tuple[int, int]，原始 PIL 尺寸 ``(width, height)``
        - mode：str，原始色彩模式（如 ``"RGB"``、``"L"``）
    """
    with Image.open(path) as img:  # Image.open：讀取 PNG 等圖片檔
        original_size = img.size  # size：原始 (寬, 高)，供終端顯示
        mode = img.mode  # mode：色彩模式，如 RGB、L（灰階）
        gray = img.convert("L")  # convert("L")：轉成灰階（L = luminance 亮度）
        resized = gray.resize(IMAGE_SIZE, Image.LANCZOS)  # resize：縮放至 28×28；LANCZOS 高品質插值
        raw_pixels = np.asarray(resized, dtype=np.float64)  # asarray：PIL 圖轉 NumPy 陣列
        normalized = raw_pixels / 255.0  # 除以 255，正規化到 0~1
        pixels = normalized - MNIST_MEAN  # 減去 MNIST 均值，與訓練前處理一致
    tensor = pixels.reshape(1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0])  # reshape：→ (1, 1, 28, 28)
    return tensor, original_size, mode


# === 權重載入 ===


def load_params(path: str) -> dict:
    """從 .npz 還原 CNN 的 params 字典（W、b 與卷積 metadata）。

    參數
    ----
    path : str
        權重 .npz 檔路徑（例如 ``models/cnn.npz``）。

    回傳
    ----
    dict
        模型參數字典，含：
        - ``"conv1"``：dict，``"W"``、``"b"`` 及 ``in_channels``、``out_channels``、``kernel_size``、``stride``、``"padding"`` metadata
        - ``"fc1"``：dict，``"W"`` 形狀 ``(3136, 128)``、``"b"`` 形狀 ``(128,)``
        - ``"fc2"``：dict，``"W"`` 形狀 ``(128, 10)``、``"b"`` 形狀 ``(10,)``
    """
    data = np.load(path)  # load：讀取 .npz 壓縮檔，回傳類似字典的物件
    return {
        "conv1": {
            "W": data["conv1_W"],
            "b": data["conv1_b"],
            "in_channels": CONV_IN_CHANNELS,  # 卷積層 metadata：輸入通道數
            "out_channels": CONV_OUT_CHANNELS,  # 輸出通道數（濾鏡個數）
            "kernel_size": CONV_KERNEL_SIZE,  # 卷積核邊長
            "stride": CONV_STRIDE,  # 步幅
            "padding": CONV_PADDING,  # 補零圈數
        },
        "fc1": {"W": data["fc1_W"], "b": data["fc1_b"]},  # fc1：展平→128 全連接層
        "fc2": {"W": data["fc2_W"], "b": data["fc2_b"]},  # fc2：128→10 全連接層
    }


# === im2col 輔助 ===


def _calc_output_size(size: int, kernel: int, stride: int) -> int:
    """依輸入邊長、卷積核大小、步幅，計算卷積或池化輸出邊長。

    公式：``(size - kernel) // stride + 1``。

    參數
    ----
    size : int
        輸入空間邊長（高度或寬度）。
    kernel : int
        卷積核或池化窗口邊長。
    stride : int
        步幅，每次滑動的格數。

    回傳
    ----
    int
        輸出空間邊長。
    """
    return (size - kernel) // stride + 1  # 卷積／池化輸出邊長公式


def im2col(
    x: np.ndarray, kernel_size: int, stride: int, padding: int
) -> tuple[np.ndarray, int, int]:
    """將 4D 輸入轉成 im2col 矩陣，供矩陣乘法加速卷積（推理版，無 cache）。

    參數
    ----
    x : np.ndarray
        輸入特徵圖，形狀 ``(batch, channel, height, width)``。
    kernel_size : int
        卷積核邊長（例如 3）。
    stride : int
        卷積步幅（例如 1）。
    padding : int
        四周補零圈數（例如 1）。

    回傳
    ----
    tuple[np.ndarray, int, int]
        三元組，依序為：
        - col：np.ndarray，形狀 ``(batch, channel*kernel*kernel, out_h*out_w)``
        - out_h：int，卷積輸出高度
        - out_w：int，卷積輸出寬度
    """
    batch, channel, height, width = x.shape  # shape 解包：批次大小、通道數、高、寬
    if padding > 0:
        x = np.pad(  # pad：在陣列四周補值
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),  # 只在高、寬方向補 padding 圈
            mode="constant",  # constant：補的值為 0
        )
        height += 2 * padding  # 補完後高度增加 2×padding
        width += 2 * padding  # 補完後寬度增加 2×padding

    out_h = _calc_output_size(height, kernel_size, stride)  # 卷積輸出高度
    out_w = _calc_output_size(width, kernel_size, stride)  # 卷積輸出寬度

    col = np.zeros((batch, channel, kernel_size, kernel_size, out_h, out_w), dtype=x.dtype)  # zeros：預分配 im2col 工作陣列
    for ky in range(kernel_size):  # ky：卷積核在垂直方向的偏移
        y_max = ky + stride * out_h  # 切片終點（不含）
        for kx in range(kernel_size):  # kx：卷積核在水平方向的偏移
            x_max = kx + stride * out_w  # 切片終點（不含）
            col[:, :, ky, kx, :, :] = x[:, :, ky:y_max:stride, kx:x_max:stride]  # 取出對應窗口並存入 col

    col = col.transpose(0, 1, 2, 3, 4, 5)  # transpose：重排維度，方便後續 reshape
    col = col.reshape(batch, channel * kernel_size * kernel_size, -1)  # reshape：攤平窗口 → (batch, in*kh*kw, out_h*out_w)
    return col, out_h, out_w


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


# === 卷積、池化、全連接 ===


def conv_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """卷積層前向傳播（推理版，不回傳 cache）。

    參數
    ----
    x : np.ndarray
        輸入特徵圖，形狀 ``(batch, in_channels, height, width)``。
    params : dict
        卷積層參數字典，含 ``"W"``、``"b"`` 及 ``kernel_size``、``stride``、``padding`` 等 metadata。

    回傳
    ----
    np.ndarray
        卷積輸出，形狀 ``(batch, out_channels, out_h, out_w)``。
    """
    W = params["W"]  # 卷積濾鏡權重
    b = params["b"]  # 卷積偏置
    kernel_size = params["kernel_size"]  # 卷積核邊長
    stride = params["stride"]  # 步幅
    padding = params["padding"]  # 補零圈數
    out_channels = params["out_channels"]  # 輸出通道數
    batch = x.shape[0]  # shape[0]：這批有幾張圖

    col, out_h, out_w = im2col(x, kernel_size, stride, padding)
    W_col = W.reshape(out_channels, -1)  # reshape：濾鏡攤平成 (out_ch, in*kh*kw)

    out = np.zeros((batch, out_channels, col.shape[2]), dtype=np.float64)  # zeros：預分配輸出矩陣
    for i in range(batch):
        matmul_out = W_col @ col[i]  # @：矩陣乘法 (out_ch, in*kh*kw) × (in*kh*kw, out_h*out_w)
        out[i] = matmul_out + b.reshape(-1, 1)  # 加上偏置，shape (out_ch, out_h*out_w)

    return out.reshape(batch, out_channels, out_h, out_w)  # reshape：還原為 4D 特徵圖


def maxpool_forward(
    x: np.ndarray, pool_size: int = 2, stride: int = 2
) -> np.ndarray:
    """最大池化前向傳播（推理版，不回傳 cache）。

    參數
    ----
    x : np.ndarray
        輸入特徵圖，形狀 ``(batch, channel, height, width)``。
    pool_size : int, optional
        池化窗口邊長，預設 2。
    stride : int, optional
        池化步幅，預設 2。

    回傳
    ----
    np.ndarray
        池化輸出，形狀 ``(batch, channel, out_h, out_w)``。
    """
    batch, channel, height, width = x.shape  # shape 解包：批次、通道、高、寬
    out_h = _calc_output_size(height, pool_size, stride)  # 池化輸出高度
    out_w = _calc_output_size(width, pool_size, stride)  # 池化輸出寬度

    out = np.zeros((batch, channel, out_h, out_w), dtype=np.float64)  # zeros：預分配池化輸出
    for i in range(out_h):
        for j in range(out_w):
            y0 = i * stride  # 窗口左上角 y 座標
            x0 = j * stride  # 窗口左上角 x 座標
            window = x[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size]  # 取出 pool_size×pool_size 窗口
            window_max = np.max(window, axis=(2, 3))  # max：沿高寬維取窗口最大值
            out[:, :, i, j] = window_max  # 寫入池化輸出

    return out


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


# === CNN 前向推理（含進度輸出）===


def model_forward_verbose(x: np.ndarray, params: dict) -> np.ndarray:
    """CNN 前向推理並印出各層 shape：Conv → ReLU → MaxPool → 展平 → FC → ReLU → FC → Softmax。

    參數
    ----
    x : np.ndarray
        輸入影像，形狀 ``(batch, 1, 28, 28)``，已零均值化。
    params : dict
        模型參數字典，含 ``"conv1"``、``"fc1"``、``"fc2"``。

    回傳
    ----
    np.ndarray
        各類別預測機率，形狀 ``(batch, 10)``。
        副作用：以 ``print()`` 輸出各層 shape 至終端。
    """
    c1 = conv_forward(x, params["conv1"])  # conv1：1→16 通道卷積
    r1 = relu(c1)  # ReLU：負值歸零
    print(f"      Conv1+ReLU  → shape {r1.shape}")

    p1 = maxpool_forward(r1)  # MaxPool：2×2 池化，特徵圖縮半
    print(f"      MaxPool     → shape {p1.shape}")

    flat = p1.reshape(p1.shape[0], -1)  # reshape：-1 表示其餘維度自動計算 → (batch, flat_size)
    f1 = dense_forward(flat, params["fc1"])  # fc1：展平→128 全連接
    r3 = relu(f1)  # ReLU：負值歸零
    print(f"      FC128+ReLU  → shape {r3.shape}")

    logits = dense_forward(r3, params["fc2"])  # fc2：128→10，輸出 10 個類別分數
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
    """對單張圖片執行完整 CNN 推理流程並印出五步進度。

    參數
    ----
    image_path : str
        待推理圖片路徑。
    weights_path : str
        CNN 權重 .npz 檔路徑（例如 ``models/cnn.npz``）。

    回傳
    ----
    None
        無回傳值；推理結果以 ``print()`` 輸出至終端。
    """
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
