"""
步驟 3（CNN 版）：用純 NumPy 手寫卷積神經網路（CNN），訓練 MNIST 手寫數字辨識模型。

架構：Conv(16, pad=1) → ReLU → MaxPool → FC(128) → ReLU → FC(10) → Softmax。
若只想先學全連接網路，可改跑 step_3_train_mlp.py（784→128→10 的 MLP）。

本檔案包含從資料讀取、卷積／池化、前向傳播、反向傳播到訓練迴圈的全部邏輯。
全程以純函式實作，不使用 class；權重與中間結果以字典傳遞。
執行前請先跑 step_1_download_mnist.py 下載 mnist/ 下的 IDX 原始檔。
"""

import os
import struct
import sys

import numpy as np

# === 設定常數 ===
# MNIST 原始 IDX 檔所在目錄（由 step 1 產生）
MNIST_DIR = "mnist"

# 執行前必須存在的四個 IDX 檔
REQUIRED_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

# 訓練超參數（可依需求調整）
EPOCHS = 5          # 整份訓練集掃過幾輪；越大通常越準，但越慢
BATCH_SIZE = 64     # 每次更新權重用多少筆樣本；太大可能學不細，太小可能不穩
LEARNING_RATE = 0.01  # 學習率：權重每次更新的步幅；第 4 epoch 起 ×0.5
MOMENTUM = 0.9      # Momentum SGD：累積歷史梯度方向，加速收斂
RANDOM_SEED = 42    # 固定亂數種子，讓每次執行結果可重現

# MNIST 有 10 個類別（數字 0～9）
NUM_CLASSES = 10

# MNIST 訓練集像素均值（÷255 後）；零均值正規化有助收斂
MNIST_MEAN = 0.1307

# 網路架構常數
IMAGE_SIZE = 28  # MNIST 每張圖 28×28 像素
CONV_OUT_CHANNELS = 16  # 第一層卷積輸出 16 個特徵通道（16 種濾鏡）
CONV_KERNEL_SIZE = 3  # 卷積核 3×3，在圖上滑動提取局部特徵
CONV_PADDING = 1  # 四周補 1 圈 0，讓卷積後特徵圖仍為 28×28
POOL_SIZE = 2  # 池化窗口 2×2，把相鄰區域壓縮成 1 個值
HIDDEN_SIZE = 128  # 全連接隱藏層維度

# 卷積後展平維度：16 通道 × 14 × 14（28→Conv pad1→28→Pool→14）
_conv_spatial = IMAGE_SIZE + 2 * CONV_PADDING - CONV_KERNEL_SIZE + 1  # 卷積後空間邊長（28）
_pool_spatial = _conv_spatial // POOL_SIZE  # 池化後空間邊長（14）
FLAT_SIZE = CONV_OUT_CHANNELS * _pool_spatial * _pool_spatial  # 展平後向量長度（16×14×14=3136）

# 訓練完成後保存權重的目錄與檔案（供 step 4 推理使用）
MODELS_DIR = "models"
WEIGHTS_PATH = f"{MODELS_DIR}/cnn.npz"


# === IDX 資料讀取 ===
# MNIST 官方提供的是二進位 IDX 格式，不是 PNG。這裡直接讀原始檔，比從圖片資料夾載入更快。


def read_images(path: str) -> tuple[np.ndarray, int, int, int]:
    """讀取 IDX3 圖像檔，回傳像素陣列、張數、列數、行數。"""
    with open(path, "rb") as f:
        # 大端序：magic(2051)、張數、列數、行數
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        # 每個像素是 0～255 的整數，先讀成 uint8 再轉 float
        pixels = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)  # frombuffer：將二進位 bytes 直接解讀為一維陣列
    return pixels, count, rows, cols


def read_labels(path: str) -> tuple[np.ndarray, int]:
    """讀取 IDX1 標籤檔，回傳標籤陣列與筆數。"""
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        labels = np.frombuffer(f.read(count), dtype=np.uint8)  # frombuffer：將二進位 bytes 直接解讀為一維標籤陣列
    return labels, count


def load_mnist_split(images_file: str, labels_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    載入單一分割（train 或 test），回傳：
    - X: 形狀 (N, 1, 28, 28)，像素已正規化到 0～1 並減去 MNIST 均值
    - y: 形狀 (N, 10) 的 one-hot 標籤
    """
    pixels, count, rows, cols = read_images(f"{MNIST_DIR}/{images_file}")
    labels, label_count = read_labels(f"{MNIST_DIR}/{labels_file}")
    if count != label_count:
        raise ValueError(
            f"mismatch: {count} images vs {label_count} labels in {images_file}"
        )

    # 正規化：0～255 → 0～1，再減 MNIST 均值做零均值化
    X = pixels.reshape(count, 1, rows, cols)  # reshape：一維像素重排成 (張數, 1, 高, 寬)
    X = X.astype(np.float64)  # astype：整數轉浮點數，才能做除法
    X = X / 255.0  # 除以 255，把 0~255 縮放到 0~1
    X = X - MNIST_MEAN  # 減去 MNIST 均值，做零均值正規化

    # one-hot：把類別數字（例如 3）變成 [0,0,0,1,0,0,0,0,0,0]
    y = np.zeros((count, NUM_CLASSES), dtype=np.float64)  # zeros：建立 (樣本數, 10) 的全 0 矩陣
    row_idx = np.arange(count)  # arange：產生 [0, 1, ..., count-1]，作為每張圖的列索引
    y[row_idx, labels] = 1.0  # labels[i] 是第 i 張圖的數字；在該欄設 1

    return X, y


# === im2col 輔助（卷積加速技巧）===
# 白話說明：卷積本來要在圖上滑動小窗口逐格計算。
# im2col 把每個窗口「攤平」成一根長向量，全部排成矩陣後，就能用一次矩陣乘法完成卷積。


def _calc_output_size(size: int, kernel: int, stride: int) -> int:
    """依輸入邊長、卷積核大小、步幅，計算輸出邊長。"""
    return (size - kernel) // stride + 1  # 卷積／池化輸出邊長公式


def im2col(
    x: np.ndarray, kernel_size: int, stride: int, padding: int
) -> tuple[np.ndarray, int, int]:
    """
    把輸入 x 轉成 im2col 矩陣。

    參數 x 形狀：(batch, channel, height, width)
    回傳 col 形狀：(batch, channel*kernel*kernel, out_h*out_w)
    """
    batch, channel, height, width = x.shape  # shape 解包：批次大小、通道數、高、寬
    if padding > 0:
        # 在四周補 0，讓卷積後的特徵圖不會縮太小
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

    # 把每個窗口攤平成一根向量
    transposed = col.transpose(0, 1, 2, 3, 4, 5)  # transpose：重排維度，方便後續 reshape
    col = transposed.reshape(batch, channel * kernel_size * kernel_size, -1)  # reshape：攤平窗口 → (batch, in*kh*kw, out_h*out_w)
    return col, out_h, out_w


def col2im(
    col: np.ndarray,
    input_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int,
    padding: int,
) -> np.ndarray:
    """
    im2col 的逆運算：把梯度從 col 格式還原回 (batch, channel, height, width)。
    反向傳播時，同一像素可能對應多個窗口，梯度要加總回去。
    """
    batch, channel, height, width = input_shape  # shape 解包：原始輸入的 batch、通道、高、寬
    out_h = _calc_output_size(height + 2 * padding, kernel_size, stride)  # 含 padding 後的輸出高度
    out_w = _calc_output_size(width + 2 * padding, kernel_size, stride)  # 含 padding 後的輸出寬度

    col = col.reshape(batch, channel, kernel_size, kernel_size, out_h, out_w)  # reshape：還原 col 的 6D 形狀
    x_padded = np.zeros(  # zeros：建立與 padded 輸入同形的梯度累加陣列
        (batch, channel, height + 2 * padding, width + 2 * padding),
        dtype=col.dtype,
    )

    for ky in range(kernel_size):
        y_max = ky + stride * out_h
        for kx in range(kernel_size):
            x_max = kx + stride * out_w
            x_padded[:, :, ky:y_max:stride, kx:x_max:stride] += col[:, :, ky, kx, :, :]  # +=：同一像素可能對應多窗口，梯度加總

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]  # 裁切：去掉 padding 區域，還原原始尺寸
    return x_padded


# === 激活函式與損失 ===
# ReLU：負值變 0，正值保留。簡單且能緩解梯度消失。
# Softmax：把 10 個分數轉成機率（加總為 1）。
# 交叉熵：衡量預測機率與正確答案的差距。


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活：max(0, x)。"""
    return np.maximum(0, x)  # maximum：逐元素取較大值，向量化實作 ReLU


def relu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """ReLU 反向傳播：x<=0 的位置梯度為 0，其餘原樣傳回。"""
    mask = x > 0  # 布林遮罩：x>0 的位置為 True
    return dout * mask  # 負值位置梯度歸零，其餘原樣傳回


def softmax(x: np.ndarray) -> np.ndarray:
    """
    對每一列（每個樣本的 10 個類別分數）做 softmax。
    先減去最大值是為了避免 exp 溢位（數值穩定技巧）。
    """
    shifted = x - np.max(x, axis=1, keepdims=True)  # max：axis=1 每列取最大；keepdims 保留維度方便相減
    exp_x = np.exp(shifted)  # exp：對每個分數取 e 的次方
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)  # sum：axis=1 每列加總，keepdims 保留維度方便相除
    return exp_x / sum_exp  # 逐元素相除，每列機率加總為 1


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """交叉熵損失：預測越準，loss 越小。"""
    batch = y_true.shape[0]  # shape[0]：第一維大小，代表這批有幾張圖
    row_idx = np.arange(batch)  # arange：產生 [0, 1, ..., batch-1]，作為每張圖的列索引
    true_labels = np.argmax(y_true, axis=1)  # argmax：axis=1 沿 10 類別找最大值位置 → 正確數字 0~9
    correct_probs = probs[row_idx, true_labels]  # 用列+欄索引，取出每張圖對正確數字的預測機率
    safe_probs = correct_probs + 1e-12  # 加微小值，避免機率為 0 時 log(0) 出錯
    log_probs = np.log(safe_probs)  # log：逐元素取自然對數
    sum_log_probs = np.sum(log_probs)  # sum：加總這批所有樣本的 log 機率
    avg_log_prob = sum_log_probs / batch  # 除以 batch 取平均
    return float(-avg_log_prob)  # 取負值得交叉熵（預測越準 loss 越小）


def cross_entropy_gradient(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    softmax + 交叉熵合併後的梯度，形式非常簡潔：(預測機率 - 正確答案) / batch。
    這是反向傳播的起點。
    """
    batch = y_true.shape[0]  # shape[0]：這批樣本數
    return (probs - y_true) / batch  # 預測機率減 one-hot 標籤，再除以 batch 取平均


# === 參數初始化 ===
# 不用 class，改以字典存放每層權重 W、bias b，以及梯度 dW、db。
# 例如 params["conv1"]["W"] 就是第一層卷積的濾鏡權重。


def init_conv_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: int = 0,
) -> dict:
    """建立卷積層參數字典。He 初始化適合 ReLU。"""
    fan_in = in_channels * kernel_size * kernel_size  # 每個濾鏡的輸入連線數
    scale = np.sqrt(2.0 / fan_in)  # sqrt：He 初始化縮放因子，適合 ReLU
    W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale  # randn：標準常態亂數 × scale
    b = np.zeros(out_channels, dtype=np.float64)  # zeros：建立長度 out_channels 的全 0 偏置向量
    return {
        "W": W,
        "b": b,
        "dW": np.zeros_like(W),  # zeros_like：建立與 W 同形狀的全 0 梯度矩陣
        "db": np.zeros_like(b),  # zeros_like：建立與 b 同形狀的全 0 梯度向量
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": 1,
        "padding": padding,
    }


def init_dense_params(in_features: int, out_features: int) -> dict:
    """建立全連接層參數字典。Xavier 初始化讓前向與反向方差平衡。"""
    scale = np.sqrt(2.0 / (in_features + out_features))  # sqrt：開平方根，得 Xavier 縮放因子
    W = np.random.randn(in_features, out_features) * scale  # randn：產生標準常態亂數，再乘以 scale
    b = np.zeros(out_features, dtype=np.float64)  # zeros：建立長度 out_features 的全 0 偏置向量
    return {
        "W": W,
        "b": b,
        "dW": np.zeros_like(W),  # zeros_like：建立與 W 同形狀的全 0 梯度矩陣
        "db": np.zeros_like(b),  # zeros_like：建立與 b 同形狀的全 0 梯度向量
    }


# === 卷積層（純函式）===
# 卷積層用多個小濾鏡在圖上滑動，提取邊緣、筆畫等局部特徵。


def conv_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    卷積前向傳播。
    輸入 x 形狀：(batch, in_channels, height, width)
    回傳輸出與 cache（反向傳播時需要的中間結果）。
    """
    W = params["W"]
    b = params["b"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = params["padding"]
    out_channels = params["out_channels"]
    batch = x.shape[0]

    col, out_h, out_w = im2col(x, kernel_size, stride, padding)
    W_col = W.reshape(out_channels, -1)  # reshape：濾鏡攤平成 (out_ch, in*kh*kw)

    # batch 向量化：col (batch, in*kh*kw, out_h*out_w)，W_col (out_ch, in*kh*kw)
    conv_out = np.einsum("oi,bil->bol", W_col, col)  # einsum：批次矩陣乘法完成卷積
    b_broadcast = b.reshape(1, -1, 1)  # reshape：偏置擴展為 (1, out_ch, 1) 方便廣播
    out = conv_out + b_broadcast  # 加上偏置

    cache = {
        "x": x,
        "col": col,
        "out_h": out_h,
        "out_w": out_w,
    }
    out_4d = out.reshape(batch, out_channels, out_h, out_w)  # reshape：還原為 (batch, out_ch, out_h, out_w)
    return out_4d, cache


def conv_backward(dout: np.ndarray, params: dict, cache: dict) -> np.ndarray:
    """
    卷積反向傳播：累積 dW、db 到 params，回傳對輸入的梯度 dx。
    """
    W = params["W"]
    x = cache["x"]
    col = cache["col"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = params["padding"]
    out_channels = params["out_channels"]
    batch = x.shape[0]

    W_col = W.reshape(out_channels, -1)  # reshape：濾鏡攤平成 (out_ch, in*kh*kw)
    dout_col = dout.reshape(batch, out_channels, -1)  # reshape：輸出梯度攤平

    dW_flat = np.einsum("bol,bil->oi", dout_col, col)  # einsum：計算權重梯度
    params["dW"] += dW_flat.reshape(W.shape)  # reshape：還原為 W 的 4D 形狀後累加
    params["db"] += dout_col.sum(axis=(0, 2))  # sum：沿 batch 與空間維加總 → 形狀 (out_ch,)

    dcol = np.einsum("oi,bol->bil", W_col, dout_col)  # einsum：計算 col 格式輸入梯度
    return col2im(dcol, x.shape, kernel_size, stride, padding)  # col2im：還原為 (batch, in_ch, H, W)


# === 最大池化層（純函式）===
# 池化把鄰近區域壓縮成一個值（這裡取最大值），縮小特徵圖、減少計算量。


def maxpool_forward(
    x: np.ndarray, pool_size: int = 2, stride: int = 2
) -> tuple[np.ndarray, dict]:
    """
    最大池化前向傳播。
    輸入 x 形狀：(batch, channel, height, width)
    """
    batch, channel, height, width = x.shape
    out_h = _calc_output_size(height, pool_size, stride)
    out_w = _calc_output_size(width, pool_size, stride)

    out = np.zeros((batch, channel, out_h, out_w), dtype=np.float64)
    max_mask = np.zeros_like(x, dtype=bool)

    for i in range(out_h):
        for j in range(out_w):
            y0 = i * stride
            x0 = j * stride
            window = x[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size]
            window_max = np.max(window, axis=(2, 3))  # max：沿高寬維取窗口最大值
            out[:, :, i, j] = window_max  # 寫入池化輸出
            max_val = out[:, :, i, j, np.newaxis, np.newaxis]  # newaxis：擴展維度以便與 window 比較
            is_max = window == max_val  # 標記窗口內哪些位置是最大值
            max_mask[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size] = is_max  # 記錄 mask 供反向傳播

    cache = {"x": x, "max_mask": max_mask, "pool_size": pool_size, "stride": stride}
    return out, cache


def maxpool_backward(dout: np.ndarray, cache: dict) -> np.ndarray:
    """最大池化反向傳播：梯度只回傳到前向時取到最大值的位置。"""
    x = cache["x"]
    max_mask = cache["max_mask"]
    pool_size = cache["pool_size"]
    stride = cache["stride"]
    dx = np.zeros_like(x)
    _, _, out_h, out_w = dout.shape

    for i in range(out_h):
        for j in range(out_w):
            y0 = i * stride
            x0 = j * stride
            window_mask = max_mask[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size]
            dout_expanded = dout[:, :, i, j, np.newaxis, np.newaxis]  # newaxis：擴展梯度至窗口大小
            grad_window = window_mask * dout_expanded  # 梯度只回傳到前向取最大值的位置
            dx[:, :, y0 : y0 + pool_size, x0 : x0 + pool_size] += grad_window  # 累加至輸入梯度

    return dx


# === 全連接層（純函式）===
# 全連接層把展平後的特徵向量，線性組合成新的特徵或最終類別分數。


def dense_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """全連接前向傳播：y = x @ W + b"""
    out = x @ params["W"]  # @：矩陣乘法 (batch, in) × (in, out)
    return out + params["b"]  # 加上偏置 b，shape (out,)


def dense_backward(dout: np.ndarray, params: dict, x: np.ndarray) -> np.ndarray:
    """全連接反向傳播：累積 dW、db，回傳 dx。"""
    dW = x.T @ dout  # T：轉置 x；@：矩陣乘法 (in, batch) × (batch, out) → 權重梯度
    params["dW"] += dW  # 累加到 params，供 SGD 更新
    params["db"] += np.sum(dout, axis=0)  # sum：axis=0 沿 batch 維加總 → 形狀 (out,)
    dx = dout @ params["W"].T  # @：dout 與 W.T 矩陣乘法；T：轉置 W
    return dx  # 梯度往前層傳遞


# === CNN 前向／反向／更新（純函式）===
# 把各層函式串起來，params 放權重，cache 放前向傳播的中間結果。


def model_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    整個 CNN 的前向傳播，回傳 softmax 機率與 cache。
    架構：Conv → ReLU → MaxPool → 展平 → FC → ReLU → FC → Softmax
    """
    cache: dict = {}

    c1, conv1_cache = conv_forward(x, params["conv1"])
    cache["conv1"] = conv1_cache
    cache["c1"] = c1

    r1 = relu(c1)
    cache["r1"] = r1

    p1, pool1_cache = maxpool_forward(r1, pool_size=POOL_SIZE, stride=POOL_SIZE)
    cache["pool1"] = pool1_cache
    cache["p1"] = p1

    flat = p1.reshape(p1.shape[0], -1)  # reshape：-1 表示其餘維度自動計算 → (batch, flat_size)
    cache["flat"] = flat

    f1 = dense_forward(flat, params["fc1"])
    cache["f1"] = f1

    r3 = relu(f1)
    cache["r3"] = r3

    logits = dense_forward(r3, params["fc2"])
    cache["logits"] = logits

    probs = softmax(logits)
    return probs, cache


def model_backward(probs: np.ndarray, y_true: np.ndarray, params: dict, cache: dict) -> None:
    """從 softmax+交叉熵的梯度開始，逐層反向傳播，梯度寫入 params 的 dW、db。"""
    dout = cross_entropy_gradient(probs, y_true)

    dout = dense_backward(dout, params["fc2"], cache["r3"])
    dout = relu_backward(cache["f1"], dout)
    dout = dense_backward(dout, params["fc1"], cache["flat"])

    p1_shape = cache["p1"].shape[1:]  # shape[1:]：去掉 batch 維，取得 (channel, H, W)
    dout = dout.reshape(dout.shape[0], *p1_shape)  # reshape：還原為池化輸出的 4D 形狀

    dout = maxpool_backward(dout, cache["pool1"])
    dout = relu_backward(cache["c1"], dout)
    conv_backward(dout, params["conv1"], cache["conv1"])


def zero_grads(params: dict) -> None:
    """把各層累積的梯度清零，準備下一個 batch。"""
    for layer in params.values():
        layer["dW"].fill(0)  # fill：原地將 dW 所有元素設為 0
        layer["db"].fill(0)  # fill：原地將 db 所有元素設為 0


def init_velocity(params: dict) -> dict:
    """為 Momentum SGD 建立與各層 W、b 同形的速度字典。"""
    return {
        name: {
            "vW": np.zeros_like(layer["W"]),  # zeros_like：與 W 同形狀的全 0 速度矩陣
            "vb": np.zeros_like(layer["b"]),  # zeros_like：與 b 同形狀的全 0 速度向量
        }
        for name, layer in params.items()
    }


def update_params(
    params: dict, velocity: dict, learning_rate: float, momentum: float
) -> None:
    """Momentum SGD：v = momentum*v - lr*grad；W += v。"""
    for name, layer in params.items():
        v = velocity[name]
        v["vW"] = momentum * v["vW"] - learning_rate * layer["dW"]  # 更新 W 的速度向量
        v["vb"] = momentum * v["vb"] - learning_rate * layer["db"]  # 更新 b 的速度向量
        layer["W"] += v["vW"]  # 沿速度方向更新權重
        layer["b"] += v["vb"]  # 沿速度方向更新偏置


def predict(x: np.ndarray, params: dict) -> np.ndarray:
    """回傳每筆樣本預測的數字類別（0～9）。"""
    probs, _ = model_forward(x, params)  # _ 表示忽略 cache
    return np.argmax(probs, axis=1)  # argmax：每列 10 個機率中取最大值 index → 預測數字


def save_params(params: dict, path: str) -> None:
    """將各層 W、b 保存為 .npz，供 step 4 推理載入。"""
    np.savez_compressed(  # savez_compressed：以壓縮格式將多個 ndarray 寫入單一 .npz
        path,
        conv1_W=params["conv1"]["W"],
        conv1_b=params["conv1"]["b"],
        fc1_W=params["fc1"]["W"],
        fc1_b=params["fc1"]["b"],
        fc2_W=params["fc2"]["W"],
        fc2_b=params["fc2"]["b"],
    )


# === 訓練與評估 ===


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
):
    """
    把資料切成一個個小批次（mini-batch）。
    每個 epoch 通常會打亂順序，避免模型記住固定順序。
    """
    n = X.shape[0]  # shape[0]：資料集總樣本數
    indices = np.arange(n)  # arange：產生 [0, 1, ..., n-1]，每筆樣本的索引
    if shuffle:
        np.random.shuffle(indices)  # shuffle：原地打亂索引順序

    for start in range(0, n, batch_size):  # range(0, n, batch_size)：每 batch_size 筆切一批
        end = min(start + batch_size, n)  # min：最後一批可能不足 batch_size
        batch_idx = indices[start:end]  # 切片取出這批的索引
        yield X[batch_idx], y[batch_idx]  # yield：每次回傳一小批 X 和 y，不一次載入全部


# === 訓練主流程 ===


def run_training() -> None:
    """載入資料、訓練 CNN、評估並保存權重，印出逐步進度。"""
    print("=== CNN Training ===")

    print("[1/5] Loading MNIST data ...")
    X_train, y_train = load_mnist_split(
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
    )
    X_test, y_test = load_mnist_split(
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    )
    print(f"      train: {X_train.shape[0]} samples")
    print(f"      test:  {X_test.shape[0]} samples")
    print(
        f"      shape: {X_train.shape}, normalized [0, 1] minus mean {MNIST_MEAN}"
    )

    # 建立整個 CNN 的參數字典
    # 架構：Conv(16, pad=1) → ReLU → MaxPool → FC(128) → ReLU → FC(10) → Softmax
    print("[2/5] Initializing model ...")
    params = {
        "conv1": init_conv_params(
            1, CONV_OUT_CHANNELS, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING
        ),
        "fc1": init_dense_params(FLAT_SIZE, HIDDEN_SIZE),
        "fc2": init_dense_params(HIDDEN_SIZE, NUM_CLASSES),
    }
    velocity = init_velocity(params)
    print(
        f"      conv1 W: {params['conv1']['W'].shape}  "
        f"fc1 W: {params['fc1']['W'].shape}  "
        f"fc2 W: {params['fc2']['W'].shape}"
    )

    num_batches = (X_train.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE  # 向上取整：每 epoch 有幾批
    print("[3/5] Training ...")
    print(
        f"      {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
        f"{num_batches} batches/epoch, lr={LEARNING_RATE}, momentum={MOMENTUM}"
    )

    for epoch in range(1, EPOCHS + 1):
        lr = LEARNING_RATE * (0.5 if epoch >= 4 else 1.0)  # 第 4 epoch 起學習率減半，避免後期震盪
        if epoch >= 4:
            print(f"      epoch {epoch}/{EPOCHS}  lr={lr}")

        total_loss = 0.0
        total_batches = 0
        correct = 0
        train_total = X_train.shape[0]

        for batch_idx, (X_batch, y_batch) in enumerate(
            iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True),
            start=1,
        ):
            zero_grads(params)
            probs, cache = model_forward(X_batch, params)
            loss = cross_entropy_loss(probs, y_batch)
            model_backward(probs, y_batch, params, cache)
            update_params(params, velocity, lr, MOMENTUM)

            total_loss += loss
            total_batches += 1
            preds = predict(X_batch, params)  # 模型預測的 0~9
            true_labels = np.argmax(y_batch, axis=1)  # argmax：axis=1 從 one-hot 取正確數字 0~9
            correct += np.sum(preds == true_labels)  # sum：布林 True 當 1 加總 → 本批猜對幾張

            if batch_idx % 100 == 0 or batch_idx == num_batches:
                avg_loss = total_loss / total_batches
                samples_seen = min(batch_idx * BATCH_SIZE, train_total)
                train_acc = correct / samples_seen
                print(
                    f"      epoch {epoch}/{EPOCHS}  batch {batch_idx}/{num_batches}  "
                    f"loss={loss:.4f}  avg_loss={avg_loss:.4f}  "
                    f"train_acc={train_acc * 100:.1f}%"
                )

        avg_loss = total_loss / total_batches
        train_acc = correct / train_total
        print(
            f"      epoch {epoch}/{EPOCHS} done  "
            f"loss={avg_loss:.4f}  train_acc={train_acc * 100:.1f}%"
        )

    print("[4/5] Evaluating on test set ...")
    test_correct = 0
    test_total = X_test.shape[0]
    eval_batch_size = 256
    test_num_batches = (test_total + eval_batch_size - 1) // eval_batch_size  # 向上取整：測試集批數

    for batch_idx, (X_batch, y_batch) in enumerate(
        iterate_minibatches(X_test, y_test, eval_batch_size, shuffle=False),
        start=1,
    ):
        preds = predict(X_batch, params)
        true_labels = np.argmax(y_batch, axis=1)  # argmax：axis=1 從 one-hot 取正確數字 0~9
        test_correct += np.sum(preds == true_labels)  # sum：布林 True 當 1 加總 → 本批猜對幾張
        if batch_idx % 100 == 0 or batch_idx == test_num_batches:
            print(
                f"      eval batch {batch_idx}/{test_num_batches}  "
                f"running_acc={test_correct / test_total * 100:.1f}%"
            )

    print(f"      test accuracy: {test_correct / test_total * 100:.1f}%")

    print("[5/5] Saving weights ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_params(params, WEIGHTS_PATH)
    print(f"      Saved to {WEIGHTS_PATH}")


# === 主程式 ===

if __name__ == "__main__":
    # 確認 step 1 已下載所需檔案
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]  # 列表推導：找出缺少的 IDX 檔
    if missing:
        print("Missing MNIST files:", ", ".join(missing))
        print("Run step_1_download_mnist.py first.")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)  # seed：固定亂數種子，使初始化與 shuffle 可重現
    run_training()
