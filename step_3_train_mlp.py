"""
步驟 3（MLP 版）：用純 NumPy 手寫全連接神經網路，訓練 MNIST 手寫數字辨識模型。

架構：展平 784 → FC(128) → ReLU → FC(10) → Softmax（不含卷積與池化）。
學會反向傳播後，可再跑 step_3_train_cnn.py 學習卷積特徵提取。

本檔案包含從資料讀取、前向傳播、反向傳播到訓練迴圈的全部邏輯。
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
EPOCHS = 100          # 整份訓練集掃過幾輪；越大通常越準，但越慢
BATCH_SIZE = 64       # 每次更新權重用多少筆樣本；太大可能學不細，太小可能不穩
LEARNING_RATE = 0.01  # 學習率：權重每次更新的步幅；太大容易發散，太小學太慢
RANDOM_SEED = 42      # 固定亂數種子，讓每次執行結果可重現

# MNIST 有 10 個類別（數字 0～9）
NUM_CLASSES = 10

# 28×28 灰階圖展平後的輸入維度
INPUT_SIZE = 28 * 28

# 唯一的全連接隱藏層維度
HIDDEN_SIZE = 128

# 訓練完成後保存權重的目錄與檔案（供 step 4 推理使用）
MODELS_DIR = "models"
WEIGHTS_PATH = f"{MODELS_DIR}/mlp.npz"


# === IDX 資料讀取 ===
# MNIST 官方提供的是二進位 IDX 格式，不是 PNG。這裡直接讀原始檔，比從圖片資料夾載入更快。


def read_images(path: str) -> tuple[np.ndarray, int, int, int]:
    """讀取 MNIST IDX3 圖像檔，回傳像素陣列與維度資訊。

    參數
    ----
    path : str
        IDX3 格式圖像檔的本機路徑（magic number 須為 2051）。

    回傳
    ----
    tuple[np.ndarray, int, int, int]
        四元組，依序為：
        - pixels：np.ndarray，形狀 ``(count * rows * cols,)``，dtype uint8
        - count：int，圖像張數
        - rows：int，每張圖列數（MNIST 為 28）
        - cols：int，每張圖行數（MNIST 為 28）
    """
    with open(path, "rb") as f:
        # 大端序：magic(2051)、張數、列數、行數
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))  # unpack：讀檔頭得張數與每張圖 28×28 的維度
        if magic != 2051:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        # 每個像素是 0～255 的整數，先讀成 uint8 再轉 float
        pixels = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)  # frombuffer：將二進位 bytes 直接解讀為一維陣列
    return pixels, count, rows, cols


def read_labels(path: str) -> tuple[np.ndarray, int]:
    """讀取 MNIST IDX1 標籤檔，回傳標籤陣列與筆數。

    參數
    ----
    path : str
        IDX1 格式標籤檔的本機路徑（magic number 須為 2049）。

    回傳
    ----
    tuple[np.ndarray, int]
        二元組，依序為：
        - labels：np.ndarray，形狀 ``(count,)``，dtype uint8，值 0～9
        - count：int，標籤筆數
    """
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))  # unpack：讀檔頭得標籤筆數
        if magic != 2049:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        labels = np.frombuffer(f.read(count), dtype=np.uint8)  # frombuffer：將二進位 bytes 直接解讀為一維標籤陣列
    return labels, count


def load_mnist_split(images_file: str, labels_file: str) -> tuple[np.ndarray, np.ndarray]:
    """載入單一分割（train 或 test），回傳正規化特徵與 one-hot 標籤。

    參數
    ----
    images_file : str
        IDX3 圖像檔名（不含 ``mnist/`` 目錄前綴）。
    labels_file : str
        IDX1 標籤檔名（不含 ``mnist/`` 目錄前綴）。

    回傳
    ----
    tuple[np.ndarray, np.ndarray]
        二元組，依序為：
        - X：np.ndarray，形狀 ``(N, 1, 28, 28)``，像素正規化至 0～1，dtype float64
        - y：np.ndarray，形狀 ``(N, 10)``，one-hot 編碼標籤，dtype float64
    """
    pixels, count, rows, cols = read_images(f"{MNIST_DIR}/{images_file}")  # 讀取圖像像素與張數、高、寬
    labels, label_count = read_labels(f"{MNIST_DIR}/{labels_file}")  # 讀取每張圖對應的數字標籤 0~9
    if count != label_count:
        raise ValueError(
            f"mismatch: {count} images vs {label_count} labels in {images_file}"
        )

    # 正規化：把 0～255 縮放到 0～1，讓梯度更新更穩定
    X = pixels.reshape(count, 1, rows, cols)  # reshape：一維像素重排成 (張數, 1, 高, 寬)
    X = X.astype(np.float64)  # astype：整數轉浮點數，才能做除法
    X = X / 255.0  # 除以 255，把 0~255 縮放到 0~1，讓梯度更新更穩定

    # one-hot：把類別數字（例如 3）變成 [0,0,0,1,0,0,0,0,0,0]，請注意這裡的1在從0開始計算的第3個位置上
    # 交叉熵或 MSE 皆需 one-hot，才能與模型輸出逐類別比較
    y = np.zeros((count, NUM_CLASSES), dtype=np.float64)  # zeros：建立 (樣本數, 10) 的全 0 矩陣
    row_idx = np.arange(count)  # arange：產生 [0, 1, ..., count-1]，作為每張圖的列索引
    y[row_idx, labels] = 1.0  # labels[i] 是第 i 張圖的數字；在該欄設 1

    return X, y


# === 激活函式與損失 ===
# ReLU：負值變 0，正值保留。簡單且能緩解梯度消失。
# Softmax：把 10 個分數轉成機率（加總為 1）；推理時用於取 argmax。
# 交叉熵：衡量 softmax 機率與 one-hot 標籤的差距（訓練迴圈可切換）。
# MSE：衡量 logits 與 one-hot 標籤的均方誤差（訓練迴圈預設使用）。


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

    先減去每列最大值以避免 exp 溢位（數值穩定技巧）。

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
    exponentiated_scores = np.exp(shifted)  # exp：對每個分數取 e 的次方
    sum_exponentiated = np.sum(exponentiated_scores, axis=1, keepdims=True)  # sum：axis=1 每列加總，keepdims 保留維度方便相除
    return exponentiated_scores / sum_exponentiated  # 逐元素相除，每列機率加總為 1


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """計算 softmax 輸出與 one-hot 標籤的交叉熵損失（批次平均）。

    參數
    ----
    probs : np.ndarray
        softmax 機率，形狀 ``(batch, num_classes)``。
    y_true : np.ndarray
        one-hot 標籤，形狀 ``(batch, num_classes)``。

    回傳
    ----
    float
        標量 loss 值；預測越準，loss 越小。
    """
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
    """softmax + 交叉熵合併後對 logits 的梯度，為反向傳播起點。

    參數
    ----
    probs : np.ndarray
        softmax 機率，形狀 ``(batch, num_classes)``。
    y_true : np.ndarray
        one-hot 標籤，形狀 ``(batch, num_classes)``。

    回傳
    ----
    np.ndarray
        梯度，形狀 ``(batch, num_classes)``，公式為 ``(probs - y_true) / batch``。
    """
    batch = y_true.shape[0]  # shape[0]：這批樣本數
    return (probs - y_true) / batch  # 預測機率減 one-hot 標籤，再除以 batch 取平均


def mse_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """計算 logits 與 one-hot 標籤的均方誤差（全元素平均）。

    參數
    ----
    logits : np.ndarray
        最後一層輸出分數，形狀 ``(batch, num_classes)``。
    y_true : np.ndarray
        one-hot 標籤，形狀 ``(batch, num_classes)``。

    回傳
    ----
    float
        標量 loss 值；預測越準，loss 越小。
    """
    return float(np.mean((logits - y_true) ** 2))  # mean：全元素平均平方差


def mse_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """MSE 對 logits 的梯度，為反向傳播起點。

    參數
    ----
    logits : np.ndarray
        最後一層輸出分數，形狀 ``(batch, num_classes)``。
    y_true : np.ndarray
        one-hot 標籤，形狀 ``(batch, num_classes)``。

    回傳
    ----
    np.ndarray
        梯度，形狀 ``(batch, num_classes)``，與 ``mse_loss`` 的 ``np.mean`` 定義一致。
    """
    batch, num_classes = y_true.shape  # shape：這批樣本數與類別數
    return 2.0 * (logits - y_true) / (batch * num_classes)  # MSE 對 logits 的偏導數


# === 參數初始化 ===
# 不用 class，改以字典存放每層 weights、bias，以及梯度 grad_weights、grad_bias。


def init_dense_params(in_features: int, out_features: int) -> dict:
    """建立全連接層參數字典，以 Xavier 初始化權重。

    參數
    ----
    in_features : int
        輸入特徵維度（例如 784 或 128）。
    out_features : int
        輸出特徵維度（例如 128 或 10）。

    回傳
    ----
    dict
        全連接層參數字典，含：
        - ``"weights"``：np.ndarray，形狀 ``(in_features, out_features)``
        - ``"bias"``：np.ndarray，形狀 ``(out_features,)``
        - ``"grad_weights"``：np.ndarray，與 ``"weights"`` 同形，梯度累積用，初始全 0
        - ``"grad_bias"``：np.ndarray，與 ``"bias"`` 同形，梯度累積用，初始全 0
    """
    scale = np.sqrt(2.0 / (in_features + out_features))  # sqrt：開平方根，得 Xavier 縮放因子
    weights = np.random.randn(in_features, out_features) * scale  # randn：產生標準常態亂數，再乘以 scale
    bias = np.zeros(out_features, dtype=np.float64)  # zeros：建立長度 out_features 的全 0 偏置向量
    return {
        "weights": weights,  # 這層的權重矩陣
        "bias": bias,  # 這層的偏置向量
        "grad_weights": np.zeros_like(weights),  # zeros_like：建立與 weights 同形狀的全 0 梯度矩陣
        "grad_bias": np.zeros_like(bias),  # zeros_like：建立與 bias 同形狀的全 0 梯度向量
    }


# === MLP 前向／反向／更新（純函式）===
# forward／backward 各一函式串起整網；params 放權重，cache 放前向中間結果。


def forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """整個 MLP 的前向傳播：展平 → FC(128) → ReLU → FC(10) → Softmax。

    參數
    ----
    x : np.ndarray
        輸入影像，形狀 ``(batch, 1, 28, 28)``，像素 0～1。
    params : dict
        模型參數字典，須含：
        - ``"fc1"``：784→128 全連接層（``init_dense_params`` 格式）
        - ``"fc2"``：128→10 全連接層

    回傳
    ----
    tuple[np.ndarray, dict]
        二元組，依序為：
        - probs：np.ndarray，形狀 ``(batch, 10)``，各類別預測機率
        - cache：dict，前向中間結果，含 ``"flattened_input"``、``"hidden_linear"``、``"hidden_relu"``、``"logits"``
    """
    cache: dict = {}  # 暫存前向中間結果，反向傳播時要用

    # 把二維影像的像素點展平成一維，變成 784 維的向量
    flattened_input = x.reshape(x.shape[0], -1)  # reshape：-1 表示其餘維度自動計算 → (batch, 784)
    cache["flattened_input"] = flattened_input  # 存展平結果，fc1 反向時要用

    # 第一層全連接層，784 維的向量乘以 128 維的權重，加上 128 維的偏置，得到 128 維的 hidden_linear
    hidden_linear = flattened_input @ params["fc1"]["weights"] + params["fc1"]["bias"]  # @：矩陣乘法 (batch, 784) × (784, 128)
    cache["hidden_linear"] = hidden_linear  # 存 ReLU 前數值，判斷哪些位置要截斷梯度

    # ReLU 激活函式，把負值變成 0，正值不變
    hidden_relu = relu(hidden_linear)  # ReLU：負值歸零，只保留正特徵
    cache["hidden_relu"] = hidden_relu  # 存隱藏層輸出，fc2 反向時要用

    # 最後一層輸出分數（logits），128 維的向量乘以 10 維的權重，加上 10 維的偏置，得到 10 維的 logits
    logits = hidden_relu @ params["fc2"]["weights"] + params["fc2"]["bias"]  # @：矩陣乘法 (batch, 128) × (128, 10)
    cache["logits"] = logits  # 存 10 類分數（尚未轉機率），切換 MSE 損失時要用

    # Softmax 激活函式
    probs = softmax(logits)  # 把 10 個分數轉成機率，每列加總為 1
    return probs, cache  # 回傳機率與暫存，供算 loss 與反向傳播


def backward(upstream_gradient: np.ndarray, params: dict, cache: dict) -> None:
    """從 fc2 輸出（logits）的梯度開始，逐層反向傳播，梯度寫入 params 的 grad_weights、grad_bias。

    參數
    ----
    upstream_gradient : np.ndarray
        損失對 logits 的梯度，形狀 ``(batch, 10)``；由 ``mse_gradient`` 或
        ``cross_entropy_gradient`` 計算。
    params : dict
        模型參數字典（``"fc1"``、``"fc2"``）；各層 ``"grad_weights"``、``"grad_bias"`` 原地累加。
    cache : dict
        ``forward`` 回傳的中間結果，含 ``"flattened_input"``、``"hidden_linear"``、``"hidden_relu"``。

    回傳
    ----
    None
        無回傳值；梯度寫入 ``params`` 各層的 ``"grad_weights"`` 與 ``"grad_bias"``。
    """
    # fc2 反向：output = input @ weights + bias
    params["fc2"]["grad_weights"] += cache["hidden_relu"].T @ upstream_gradient  # T：轉置；@：(128, batch) × (batch, 10)
    params["fc2"]["grad_bias"] += np.sum(upstream_gradient, axis=0)  # sum：沿 batch 維加總 → (10,)
    upstream_gradient = upstream_gradient @ params["fc2"]["weights"].T  # @：梯度傳回 fc2 輸入 → (batch, 128)

    # ReLU 反向：前向輸入 <=0 的位置梯度歸零
    upstream_gradient = upstream_gradient * (cache["hidden_linear"] > 0)  # 布林遮罩逐元素相乘，向量化實作

    # fc1 反向
    params["fc1"]["grad_weights"] += cache["flattened_input"].T @ upstream_gradient  # T：轉置；@：(784, batch) × (batch, 128)
    params["fc1"]["grad_bias"] += np.sum(upstream_gradient, axis=0)  # sum：沿 batch 維加總 → (128,)


def predict(x: np.ndarray, params: dict) -> np.ndarray:
    """對輸入批次做前向推理，回傳每筆樣本預測的數字類別。

    參數
    ----
    x : np.ndarray
        輸入影像，形狀 ``(batch, 1, 28, 28)``。
    params : dict
        模型參數字典（``"fc1"``、``"fc2"``）。

    回傳
    ----
    np.ndarray
        預測類別，形狀 ``(batch,)``，dtype int64，值 0～9。
    """
    probs, _ = forward(x, params)  # _ 表示忽略 cache
    return np.argmax(probs, axis=1)  # argmax：每列 10 個機率中取最大值 index → 預測數字


# === 主程式 ===
if __name__ == "__main__":
    # 確認 step 1 已下載所需檔案
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]  # 列出缺少的 IDX 檔名
    if missing:
        print("Missing MNIST files:", ", ".join(missing))
        print("Run step_1_download_mnist.py first.")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)  # seed：固定亂數種子，使初始化與 shuffle 可重現
    print("=== MLP Training ===")

    print("[1/5] Loading MNIST data ...")
    X_train, y_train = load_mnist_split(
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
    )  # 載入 60000 張訓練圖與 one-hot 標籤
    X_test, y_test = load_mnist_split(
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    )  # 載入 10000 張測試圖與 one-hot 標籤
    print(f"      train: {X_train.shape[0]} samples")
    print(f"      test:  {X_test.shape[0]} samples")
    print(
        f"      shape: {X_train.shape}, normalized [0, 1]"
    )

    # 建立 MLP 的參數字典
    # 架構：FC(784→128) → ReLU → FC(128→10) → Softmax
    print("[2/5] Initializing model ...")
    params = {
        "fc1": init_dense_params(INPUT_SIZE, HIDDEN_SIZE),  # 784→128 全連接層權重
        "fc2": init_dense_params(HIDDEN_SIZE, NUM_CLASSES),  # 128→10 全連接層權重
    }
    print(
        f"      fc1 weights: {params['fc1']['weights'].shape}  "
        f"fc2 weights: {params['fc2']['weights'].shape}"
    )

    # +BATCH_SIZE-1 再 // 是向上取整除法，最後不足一批也算一批
    num_batches = (X_train.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE  # 每 epoch 要跑幾批（向上取整）
    print("[3/5] Training ...")
    print(
        f"      {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
        f"{num_batches} batches/epoch, lr={LEARNING_RATE}"
    )

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0  # 本 epoch 累積的 loss 總和
        total_batches = 0  # 本 epoch 已處理的批數
        correct = 0  # 本 epoch 猜對的張數
        train_total = X_train.shape[0]  # 訓練集總張數（60000）

        batch_idx = 0  # 本 epoch 目前跑到第幾批
        indices = np.arange(train_total)  # arange：產生 [0, 1, ..., train_total-1]
        np.random.shuffle(indices)  # shuffle：每 epoch 打亂樣本順序

        for start in range(0, train_total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, train_total)  # 這批結束位置（最後一批可能不足 64 張）
            batch_indices = indices[start:end]  # 切片取出這批的索引
            X_batch = X_train[batch_indices]  # 依索引取出這批的輸入圖
            y_batch = y_train[batch_indices]  # 依索引取出這批的 one-hot 標籤
            batch_idx += 1  # 批數加 1

            for layer in ("fc1", "fc2"):
                params[layer]["grad_weights"].fill(0)  # fill：梯度清零，準備本批累加
                params[layer]["grad_bias"].fill(0)  # fill：偏置梯度清零

            probs, cache = forward(X_batch, params)  # 前向傳播：算預測機率並暫存中間結果

            # --- Cross entropy（切換時註釋 MSE 區塊、取消本區註釋）---
            loss = cross_entropy_loss(probs, y_batch)  # 算這批「猜錯有多嚴重」；數字越小越準
            upstream_gradient = cross_entropy_gradient(probs, y_batch)  # 算出調整方向的起點，交給 backward 往回傳

            # --- MSE on logits（切換時註釋上方 Cross entropy 區塊、取消本區註釋）---
            # loss = mse_loss(cache["logits"], y_batch)  # 均方誤差：比對分數與 one-hot 標籤的差距
            # upstream_gradient = mse_gradient(cache["logits"], y_batch)  # MSE 對分數的梯度，作為反向起點

            backward(upstream_gradient, params, cache)  # 反向傳播：把梯度寫入各層 grad_weights、grad_bias

            # 更新權重和偏置
            for layer in ("fc1", "fc2"):
                layer_params = params[layer]  # 取出這一層的權重、偏置與梯度
                layer_params["weights"] -= LEARNING_RATE * layer_params["grad_weights"]  # SGD：沿梯度反方向更新權重
                layer_params["bias"] -= LEARNING_RATE * layer_params["grad_bias"]  # SGD：沿梯度反方向更新偏置

            total_loss += loss  # 把本批 loss 加進 epoch 累積
            total_batches += 1  # 已處理批數加 1
            preds = predict(X_batch, params)  # 用更新後的權重預測這批數字 0~9
            true_labels = np.argmax(y_batch, axis=1)  # argmax：從 one-hot 取出每張圖的正確數字
            correct += np.sum(preds == true_labels)  # 比對預測與正確答案，累加本批猜對張數

            # 每 100 批或最後一批印出進度
            if batch_idx % 100 == 0 or batch_idx == num_batches:
                avg_loss = total_loss / total_batches  # 到目前為止的平均 loss
                samples_seen = min(batch_idx * BATCH_SIZE, train_total)  # 本 epoch 已看過幾張圖
                train_acc = correct / samples_seen  # 到目前為止的訓練準確率
                print(
                    f"      epoch {epoch}/{EPOCHS}  batch {batch_idx}/{num_batches}  "
                    f"loss={loss:.4f}  avg_loss={avg_loss:.4f}  "
                    f"train_acc={train_acc * 100:.1f}%"
                )

        avg_loss = total_loss / total_batches  # 本 epoch 全部批次的平均 loss
        train_acc = correct / train_total  # 本 epoch 在整份訓練集上的準確率
        print(
            f"      epoch {epoch}/{EPOCHS} done  "
            f"loss={avg_loss:.4f}  train_acc={train_acc * 100:.1f}%"
        )

    print("[4/5] Evaluating on test set ...")
    test_correct = 0  # 測試集猜對張數
    test_total = X_test.shape[0]  # 測試集總張數（10000）
    eval_batch_size = 256  # 評估時每批取幾張（不必與訓練 batch 相同）
    test_num_batches = (test_total + eval_batch_size - 1) // eval_batch_size  # 測試集要跑幾批

    for batch_idx in range(test_num_batches):
        start = batch_idx * eval_batch_size  # 這批在測試集的起始索引
        end = min(start + eval_batch_size, test_total)  # 這批結束索引（最後一批可能較短）
        X_batch = X_test[start:end]  # 取出這批測試圖
        y_batch = y_test[start:end]  # 取出這批 one-hot 標籤

        preds = predict(X_batch, params)  # 對測試圖做預測
        true_labels = np.argmax(y_batch, axis=1)  # argmax：從 one-hot 取正確數字 0~9
        test_correct += np.sum(preds == true_labels)  # 累加本批猜對張數
        print(
            f"      eval batch {batch_idx + 1}/{test_num_batches}  "
            f"running_acc={test_correct / test_total * 100:.1f}%"
        )

    print(f"      test accuracy: {test_correct / test_total * 100:.1f}%")

    print("[5/5] Saving weights ...")
    os.makedirs(MODELS_DIR, exist_ok=True)  # 建立 models/ 目錄（已存在不報錯）
    np.savez_compressed(  # savez_compressed：以壓縮格式將多個 ndarray 寫入單一 .npz
        WEIGHTS_PATH,
        fc1_W=params["fc1"]["weights"],  # 第一層權重矩陣
        fc1_b=params["fc1"]["bias"],  # 第一層偏置向量
        fc2_W=params["fc2"]["weights"],  # 第二層權重矩陣
        fc2_b=params["fc2"]["bias"],  # 第二層偏置向量
    )
    print(f"      Saved to {WEIGHTS_PATH}")
