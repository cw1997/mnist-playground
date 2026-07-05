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
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
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
        magic, count = struct.unpack(">II", f.read(8))
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
    pixels, count, rows, cols = read_images(f"{MNIST_DIR}/{images_file}")
    labels, label_count = read_labels(f"{MNIST_DIR}/{labels_file}")
    if count != label_count:
        raise ValueError(
            f"mismatch: {count} images vs {label_count} labels in {images_file}"
        )

    # 正規化：把 0～255 縮放到 0～1，讓梯度更新更穩定
    X = pixels.reshape(count, 1, rows, cols)  # reshape：一維像素重排成 (張數, 1, 高, 寬)
    X = X.astype(np.float64)  # astype：整數轉浮點數，才能做除法
    X = X / 255.0  # 除以 255，把 0~255 縮放到 0~1，讓梯度更新更穩定

    # one-hot：把類別數字（例如 3）變成 [0,0,0,1,0,0,0,0,0,0]，請注意這裡的1在從0開始計算的第3個位置上
    # 交叉熵損失需要one-hot這種格式來比較「預測機率」與「正確答案」
    y = np.zeros((count, NUM_CLASSES), dtype=np.float64)  # zeros：建立 (樣本數, 10) 的全 0 矩陣
    row_idx = np.arange(count)  # arange：產生 [0, 1, ..., count-1]，作為每張圖的列索引
    y[row_idx, labels] = 1.0  # labels[i] 是第 i 張圖的數字；在該欄設 1

    return X, y


# === 激活函式與損失 ===
# ReLU：負值變 0，正值保留。簡單且能緩解梯度消失。
# Softmax：把 10 個分數轉成機率（加總為 1）。
# 交叉熵：衡量預測機率與正確答案的差距。


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
    exp_x = np.exp(shifted)  # exp：對每個分數取 e 的次方
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)  # sum：axis=1 每列加總，keepdims 保留維度方便相除
    return exp_x / sum_exp  # 逐元素相除，每列機率加總為 1


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


# === 參數初始化 ===
# 不用 class，改以字典存放每層權重 W、bias b，以及梯度 dW、db。


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
        - ``"W"``：np.ndarray，形狀 ``(in_features, out_features)``
        - ``"b"``：np.ndarray，形狀 ``(out_features,)``
        - ``"dW"``：np.ndarray，與 ``"W"`` 同形，梯度累積用，初始全 0
        - ``"db"``：np.ndarray，與 ``"b"`` 同形，梯度累積用，初始全 0
    """
    scale = np.sqrt(2.0 / (in_features + out_features))  # sqrt：開平方根，得 Xavier 縮放因子
    W = np.random.randn(in_features, out_features) * scale  # randn：產生標準常態亂數，再乘以 scale
    b = np.zeros(out_features, dtype=np.float64)  # zeros：建立長度 out_features 的全 0 偏置向量
    return {
        "W": W,
        "b": b,
        "dW": np.zeros_like(W),  # zeros_like：建立與 W 同形狀的全 0 梯度矩陣
        "db": np.zeros_like(b),  # zeros_like：建立與 b 同形狀的全 0 梯度向量
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
        - cache：dict，前向中間結果，含 ``"flat"``、``"f1"``、``"r1"``、``"logits"``
    """
    cache: dict = {}

    flat = x.reshape(x.shape[0], -1)  # reshape：-1 表示其餘維度自動計算 → (batch, 784)
    cache["flat"] = flat

    f1 = flat @ params["fc1"]["W"] + params["fc1"]["b"]  # @：矩陣乘法 (batch, 784) × (784, 128)
    cache["f1"] = f1

    r1 = relu(f1)
    cache["r1"] = r1

    logits = r1 @ params["fc2"]["W"] + params["fc2"]["b"]  # @：矩陣乘法 (batch, 128) × (128, 10)
    cache["logits"] = logits

    probs = softmax(logits)
    return probs, cache


def backward(probs: np.ndarray, y_true: np.ndarray, params: dict, cache: dict) -> None:
    """從 softmax+交叉熵梯度開始，逐層反向傳播，梯度寫入 params 的 dW、db。

    參數
    ----
    probs : np.ndarray
        前向傳播的 softmax 機率，形狀 ``(batch, 10)``。
    y_true : np.ndarray
        one-hot 標籤，形狀 ``(batch, 10)``。
    params : dict
        模型參數字典（``"fc1"``、``"fc2"``）；各層 ``"dW"``、``"db"`` 原地累加。
    cache : dict
        ``forward`` 回傳的中間結果，含 ``"flat"``、``"f1"``、``"r1"``。

    回傳
    ----
    None
        無回傳值；梯度寫入 ``params`` 各層的 ``"dW"`` 與 ``"db"``。
    """
    batch = y_true.shape[0]  # shape[0]：這批樣本數
    dout = (probs - y_true) / batch  # softmax+交叉熵合併梯度，反向傳播起點

    # fc2 反向：y = x @ W + b
    params["fc2"]["dW"] += cache["r1"].T @ dout  # T：轉置；@：(128, batch) × (batch, 10)
    params["fc2"]["db"] += np.sum(dout, axis=0)  # sum：沿 batch 維加總 → (10,)
    dout = dout @ params["fc2"]["W"].T  # @：梯度傳回 fc2 輸入 → (batch, 128)

    # ReLU 反向：前向輸入 <=0 的位置梯度歸零
    dout = dout * (cache["f1"] > 0)  # 布林遮罩逐元素相乘，向量化實作

    # fc1 反向
    params["fc1"]["dW"] += cache["flat"].T @ dout  # T：轉置；@：(784, batch) × (batch, 128)
    params["fc1"]["db"] += np.sum(dout, axis=0)  # sum：沿 batch 維加總 → (128,)


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
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]
    if missing:
        print("Missing MNIST files:", ", ".join(missing))
        print("Run step_1_download_mnist.py first.")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)  # seed：固定亂數種子，使初始化與 shuffle 可重現
    print("=== MLP Training ===")

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
        f"      shape: {X_train.shape}, normalized [0, 1]"
    )

    # 建立 MLP 的參數字典
    # 架構：FC(784→128) → ReLU → FC(128→10) → Softmax
    print("[2/5] Initializing model ...")
    params = {
        "fc1": init_dense_params(INPUT_SIZE, HIDDEN_SIZE),
        "fc2": init_dense_params(HIDDEN_SIZE, NUM_CLASSES),
    }
    print(
        f"      fc1 W: {params['fc1']['W'].shape}  "
        f"fc2 W: {params['fc2']['W'].shape}"
    )

    # +BATCH_SIZE-1 再 // 是向上取整除法，最後不足一批也算一批
    num_batches = (X_train.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    print("[3/5] Training ...")
    print(
        f"      {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
        f"{num_batches} batches/epoch, lr={LEARNING_RATE}"
    )

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_batches = 0
        correct = 0
        train_total = X_train.shape[0]

        batch_idx = 0
        indices = np.arange(train_total)  # arange：產生 [0, 1, ..., train_total-1]
        np.random.shuffle(indices)  # shuffle：每 epoch 打亂樣本順序

        for start in range(0, train_total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, train_total)
            batch_indices = indices[start:end]  # 切片取出這批的索引
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            batch_idx += 1

            for layer in ("fc1", "fc2"):
                params[layer]["dW"].fill(0)  # fill：梯度清零，準備本批累加
                params[layer]["db"].fill(0)

            probs, cache = forward(X_batch, params)
            loss = cross_entropy_loss(probs, y_batch)
            backward(probs, y_batch, params, cache)

            for layer in ("fc1", "fc2"):
                p = params[layer]
                p["W"] -= LEARNING_RATE * p["dW"]  # SGD：沿梯度反方向更新權重
                p["b"] -= LEARNING_RATE * p["db"]

            total_loss += loss
            total_batches += 1
            preds = predict(X_batch, params)  # 模型預測的 0~9
            true_labels = np.argmax(y_batch, axis=1)  # argmax：axis=1 從 one-hot 取正確數字 0~9
            correct += np.sum(preds == true_labels)  # sum：布林 True 當 1 加總 → 本批猜對幾張

            # 每 100 批或最後一批印出進度
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
    test_num_batches = (test_total + eval_batch_size - 1) // eval_batch_size

    for batch_idx in range(test_num_batches):
        start = batch_idx * eval_batch_size
        end = min(start + eval_batch_size, test_total)
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        preds = predict(X_batch, params)
        true_labels = np.argmax(y_batch, axis=1)  # argmax：axis=1 從 one-hot 取正確數字 0~9
        test_correct += np.sum(preds == true_labels)  # sum：布林 True 當 1 加總 → 本批猜對幾張
        print(
            f"      eval batch {batch_idx + 1}/{test_num_batches}  "
            f"running_acc={test_correct / test_total * 100:.1f}%"
        )

    print(f"      test accuracy: {test_correct / test_total * 100:.1f}%")

    print("[5/5] Saving weights ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.savez_compressed(  # savez_compressed：以壓縮格式將多個 ndarray 寫入單一 .npz
        WEIGHTS_PATH,
        fc1_W=params["fc1"]["W"],
        fc1_b=params["fc1"]["b"],
        fc2_W=params["fc2"]["W"],
        fc2_b=params["fc2"]["b"],
    )
    print(f"      Saved to {WEIGHTS_PATH}")
