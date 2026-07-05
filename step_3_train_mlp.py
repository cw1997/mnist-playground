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
    """讀取 IDX3 圖像檔，回傳像素陣列、張數、列數、行數。"""
    with open(path, "rb") as f:
        # 大端序：magic(2051)、張數、列數、行數
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        # 每個像素是 0～255 的整數，先讀成 uint8 再轉 float
        pixels = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)
    return pixels, count, rows, cols


def read_labels(path: str) -> tuple[np.ndarray, int]:
    """讀取 IDX1 標籤檔，回傳標籤陣列與筆數。"""
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected magic number in {path}: {magic}")
        labels = np.frombuffer(f.read(count), dtype=np.uint8)
    return labels, count


def load_mnist_split(images_file: str, labels_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    載入單一分割（train 或 test），回傳：
    - X: 形狀 (N, 1, 28, 28)，像素已正規化到 0～1
    - y: 形狀 (N, 10) 的 one-hot 標籤
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
    y = np.zeros((count, NUM_CLASSES), dtype=np.float64)  # 形狀 (樣本數, 10)，全部填 0
    row_idx = np.arange(count)  # [0, 1, ..., count-1]，每張圖對應的列號
    y[row_idx, labels] = 1.0  # labels[i] 是第 i 張圖的數字；在該欄設 1

    return X, y


# === 激活函式與損失 ===
# ReLU：負值變 0，正值保留。簡單且能緩解梯度消失。
# Softmax：把 10 個分數轉成機率（加總為 1）。
# 交叉熵：衡量預測機率與正確答案的差距。


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活：max(0, x)。"""
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """ReLU 反向傳播：x<=0 的位置梯度為 0，其餘原樣傳回。"""
    return dout * (x > 0)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    對每一列（每個樣本的 10 個類別分數）做 softmax。
    先減去最大值是為了避免 exp 溢位（數值穩定技巧）。
    """
    shifted = x - np.max(x, axis=1, keepdims=True)  # axis=1：每列取最大；keepdims 保留維度方便相減
    exp_x = np.exp(shifted)  # 對每個分數取 e 的次方
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # 每列除以該列總和，得到機率（加總為 1）


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """交叉熵損失：預測越準，loss 越小。"""
    batch = y_true.shape[0]  # shape[0]：第一維大小，代表這批有幾張圖
    row_idx = np.arange(batch)  # [0, 1, ..., batch-1]，每張圖在第幾列
    true_labels = np.argmax(y_true, axis=1)  # axis=1 沿 10 類別找 1 的位置 → 正確數字 0~9
    correct_probs = probs[row_idx, true_labels]  # 用列+欄索引，取出每張圖對正確數字的預測機率
    # 交叉熵 = -平均 ln(機率)；1e-12 避免機率為 0 時 log(0) 出錯
    return float(-np.sum(np.log(correct_probs + 1e-12)) / batch)


def cross_entropy_gradient(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    softmax + 交叉熵合併後的梯度，形式非常簡潔：(預測機率 - 正確答案) / batch。
    這是反向傳播的起點。
    """
    batch = y_true.shape[0]  # shape[0]：這批樣本數
    return (probs - y_true) / batch  # 預測機率減 one-hot 標籤，再除以 batch 取平均


# === 參數初始化 ===
# 不用 class，改以字典存放每層權重 W、bias b，以及梯度 dW、db。


def init_dense_params(in_features: int, out_features: int) -> dict:
    """建立全連接層參數字典。Xavier 初始化讓前向與反向方差平衡。"""
    scale = np.sqrt(2.0 / (in_features + out_features))  # Xavier 縮放因子，控制初始權重大小
    W = np.random.randn(in_features, out_features) * scale  # randn：標準常態亂數，再乘以 scale
    b = np.zeros(out_features, dtype=np.float64)  # 偏置初始為 0，長度等於輸出維度
    return {
        "W": W,
        "b": b,
        "dW": np.zeros_like(W),  # zeros_like：建立與 W 同形狀的全 0 梯度矩陣
        "db": np.zeros_like(b),
    }


# === 全連接層（純函式）===
# 全連接層把輸入向量做線性組合，得到新的特徵或最終類別分數。


def dense_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """全連接前向傳播：y = x @ W + b"""
    return x @ params["W"] + params["b"]  # @ 矩陣乘法：(batch, in) × (in, out)；+ b 加上偏置


def dense_backward(dout: np.ndarray, params: dict, x: np.ndarray) -> np.ndarray:
    """全連接反向傳播：累積 dW、db，回傳 dx。"""
    params["dW"] += x.T @ dout  # .T 轉置 x；(in, batch) × (batch, out) → 權重梯度
    params["db"] += np.sum(dout, axis=0)  # axis=0 沿 batch 維求和 → 形狀 (out,)
    return dout @ params["W"].T  # 梯度繼續往前層傳遞


# === MLP 前向／反向／更新（純函式）===
# 把各層函式串起來，params 放權重，cache 放前向傳播的中間結果。


def model_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    整個 MLP 的前向傳播，回傳 softmax 機率與 cache。
    架構：展平 → FC(128) → ReLU → FC(10) → Softmax
    """
    cache: dict = {}

    flat = x.reshape(x.shape[0], -1)  # -1 表示「其餘維度自動算」→ (batch, 784)
    cache["flat"] = flat

    f1 = dense_forward(flat, params["fc1"])
    cache["f1"] = f1

    r1 = relu(f1)
    cache["r1"] = r1

    logits = dense_forward(r1, params["fc2"])
    cache["logits"] = logits

    probs = softmax(logits)
    return probs, cache


def model_backward(probs: np.ndarray, y_true: np.ndarray, params: dict, cache: dict) -> None:
    """從 softmax+交叉熵的梯度開始，逐層反向傳播，梯度寫入 params 的 dW、db。"""
    dout = cross_entropy_gradient(probs, y_true)

    dout = dense_backward(dout, params["fc2"], cache["r1"])
    dout = relu_backward(cache["f1"], dout)
    dense_backward(dout, params["fc1"], cache["flat"])


def zero_grads(params: dict) -> None:
    """把各層累積的梯度清零，準備下一個 batch。"""
    for layer in ("fc1", "fc2"):
        params[layer]["dW"].fill(0)
        params[layer]["db"].fill(0)


def update_params(params: dict, learning_rate: float) -> None:
    """SGD：所有層沿梯度反方向更新權重。"""
    for layer in ("fc1", "fc2"):
        p = params[layer]
        p["W"] -= learning_rate * p["dW"]
        p["b"] -= learning_rate * p["db"]


def predict(x: np.ndarray, params: dict) -> np.ndarray:
    """回傳每筆樣本預測的數字類別（0～9）。"""
    probs, _ = model_forward(x, params)  # _ 表示忽略 cache
    return np.argmax(probs, axis=1)  # 每列 10 個機率中取最大值的 index → 預測數字


def save_params(params: dict, path: str) -> None:
    """將各層 W、b 保存為 .npz，供 step 4 推理載入。"""
    np.savez_compressed(
        path,
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
    n = X.shape[0]
    indices = np.arange(n)  # [0, 1, ..., n-1]，每筆樣本的索引
    if shuffle:
        np.random.shuffle(indices)  # 原地打亂索引順序

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]  # 切片取出這批的索引
        yield X[batch_idx], y[batch_idx]  # 用索引取出對應的 X 和 y


# === 主程式 ===
if __name__ == "__main__":
    # 確認 step 1 已下載所需檔案
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f"{MNIST_DIR}/{f}")]
    if missing:
        print("Missing MNIST files:", ", ".join(missing))
        print("Run step_1_download_mnist.py first.")
        sys.exit(1)

    np.random.seed(RANDOM_SEED)
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

        for batch_idx, (X_batch, y_batch) in enumerate(
            iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True),
            start=1,
        ):
            zero_grads(params)
            probs, cache = model_forward(X_batch, params)
            loss = cross_entropy_loss(probs, y_batch)
            model_backward(probs, y_batch, params, cache)
            update_params(params, LEARNING_RATE)

            total_loss += loss
            total_batches += 1
            preds = predict(X_batch, params)  # 模型預測的 0~9
            true_labels = np.argmax(y_batch, axis=1)  # one-hot 轉回正確數字
            correct += np.sum(preds == true_labels)  # True 當 1 加總 → 本批猜對幾張

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

    for batch_idx, (X_batch, y_batch) in enumerate(
        iterate_minibatches(X_test, y_test, eval_batch_size, shuffle=False),
        start=1,
    ):
        preds = predict(X_batch, params)
        true_labels = np.argmax(y_batch, axis=1)
        test_correct += np.sum(preds == true_labels)
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
