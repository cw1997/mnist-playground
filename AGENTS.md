# mnist-playground — AI 代理協作規則

供 Cursor 等 AI 代理遵循。維持「逐步教學、單檔自包含、零基礎可讀」風格。以 Python 逐步探索 MNIST：下載、匯 PNG、訓練模型；每步一個可獨立執行的 `step_N_*.py`。

## 專案與步驟架構

| 步驟 | 腳本 | 依賴 | 產出 |
|------|------|------|------|
| 1 | `step_1_download_mnist.py` | 無 | `mnist/` 下的 IDX 原始檔 |
| 2 | `step_2_show_image.py` | step 1 | `images/` 下的 PNG 圖片 |
| 3a | `step_3_train_mlp.py` | step 1 | 終端機輸出 MLP 訓練 loss 與測試準確率；`models/mlp.npz` |
| 3b | `step_3_train_cnn.py` | step 1 | 終端機輸出 CNN 訓練 loss 與測試準確率；`models/cnn.npz` |
| 4a | `step_4_inference_mlp.py` | step 3a | 終端機輸出 MLP 推理進度、10 類機率與預測結果 |
| 4b | `step_4_inference_cnn.py` | step 3b | 終端機輸出 CNN 推理進度、10 類機率與預測結果 |

- 新功能優先以 **新增下一步**（`step_4_*.py`）擴充，而非改動既有步驟的核心行為。
- 每步執行前檢查前置檔案；缺檔時印出明確提示並 `sys.exit(1)`。

**資料路徑：** `mnist/`（IDX，step 1 寫入、step 2/3 讀取）；`images/`（PNG，僅 step 2 寫入；step 3 直接讀 IDX，不依賴 PNG）；`models/`（`.npz` 權重，step 3 寫入、step 4 讀取）。`mnist/`、`images/`、`models/` 已在 `.gitignore`，勿提交。

## 技術棧與單檔原則

| 步驟 | 允許的套件 | 禁止 |
|------|-----------|------|
| step 1 | Python 標準函式庫 | 第三方套件 |
| step 2 | Pillow | 深度學習框架 |
| step 3 | NumPy（純手寫 MLP／CNN） | PyTorch、TensorFlow、Keras、scikit-learn 等 |
| step 4 | Pillow、NumPy（純手寫推理） | PyTorch、TensorFlow、Keras、scikit-learn 等 |

- 每一步邏輯全在對應單一 `.py`，不得拆出 `layers.py`、`utils.py` 等額外模組。
- 可更新：`requirements.txt`、`README.md`、`AGENTS.md`、`.gitignore`。

`step_3_train_mlp.py`：全連接層與反向傳播。`step_3_train_cnn.py`：手寫卷積、池化、im2col。兩者均不得呼叫高階深度學習 API。

## 純函式（禁止 class）

所有 `step_N_*.py` 不得以 `class` 實作邏輯，一律使用純函式。

| 做法 | 說明 |
|------|------|
| 允許 | `def` 函式、模組頂層常數、`dict`／`tuple` 等資料結構傳遞狀態 |
| 禁止 | 定義 `class`（含層類別、模型類別、資料類別等） |

step 3 的權重與梯度以字典存放（例如 `params["conv1"]["W"]`），前向傳播的中間結果以 cache 字典回傳，由 `model_forward`／`model_backward`／`update_params` 等函式串接。

## 註釋、風格與 README

**註釋與風格**

- 繁體中文（程式註釋、docstring、AI 代理與使用者溝通）；**不含** `print()` 終端輸出。
- 零基礎導向：先白話解釋概念再寫程式，說明「為什麼」；陣列標註 `shape`，例如 `(batch, channel, height, width)`。
- 頂部模組 docstring 說明用途與前置條件；常數集中於檔案前段；以 `# === 區塊標題 ===` 分隔邏輯；函式附簡短 docstring；以 `if __name__ == "__main__":` 為進入點。

**終端輸出（print）**

- 所有 `step_N_*.py` 的 `print()` 使用**專業標準英文**；註釋／docstring 仍用繁中，兩者分工不混用。
- 語氣簡潔、陳述式；進度訊息用現在分詞或動名詞（例如 `Loading`、`Evaluating`）。
- 一般訊息句首大寫；錯誤／缺檔提示句首大寫且結尾加句點（`Run step_1_download_mnist.py first.`）。
- 術語一致：`Predicted digit`、`Confidence`、`test accuracy`；數值與路徑以 f-string 插值。
- 增刪改 `print()` 時，同步更新 README 對應小節的「預期輸出範例」code block。

**步驟式輸出格式（全步驟統一）**

以 [`step_4_inference_mlp.py`](step_4_inference_mlp.py) 為參考；所有 `step_N_*.py` 的終端輸出皆採相同結構。

| 元素 | 格式 | 範例 |
|------|------|------|
| 標題 | `=== {任務名} ===` 或 `=== {任務名}: {關鍵路徑} ===` | `=== MLP Training ===`、`=== Inference: test.png ===` |
| 主步驟 | `[N/Total] {動名詞} ...` | `[1/5] Loading MNIST data ...` |
| 子細節 | 前綴 **6 個空格** | `      train: 60000 samples` |
| 層級／結果 | 子細節行內用 `→` 連接 | `      Flatten     → shape (1, 784)` |
| 錯誤訊息 | 頂層、無縮排 | `Missing MNIST files: ...` |

**程式結構**

- 主流程抽成 `run_*()` 函式（例如 `run_download()`、`run_export()`、`run_training()`、`run_inference()`）。
- `if __name__ == "__main__":` 只做：參數解析（若有）、前置檔案檢查、呼叫 `run_*()`。
- 缺檔檢查的 `print()` 留在 `main` 區塊（錯誤時直接 `sys.exit(1)`，不進入 `run_*()`）。
- 訓練／推理的 epoch、batch、eval 進度與各層 shape 等細節，一律以 6 空格縮排，歸在對應主步驟之下。

**各步驟主步驟對照**

| 腳本 | 標題 | 主步驟 |
|------|------|--------|
| step 1 | `=== MNIST Download ===` | `[1/4]`～`[4/4]` 每個 .gz 檔一個步驟 |
| step 2 | `=== MNIST PNG Export ===` | `[1/3]` 檢查 → `[2/3]` train → `[3/3]` test |
| step 3a | `=== MLP Training ===` | `[1/5]` 載入 → `[2/5]` 初始化 → `[3/5]` 訓練 → `[4/5]` 評估 → `[5/5]` 存檔 |
| step 3b | `=== CNN Training ===` | 同 3a；epoch 4 起可印 `lr=` 縮排細節 |
| step 4a/4b | `=== Inference: {path} ===` | `[1/5]` 讀圖 → `[2/5]` 前處理 → `[3/5]` 載入權重 → `[4/5]` 前向 → `[5/5]` 結果 |

**範例（推理）**

```
=== Inference: test.png ===
[1/5] Loading image ...
      Original size: 280×280, mode: RGB
[4/5] Forward pass ...
      Flatten     → shape (1, 784)
      FC128+ReLU  → shape (1, 128)
[5/5] Inference result
      Predicted digit: 4
      Confidence:      97.35%
```

**範例（訓練）**

```
=== MLP Training ===
[1/5] Loading MNIST data ...
      train: 60000 samples
      test:  10000 samples
[3/5] Training ...
      100 epochs, batch_size=64, 938 batches/epoch, lr=0.01
      epoch 1/100  batch 100/938  loss=0.6200  avg_loss=0.7800  train_acc=78.5%
[4/5] Evaluating on test set ...
      test accuracy: 93.1%
[5/5] Saving weights ...
      Saved to models/mlp.npz
```

**README 同步**

`README.md` 面向人類讀者，與程式內繁中註釋互補。強制：每個 `*.py` 在「程式碼說明」章節有 `### 檔名.py` 小節（例如 `### step_3_train_cnn.py`）；章節開頭維護腳本索引表。

| 項目 | 說明 |
|------|------|
| 用途 | 這個腳本做什麼、產出什麼 |
| 前置條件 | 需先執行哪些步驟、需要哪些檔案或套件 |
| 執行流程 | 主程式依序做了哪些事（可列 numbered list） |
| 關鍵函式 | 重要函式的原理，附程式碼片段與表格 |
| 圖示 | 有資料流或網路架構時，用 mermaid 或 ASCII 示意 |
| 設計要點 | 為何這樣設計的簡短總結 |

增刪改任何 `.py` 時同步更新 README：維護對應小節；涉及資料格式或網路架構時附表格或 mermaid；新增步驟時一併更新執行指令、相依套件表、專案結構表；刪除時移除對應小節與引用。小節深度對齊 step 1／step 2，讓零基礎讀者不看原始碼也能理解設計意圖。

## 禁止事項

- 不要引入與教學目標無關的複雜抽象（過度封裝、多層繼承、插件架構等）。
- 不要為「可能以後用到」的功能預先建立檔案或框架。
- 不要在未經使用者要求時建立 git commit 或 push。
- 不要修改本規則檔所描述的核心原則，除非使用者明確要求。
