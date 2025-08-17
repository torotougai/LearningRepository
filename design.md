# PyTorch MNISTプロジェクト 設計書

## 1. 概要

本プロジェクトは、PyTorchを使用してMNISTデータセットの分類モデルを構築するためのリポジトリです。
PyTorchの基本的な使い方、CUDAの検証、データセットのダウンロード、および簡単なニューラルネットワークモデルの定義を含みます。

## 2. プロジェクト構成

```
C:\Users\touga\Documents\PyTorch-Test/
├── .gitignore
├── cuda_test.py
├── download_dataset.py
├── design.md
├── models/
│   └── simple_model.py
└── .vscode/
    ├── launch.json
    └── settings.json
```

## 3. 各ファイルの詳細

### 3.1. `cuda_test.py`

**目的:**

PyTorchが正しくインストールされ、CUDA（GPU）が利用可能かどうかを検証します。

**コード:**

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")

# 簡単なGPUテスト
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"GPU computation successful: {z.shape}")
    print("✅ PyTorch with CUDA is working correctly!")
else:
    print("❌ CUDA not available")
```

### 3.2. `download_dataset.py`

**目的:**

`torchvision`ライブラリを使用して、機械学習のベンチマークとして有名なMNISTデータセットをダウンロードします。

**コード:**

```python
# 必要なライブラリをインポートします
import torch
from torchvision import datasets, transforms

# 画像データをPyTorchのテンソルに変換するためのルールを定義します
# これを「前処理」と呼びます
transform = transforms.Compose([
    transforms.ToTensor() # 画像をテンソルに変換する
])

# MNISTデータセットの訓練用データをダウンロード（または読み込み）します
# root='./data': './data'というフォルダにデータを保存します
# train=True: 「訓練用」のデータを指定します
# download=True: データがなければ自動でダウンロードします
# transform=transform: 上で定義した変換ルールを適用します
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# データセットがどんなものか、少しだけ中身を見てみましょう
print(f"訓練データの総数: {len(train_dataset)}")

# 最初の1枚のデータを取り出してみます
image, label = train_dataset[0]

print(f"画像のサイズ: {image.shape}")
print(f"画像のラベル: {label}")
```

### 3.3. `models/simple_model.py`

**目的:**

MNISTの画像を分類するための、基本的な全結合層からなるニューラルネットワークモデルを定義します。

**コード:**

```python
# 必要なライブラリをインポートします
import torch.nn as nn
import torch.nn.functional as F


# ニューラルネットワークの設計図（クラス）を定義します
class SimpleNet(nn.Module):
    def __init__(self):
        # 親クラスの初期化を呼び出します
        super().__init__()

        # ネットワークの層を定義します
        # 第1層: 入力784 (28*28), 出力128
        self.fc1 = nn.Linear(28 * 28, 128)
        # 第2層: 入力128, 出力10
        self.fc2 = nn.Linear(128, 10)

    # データがネットワークを流れる順序を定義します
    def forward(self, x):
        # xは画像データです (shape: [バッチサイズ, 1, 28, 28])

        # 1. 画像を1次元に平坦化します
        x = x.view(-1, 28 * 28)

        # 2. 第1層に通し、活性化関数ReLUを適用します
        x = self.fc1(x)
        x = F.relu(x)

        # 3. 第2層に通します
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # 設計図からモデルのインスタンスを作成します
    model = SimpleNet()
    print(model)
```

## 4. セットアップと実行方法

### 4.1. VS Codeでの実行

1.  VS Codeでリポジトリのルートフォルダを開きます。
2.  Microsoft製の`Python`および`Ruff`拡張機能をインストールします。
3.  実行したいPythonファイル（例: `cuda_test.py`）を開きます。
4.  `F5`キーを押すか、デバッグパネルから「Python: Current File」を実行すると、ファイルが実行されます。

## 5. 今後の展望

-   訓練ループの実装
-   評価（テスト）ループの実装
-   ハイパーパラメータの調整
-   モデルの保存と読み込み機能

```