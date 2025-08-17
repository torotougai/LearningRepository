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
