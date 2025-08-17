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
