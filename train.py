# (これまでのデータセット準備とモデル定義のコードが実行済みであるとします)
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simple_model import SimpleNet
import torch.nn as nn
from torchvision import datasets, transforms

# --- 1. 学習の準備 ---
model = SimpleNet()
criterion = nn.CrossEntropyLoss()  # 採点官：クロスエントロピー損失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # コーチ：Adam

# 画像データをPyTorchのテンソルに変換するためのルールを定義します
# これを「前処理」と呼びます
transform = transforms.Compose(
    [
        transforms.ToTensor()  # 画像をテンソルに変換する
    ]
)

# MNISTデータセットの訓練用データをダウンロード（または読み込み）します
# root='./data': './data'というフォルダにデータを保存します
# train=True: 「訓練用」のデータを指定します
# download=True: データがなければ自動でダウンロードします
# transform=transform: 上で定義した変換ルールを適用します
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)


# データを効率的に使うためのDataLoaderを作成
# batch_size=64: 64枚ずつの束にまとめる
# shuffle=True: データをランダムにかき混ぜる（学習効果を高めるため）
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# --- 2. 学習ループ（反復練習） ---
epochs = 3  # データセット全体を3周学習させます

for epoch in range(epochs):  # エポックのループ
    running_loss = 0.0

    # train_loaderからバッチ（画像の小さな束）を一つずつ取り出す
    for i, data in enumerate(train_loader, 0):
        # dataには [入力画像, 正解ラベル] が入っています
        inputs, labels = data

        # --- ここが学習サイクルの心臓部 ---
        # a. 前回の計算結果をリセット
        optimizer.zero_grad()

        # b. 予測
        outputs = model(inputs)

        # c. 採点（損失の計算）
        loss = criterion(outputs, labels)

        # d. 改善点の分析（誤差逆伝播）
        loss.backward()

        # e. モデルの重みを更新
        optimizer.step()
        # ------------------------------------

        # 学習の進捗を表示します
        running_loss += loss.item()
        if i % 200 == 199:  # 200バッチごとに損失の平均を表示
            print(
                f"[エポック {epoch + 1}, バッチ {i + 1}] 損失: {running_loss / 200:.3f}"
            )
            running_loss = 0.0

print("学習が完了しました")

# 学習済みモデルの重みを保存
MODEL_PATH = "mnist_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"モデルを {MODEL_PATH} に保存しました")
