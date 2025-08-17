
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from models.simple_model import SimpleNet

def predict():
    # 1. 定数とモデルの準備
    MODEL_PATH = "mnist_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルのインスタンスを作成し、学習済みの重みを読み込む
    model = SimpleNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # 推論モードに設定

    # 2. テストデータの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    print(f"Using device: {device}")
    print("Inference started...")

    # 3. 推論の実行と結果の表示
    with torch.no_grad(): # 勾配計算を無効化
        # テストデータから最初のバッチを取得
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)

        # モデルで予測
        outputs = model(images)
        # 予測結果から最も確率の高いクラスを取得
        _, predicted = torch.max(outputs, 1)

        # 結果の表示
        fig = plt.figure(figsize=(10, 4))
        for i in range(10):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            # 画像の次元をmatplotlibで表示できるように調整
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.squeeze(img)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}",
                         color=("green" if predicted[i] == labels[i] else "red"))
        plt.show()

    print("Inference finished.")

if __name__ == '__main__':
    predict()
