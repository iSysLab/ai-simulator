import csv
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


# =========================
# 모델 정의 (Conv층 수, 필터 수 가변)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_filters, num_conv_layers):
        super(SimpleCNN, self).__init__()

        conv_layers = []
        in_channels = 1  # MNIST 흑백

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** i)
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            if (i + 1) % 2 == 0:
                conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)

        # flat_size 자동 계산
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            flat_size = self.features(dummy).view(1, -1).size(1)

        # FC 부분 (고정)
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =========================
# 단일 실험 실행 함수
# =========================
def run_experiment(num_filters, num_conv_layers, device, train_batches, test_batches, epochs):
    model = SimpleCNN(num_filters, num_conv_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 시간 측정
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_train = time.time()

    for _ in range(epochs):
        model.train()
        for images, labels in train_batches:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.time() - start_train

    # 추론 시간 측정
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_infer = time.time()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_batches:
            preds = model(images).argmax(dim=1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start_infer

    accuracy = 100.0 * correct / total
    return training_time, inference_time, accuracy


# =========================
# 메인
# =========================
epochs           = 3
repeat           = 3
train_batch_size = 64
test_batch_size  = 1000

# (필터 수, Conv 층수)
configs = [
    (32,  2), (32,  3), (32,  5),
    (64,  2), (64,  3), (64,  5),
    (128, 2), (128, 3), (128, 5),
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size,  shuffle=False, num_workers=0, pin_memory=True)

devices = []
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
else:
    print("⚠️  GPU를 사용할 수 없습니다. CPU만 측정합니다.")
devices.append(torch.device("cpu"))

os.makedirs("results", exist_ok=True)
csv_path = "results/cnn_mnist_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["device", "filters", "conv_layers", "avg_train_time_s", "avg_infer_time_s", "avg_accuracy_pct"])
    writer.writeheader()

    total_runs = len(devices) * len(configs)
    run_idx = 0

    for device in devices:
        print(f"\n  데이터를 {device.type}에 사전 로딩 중...")
        train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
        test_batches  = [(d.to(device), t.to(device)) for d, t in test_loader]
        print(f"  사전 로딩 완료.\n")

        for num_filters, num_conv_layers in configs:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] device={device.type}, filters={num_filters}, conv_layers={num_conv_layers}")

            train_times, infer_times, accuracies = [], [], []

            for rep in range(repeat):
                train_time, infer_time, accuracy = run_experiment(
                    num_filters, num_conv_layers, device, train_batches, test_batches, epochs
                )
                train_times.append(train_time)
                infer_times.append(infer_time)
                accuracies.append(accuracy)
                print(f"  [{rep+1}/{repeat}회차] Train: {train_time:.4f}s | Infer: {infer_time:.4f}s | Acc: {accuracy:.2f}%")

            avg_train = round(float(np.mean(train_times)), 4)
            avg_infer = round(float(np.mean(infer_times)), 4)
            avg_acc   = round(float(np.mean(accuracies)), 2)
            print(f"  >> 평균 Train: {avg_train}s | 평균 Infer: {avg_infer}s | 평균 Acc: {avg_acc}%")

            writer.writerow({
                "device":           device.type,
                "filters":          num_filters,
                "conv_layers":      num_conv_layers,
                "avg_train_time_s": avg_train,
                "avg_infer_time_s": avg_infer,
                "avg_accuracy_pct": avg_acc,
            })
            f.flush()

print(f"\n✅ 전체 실험 완료! 결과 저장: {csv_path}")
