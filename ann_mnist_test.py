import csv
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


# =========================
# 모델 정의 (은닉층 수, 뉴런 수 가변)
# =========================
class SimpleANN(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super(SimpleANN, self).__init__()
        layers = [nn.Flatten(), nn.Linear(784, num_neurons), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
        layers.append(nn.Linear(num_neurons, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =========================
# 단일 실험 실행 함수
# =========================
def run_experiment(num_layers, num_neurons, device, train_batches, test_batches, epochs):
    model = SimpleANN(num_layers, num_neurons).to(device)
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
            total += labels.size(0)
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
repeat           = 10
train_batch_size = 64
test_batch_size  = 1000
lr               = 0.001

configs = [
    (2,  64), (2, 128), (2, 256),
    (3,  64), (3, 128), (3, 256),
    (5,  64), (5, 128), (5, 256),
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
csv_path = "results/ann_mnist_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["device", "hidden_layers", "neurons", "avg_train_time_s", "avg_infer_time_s", "avg_accuracy_pct"])
    writer.writeheader()

    total_runs = len(devices) * len(configs)
    run_idx = 0

    for device in devices:
        print(f"\n  데이터를 {device.type}에 사전 로딩 중...")
        train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
        test_batches  = [(d.to(device), t.to(device)) for d, t in test_loader]
        print(f"  사전 로딩 완료.\n")

        for num_layers, num_neurons in configs:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] device={device.type}, layers={num_layers}, neurons={num_neurons}")

            train_times, infer_times, accuracies = [], [], []

            for rep in range(repeat):
                train_time, infer_time, accuracy = run_experiment(
                    num_layers, num_neurons, device, train_batches, test_batches, epochs
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
                "hidden_layers":    num_layers,
                "neurons":          num_neurons,
                "avg_train_time_s": avg_train,
                "avg_infer_time_s": avg_infer,
                "avg_accuracy_pct": avg_acc,
            })
            f.flush()  # config마다 즉시 저장

print(f"\n✅ 전체 실험 완료! 결과 저장: {csv_path}")
