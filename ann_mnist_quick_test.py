import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# =========================
# 설정 (빠른 테스트용)
# =========================
epochs          = 1
batch_size      = 64
test_batch_size = 1000
lr              = 0.001
num_layers      = 2
num_neurons     = 64

# =========================
# 모델 정의
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
# 데이터 준비
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,      shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

# =========================
# 장치 설정
# =========================
devices = []
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
else:
    print("⚠️  GPU를 사용할 수 없습니다. CPU만 측정합니다.")
devices.append(torch.device("cpu"))

# =========================
# 실험 실행
# =========================
for device in devices:
    print(f"\n{'='*50}")
    print(f"device={device.type} | layers={num_layers} | neurons={num_neurons}")
    print(f"{'='*50}")

    print(f"  데이터 사전 로딩 중...")
    train_batches = [(d.to(device), t.to(device)) for d, t in train_loader]
    test_batches  = [(d.to(device), t.to(device)) for d, t in test_loader]

    model     = SimpleANN(num_layers, num_neurons).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    train_time = time.time() - start_train

    # 추론 시간 측정
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_infer = time.time()

    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_batches:
            preds = model(images).argmax(dim=1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.time() - start_infer

    accuracy = 100.0 * correct / total
    print(f"  Train: {train_time:.4f}s | Infer: {infer_time:.4f}s | Acc: {accuracy:.2f}%")

print("\n✅ 테스트 완료!")
