import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
import ssl  # 이 줄 추가

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ann_models import SimpleANN, create_model_variants
from utils.timer import TimeEstimator


def prepare_mnist_data(batch_size=64):
    """
    MNIST 데이터 준비
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download=False로 변경하고, 데이터를 먼저 수동으로 다운로드
    try:
        train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, download=False, transform=transform)
    except:
        print("MNIST 데이터를 다운로드합니다...")
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, device='mps', epochs=3):
    """
    모델 학습
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if device == 'mps':
                torch.mps.synchronize()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model


def run_experiment():
    """
    전체 실험 실행
    """
    print("=" * 80)
    print("ANN MNIST Time Estimation Experiment")
    print("=" * 80)

    # 데이터 준비
    print("\n1. Preparing MNIST data...")
    train_loader, test_loader = prepare_mnist_data(batch_size=64)

    # TimeEstimator 생성 (CPU와 MPS 둘 다 측정)
    devices = ['cpu', 'mps']
    results = []

    # 모델 variants 생성
    print("\n2. Creating model variants...")
    model_variants = create_model_variants()
    print(f"Total {len(model_variants)} model variants created")

    # 각 모델에 대해 실험
    for idx, (model, model_info) in enumerate(model_variants):
        print(f"\n{'=' * 80}")
        print(f"Model {idx + 1}/{len(model_variants)}")
        print(f"Configuration: {model_info['config']}")
        print(f"Total Parameters: {model_info['total_params']:,}")
        print(f"{'=' * 80}")

        for device in devices:
            print(f"\n[Device: {device.upper()}]")

            # 새로운 모델 인스턴스 생성 (각 device마다)
            fresh_model = SimpleANN(hidden_sizes=model_info['config'])

            # 학습
            print("Training model...")
            trained_model = train_model(fresh_model, train_loader, device=device, epochs=2)

            # 시간 측정
            estimator = TimeEstimator(device=device, warmup_runs=5, measure_runs=10)

            print("Measuring inference time...")
            inference_mean, inference_std = estimator.measure_inference_time(
                trained_model,
                input_shape=(64, 1, 28, 28)
            )

            print("Measuring training time...")
            # 학습 시간은 1 epoch만 측정
            training_mean, training_std = estimator.measure_training_time(
                trained_model,
                train_loader,
                epochs=1
            )

            # 결과 저장
            result = {
                'model_idx': idx,
                'config': str(model_info['config']),
                'num_layers': model_info['num_layers'],
                'total_params': model_info['total_params'],
                'device': device,
                'inference_time_mean_ms': inference_mean,
                'inference_time_std_ms': inference_std,
                'training_time_mean_sec': training_mean,
                'training_time_std_sec': training_std,
            }
            results.append(result)

            print(f"Inference Time: {inference_mean:.2f} ± {inference_std:.2f} ms")
            print(f"Training Time: {training_mean:.2f} ± {training_std:.2f} sec")

    # 결과를 DataFrame으로 변환 및 저장
    print("\n" + "=" * 80)
    print("Saving results...")
    df = pd.DataFrame(results)

    # data 폴더가 없으면 생성
    os.makedirs('../data', exist_ok=True)

    output_file = '../data/ann_mnist_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(df.to_string())

    return df


if __name__ == "__main__":
    results_df = run_experiment()