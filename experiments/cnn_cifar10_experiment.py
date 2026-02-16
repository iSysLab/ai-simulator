import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_models import create_cnn_variants
from utils.timer import TimeEstimator


def prepare_cifar10_data(batch_size=64):
    """
    CIFAR-10 데이터 준비
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        train_dataset = datasets.CIFAR10('../data', train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False, download=False, transform=transform_test)
    except:
        print("CIFAR-10 데이터를 다운로드합니다...")
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)

    # num_workers=0으로 변경 (Mac 권한 문제 해결)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

            if batch_idx % 100 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model


def run_experiment():
    """
    전체 실험 실행
    """
    print("=" * 80)
    print("CNN CIFAR-10 Time Estimation Experiment")
    print("=" * 80)

    # 데이터 준비
    print("\n1. Preparing CIFAR-10 data...")
    train_loader, test_loader = prepare_cifar10_data(batch_size=64)

    # TimeEstimator 생성
    devices = ['cpu', 'mps']
    results = []

    # 모델 variants 생성
    print("\n2. Creating CNN model variants...")
    model_variants = create_cnn_variants()
    print(f"Total {len(model_variants)} model variants created")

    # 각 모델에 대해 실험
    for idx, (model, model_info) in enumerate(model_variants):
        print(f"\n{'=' * 80}")
        print(f"Model {idx + 1}/{len(model_variants)}: {model_info['name']}")
        print(f"Total Parameters: {model_info['total_params']:,}")
        print(f"{'=' * 80}")

        for device in devices:
            print(f"\n[Device: {device.upper()}]")

            # 새로운 모델 인스턴스 생성
            if model_info['model_type'] == 'SimpleCNN':
                from models.cnn_models import SimpleCNN
                fresh_model = SimpleCNN(
                    num_classes=10,
                    num_conv_layers=model_info['num_conv_layers'],
                    base_channels=model_info['base_channels']
                )
            elif model_info['model_type'] == 'ResNet18':
                from models.cnn_models import get_resnet18
                fresh_model = get_resnet18(num_classes=10, pretrained=False)
            elif model_info['model_type'] == 'MobileNetV2':
                from models.cnn_models import get_mobilenetv2
                fresh_model = get_mobilenetv2(num_classes=10, pretrained=False)

            # 학습 (에포크 수를 2로 줄여서 시간 단축)
            print("Training model...")
            trained_model = train_model(fresh_model, train_loader, device=device, epochs=2)

            # 시간 측정
            estimator = TimeEstimator(device=device, warmup_runs=5, measure_runs=10)

            print("Measuring inference time...")
            inference_mean, inference_std = estimator.measure_inference_time(
                trained_model,
                input_shape=(64, 3, 32, 32)
            )

            print("Measuring training time...")
            training_mean, training_std = estimator.measure_training_time(
                trained_model,
                train_loader,
                epochs=1
            )

            # 결과 저장
            result = {
                'model_idx': idx,
                'model_name': model_info['name'],
                'model_type': model_info['model_type'],
                'total_params': model_info['total_params'],
                'device': device,
                'inference_time_mean_ms': inference_mean,
                'inference_time_std_ms': inference_std,
                'training_time_mean_sec': training_mean,
                'training_time_std_sec': training_std,
            }

            # SimpleCNN의 경우 추가 정보
            if model_info['model_type'] == 'SimpleCNN':
                result['num_conv_layers'] = model_info['num_conv_layers']
                result['base_channels'] = model_info['base_channels']

            results.append(result)

            print(f"Inference Time: {inference_mean:.2f} ± {inference_std:.2f} ms")
            print(f"Training Time: {training_mean:.2f} ± {training_std:.2f} sec")

    # 결과를 DataFrame으로 변환 및 저장
    print("\n" + "=" * 80)
    print("Saving results...")
    df = pd.DataFrame(results)

    os.makedirs('../data', exist_ok=True)
    output_file = '../data/cnn_cifar10_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    print(df[['model_name', 'total_params', 'device', 'inference_time_mean_ms', 'training_time_mean_sec']].to_string())

    return df


if __name__ == "__main__":
    results_df = run_experiment()