"""
experiments/cnn_cifar10_experiment.py - CNN 모델 × CIFAR-10 실행 시간 측정 실험

[이 파일의 역할]
Stage 2: 다양한 CNN 모델 구조로 CIFAR-10 데이터셋을 학습한 뒤,
각 모델의 추론(Inference) 시간과 학습(Training) 시간을 CPU/MPS 양쪽에서 측정합니다.
측정 결과는 CSV 파일로 저장되어 Stage 3 예측 모델의 학습 데이터로 사용됩니다.

[CIFAR-10이란?]
- 10가지 클래스(비행기, 자동차, 새, 고양이 등)의 32×32 컬러 이미지 데이터셋
- 학습용 50,000장 + 테스트용 10,000장
- CNN이 처음 등장했을 때부터 표준 벤치마크로 사용됨

[실험 흐름]
1. CIFAR-10 데이터 로드 (없으면 자동 다운로드)
2. 11가지 CNN 모델 변형 생성 (SimpleCNN×9 + ResNet18 + MobileNetV2)
3. 각 모델을 CPU, MPS에서 순서대로:
   a. 2 에포크 학습 (가중치 초기화 후 새로운 인스턴스로 시작)
   b. 추론 시간 측정 (warmup 5회 + 측정 10회 평균)
   c. 학습 시간 측정 (1 에포크)
4. 결과를 DataFrame으로 정리 후 CSV 저장

[생성되는 데이터 파일]
- ../data/cnn_cifar10_results.csv: 22행 × 11컬럼 (11모델 × 2디바이스)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

# 프로젝트 루트 디렉터리를 Python 경로에 추가
# 이렇게 해야 models/, utils/ 등 상위 패키지를 import할 수 있음
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_models import create_cnn_variants
from utils.timer import TimeEstimator


def prepare_cifar10_data(batch_size=64):
    """
    CIFAR-10 데이터셋을 로드하고 DataLoader를 생성합니다.

    [데이터 증강 (Data Augmentation) - 학습용만 적용]
    - RandomCrop: 32×32 이미지를 padding=4로 늘린 뒤 랜덤하게 32×32 크롭
      → 이미지 위치 변화에 강건한 모델 학습
    - RandomHorizontalFlip: 50% 확률로 이미지를 좌우 반전
      → 좌우 대칭적 특징 학습 (비행기, 자동차 등은 방향에 무관)
    (테스트용에는 데이터 증강을 적용하지 않아 일관된 평가 기준 유지)

    [정규화 (Normalization)]
    - CIFAR-10 데이터셋의 채널별 평균과 표준편차로 픽셀값 정규화
    - 평균: (0.4914, 0.4822, 0.4465) ← R, G, B 채널 각각
    - 표준편차: (0.2023, 0.1994, 0.2010)
    - 정규화 이유: 학습 안정성 향상, gradient 흐름 개선

    Args:
        batch_size (int): 한 번에 처리할 이미지 수. 기본값 64.
            - 클수록 GPU 효율 향상 (병렬 처리)
            - 작을수록 메모리 사용량 감소

    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: 학습 데이터를 배치 단위로 반환하는 DataLoader
            - test_loader: 테스트 데이터를 배치 단위로 반환하는 DataLoader
    """
    # 학습용 전처리: 데이터 증강 + 정규화
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 랜덤 크롭으로 위치 불변성 학습
        transforms.RandomHorizontalFlip(),           # 랜덤 좌우 반전으로 대칭성 학습
        transforms.ToTensor(),                       # PIL Image → [0,1] 범위의 Tensor로 변환
        transforms.Normalize(                        # 정규화: (pixel - mean) / std
            (0.4914, 0.4822, 0.4465),               # CIFAR-10 R,G,B 채널 평균
            (0.2023, 0.1994, 0.2010)                # CIFAR-10 R,G,B 채널 표준편차
        ),
    ])

    # 테스트용 전처리: 정규화만 (데이터 증강 없음)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    # CIFAR-10 데이터셋 로드 (이미 다운로드된 경우 기존 파일 사용)
    try:
        # '../data' 폴더에 이미 다운로드된 데이터가 있으면 바로 로드
        train_dataset = datasets.CIFAR10('../data', train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False, download=False, transform=transform_test)
    except:
        # 데이터가 없으면 자동 다운로드 (약 170MB)
        print("CIFAR-10 데이터를 다운로드합니다...")
        # Mac에서 SSL 인증서 오류를 우회하기 위한 임시 설정
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)

    # DataLoader 생성
    # num_workers=0: Mac에서 멀티프로세싱 권한 문제를 방지하기 위해 단일 프로세스 사용
    # (num_workers > 0이면 자식 프로세스를 생성하는데, macOS에서 권한 오류 발생 가능)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_model(model, train_loader, device='mps', epochs=3):
    """
    CNN 모델을 CIFAR-10 데이터로 학습합니다.

    [학습 과정 (한 배치 기준)]
    1. optimizer.zero_grad(): 이전 배치의 gradient 초기화
    2. output = model(data): 순전파 → 예측값 계산
    3. loss = criterion(output, target): 예측값과 정답의 차이(손실) 계산
    4. loss.backward(): 역전파 → 각 파라미터의 gradient 계산
    5. optimizer.step(): gradient를 이용해 파라미터 업데이트

    [손실 함수: CrossEntropyLoss]
    - 분류 문제의 표준 손실 함수
    - 모델이 정답 클래스에 높은 확률을 할당할수록 손실이 낮아짐
    - 내부적으로 Softmax → Log → NLLLoss 순서로 계산

    [옵티마이저: Adam (lr=0.001)]
    - 각 파라미터마다 개별 학습률을 적응적으로 조정
    - SGD보다 빠른 수렴, 하이퍼파라미터 조정이 덜 필요함

    Args:
        model: 학습할 PyTorch CNN 모델
        train_loader: CIFAR-10 학습 데이터 DataLoader
        device (str): 학습 디바이스 ('cpu' 또는 'mps')
        epochs (int): 학습 에포크 수

    Returns:
        nn.Module: 학습이 완료된 모델
    """
    # 모델을 지정 디바이스로 이동
    model = model.to(device)

    # 분류 문제의 표준 손실 함수
    criterion = nn.CrossEntropyLoss()

    # Adam 옵티마이저: lr=0.001은 일반적인 기본값
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습 모드 전환 (Dropout 활성화, BatchNorm이 배치 통계 사용)
    model.train()

    for epoch in range(epochs):
        total_loss = 0    # 에포크 누적 손실
        correct = 0       # 정확히 맞춘 샘플 수
        total = 0         # 전체 샘플 수

        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터를 지정 디바이스로 이동
            data, target = data.to(device), target.to(device)

            # gradient 초기화 (누적을 방지)
            optimizer.zero_grad()

            # 순전파: (batch, 3, 32, 32) → (batch, num_classes)
            output = model(data)

            # 손실 계산: 예측값과 실제 레이블 비교
            loss = criterion(output, target)

            # 역전파: 손실에 대한 gradient 계산
            loss.backward()

            # 파라미터 업데이트
            optimizer.step()

            # 학습 통계 누적
            total_loss += loss.item()
            _, predicted = output.max(1)   # 가장 높은 점수의 클래스를 예측값으로
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # MPS 동기화: GPU 연산 완료 대기
            if device == 'mps':
                torch.mps.synchronize()

            # 100 배치마다 중간 진행상황 출력
            if batch_idx % 100 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 에포크 완료 후 요약 출력
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model


def run_experiment():
    """
    전체 CNN 실행 시간 측정 실험을 수행합니다.

    [실험 설계]
    - 모델: 11가지 (SimpleCNN×9 + ResNet18 + MobileNetV2)
    - 디바이스: CPU, MPS 각각
    - 총 측정 횟수: 11 × 2 = 22개 데이터 포인트
    - 각 측정: warmup 5회 + 실제 측정 10회 평균

    [중요: 왜 매번 새 모델 인스턴스를 만드는가?]
    CPU에서 학습한 모델 가중치와 MPS에서 학습한 가중치가 서로 영향을 주지 않도록,
    디바이스를 바꿀 때마다 새로운 모델 인스턴스(초기화된 파라미터)로 학습합니다.
    이렇게 해야 "CPU에서의 실행 시간"과 "MPS에서의 실행 시간"을 독립적으로 비교 가능합니다.

    Returns:
        pd.DataFrame: 모든 실험 결과 (22행 × 여러 컬럼)
    """
    print("=" * 80)
    print("CNN CIFAR-10 Time Estimation Experiment")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────
    # Step 1: CIFAR-10 데이터 준비
    # ─────────────────────────────────────────────────────────────
    print("\n1. Preparing CIFAR-10 data...")
    train_loader, test_loader = prepare_cifar10_data(batch_size=64)

    # 측정할 디바이스 목록 (CPU와 M1 MPS GPU)
    devices = ['cpu', 'mps']
    results = []   # 모든 실험 결과를 담을 리스트

    # ─────────────────────────────────────────────────────────────
    # Step 2: CNN 모델 변형 생성
    # ─────────────────────────────────────────────────────────────
    print("\n2. Creating CNN model variants...")
    model_variants = create_cnn_variants()
    print(f"Total {len(model_variants)} model variants created")

    # ─────────────────────────────────────────────────────────────
    # Step 3: 각 모델 × 각 디바이스 조합으로 실험 수행
    # ─────────────────────────────────────────────────────────────
    for idx, (model, model_info) in enumerate(model_variants):
        print(f"\n{'=' * 80}")
        print(f"Model {idx + 1}/{len(model_variants)}: {model_info['name']}")
        print(f"Total Parameters: {model_info['total_params']:,}")
        print(f"{'=' * 80}")

        for device in devices:
            print(f"\n[Device: {device.upper()}]")

            # [중요] 매번 새로운 모델 인스턴스를 생성하여 독립적인 실험 보장
            # 모델 타입에 따라 적절한 생성 함수 호출
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

            # 모델 학습 (에포크 수=2: 학습 데이터 수집 시간 단축을 위해 축소)
            # 완전한 학습이 아닌 "적당히 수렴한" 상태에서 시간을 측정하는 것이 목적
            print("Training model...")
            trained_model = train_model(fresh_model, train_loader, device=device, epochs=2)

            # TimeEstimator 생성: warmup 5회, 실제 측정 10회
            estimator = TimeEstimator(device=device, warmup_runs=5, measure_runs=10)

            # 추론 시간 측정
            # input_shape=(64, 3, 32, 32): batch_size=64, RGB 3채널, 32×32 이미지
            print("Measuring inference time...")
            inference_mean, inference_std = estimator.measure_inference_time(
                trained_model,
                input_shape=(64, 3, 32, 32)
            )

            # 학습 시간 측정 (1 에포크 기준)
            print("Measuring training time...")
            training_mean, training_std = estimator.measure_training_time(
                trained_model,
                train_loader,
                epochs=1
            )

            # 결과 딕셔너리 구성
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

            # SimpleCNN의 경우 구조 정보를 추가로 기록
            # (ResNet18, MobileNetV2는 고정 구조이므로 불필요)
            if model_info['model_type'] == 'SimpleCNN':
                result['num_conv_layers'] = model_info['num_conv_layers']
                result['base_channels'] = model_info['base_channels']

            results.append(result)

            # 결과 출력
            print(f"Inference Time: {inference_mean:.2f} ± {inference_std:.2f} ms")
            print(f"Training Time: {training_mean:.2f} ± {training_std:.2f} sec")

    # ─────────────────────────────────────────────────────────────
    # Step 4: 결과 저장
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Saving results...")
    df = pd.DataFrame(results)

    # 결과 저장 디렉터리 생성 (없으면 자동 생성)
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
    # 이 스크립트를 직접 실행하면 전체 실험이 시작됩니다.
    # 예상 소요 시간: CPU + MPS 합산 약 4~6시간 (M1 MacBook Air 기준)
    results_df = run_experiment()
