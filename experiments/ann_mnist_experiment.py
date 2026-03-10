# ============================================================
# ANN + MNIST 실행 시간 측정 실험 (확장 버전)
# ============================================================
#
# 목적:
#   다양한 구조의 ANN 모델을 MNIST 데이터셋으로 학습하고,
#   CPU와 MPS(M1 GPU)에서의 추론 시간 및 학습 시간을 측정
#
# 기존 데이터와 비교:
#   - 기존(1단계): 14가지 조합 × 2 디바이스 = 28개 데이터 포인트
#   - 추가(3단계 개선): 12가지 신규 조합 × 2 디바이스 = 24개 데이터 포인트 추가
#
# 교수님 유의사항:
#   - 레이어 수와 뉴런 수를 다양하게 구성할 것
#   - 10회 이상 반복 측정 후 평균 사용 (Warmup 포함)
#   - 하드웨어 사양 및 실행 결과 기록
#
# 작성자: 김홍근 / 하드웨어: MacBook Air M1
# ============================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
import ssl

# 상위 디렉토리(dnn/)를 Python 경로에 추가
# → models/, utils/ 폴더의 모듈을 import할 수 있게 됨
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ann_models import SimpleANN, create_extended_variants
from utils.timer import TimeEstimator

# ──────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
STAGE1_DIR  = os.path.join(DATA_DIR, 'stage1')   # 1단계 ANN 실험 결과 저장 위치
OUTPUT_FILE = os.path.join(STAGE1_DIR, 'ann_mnist_results.csv')


# ──────────────────────────────────────────────────────────
# 기존 CSV 확인 함수
# ──────────────────────────────────────────────────────────

def load_existing_configs(output_file):
    """
    기존 CSV 파일에서 이미 측정된 config 목록을 불러오는 함수

    이 함수를 통해 이미 측정한 조합은 건너뛰고,
    새로운 조합만 추가 측정하여 시간을 절약할 수 있음

    Args:
        output_file (str): 기존 CSV 파일 경로

    Returns:
        tuple: (이미 측정된 config 문자열 집합, 다음 model_idx 시작값)
    """
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        # 이미 측정된 config 문자열 목록 (예: "[64]", "[128, 256]" 등)
        existing_configs = set(df['config'].unique())
        # 다음 model_idx는 기존 최대값 + 1부터 시작
        next_idx = int(df['model_idx'].max()) + 1
        print(f"  기존 CSV 발견: {len(df)}행, {len(existing_configs)}가지 config 측정 완료")
        print(f"  다음 model_idx 시작: {next_idx}")
        return existing_configs, next_idx
    else:
        print("  기존 CSV 없음 → 새로 시작")
        return set(), 0


# ──────────────────────────────────────────────────────────
# MNIST 데이터 준비 함수
# ──────────────────────────────────────────────────────────

def prepare_mnist_data(batch_size=64):
    """
    MNIST 데이터셋을 불러오는 함수

    MNIST: 손글씨 숫자(0~9) 인식 데이터셋
    - 훈련 데이터: 60,000장 (28×28 흑백 이미지)
    - 테스트 데이터: 10,000장

    Args:
        batch_size (int): 한 번에 처리할 이미지 수 (기본값 64)

    Returns:
        tuple: (train_loader, test_loader)
    """
    # 이미지 전처리: 텐서 변환 + 정규화
    # Normalize((평균,), (표준편차,)): MNIST 데이터셋의 통계값 사용
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        # 이미 다운로드된 데이터 사용 (download=False)
        train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=False, transform=transform)
        test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    except Exception:
        # 데이터가 없으면 다운로드 (SSL 인증서 오류 우회)
        print("  MNIST 데이터 다운로드 중...")
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
        test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # DataLoader: 데이터를 배치 단위로 제공
    # shuffle=True: 매 epoch마다 데이터 순서를 섞어 과적합 방지
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────
# 모델 학습 함수
# ──────────────────────────────────────────────────────────

def train_model(model, train_loader, device='cpu', epochs=2):
    """
    ANN 모델을 MNIST 데이터로 학습하는 함수

    학습 목적:
    - MNIST를 식별할 수 있는 정도로만 학습 (완벽한 학습이 목적이 아님)
    - 학습된 상태에서의 추론 시간을 측정하기 위한 준비

    Args:
        model: 학습할 ANN 모델 인스턴스
        train_loader: MNIST 훈련 데이터 로더
        device (str): 학습에 사용할 디바이스 ('cpu' 또는 'mps')
        epochs (int): 학습 epoch 수 (기본값 2 - 빠른 실험을 위해 짧게 설정)

    Returns:
        model: 학습 완료된 모델
    """
    # 모델을 지정된 디바이스(CPU 또는 MPS)로 이동
    model = model.to(device)

    # 손실 함수: CrossEntropyLoss (분류 문제에 적합)
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저: Adam (학습률 0.001)
    # Adam: 적응적 학습률을 사용하는 고성능 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()  # 학습 모드 설정 (Dropout 등이 활성화됨)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터를 해당 디바이스로 이동
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()        # 이전 gradient 초기화
            output = model(data)          # 순전파(Forward pass)
            loss = criterion(output, target)  # 손실 계산
            loss.backward()              # 역전파(Backward pass) - gradient 계산
            optimizer.step()             # 파라미터 업데이트

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # MPS(M1 GPU)는 비동기로 작동하므로 정확한 시간 측정을 위해 동기화
            if device == 'mps':
                torch.mps.synchronize()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'    Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%')

    return model


# ──────────────────────────────────────────────────────────
# 메인 실험 함수
# ──────────────────────────────────────────────────────────

def run_experiment():
    """
    ANN 모델들의 실행 시간을 측정하는 메인 실험 함수

    동작 순서:
    1. 기존 CSV에서 이미 측정된 config 확인
    2. 새로운 config만 선별
    3. 각 config × 각 디바이스(CPU/MPS) 조합으로 측정
    4. 새 결과를 기존 CSV에 추가(append)하여 저장

    측정 항목:
    - 추론 시간 (ms): warmup 5회 후, 10회 반복 측정 평균
    - 학습 시간 (초/epoch): 1 epoch 기준

    교수님 유의사항: 10회 이상 반복 측정 후 평균 사용
    """
    print("=" * 70)
    print("  ANN + MNIST 실행 시간 측정 실험 (추가 데이터 수집)")
    print("  하드웨어: MacBook Air M1 (CPU + MPS)")
    print("=" * 70)

    # ── Step 1. 기존 측정 데이터 확인 ────────────────────────
    print("\n[Step 1] 기존 측정 데이터 확인")
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_configs, next_model_idx = load_existing_configs(OUTPUT_FILE)

    # ── Step 2. 새 모델 조합 준비 ────────────────────────────
    print("\n[Step 2] 추가 ANN 모델 조합 준비")
    all_variants = create_extended_variants()

    # 이미 측정된 config는 건너뜀 (중복 측정 방지)
    new_variants = [
        (model, info) for model, info in all_variants
        if str(info['config']) not in existing_configs
    ]

    if not new_variants:
        print("  모든 새로운 조합이 이미 측정됨. 추가 실험 불필요!")
        return pd.read_csv(OUTPUT_FILE)

    print(f"  전체 추가 조합: {len(all_variants)}개")
    print(f"  이미 측정됨: {len(all_variants) - len(new_variants)}개")
    print(f"  새로 측정할 조합: {len(new_variants)}개")
    print(f"  예상 새 데이터 포인트: {len(new_variants) * 2}개 (CPU + MPS 각각)")

    # ── Step 3. MNIST 데이터 준비 ────────────────────────────
    print("\n[Step 3] MNIST 데이터 로드")
    train_loader, test_loader = prepare_mnist_data(batch_size=64)
    print("  MNIST 데이터 준비 완료")

    # ── Step 4. 실험 실행 ────────────────────────────────────
    print("\n[Step 4] 실험 시작")
    devices = ['cpu', 'mps']
    new_results = []

    for i, (_, model_info) in enumerate(new_variants):
        config = model_info['config']
        total_params = model_info['total_params']
        current_idx = next_model_idx + i

        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(new_variants)}] Config: {config}")
        print(f"  num_layers={model_info['num_layers']}, total_params={total_params:,}")
        print(f"  model_idx: {current_idx}")
        print(f"{'─' * 70}")

        for device in devices:
            print(f"\n  [디바이스: {device.upper()}]")

            # 각 디바이스마다 새로운 모델 인스턴스 생성
            # (이전 디바이스 학습 상태와 독립적으로 측정하기 위함)
            fresh_model = SimpleANN(hidden_sizes=config)

            # 모델 학습 (MNIST 식별 가능한 수준까지)
            print("  학습 중...")
            trained_model = train_model(fresh_model, train_loader, device=device, epochs=2)

            # 시간 측정기 생성
            # warmup_runs=5: 처음 5번은 GPU 캐시 워밍업으로 제외
            # measure_runs=10: 10번 반복 측정 후 평균 계산
            estimator = TimeEstimator(device=device, warmup_runs=5, measure_runs=10)

            # 추론 시간 측정 (배치 크기 64, 이미지 28×28 흑백)
            print("  추론 시간 측정 중 (warmup 5회 + 측정 10회)...")
            inference_mean, inference_std = estimator.measure_inference_time(
                trained_model,
                input_shape=(64, 1, 28, 28)  # (배치, 채널, 높이, 너비)
            )

            # 학습 시간 측정 (1 epoch 기준)
            print("  학습 시간 측정 중 (1 epoch)...")
            training_mean, training_std = estimator.measure_training_time(
                trained_model,
                train_loader,
                epochs=1
            )

            # 결과 저장
            result = {
                'model_idx': current_idx,
                'config': str(config),
                'num_layers': model_info['num_layers'],
                'total_params': total_params,
                'device': device,
                'inference_time_mean_ms': inference_mean,
                'inference_time_std_ms': inference_std,
                'training_time_mean_sec': training_mean,
                'training_time_std_sec': training_std,
            }
            new_results.append(result)

            print(f"  추론 시간: {inference_mean:.3f} ± {inference_std:.3f} ms")
            print(f"  학습 시간: {training_mean:.2f} ± {training_std:.2f} 초/epoch")

    # ── Step 5. 결과 저장 (기존 CSV에 추가) ──────────────────
    print(f"\n{'=' * 70}")
    print("[Step 5] 결과 저장")

    # stage1 폴더가 없으면 생성
    os.makedirs(STAGE1_DIR, exist_ok=True)

    new_df = pd.DataFrame(new_results)

    if os.path.exists(OUTPUT_FILE):
        # 기존 CSV에 새 결과를 이어 붙여 저장 (append)
        existing_df = pd.read_csv(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  기존 {len(existing_df)}행 + 새 {len(new_df)}행 = 총 {len(combined_df)}행")
    else:
        # 기존 CSV 없으면 새로 생성
        new_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  새 CSV 생성: {len(new_df)}행")

    print(f"  저장 완료: {OUTPUT_FILE}")

    # 새 결과 요약 출력
    print(f"\n[새로 수집된 데이터 요약]")
    print(new_df[['config', 'num_layers', 'total_params', 'device',
                  'inference_time_mean_ms', 'training_time_mean_sec']].to_string(index=False))

    print(f"\n{'=' * 70}")
    print("  실험 완료!")
    print(f"{'=' * 70}")

    return new_df


# ──────────────────────────────────────────────────────────
# 직접 실행 시 실험 시작
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_df = run_experiment()
