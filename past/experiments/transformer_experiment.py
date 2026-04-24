        # ============================================================
# 4단계: Transformer 모델 실행 시간 측정 실험
# ============================================================
#
# 목적:
#   다양한 구조의 Transformer 모델(SimpleViT)을 CIFAR-10으로 학습하고,
#   CPU와 MPS에서의 추론 시간 및 학습 시간을 측정하여 CSV로 저장
#
# 데이터셋: CIFAR-10 (32×32 컬러 이미지, 10개 클래스)
# 하드웨어: MacBook Air M1 (CPU + MPS)
#
# 측정 방식 (교수님 유의사항):
#   - Warmup 5회 후 10회 반복 측정 → 평균값 사용
#   - CPU와 MPS 각각 측정
#
# 실행 전 준비:
#   - CIFAR-10 데이터가 data/raw/ 폴더에 있어야 함
#   - 가상환경 활성화 후 실행
#
# 작성자: 김홍근
# ============================================================

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 상위 디렉토리를 Python 경로에 추가 (models/, utils/ 접근)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_models import create_transformer_variants
from benchmark.support.timer import TimeEstimator

# ──────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
RAW_DIR    = os.path.join(DATA_DIR, 'raw')       # CIFAR-10 원본 데이터셋
STAGE4_DIR = os.path.join(DATA_DIR, 'stage4')    # 4단계 결과 저장
OUTPUT_FILE = os.path.join(STAGE4_DIR, 'transformer_results.csv')

# ──────────────────────────────────────────────────────────
# CIFAR-10 데이터 준비
# ──────────────────────────────────────────────────────────

def prepare_cifar10_data(batch_size=64):
    """
    CIFAR-10 데이터셋 로드 함수

    CIFAR-10: 10개 카테고리의 32×32 컬러 이미지 데이터셋
      - 훈련: 50,000장, 테스트: 10,000장

    Args:
        batch_size (int): 배치 크기

    Returns:
        tuple: (train_loader, test_loader)
    """
    # 이미지 전처리: 정규화 (CIFAR-10 통계값 사용)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 R, G, B 평균
            std=(0.2023, 0.1994, 0.2010)    # CIFAR-10 R, G, B 표준편차
        )
    ])

    try:
        train_dataset = datasets.CIFAR10(RAW_DIR, train=True,  download=False, transform=transform)
        test_dataset  = datasets.CIFAR10(RAW_DIR, train=False, download=False, transform=transform)
    except Exception:
        print("  CIFAR-10 다운로드 중...")
        train_dataset = datasets.CIFAR10(RAW_DIR, train=True,  download=True,  transform=transform)
        test_dataset  = datasets.CIFAR10(RAW_DIR, train=False, download=True,  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

# ──────────────────────────────────────────────────────────
# 모델 학습 함수
# ──────────────────────────────────────────────────────────

def train_model(model, train_loader, device='cpu', epochs=3):
    """
    Transformer 모델을 CIFAR-10으로 학습하는 함수

    학습 목적:
    - CIFAR-10을 어느 정도 식별 가능한 수준까지만 학습 (완벽 정확도 불필요)
    - 학습된 상태에서 추론 시간 측정하기 위한 준비

    Args:
        model: 학습할 Transformer 모델
        train_loader: CIFAR-10 훈련 데이터 로더
        device (str): 'cpu' 또는 'mps'
        epochs (int): 학습 epoch 수 (기본 3)

    Returns:
        model: 학습 완료된 모델
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # AdamW: Adam + weight_decay (L2 정규화). Transformer 학습에 자주 사용됨.

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct    = 0
        total      = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total   += target.size(0)
            correct += predicted.eq(target).sum().item()

            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%')

    return model

# ──────────────────────────────────────────────────────────
# 기존 측정 데이터 확인 (중복 측정 방지)
# ──────────────────────────────────────────────────────────

def load_existing_pairs(output_file):
    """
    이미 측정된 (config_str, device) 조합을 CSV에서 불러옵니다.
    Mac(MPS)에서 측정한 config도 Windows(CUDA)에서 cuda 데이터를 추가할 수 있도록
    (config_str, device) 기준으로 중복 체크합니다.
    """
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        existing_pairs = set(zip(df['config_str'].astype(str), df['device'].astype(str)))
        config_to_idx = df.groupby('config_str')['model_idx'].first().to_dict()
        next_idx = int(df['model_idx'].max()) + 1
        print(f"  기존 CSV 발견: {len(df)}행, (config_str, device) 조합 {len(existing_pairs)}개")
        return existing_pairs, config_to_idx, next_idx
    print("  기존 CSV 없음 → 새로 시작")
    return set(), {}, 0

# ──────────────────────────────────────────────────────────
# 메인 실험 함수
# ──────────────────────────────────────────────────────────

def run_experiment():
    """
    Transformer 모델 실행 시간 측정 메인 함수

    동작 순서:
    1. 기존 측정 데이터 확인 (이미 측정한 config는 건너뜀)
    2. 새로운 Transformer 조합 준비
    3. CIFAR-10 데이터 로드
    4. 각 조합 × CPU/MPS 측정 (warmup 5회 + 10회 반복)
    5. 결과를 CSV에 저장 (기존 데이터에 추가)
    """
    print("=" * 70)
    print("  4단계: Transformer 실행 시간 측정 실험")
    print("  하드웨어: MacBook Air M1 (CPU + MPS)")
    print("=" * 70)

    os.makedirs(STAGE4_DIR, exist_ok=True)

    if torch.cuda.is_available():
        devices = ['cpu', 'cuda']
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        devices = ['cpu', 'mps']
    else:
        devices = ['cpu']
    print(f"  사용 디바이스: {devices}")

    # 기존 측정 확인
    print("\n[Step 1] 기존 측정 데이터 확인")
    existing_pairs, config_to_idx, next_idx = load_existing_pairs(OUTPUT_FILE)

    # 측정할 (config_str, device) work_list 구성
    print("\n[Step 2] Transformer 모델 조합 준비")
    all_variants = create_transformer_variants(img_size=32, in_channels=3, num_classes=10)
    work_list = []
    for model, info in all_variants:
        config_str = info['config_str']
        for device in devices:
            if (config_str, device) not in existing_pairs:
                model_idx = config_to_idx.get(config_str, next_idx)
                if config_str not in config_to_idx:
                    config_to_idx[config_str] = model_idx
                    next_idx += 1
                work_list.append((info, device, config_to_idx[config_str]))

    if not work_list:
        print("  모든 (config_str, device) 조합이 이미 측정됨!")
        return pd.read_csv(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else pd.DataFrame()

    print(f"  전체 조합: {len(all_variants)}개 | 측정할 항목: {len(work_list)}개")

    # CIFAR-10 데이터 로드
    print("\n[Step 3] CIFAR-10 데이터 로드")
    train_loader, test_loader = prepare_cifar10_data(batch_size=64)
    print("  CIFAR-10 데이터 준비 완료")

    # 실험 실행
    print("\n[Step 4] 실험 시작")
    new_results = []

    for i, (info, device, current_idx) in enumerate(work_list):
        config_str = info['config_str']

        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(work_list)}] {config_str} @ {device.upper()}")
        print(f"  embed_dim={info['embed_dim']}, layers={info['num_layers']}, "
              f"heads={info['num_heads']}, patch={info['patch_size']}")
        print(f"  params={info['total_params']:,} | patches={info['num_patches']}")
        print(f"{'─' * 70}")

        print(f"\n  [디바이스: {device.upper()}]")

        # 각 디바이스마다 새 모델 인스턴스 (독립적 측정)
        model_obj = next(m for m, inf in all_variants if inf['config_str'] == config_str)
        fresh_model = type(model_obj)(
            img_size=info['img_size'],
            patch_size=info['patch_size'],
            in_channels=3,
            num_classes=10,
            embed_dim=info['embed_dim'],
            num_layers=info['num_layers'],
            num_heads=info['num_heads'],
        )

        # 학습
        print("  학습 중 (3 epoch)...")
        trained_model = train_model(fresh_model, train_loader, device=device, epochs=3)

        # 시간 측정 (warmup 5회 + 10회 반복)
        estimator = TimeEstimator(device=device, warmup_runs=5, measure_runs=10)

        print("  추론 시간 측정 중...")
        inference_mean, inference_std = estimator.measure_inference_time(
            trained_model,
            input_shape=(64, 3, 32, 32)  # CIFAR-10: 32×32 컬러
        )

        print("  학습 시간 측정 중...")
        training_mean, training_std = estimator.measure_training_time(
            trained_model, train_loader, epochs=1
        )

        # 결과 기록
        result = {
            'model_idx': current_idx,
            'config_str': config_str,
            'model_type': 'Transformer',
            'embed_dim': info['embed_dim'],
            'num_layers': info['num_layers'],
            'num_heads': info['num_heads'],
            'patch_size': info['patch_size'],
            'num_patches': info['num_patches'],
            'total_params': info['total_params'],
            'device': device,
            'inference_time_mean_ms': inference_mean,
            'inference_time_std_ms': inference_std,
            'training_time_mean_sec': training_mean,
            'training_time_std_sec': training_std,
            'dataset': 'CIFAR-10',
            'img_size': 32,
            'img_channels': 3,
            'num_classes': 10,
            'batch_size': 64,
        }
        new_results.append(result)

        print(f"  추론: {inference_mean:.3f} ± {inference_std:.3f} ms")
        print(f"  학습: {training_mean:.2f} ± {training_std:.2f} 초/epoch")

    # 결과 저장
    print(f"\n{'=' * 70}")
    print("[Step 5] 결과 저장")
    new_df = pd.DataFrame(new_results)

    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  기존 {len(existing_df)}행 + 새 {len(new_df)}행 = 총 {len(combined_df)}행")
    else:
        new_df.to_csv(OUTPUT_FILE, index=False)
        print(f"  새 CSV 생성: {len(new_df)}행")

    print(f"  저장: {OUTPUT_FILE}")
    print(f"\n{'=' * 70}")
    print("  Transformer 실험 완료!")
    print(f"{'=' * 70}")
    return new_df


if __name__ == '__main__':
    run_experiment()
