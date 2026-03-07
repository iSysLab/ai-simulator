"""CNN + MNIST 실행 시간 측정 실험

실험 설정:
    - num_filters     : 8, 16, 32, 64, 128
    - num_conv_layers : 1, 2, 3, 4, 5, 6
    - use_batchnorm   : False, True
    - 조합 수          : 5 × 6 × 2 = 60가지
    - device          : cpu, cuda
    - warmup          : 3회
    - 반복 측정        : 10회
    - batch_size      : 64
    - epochs          : 1

사용법:
    python collect/cnn_collector.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from itertools import product
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 루트 디렉토리를 경로에 추가 (models, features 임포트용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import SimpleCNN
from features.extractor import extract_features

# ── 실험 설정 ─────────────────────────────────────────────
NUM_FILTERS      = [8, 16, 32, 64, 128]
NUM_CONV_LAYERS  = [1, 2, 3, 4, 5, 6]
USE_BATCHNORM    = [False, True]
BATCH_SIZE       = 64
EPOCHS           = 1
WARMUP_RUNS      = 3
MEASURE_RUNS     = 10

INPUT_CONFIG = {
    'batch_size':     BATCH_SIZE,
    'input_channels': 1,
    'input_height':   28,
    'input_width':    28,
    'num_classes':    10,
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'cnn_results.csv')


# ── 유틸 함수 ─────────────────────────────────────────────

def get_devices():
    """사용 가능한 디바이스 목록 반환"""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def sync(device_str):
    """GPU 연산 완료 대기 (정확한 시간 측정을 위해 필요)"""
    if device_str == 'cuda':
        torch.cuda.synchronize()


def load_mnist():
    """MNIST 데이터셋 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(ROOT_DIR, 'data')
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# ── 시간 측정 ─────────────────────────────────────────────

def measure_times(model_fn, train_batches, test_batches, device_str):
    """워밍업 후 학습/추론 시간 측정

    Args:
        model_fn: 모델을 새로 생성하는 함수 (매 측정마다 fresh 모델 사용)
        train_batches: 디바이스에 사전 로딩된 학습 배치 리스트
        test_batches:  디바이스에 사전 로딩된 테스트 배치 리스트
        device_str: 'cpu' 또는 'cuda'

    Returns:
        tuple: (train_times 리스트, infer_times 리스트)
    """
    criterion = nn.CrossEntropyLoss()
    device = torch.device(device_str)

    def single_run():
        """모델 새로 생성 → 학습 시간 측정 → 추론 시간 측정"""
        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters())

        # 학습 시간 측정 (1 epoch)
        model.train()
        sync(device_str)
        t0 = time.perf_counter()
        for data, target in train_batches:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        sync(device_str)
        train_time = time.perf_counter() - t0

        # 추론 시간 측정
        model.eval()
        sync(device_str)
        t0 = time.perf_counter()
        with torch.no_grad():
            for data, _ in test_batches:
                model(data)
        sync(device_str)
        infer_time = time.perf_counter() - t0

        return train_time, infer_time

    # 워밍업: GPU 캐시 준비 (측정값에 포함하지 않음)
    for _ in range(WARMUP_RUNS):
        single_run()

    # 본 측정
    train_times, infer_times = [], []
    for _ in range(MEASURE_RUNS):
        t_train, t_infer = single_run()
        train_times.append(t_train)
        infer_times.append(t_infer)

    return train_times, infer_times


# ── 메인 실험 ─────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CNN + MNIST 실행 시간 측정")
    print(f"  조합: {len(NUM_FILTERS)} filters × {len(NUM_CONV_LAYERS)} layers"
          f" × {len(USE_BATCHNORM)} batchnorm = {len(NUM_FILTERS)*len(NUM_CONV_LAYERS)*len(USE_BATCHNORM)}가지")
    print("=" * 60)

    devices = get_devices()
    print(f"측정 장치: {devices}\n")

    print("MNIST 데이터 로드 중...")
    train_loader, test_loader = load_mnist()
    print("완료\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    total = len(NUM_FILTERS) * len(NUM_CONV_LAYERS) * len(USE_BATCHNORM) * len(devices)
    count = 0

    for device_str in devices:
        device = torch.device(device_str)

        # 데이터를 디바이스에 사전 로딩 (데이터 전송 시간 제외)
        print(f"[{device_str.upper()}] 데이터 사전 로딩 중...")
        train_batches = [(x.to(device), y.to(device)) for x, y in train_loader]
        test_batches  = [(x.to(device), y.to(device)) for x, y in test_loader]
        print("완료\n")

        for num_filters, num_conv_layers, use_batchnorm in product(
            NUM_FILTERS, NUM_CONV_LAYERS, USE_BATCHNORM
        ):
            count += 1
            bn_str = 'BN' if use_batchnorm else 'noBN'

            # 매 측정마다 새 모델을 생성하는 함수
            def model_fn(f=num_filters, l=num_conv_layers, bn=use_batchnorm):
                return SimpleCNN(num_filters=f, num_conv_layers=l, use_batchnorm=bn)

            # feature 추출 (모델 구조 정보)
            dummy_model = model_fn()
            model_config = {
                'num_filters':     num_filters,
                'num_conv_layers': num_conv_layers,
                'has_batchnorm':   1 if use_batchnorm else 0,
                'has_pooling':     1,   # MaxPool2d 항상 사용
                'kernel_size':     3,   # Conv 커널 크기 고정
                'num_fc_layers':   1,   # classifier Linear 1개
            }
            features = extract_features(
                dummy_model, 'cnn', model_config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] filters={num_filters}, layers={num_conv_layers}, "
                  f"{bn_str}, device={device_str}, params={features['total_params']:,}")

            # 시간 측정
            train_times, infer_times = measure_times(
                model_fn, train_batches, test_batches, device_str
            )

            result = {
                **features,
                'train_time_mean': round(float(np.mean(train_times)), 5),
                'train_time_std':  round(float(np.std(train_times)),  5),
                'infer_time_mean': round(float(np.mean(infer_times)), 5),
                'infer_time_std':  round(float(np.std(infer_times)),  5),
            }
            results.append(result)

            print(f"  학습: {result['train_time_mean']:.4f}s "
                  f"(±{result['train_time_std']:.4f}) | "
                  f"추론: {result['infer_time_mean']:.4f}s "
                  f"(±{result['infer_time_std']:.4f})\n")

        del train_batches, test_batches
        if device_str == 'cuda':
            torch.cuda.empty_cache()

    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {OUTPUT_PATH} ({len(df)}행)")


if __name__ == '__main__':
    run()
