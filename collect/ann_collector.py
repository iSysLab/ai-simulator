"""ANN + MNIST 실행 시간 측정 실험

실험 설정:
    - hidden_size : 16, 32, 64, 128, 256, 512, 1024
    - num_hidden_layers: 1, 2, 3, 4, 5, 6, 7
    - warmup   : 3회
    - 반복 측정 : 10회
    - batch_size: 64
    - epochs   : 1

사용법:
    python collect/ann_collector.py
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

from models.ann import SimpleANN
from features.extractor import extract_features

# ── 실험 설정 ─────────────────────────────────────────────
HIDDEN_SIZES        = [16, 32, 64, 128, 256, 512, 1024]
NUM_HIDDEN_LAYERS   = [1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE          = 64
EPOCHS              = 1
WARMUP_RUNS         = 1
MEASURE_RUNS        = 1

INPUT_CONFIG = {
    'batch_size':     BATCH_SIZE,
    'input_channels': 1,
    'input_height':   28,
    'input_width':    28,
    'num_classes':    10,
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'ann_results.csv')


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
            optimizer.zero_grad() # 이전 배치에서 계산된 gradient 초기화
            loss = criterion(model(data), target) # loss를 기준으로 gradient 계산 (backpropagation)
            loss.backward() # loss를 기준으로 gradient 계산 (backpropagation)
            optimizer.step() # 계산된 gradient를 이용해 모델 가중치 업데이트
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
    print("  ANN + MNIST 실행 시간 측정")
    print(f"  조합: {len(HIDDEN_SIZES)} hidden_size × {len(NUM_HIDDEN_LAYERS)} num_layers")
    print("=" * 60)

    devices = get_devices()
    print(f"측정 장치: {devices}\n")

    print("MNIST 데이터 로드 중...")
    train_loader, test_loader = load_mnist()
    print("완료\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    total = len(HIDDEN_SIZES) * len(NUM_HIDDEN_LAYERS) * len(devices)
    count = 0

    for device_str in devices:
        device = torch.device(device_str)

        # 데이터를 디바이스에 사전 로딩 (데이터 전송 시간 제외)
        print(f"[{device_str.upper()}] 데이터 사전 로딩 중...")
        train_batches = [(x.to(device), y.to(device)) for x, y in train_loader]
        test_batches  = [(x.to(device), y.to(device)) for x, y in test_loader]
        print("완료\n")

        for hidden_size, num_hidden_layers in product(HIDDEN_SIZES, NUM_HIDDEN_LAYERS):
            count += 1

            # 매 측정마다 새 모델을 생성하는 함수
            def model_fn(h=hidden_size, l=num_hidden_layers):
                return SimpleANN(hidden_size=h, num_hidden_layers=l)

            # feature 추출 (모델 구조 정보)
            dummy_model = model_fn()
            model_config = {
                'hidden_size':       hidden_size,
                'num_hidden_layers': num_hidden_layers,
            }
            features = extract_features(
                dummy_model, 'ann', model_config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] hidden={hidden_size}, layers={num_hidden_layers}, "
                  f"device={device_str}, params={features['total_params']:,}")

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
