"""Swin Transformer + CIFAR-10 실행 시간 측정 실험

Swin Transformer는 윈도우 기반 Attention과 계층적 구조를 사용하는 모델.
timm 라이브러리에서 Tiny / Small / Base 3가지 변형을 가져와 실험.

실험 설정:
    - 모델    : swin_tiny / swin_small / swin_base
    - 입력    : CIFAR-10 (3×32×32) → 224×224로 리사이즈
    - warmup  : 5회
    - 반복 측정: 10회
    - batch_size: 64

사용법:
    python collect/swin_collector.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swin import create_swin_variants
from features.extractor import extract_features
from utils.device_utils import get_best_device, synchronize_device

# ── 실험 설정 ─────────────────────────────────────────────
BATCH_SIZE   = 64
WARMUP_RUNS  = 5
MEASURE_RUNS = 10

# Swin은 224×224 입력 기준으로 설계됨
INPUT_CONFIG = {
    'batch_size':      BATCH_SIZE,
    'input_channels':  1,
    'input_height':    224,
    'input_width':     224,
    'num_classes':     10,
    'dataset_encoded': 0,   # MNIST=0
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'swin_results.csv')


def get_devices():
    best = get_best_device()
    return ['cpu'] if best == 'cpu' else ['cpu', best]


def load_mnist():
    """MNIST 데이터셋 로드 (224×224로 리사이즈, 흑백 1채널)"""
    transform = transforms.Compose([
        transforms.Resize(224),          # 28×28 → 224×224
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir     = os.path.join(ROOT_DIR, 'data')
    train_ds     = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds      = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def measure_times(model_fn, train_batches, test_batches, device_str):
    """학습 시간 + 추론 시간 측정

    warmup 5회 후 10회 반복 측정하여 평균/표준편차 반환.
    """
    criterion = nn.CrossEntropyLoss()
    device    = torch.device(device_str)

    def single_run():
        model     = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters())

        # 학습 시간 측정
        model.train()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        for data, target in train_batches:
            optimizer.zero_grad()
            loss = criterion(model(data.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        synchronize_device(device_str)
        train_time = time.perf_counter() - t0

        # 추론 시간 측정
        model.eval()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        with torch.no_grad():
            for data, _ in test_batches:
                model(data.to(device))
        synchronize_device(device_str)
        infer_time = time.perf_counter() - t0

        return train_time, infer_time

    # 웜업 (초기 오버헤드 제거)
    for _ in range(WARMUP_RUNS):
        single_run()

    train_times, infer_times = [], []
    for _ in range(MEASURE_RUNS):
        t_train, t_infer = single_run()
        train_times.append(t_train)
        infer_times.append(t_infer)

    return train_times, infer_times


def run():
    print("=" * 60)
    print("  Swin Transformer + MNIST 실행 시간 측정")
    print("=" * 60)

    variants = create_swin_variants()
    devices  = get_devices()
    print(f"조합 수: {len(variants)}  |  측정 장치: {devices}\n")

    print("MNIST 데이터 로드 중...")
    train_loader, test_loader = load_mnist()
    print("완료\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    total   = len(variants) * len(devices)
    count   = 0

    for device_str in devices:
        device = torch.device(device_str)

        print(f"[{device_str.upper()}] 데이터 사전 로딩 중...")
        train_batches = [(x.to(device), y.to(device)) for x, y in train_loader]
        test_batches  = [(x.to(device), y.to(device)) for x, y in test_loader]
        print("완료\n")

        for model_tmpl, config in variants:
            count += 1
            model_name = config['model_name']
            embed_dim  = config['embed_dim']
            num_layers = config['num_transformer_layers']

            def model_fn(mn=model_name):
                import timm
                return timm.create_model(mn, pretrained=False, num_classes=10, img_size=224, in_chans=1)

            dummy_model = model_fn()
            features = extract_features(
                dummy_model, 'transformer', config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] {model_name}, embed={embed_dim}, "
                  f"layers={num_layers}, device={device_str}, "
                  f"params={features['total_params']:,}")

            train_times, infer_times = measure_times(
                model_fn, train_batches, test_batches, device_str
            )

            result = {
                **features,
                'training_time_mean_sec': round(float(np.mean(train_times)), 5),
                'training_time_std_sec':  round(float(np.std(train_times)),  5),
                'inference_time_mean_ms': round(float(np.mean(infer_times)) * 1000, 4),
                'inference_time_std_ms':  round(float(np.std(infer_times))  * 1000, 4),
            }
            results.append(result)

            print(f"  학습: {result['training_time_mean_sec']:.4f}s "
                  f"(±{result['training_time_std_sec']:.4f}) | "
                  f"추론: {result['inference_time_mean_ms']:.2f}ms "
                  f"(±{result['inference_time_std_ms']:.2f})\n")

        del train_batches, test_batches
        if device_str == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {OUTPUT_PATH} ({len(df)}행)")


if __name__ == '__main__':
    run()
