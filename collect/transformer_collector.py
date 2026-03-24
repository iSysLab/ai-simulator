"""Transformer + CIFAR-10 실행 시간 측정 실험

실험 설정:
    - embed_dim    : 64, 128, 256
    - num_layers   : 2, 4, 6
    - num_heads    : 4, 8
    - patch_size   : 4, 8
    - warmup  : 2회
    - 반복 측정: 3회
    - batch_size: 64

사용법:
    python collect/transformer_collector.py
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

from models.transformer import create_transformer_variants
from features.extractor import extract_features
from utils.device_utils import get_best_device, synchronize_device

# ── 실험 설정 ─────────────────────────────────────────────
BATCH_SIZE   = 64
WARMUP_RUNS  = 5
MEASURE_RUNS = 10

INPUT_CONFIG = {
    'batch_size':      BATCH_SIZE,
    'input_channels':  3,
    'input_height':    32,
    'input_width':     32,
    'num_classes':     10,
    'dataset_encoded': 1,   # CIFAR-10=1
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'transformer_results.csv')


def get_devices():
    best = get_best_device()
    return ['cpu'] if best == 'cpu' else ['cpu', best]


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    data_dir = os.path.join(ROOT_DIR, 'data')
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def measure_times(model_fn, train_batches, test_batches, device_str):
    criterion = nn.CrossEntropyLoss()
    device    = torch.device(device_str)

    def single_run():
        model     = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        for data, target in train_batches:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        synchronize_device(device_str)
        train_time = time.perf_counter() - t0

        model.eval()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        with torch.no_grad():
            for data, _ in test_batches:
                model(data)
        synchronize_device(device_str)
        infer_time = time.perf_counter() - t0

        return train_time, infer_time

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
    print("  Transformer + CIFAR-10 실행 시간 측정")
    print("=" * 60)

    variants = create_transformer_variants(img_size=32, in_channels=3, num_classes=10)
    devices  = get_devices()
    print(f"조합 수: {len(variants)}  |  측정 장치: {devices}\n")

    print("CIFAR-10 데이터 로드 중...")
    train_loader, test_loader = load_cifar10()
    print("완료\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    total = len(variants) * len(devices)
    count = 0

    for device_str in devices:
        device = torch.device(device_str)

        print(f"[{device_str.upper()}] 데이터 사전 로딩 중...")
        train_batches = [(x.to(device), y.to(device)) for x, y in train_loader]
        test_batches  = [(x.to(device), y.to(device)) for x, y in test_loader]
        print("완료\n")

        for model_tmpl, config in variants:
            count += 1

            embed_dim   = config['embed_dim']
            num_layers  = config['num_transformer_layers']
            num_heads   = config['num_heads']
            patch_size  = config['patch_size']

            def model_fn(ed=embed_dim, nl=num_layers, nh=num_heads, ps=patch_size):
                from models.transformer import SimpleViT
                return SimpleViT(
                    img_size=32, patch_size=ps, in_channels=3, num_classes=10,
                    embed_dim=ed, num_layers=nl, num_heads=nh,
                )

            dummy_model = model_fn()
            features = extract_features(
                dummy_model, 'transformer', config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] embed={embed_dim}, layers={num_layers}, "
                  f"heads={num_heads}, patch={patch_size}, device={device_str}, "
                  f"params={features['total_params']:,}")

            train_times, infer_times = measure_times(
                model_fn, train_batches, test_batches, device_str
            )

            result = {
                **features,
                'training_time_mean_sec':  round(float(np.mean(train_times)), 5),
                'training_time_std_sec':   round(float(np.std(train_times)),  5),
                'inference_time_mean_ms':  round(float(np.mean(infer_times)) * 1000, 4),
                'inference_time_std_ms':   round(float(np.std(infer_times))  * 1000, 4),
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
