"""DCGAN + MNIST 실행 시간 측정 실험

DCGAN은 Conv/ConvTranspose 레이어 기반 GAN으로
기존 Linear 기반 GAN보다 이미지 품질이 높고 학습이 안정적.

실험 설정:
    - 모델    : Small(64) / Base(100) / Large(128) latent_dim
    - 입력    : MNIST (1×28×28) 흑백
    - warmup  : 5회
    - 반복 측정: 10회
    - batch_size: 64

사용법:
    python collect/dcgan_collector.py
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

from models.dcgan import create_dcgan_variants
from features.extractor import extract_features
from utils.device_utils import get_best_device, synchronize_device

# ── 실험 설정 ─────────────────────────────────────────────
BATCH_SIZE   = 64
WARMUP_RUNS  = 5
MEASURE_RUNS = 10

INPUT_CONFIG = {
    'batch_size':      BATCH_SIZE,
    'input_channels':  1,
    'input_height':    28,
    'input_width':     28,
    'num_classes':     0,   # GAN은 분류 모델이 아님
    'dataset_encoded': 0,   # MNIST=0
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'dcgan_results.csv')


def get_devices():
    best = get_best_device()
    return ['cpu'] if best == 'cpu' else ['cpu', best]


def load_mnist():
    """MNIST 데이터셋 로드 (흑백 1채널)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1] 범위로 정규화 (DCGAN 기준)
    ])
    data_dir     = os.path.join(ROOT_DIR, 'data')
    train_ds     = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds      = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def measure_times(model_fn, train_batches, test_batches, device_str):
    """학습 시간 + 추론 시간 측정

    DCGAN 학습: Generator와 Discriminator를 교대로 학습.
    warmup 5회 후 10회 반복 측정.
    """
    criterion = nn.BCELoss()  # Binary Cross Entropy (진짜/가짜 판별)
    device    = torch.device(device_str)

    def single_run():
        model         = model_fn().to(device)
        opt_g         = torch.optim.Adam(model.generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
        opt_d         = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        latent_dim    = model.latent_dim

        # 학습 시간 측정
        model.train()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        for real_imgs, _ in train_batches:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Discriminator 학습
            z        = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = model.generator(z).detach()
            real_loss = criterion(model.discriminator(real_imgs), torch.ones(batch_size, 1).to(device))
            fake_loss = criterion(model.discriminator(fake_imgs), torch.zeros(batch_size, 1).to(device))
            d_loss    = (real_loss + fake_loss) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Generator 학습
            z        = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = model.generator(z)
            g_loss    = criterion(model.discriminator(fake_imgs), torch.ones(batch_size, 1).to(device))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        synchronize_device(device_str)
        train_time = time.perf_counter() - t0

        # 추론 시간 측정 (Generator만)
        model.eval()
        synchronize_device(device_str)
        t0 = time.perf_counter()
        with torch.no_grad():
            for real_imgs, _ in test_batches:
                batch_size = real_imgs.size(0)
                z = torch.randn(batch_size, latent_dim).to(device)
                model.generator(z)
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
    print("  DCGAN + MNIST 실행 시간 측정")
    print("=" * 60)

    variants = create_dcgan_variants(img_channels=1)
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
            latent_dim    = config['latent_dim']
            base_channels = config['base_channels']

            def model_fn(ld=latent_dim, bc=base_channels):
                from models.dcgan import SimpleDCGAN
                return SimpleDCGAN(latent_dim=ld, base_channels=bc, img_channels=1)

            dummy_model = model_fn()
            features = extract_features(
                dummy_model, 'gan', config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] DCGAN, latent={latent_dim}, "
                  f"base_ch={base_channels}, device={device_str}, "
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
