"""GAN + MNIST 실행 시간 측정 실험

실험 설정:
    - latent_dim      : 64, 128, 256
    - g_hidden_dims   : 다양한 조합 (8가지)
    - warmup  : 2회
    - 반복 측정: 3회
    - batch_size: 64

GAN 측정 방식:
    학습 시간: G forward → D forward → 손실 계산 → 역전파 전체 1 epoch
    추론 시간: Generator forward만 (노이즈 → 이미지)

사용법:
    python collect/gan_collector.py
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

from models.gan import create_gan_variants
from features.extractor import extract_features

# ── 실험 설정 ─────────────────────────────────────────────
BATCH_SIZE   = 64
IMG_SIZE     = 28
IMG_CHANNELS = 1
WARMUP_RUNS  = 1
MEASURE_RUNS = 1

INPUT_CONFIG = {
    'batch_size':     BATCH_SIZE,
    'input_channels': IMG_CHANNELS,
    'input_height':   IMG_SIZE,
    'input_width':    IMG_SIZE,
    'num_classes':    1,   # GAN은 분류 클래스 없음 (1로 설정)
}

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'gan_results.csv')


def get_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def sync(device_str):
    if device_str == 'cuda':
        torch.cuda.synchronize()


def load_mnist():
    # GAN용 정규화: 픽셀 0~1 → -1~1 (Generator Tanh 출력과 맞춤)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(ROOT_DIR, 'data')
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


def measure_times(model_fn, train_batches, device_str, latent_dim):
    criterion = nn.BCELoss()
    device    = torch.device(device_str)

    def single_run():
        gan           = model_fn().to(device)
        g_optimizer   = torch.optim.Adam(gan.generator.parameters(),     lr=0.0002)
        d_optimizer   = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002)

        real_label = torch.ones(BATCH_SIZE,  1, device=device)
        fake_label = torch.zeros(BATCH_SIZE, 1, device=device)

        # 학습 시간 측정 (1 epoch)
        gan.train()
        sync(device_str)
        t0 = time.perf_counter()
        for real_imgs, _ in train_batches:
            bs = real_imgs.size(0)
            if bs < BATCH_SIZE:
                continue

            # Discriminator 학습
            d_optimizer.zero_grad()
            d_real = gan.discriminator(real_imgs)
            d_loss_real = criterion(d_real, real_label)

            noise  = torch.randn(bs, latent_dim, device=device)
            fake   = gan.generator(noise).detach()
            d_fake = gan.discriminator(fake)
            d_loss_fake = criterion(d_fake, fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Generator 학습
            g_optimizer.zero_grad()
            noise  = torch.randn(bs, latent_dim, device=device)
            fake   = gan.generator(noise)
            g_loss = criterion(gan.discriminator(fake), real_label)
            g_loss.backward()
            g_optimizer.step()

        sync(device_str)
        train_time = time.perf_counter() - t0

        # 추론 시간 측정 (Generator forward만)
        gan.generator.eval()
        sync(device_str)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in train_batches:
                noise = torch.randn(BATCH_SIZE, latent_dim, device=device)
                gan.generator(noise)
        sync(device_str)
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
    print("  GAN + MNIST 실행 시간 측정")
    print("=" * 60)

    variants = create_gan_variants(img_size=IMG_SIZE, img_channels=IMG_CHANNELS)
    devices  = get_devices()
    print(f"조합 수: {len(variants)}  |  측정 장치: {devices}\n")

    print("MNIST 데이터 로드 중...")
    train_loader = load_mnist()
    print("완료\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    total = len(variants) * len(devices)
    count = 0

    for device_str in devices:
        device = torch.device(device_str)

        print(f"[{device_str.upper()}] 데이터 사전 로딩 중...")
        train_batches = [(x.to(device), y.to(device)) for x, y in train_loader]
        print("완료\n")

        for gan_tmpl, config in variants:
            count += 1
            latent_dim    = config['latent_dim']
            g_hidden_dims = config['g_hidden_dims']

            def model_fn(ld=latent_dim, ghd=g_hidden_dims):
                from models.gan import SimpleGAN
                return SimpleGAN(
                    latent_dim=ld,
                    img_size=IMG_SIZE,
                    img_channels=IMG_CHANNELS,
                    g_hidden_dims=ghd,
                )

            dummy_model = model_fn()
            features = extract_features(
                dummy_model, 'gan', config, device_str, INPUT_CONFIG
            )
            del dummy_model

            print(f"[{count}/{total}] latent={latent_dim}, "
                  f"g_dims={g_hidden_dims}, device={device_str}, "
                  f"params={features['total_params']:,}")

            train_times, infer_times = measure_times(
                model_fn, train_batches, device_str, latent_dim
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

        del train_batches
        if device_str == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {OUTPUT_PATH} ({len(df)}행)")


if __name__ == '__main__':
    run()
