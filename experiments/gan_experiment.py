# ============================================================
# 4단계: GAN 모델 실행 시간 측정 실험
# ============================================================
#
# 목적:
#   다양한 구조의 GAN 모델을 MNIST로 학습하고,
#   CPU와 MPS에서의 추론 시간 및 학습 시간을 측정하여 CSV로 저장
#
# 데이터셋: MNIST (28×28 흑백 이미지)
#   - CIFAR-10보다 단순하여 GAN 학습이 빠름
#   - GAN의 성능(이미지 품질)보다 실행 시간 측정이 목적
#
# GAN 학습 특성:
#   - Generator와 Discriminator를 번갈아가며 학습
#   - 일반 분류 모델보다 학습이 불안정할 수 있음
#   - 실험 목적은 "얼마나 걸리는가"이므로 정확도 불필요
#
# 측정 방식 (교수님 유의사항):
#   - Warmup 5회 후 10회 반복 측정 → 평균값 사용
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
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gan_models import create_gan_variants

# ──────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RAW_DIR     = os.path.join(DATA_DIR, 'raw')
STAGE4_DIR  = os.path.join(DATA_DIR, 'stage4')
OUTPUT_FILE = os.path.join(STAGE4_DIR, 'gan_results.csv')

# GAN 학습 설정
LATENT_DIM_DEFAULT = 128  # 노이즈 벡터 기본 차원 (모델마다 달라짐)
IMG_SIZE    = 28           # MNIST 이미지 크기 (28×28)
IMG_CHANNELS = 1           # 흑백
BATCH_SIZE  = 64
EPOCHS_TRAIN = 3           # 시간 측정용 학습 epoch (짧게)

# ──────────────────────────────────────────────────────────
# MNIST 데이터 준비
# ──────────────────────────────────────────────────────────

def prepare_mnist_data(batch_size=64):
    """
    MNIST 데이터셋 로드

    GAN용 정규화: 픽셀값을 -1~1 범위로 변환 (Generator의 Tanh 출력과 맞춤)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 0~1 → -1~1 변환
    ])

    try:
        train_dataset = datasets.MNIST(RAW_DIR, train=True,  download=False, transform=transform)
    except Exception:
        print("  MNIST 다운로드 중...")
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = datasets.MNIST(RAW_DIR, train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader

# ──────────────────────────────────────────────────────────
# GAN 학습 함수
# ──────────────────────────────────────────────────────────

def train_gan(gan, train_loader, device='cpu', epochs=3):
    """
    GAN을 MNIST 데이터로 학습하는 함수

    GAN 학습 방식:
      1단계 (Discriminator 학습):
        - 진짜 이미지 → Discriminator → "진짜(1)" 판별 학습
        - 가짜 이미지 → Discriminator → "가짜(0)" 판별 학습

      2단계 (Generator 학습):
        - 가짜 이미지를 만들어 Discriminator에 통과
        - Discriminator가 "진짜(1)"라고 판별하도록 Generator 학습

    Args:
        gan: SimpleGAN 인스턴스
        train_loader: MNIST 데이터 로더
        device: 'cpu' 또는 'mps'
        epochs: 학습 epoch 수

    Returns:
        gan: 학습 완료된 GAN
    """
    gan = gan.to(device)

    # 손실 함수: Binary Cross Entropy (진짜/가짜 이진 분류)
    criterion = nn.BCELoss()

    # Generator와 Discriminator 각각의 옵티마이저
    g_optimizer = torch.optim.Adam(gan.generator.parameters(),     lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # betas=(0.5, 0.999): GAN 학습에서 흔히 사용하는 Adam 설정값

    for epoch in range(epochs):
        d_losses, g_losses = [], []

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)

            # ── 1단계: Discriminator 학습 ──────────────────
            d_optimizer.zero_grad()

            # 진짜 이미지에 대한 손실 (진짜=1로 판별하도록)
            real_labels = torch.ones(current_batch_size, 1, device=device)
            d_real_loss = criterion(gan.discriminator(real_imgs), real_labels)

            # 가짜 이미지 생성 후 손실 (가짜=0으로 판별하도록)
            noise = torch.randn(current_batch_size, gan.latent_dim, device=device)
            fake_imgs   = gan.generator(noise).detach()  # .detach(): G의 gradient 차단
            fake_labels = torch.zeros(current_batch_size, 1, device=device)
            d_fake_loss = criterion(gan.discriminator(fake_imgs), fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ── 2단계: Generator 학습 ───────────────────────
            g_optimizer.zero_grad()

            # 가짜 이미지를 생성하고 Discriminator가 진짜라고 판별하도록 학습
            noise    = torch.randn(current_batch_size, gan.latent_dim, device=device)
            fake_imgs = gan.generator(noise)
            g_loss    = criterion(gan.discriminator(fake_imgs), real_labels)  # 진짜처럼 보이도록

            g_loss.backward()
            g_optimizer.step()

            if device == 'mps':
                torch.mps.synchronize()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f'    Epoch {epoch+1}/{epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}')

    return gan

# ──────────────────────────────────────────────────────────
# GAN 시간 측정 함수
# ──────────────────────────────────────────────────────────

def measure_gan_time(gan, device, batch_size=64, warmup=5, n_runs=10):
    """
    GAN의 추론 시간과 학습 시간을 측정하는 함수

    GAN 추론 시간 = Generator가 노이즈 → 이미지를 생성하는 시간
    GAN 학습 시간 = G + D 한 번의 업데이트(미니배치 전체 루프) 시간

    Args:
        gan: SimpleGAN 인스턴스 (학습 완료 상태)
        device: 측정 디바이스
        batch_size: 한 번에 생성할 이미지 수
        warmup: 워밍업 실행 횟수
        n_runs: 실제 측정 횟수

    Returns:
        tuple: (추론_평균_ms, 추론_표준편차_ms, 학습_평균_초, 학습_표준편차_초)
    """
    gan.eval()  # 추론 모드 (BatchNorm, Dropout 비활성화)

    # ── 추론 시간 측정 (Generator forward pass) ────────────
    noise_batch = torch.randn(batch_size, gan.latent_dim, device=device)

    # 워밍업 (첫 실행은 캐시 없어 느림 → 제외)
    for _ in range(warmup):
        with torch.no_grad():
            _ = gan.generator(noise_batch)
        if device == 'mps':
            torch.mps.synchronize()

    # 본 측정
    inference_times = []
    for _ in range(n_runs):
        noise = torch.randn(batch_size, gan.latent_dim, device=device)
        if device == 'mps':
            torch.mps.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = gan.generator(noise)
        if device == 'mps':
            torch.mps.synchronize()
        end = time.perf_counter()

        inference_times.append((end - start) * 1000)  # ms 변환

    inf_mean = float(sum(inference_times) / len(inference_times))
    inf_std  = float((sum((t - inf_mean) ** 2 for t in inference_times) / len(inference_times)) ** 0.5)

    # ── 학습 시간 측정 (1 epoch G+D 업데이트 루프) ─────────
    gan.train()
    criterion   = nn.BCELoss()
    g_optimizer = torch.optim.Adam(gan.generator.parameters(),     lr=0.0002)
    d_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002)

    # 워밍업
    for _ in range(warmup):
        noise     = torch.randn(batch_size, gan.latent_dim, device=device)
        fake_imgs = gan.generator(noise)
        labels    = torch.ones(batch_size, 1, device=device)
        g_loss    = criterion(gan.discriminator(fake_imgs), labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if device == 'mps':
            torch.mps.synchronize()

    # 본 측정 (배치 1회 기준)
    training_times = []
    for _ in range(n_runs):
        noise     = torch.randn(batch_size, gan.latent_dim, device=device)
        real_imgs = torch.randn(batch_size, gan.img_channels,
                                gan.img_size, gan.img_size, device=device)
        real_labels = torch.ones(batch_size,  1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        if device == 'mps':
            torch.mps.synchronize()

        start = time.perf_counter()

        # D 업데이트
        d_optimizer.zero_grad()
        d_real = criterion(gan.discriminator(real_imgs), real_labels)
        fake   = gan.generator(noise).detach()
        d_fake = criterion(gan.discriminator(fake), fake_labels)
        (d_real + d_fake).backward()
        d_optimizer.step()

        # G 업데이트
        g_optimizer.zero_grad()
        noise2 = torch.randn(batch_size, gan.latent_dim, device=device)
        g_loss = criterion(gan.discriminator(gan.generator(noise2)), real_labels)
        g_loss.backward()
        g_optimizer.step()

        if device == 'mps':
            torch.mps.synchronize()

        end = time.perf_counter()
        training_times.append(end - start)

    tr_mean = float(sum(training_times) / len(training_times))
    tr_std  = float((sum((t - tr_mean) ** 2 for t in training_times) / len(training_times)) ** 0.5)

    return inf_mean, inf_std, tr_mean, tr_std

# ──────────────────────────────────────────────────────────
# 기존 측정 확인
# ──────────────────────────────────────────────────────────

def load_existing_configs(output_file):
    """이미 측정된 config 목록 로드"""
    if os.path.exists(output_file):
        df       = pd.read_csv(output_file)
        existing = set(df['config_str'].unique())
        next_idx = int(df['model_idx'].max()) + 1
        print(f"  기존 CSV: {len(df)}행, {len(existing)}가지 config 완료")
        return existing, next_idx
    print("  기존 CSV 없음 → 새로 시작")
    return set(), 0

# ──────────────────────────────────────────────────────────
# 메인 실험 함수
# ──────────────────────────────────────────────────────────

def run_experiment():
    """
    GAN 모델 실행 시간 측정 메인 함수

    측정 항목:
    - 추론 시간: Generator가 노이즈 → 이미지 생성 (batch=64 기준)
    - 학습 시간: G + D 미니배치 1회 업데이트 (초 단위)
    """
    print("=" * 70)
    print("  4단계: GAN 실행 시간 측정 실험")
    print("  데이터셋: MNIST (28×28 흑백)")
    print("  하드웨어: MacBook Air M1 (CPU + MPS)")
    print("=" * 70)

    os.makedirs(STAGE4_DIR, exist_ok=True)

    # 기존 측정 확인
    print("\n[Step 1] 기존 측정 데이터 확인")
    existing_configs, next_idx = load_existing_configs(OUTPUT_FILE)

    # GAN 조합 준비 (MNIST: 28×28 흑백)
    print("\n[Step 2] GAN 모델 조합 준비")
    all_variants = create_gan_variants(img_size=IMG_SIZE, img_channels=IMG_CHANNELS)
    new_variants = [
        (gan, info) for gan, info in all_variants
        if info['config_str'] not in existing_configs
    ]

    if not new_variants:
        print("  모든 조합이 이미 측정됨!")
        return pd.read_csv(OUTPUT_FILE)

    print(f"  전체: {len(all_variants)}개 | 새로 측정: {len(new_variants)}개")

    # MNIST 데이터 로드
    print("\n[Step 3] MNIST 데이터 로드")
    train_loader = prepare_mnist_data(batch_size=BATCH_SIZE)
    print("  MNIST 준비 완료")

    # 실험
    print("\n[Step 4] 실험 시작")
    devices     = ['cpu', 'mps']
    new_results = []

    for i, (_, info) in enumerate(new_variants):
        config_str  = info['config_str']
        current_idx = next_idx + i

        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(new_variants)}] {config_str}")
        print(f"  latent_dim={info['latent_dim']}, "
              f"G_dims={info['g_hidden_dims']}, "
              f"total_params={info['total_params']:,}")
        print(f"{'─' * 70}")

        for device in devices:
            print(f"\n  [디바이스: {device.upper()}]")

            # 새 GAN 인스턴스 생성
            from models.gan_models import SimpleGAN
            fresh_gan = SimpleGAN(
                latent_dim=info['latent_dim'],
                img_size=IMG_SIZE,
                img_channels=IMG_CHANNELS,
                g_hidden_dims=info['g_hidden_dims'],
                d_hidden_dims=info['d_hidden_dims'],
            )

            # 짧은 학습 (3 epoch)
            print("  학습 중 (3 epoch)...")
            trained_gan = train_gan(fresh_gan, train_loader, device=device, epochs=EPOCHS_TRAIN)

            # 시간 측정
            print("  시간 측정 중 (warmup 5회 + 10회 반복)...")
            inf_mean, inf_std, tr_mean, tr_std = measure_gan_time(
                trained_gan, device, batch_size=BATCH_SIZE, warmup=5, n_runs=10
            )

            result = {
                'model_idx': current_idx,
                'config_str': config_str,
                'model_type': 'GAN',
                'latent_dim': info['latent_dim'],
                'g_hidden_dims': str(info['g_hidden_dims']),
                'd_hidden_dims': str(info['d_hidden_dims']),
                'num_layers': info['num_layers'],
                'total_params': info['total_params'],
                'g_total_params': info['g_total_params'],
                'd_total_params': info['d_total_params'],
                'device': device,
                'inference_time_mean_ms': inf_mean,
                'inference_time_std_ms': inf_std,
                'training_time_mean_sec': tr_mean,
                'training_time_std_sec': tr_std,
                'dataset': 'MNIST',
                'img_size': IMG_SIZE,
                'img_channels': IMG_CHANNELS,
                'batch_size': BATCH_SIZE,
            }
            new_results.append(result)

            print(f"  추론(G forward): {inf_mean:.3f} ± {inf_std:.3f} ms")
            print(f"  학습(G+D 배치): {tr_mean:.4f} ± {tr_std:.4f} 초")

    # 저장
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
    print("  GAN 실험 완료!")
    print(f"{'=' * 70}")
    return new_df


if __name__ == '__main__':
    run_experiment()
