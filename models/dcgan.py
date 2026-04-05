"""DCGAN (Deep Convolutional GAN) 모델 정의

DCGAN은 2015년 Radford et al.이 발표한 GAN의 개선 버전.
기존 GAN의 Linear 레이어 대신 Conv/ConvTranspose 레이어를 사용.

── 기존 GAN과 비교 ───────────────────────────────────────
  SimpleGAN:  Linear 레이어로 노이즈 → 이미지 생성 (1D 처리)
  DCGAN:      ConvTranspose 레이어로 공간적 구조 유지 (2D 처리)
  → DCGAN이 이미지 품질이 더 높고 학습이 안정적

── 핵심 구성 요소 ────────────────────────────────────────
  Generator:
    노이즈 벡터 (latent_dim,)
    → ConvTranspose2d + BatchNorm + ReLU (업샘플링)
    → 이미지 (C, H, W)

  Discriminator:
    이미지 (C, H, W)
    → Conv2d + BatchNorm + LeakyReLU (다운샘플링)
    → 진짜/가짜 확률 (1,)

── 입력 처리 흐름 (Generator) ───────────────────────────
  z (B, latent_dim)
    → reshape (B, latent_dim, 1, 1)
    → ConvTranspose2d × N (업샘플링: 1×1 → H×W)
    → Tanh → 이미지 (B, C, H, W)

── MNIST 기준 출력 크기 ──────────────────────────────────
  28×28 이미지 생성을 위해
  ConvTranspose 레이어로 1×1 → 7×7 → 14×14 → 28×28 업샘플링
"""

import torch.nn as nn


class DCGANGenerator(nn.Module):
    """DCGAN Generator: 노이즈 → 이미지 (ConvTranspose 기반)

    Args:
        latent_dim (int): 입력 노이즈 벡터 차원
        base_channels (int): 기본 채널 수 (레이어마다 절반씩 줄어듦)
        img_channels (int): 출력 이미지 채널 수
    """

    def __init__(self, latent_dim=100, base_channels=64, img_channels=1):
        super().__init__()

        # 1×1 → 7×7 → 14×14 → 28×28 업샘플링 (MNIST 기준)
        self.net = nn.Sequential(
            # (B, latent_dim, 1, 1) → (B, base_channels*4, 7, 7)
            nn.ConvTranspose2d(latent_dim, base_channels * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # (B, base_channels*4, 7, 7) → (B, base_channels*2, 14, 14)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # (B, base_channels*2, 14, 14) → (B, img_channels, 28, 28)
            nn.ConvTranspose2d(base_channels * 2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # 출력 범위 [-1, 1]
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)  # (B, latent_dim) → (B, latent_dim, 1, 1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator: 이미지 → 진짜/가짜 확률 (Conv 기반)

    Args:
        base_channels (int): 기본 채널 수 (레이어마다 2배씩 증가)
        img_channels (int): 입력 이미지 채널 수
    """

    def __init__(self, base_channels=64, img_channels=1):
        super().__init__()

        # 28×28 → 14×14 → 7×7 → 1×1 다운샘플링 (MNIST 기준)
        self.net = nn.Sequential(
            # (B, img_channels, 28, 28) → (B, base_channels, 14, 14)
            nn.Conv2d(img_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, base_channels, 14, 14) → (B, base_channels*2, 7, 7)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, base_channels*2, 7, 7) → (B, 1, 1, 1)
            nn.Conv2d(base_channels * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid(),  # 진짜/가짜 확률
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)


class SimpleDCGAN(nn.Module):
    """DCGAN Generator + Discriminator 묶음

    Args:
        latent_dim (int): 노이즈 벡터 차원
        base_channels (int): 기본 채널 수
        img_channels (int): 이미지 채널 수
    """

    def __init__(self, latent_dim=100, base_channels=64, img_channels=1):
        super().__init__()
        self.latent_dim    = latent_dim
        self.generator     = DCGANGenerator(latent_dim, base_channels, img_channels)
        self.discriminator = DCGANDiscriminator(base_channels, img_channels)

    def forward(self, z):
        return self.generator(z)


def create_dcgan_variants(img_channels=1):
    """다양한 DCGAN 조합 생성

    latent_dim과 base_channels 조합으로 Small/Base/Large 변형 생성.

    Returns:
        list: (dcgan, config_dict) 튜플 리스트
    """
    # (latent_dim, base_channels)
    configs = [
        (64,  32),   # Small - 경량
        (100, 64),   # Base  - 기본 (논문 기준)
        (128, 128),  # Large - 대형
    ]

    variants = []
    for latent_dim, base_channels in configs:
        dcgan = SimpleDCGAN(
            latent_dim=latent_dim,
            base_channels=base_channels,
            img_channels=img_channels,
        )
        config = {
            'latent_dim':    latent_dim,
            'base_channels': base_channels,
            'g_hidden_dims': [base_channels * 4, base_channels * 2],  # extractor 호환용
        }
        variants.append((dcgan, config))

    return variants
