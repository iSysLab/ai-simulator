# ============================================================
# 4단계: GAN 모델 정의
# ============================================================
#
# GAN(Generative Adversarial Network)이란?
#   두 개의 신경망(Generator, Discriminator)이 서로 경쟁하며 학습하는 구조.
#
#   - Generator (생성자): 노이즈를 입력받아 가짜 이미지를 생성
#   - Discriminator (판별자): 이미지를 보고 진짜/가짜를 판별
#
#   학습 과정:
#     Generator는 Discriminator를 속이려 하고,
#     Discriminator는 Generator를 꿰뚫으려 한다.
#     이 경쟁 과정에서 Generator가 점점 진짜 같은 이미지를 생성하게 됨.
#
# 이 연구에서의 목적:
#   GAN의 실행 시간을 측정하여 예측 모델의 학습 데이터로 사용.
#   (GAN이 이미지를 잘 생성하는지는 부차적 목표)
#
# 교수님 유의사항:
#   - 레이어 수, 히든 뉴런 수를 다양하게 구성할 것
#   - 상세한 한국어 주석 작성
#
# 하드웨어: MacBook Air M1 (CPU + MPS)
# 작성자: 김홍근
# ============================================================

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────
# Generator (생성자)
# ──────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    GAN의 Generator (생성자)

    역할: 무작위 노이즈 벡터를 입력받아 가짜 이미지를 생성

    구조:
      노이즈 벡터 (latent_dim 차원)
        ↓  Linear → BatchNorm → ReLU  (반복 × num_layers)
        ↓  Linear → Tanh
      출력 이미지 (img_size × img_size × channels)

    BatchNorm이란?
      각 레이어의 출력을 정규화하여 학습을 안정화시키는 기법.

    Tanh 활성화 함수:
      출력 범위를 -1~1로 제한. 이미지 픽셀값 범위와 맞춤.

    Args:
        latent_dim (int): 입력 노이즈 벡터의 차원 (64, 128, 256)
        img_size (int): 출력 이미지의 가로/세로 크기
        img_channels (int): 출력 이미지 채널 수 (흑백=1, 컬러=3)
        hidden_dims (list): 각 중간 레이어의 뉴런 수 리스트
    """
    def __init__(self, latent_dim=128, img_size=32, img_channels=3, hidden_dims=None):
        super().__init__()

        self.latent_dim   = latent_dim
        self.img_size     = img_size
        self.img_channels = img_channels
        self.output_dim   = img_size * img_size * img_channels

        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        self.hidden_dims = hidden_dims

        # 레이어 순서대로 쌓기
        layers = []
        in_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 정규화
                nn.ReLU(inplace=True),       # 활성화
            ])
            in_dim = hidden_dim

        # 최종 출력 레이어 (이미지 크기 × 채널 수)
        layers.extend([
            nn.Linear(in_dim, self.output_dim),
            nn.Tanh(),  # 출력 범위 -1~1 제한
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: 노이즈 벡터 (배치 크기, latent_dim)

        Returns:
            가짜 이미지 텐서 (배치 크기, 채널, 높이, 너비)
        """
        output = self.model(z)  # (배치, img_size×img_size×channels)
        # 이미지 형태로 reshape
        output = output.view(-1, self.img_channels, self.img_size, self.img_size)
        return output

    def get_model_info(self):
        """모델 구조 정보 반환 (Feature 추출용)"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_type': 'GAN_Generator',
            'latent_dim': self.latent_dim,
            'img_size': self.img_size,
            'img_channels': self.img_channels,
            'num_layers': len(self.hidden_dims) + 1,
            'hidden_dims': self.hidden_dims,
            'max_width': max(self.hidden_dims) if self.hidden_dims else self.latent_dim,
            'min_width': min(self.hidden_dims) if self.hidden_dims else self.latent_dim,
            'total_params': total_params,
        }


# ──────────────────────────────────────────────────────────
# Discriminator (판별자)
# ──────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    GAN의 Discriminator (판별자)

    역할: 이미지를 입력받아 진짜(1) / 가짜(0) 확률을 출력

    구조:
      이미지 (img_size × img_size × channels)
        ↓  Linear → LeakyReLU  (반복 × num_layers)
        ↓  Linear → Sigmoid
      출력: 진짜일 확률 (0~1)

    LeakyReLU란?
      ReLU와 비슷하지만 음수 입력에도 작은 기울기를 허용.
      GAN의 Discriminator에서 자주 사용됨 (기울기 소실 방지).

    Sigmoid:
      출력을 0~1 범위로 변환 (0: 가짜, 1: 진짜)

    Args:
        img_size (int): 입력 이미지 크기
        img_channels (int): 입력 이미지 채널 수
        hidden_dims (list): 각 중간 레이어의 뉴런 수 리스트
        negative_slope (float): LeakyReLU의 음수 기울기 계수
    """
    def __init__(self, img_size=32, img_channels=3, hidden_dims=None, negative_slope=0.2):
        super().__init__()

        self.img_size     = img_size
        self.img_channels = img_channels
        self.input_dim    = img_size * img_size * img_channels

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        self.hidden_dims = hidden_dims

        # 레이어 쌓기
        layers = []
        in_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(negative_slope, inplace=True),  # 음수 기울기 허용
            ])
            in_dim = hidden_dim

        # 최종 출력: 진짜/가짜 확률 (1개 값)
        layers.extend([
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),  # 0~1 범위로 변환
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        """
        Args:
            img: 이미지 텐서 (배치 크기, 채널, 높이, 너비)

        Returns:
            진짜일 확률 (배치 크기, 1)
        """
        # 이미지를 1D 벡터로 펼치기
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

    def get_model_info(self):
        """모델 구조 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_type': 'GAN_Discriminator',
            'img_size': self.img_size,
            'img_channels': self.img_channels,
            'num_layers': len(self.hidden_dims) + 1,
            'hidden_dims': self.hidden_dims,
            'max_width': max(self.hidden_dims) if self.hidden_dims else self.input_dim,
            'min_width': min(self.hidden_dims) if self.hidden_dims else 1,
            'total_params': total_params,
        }


# ──────────────────────────────────────────────────────────
# GAN 전체 (Generator + Discriminator 묶음)
# ──────────────────────────────────────────────────────────

class SimpleGAN(nn.Module):
    """
    Generator와 Discriminator를 묶은 전체 GAN 모듈

    시간 측정 시에는 Generator와 Discriminator를 분리하여 각각 측정.
    (실험에서 G_forward + D_forward + 손실 계산 + 역전파 전체를 측정)

    Args:
        latent_dim (int): 노이즈 벡터 차원
        img_size (int): 이미지 크기
        img_channels (int): 이미지 채널 수
        g_hidden_dims (list): Generator의 히든 레이어 뉴런 수
        d_hidden_dims (list): Discriminator의 히든 레이어 뉴런 수
    """
    def __init__(
        self,
        latent_dim=128, img_size=32, img_channels=3,
        g_hidden_dims=None, d_hidden_dims=None
    ):
        super().__init__()

        if g_hidden_dims is None:
            g_hidden_dims = [256, 512, 1024]
        if d_hidden_dims is None:
            d_hidden_dims = [1024, 512, 256]

        self.latent_dim   = latent_dim
        self.img_size     = img_size
        self.img_channels = img_channels

        self.generator     = Generator(latent_dim, img_size, img_channels, g_hidden_dims)
        self.discriminator = Discriminator(img_size, img_channels, d_hidden_dims)

    def get_model_info(self):
        """전체 GAN의 구조 정보 반환"""
        g_info = self.generator.get_model_info()
        d_info = self.discriminator.get_model_info()
        return {
            'model_type': 'GAN',
            'latent_dim': self.latent_dim,
            'img_size': self.img_size,
            'img_channels': self.img_channels,
            'g_hidden_dims': self.generator.hidden_dims,
            'd_hidden_dims': self.discriminator.hidden_dims,
            'g_total_params': g_info['total_params'],
            'd_total_params': d_info['total_params'],
            'total_params': g_info['total_params'] + d_info['total_params'],
            'num_layers': g_info['num_layers'],  # Generator 기준
            'max_width': g_info['max_width'],
            'min_width': g_info['min_width'],
        }


# ──────────────────────────────────────────────────────────
# 다양한 GAN 조합 생성 함수
# ──────────────────────────────────────────────────────────

def create_gan_variants(img_size=32, img_channels=3):
    """
    교수님 유의사항: 레이어 수, 히든 뉴런 수를 다양하게 구성

    조합 기준:
    - latent_dim: 64, 128, 256
    - Generator 히든 레이어: 다양한 크기
    - Discriminator는 Generator의 역순 구조 사용

    Returns:
        list: (GAN 인스턴스, 모델 정보 딕셔너리) 튜플 리스트
    """
    # (latent_dim, generator_hidden_dims) 조합
    configs = [
        # 소형 GAN
        (64,  [128, 256]),
        (64,  [128, 256, 512]),
        (128, [256, 512]),
        # 중형 GAN
        (128, [256, 512, 1024]),
        (128, [128, 256, 512, 256]),
        (256, [256, 512, 1024]),
        # 대형 GAN
        (256, [512, 1024, 512]),
        (256, [256, 512, 1024, 512]),
    ]

    variants = []
    for latent_dim, g_hidden_dims in configs:
        # Discriminator는 Generator의 역순
        d_hidden_dims = list(reversed(g_hidden_dims))

        gan = SimpleGAN(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            g_hidden_dims=g_hidden_dims,
            d_hidden_dims=d_hidden_dims,
        )
        info = gan.get_model_info()
        info['config_str'] = f"z{latent_dim}_G{'_'.join(str(d) for d in g_hidden_dims)}"
        variants.append((gan, info))

    return variants


# ──────────────────────────────────────────────────────────
# 직접 실행 시 모델 구조 확인
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    variants = create_gan_variants()
    print(f"총 {len(variants)}가지 GAN 조합 생성됨\n")
    for gan, info in variants:
        print(f"  {info['config_str']:40s} | "
              f"G params: {info['g_total_params']:>7,} | "
              f"D params: {info['d_total_params']:>7,} | "
              f"Total: {info['total_params']:>8,}")
