"""GAN (Generative Adversarial Network) 모델

Generator + Discriminator 쌍으로 구성.
khg9859 브랜치의 SimpleGAN을 모듈형 구조에 맞게 포팅.
"""
import torch.nn as nn
from .registry import register_model


class Generator(nn.Module):
    """노이즈 벡터 → 가짜 이미지 생성

    Args:
        latent_dim: 입력 노이즈 벡터 차원
        img_size: 출력 이미지 크기 (정사각형)
        img_channels: 출력 채널 수
        hidden_dims: 각 히든 레이어 뉴런 수 리스트
    """

    def __init__(self, latent_dim=128, img_size=32, img_channels=3,
                 hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.output_dim = img_size * img_size * img_channels

        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
        self.hidden_dims = hidden_dims

        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ])
            in_dim = h

        layers.extend([
            nn.Linear(in_dim, self.output_dim),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    """이미지 → 진짜/가짜 확률 판별

    Args:
        img_size: 입력 이미지 크기
        img_channels: 입력 채널 수
        hidden_dims: 히든 레이어 뉴런 수 리스트
    """

    def __init__(self, img_size=32, img_channels=3, hidden_dims=None):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.input_dim = img_size * img_size * img_channels

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        self.hidden_dims = hidden_dims

        layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = h

        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


class SimpleGAN(nn.Module):
    """Generator + Discriminator를 묶은 GAN 모듈

    Args:
        latent_dim: 노이즈 벡터 차원
        img_size: 이미지 크기
        img_channels: 채널 수
        g_hidden_dims: Generator 히든 레이어
        d_hidden_dims: Discriminator 히든 레이어
    """

    def __init__(self, latent_dim=128, img_size=32, img_channels=3,
                 g_hidden_dims=None, d_hidden_dims=None):
        super().__init__()
        if g_hidden_dims is None:
            g_hidden_dims = [256, 512, 1024]
        if d_hidden_dims is None:
            d_hidden_dims = list(reversed(g_hidden_dims))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels

        self.generator = Generator(latent_dim, img_size, img_channels, g_hidden_dims)
        self.discriminator = Discriminator(img_size, img_channels, d_hidden_dims)

    def forward(self, z):
        """Generator forward만 (벤치마크용)"""
        return self.generator(z)


@register_model('gan')
def create_gan(latent_dim=128, img_size=32, img_channels=3,
               g_hidden_dims=None, d_hidden_dims=None):
    return SimpleGAN(
        latent_dim=latent_dim, img_size=img_size,
        img_channels=img_channels,
        g_hidden_dims=g_hidden_dims, d_hidden_dims=d_hidden_dims,
    )
