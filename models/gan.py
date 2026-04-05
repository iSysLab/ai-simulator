import torch
import torch.nn as nn


class Generator(nn.Module):
    """GAN Generator: 노이즈 → 가짜 이미지

    Args:
        latent_dim: 입력 노이즈 벡터 차원
        img_size: 출력 이미지 크기
        img_channels: 출력 이미지 채널 수
        hidden_dims: 중간 레이어 뉴런 수 리스트
    """

    def __init__(self, latent_dim=128, img_size=28, img_channels=1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        self.latent_dim   = latent_dim
        self.img_size     = img_size
        self.img_channels = img_channels
        self.hidden_dims  = hidden_dims
        self.output_dim   = img_size * img_size * img_channels # 숫자 이미지를 한 줄로 표현

        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, self.output_dim), nn.Tanh()]
        self.model = nn.Sequential(*layers) # 순서대로 묶음

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    """GAN Discriminator: 이미지 → 진짜/가짜 확률

    Args:
        img_size: 입력 이미지 크기
        img_channels: 입력 이미지 채널 수
        hidden_dims: 중간 레이어 뉴런 수 리스트
    """

    def __init__(self, img_size=28, img_channels=1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        self.input_dim = img_size * img_size * img_channels

        layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2, inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


class SimpleGAN(nn.Module):
    """Generator + Discriminator 묶음

    Args:
        latent_dim: 노이즈 벡터 차원
        img_size: 이미지 크기
        img_channels: 이미지 채널 수
        g_hidden_dims: Generator 히든 레이어
        d_hidden_dims: Discriminator 히든 레이어
    """

    def __init__(
        self,
        latent_dim=128, img_size=28, img_channels=1,
        g_hidden_dims=None, d_hidden_dims=None
    ):
        super().__init__()
        if g_hidden_dims is None:
            g_hidden_dims = [256, 512, 1024]
        if d_hidden_dims is None:
            d_hidden_dims = list(reversed(g_hidden_dims))

        self.latent_dim = latent_dim # 클수록 다양한 이미지 생성
        self.generator     = Generator(latent_dim, img_size, img_channels, g_hidden_dims)
        self.discriminator = Discriminator(img_size, img_channels, d_hidden_dims)

    def forward(self, z):
        return self.generator(z)


def create_gan_variants(img_size=28, img_channels=1):
    """다양한 GAN 조합 생성

    Returns:
        list: (gan, config_dict) 튜플 리스트
    """
    # (latent_dim, g_hidden_dims)
    configs = [
        (64,  [128, 256]),
        (64,  [128, 256, 512]),
        (128, [256, 512]),
        (128, [256, 512, 1024]),
        (128, [128, 256, 512, 256]),
        (256, [256, 512, 1024]),
        (256, [512, 1024, 512]),
        (256, [256, 512, 1024, 512]),
    ]

    variants = []
    for latent_dim, g_hidden_dims in configs:
        d_hidden_dims = list(reversed(g_hidden_dims))
        gan = SimpleGAN(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            g_hidden_dims=g_hidden_dims, # 이미지 생성
            d_hidden_dims=d_hidden_dims, # 진짜/가짜 판별
        )
        config = {
            'latent_dim': latent_dim,
            'g_hidden_dims': g_hidden_dims,
        }
        variants.append((gan, config))

    return variants
