import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """이미지를 패치로 나누고 각 패치를 벡터로 변환"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size({img_size})는 patch_size({patch_size})로 나누어 떨어져야 함"
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)   # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)         # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)    # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer 기본 블록: LayerNorm + Attention + MLP + 잔차 연결"""

    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """Vision Transformer (이미지 분류용)

    Args:
        img_size: 입력 이미지 크기 (기본 32 for CIFAR-10)
        patch_size: 패치 크기
        in_channels: 입력 채널 수
        num_classes: 분류 클래스 수
        embed_dim: 임베딩 차원
        num_layers: Transformer Encoder Block 수
        num_heads: Attention Head 수
    """

    def __init__(
        self,
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=128, num_layers=4, num_heads=4,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.num_layers = num_layers
        self.num_heads  = num_heads

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def create_transformer_variants(img_size=32, in_channels=3, num_classes=10):
    """다양한 Transformer 조합 생성

    Returns:
        list: (model, config_dict) 튜플 리스트
    """
    # (embed_dim, num_layers, num_heads, patch_size)
    # embed_dim  : 조각 1개를 숫자 몇 개로 표현 (클수록 풍부, 느림)
    # num_layers : 트랜스포머 층 개수 (많을수록 정확, 느림)
    # num_heads  : 몇 가지 관점으로 분석 (많을수록 다양, 느림)
    # patch_size : 이미지 조각 크기 (작을수록 세밀, 느림)
    configs = [
        (64,  2, 4, 4),
        (64,  4, 4, 4),
        (64,  2, 4, 8),
        (128, 2, 4, 4),
        (128, 4, 4, 4),
        (128, 6, 4, 4),
        (128, 2, 8, 4),
        (128, 4, 8, 4),
        (256, 2, 8, 4),
        (256, 4, 8, 4),
        (256, 6, 8, 4),
        (256, 4, 8, 8),
    ]

    variants = []
    for embed_dim, num_layers, num_heads, patch_size in configs:
        model = SimpleViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        config = {
            'embed_dim': embed_dim,
            'num_transformer_layers': num_layers,
            'num_heads': num_heads,
            'patch_size': patch_size,
        }
        variants.append((model, config))

    return variants
