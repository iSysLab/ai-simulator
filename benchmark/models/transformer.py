"""Vision Transformer (ViT) 모델 — CIFAR-10/MNIST 대응

이미지를 패치로 나누고 Transformer Encoder로 분류.
khg9859 브랜치의 SimpleViT를 모듈형 구조에 맞게 포팅.
"""
import math
import torch
import torch.nn as nn
from .registry import register_model


class PatchEmbedding(nn.Module):
    """이미지를 패치로 나누고 임베딩 벡터로 변환"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)   # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)         # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)    # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.out_proj(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """LayerNorm → Attention → 잔차 → LayerNorm → MLP → 잔차"""

    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """소형 Vision Transformer (이미지 분류용)

    Args:
        img_size: 입력 이미지 크기 (정사각형)
        patch_size: 패치 크기
        in_channels: 입력 채널 수
        num_classes: 분류 클래스 수
        embed_dim: 임베딩 차원
        num_layers: Encoder Block 수
        num_heads: Attention Head 수
        mlp_ratio: MLP 중간 차원 비율
        dropout: Dropout 비율
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=128, num_layers=4,
                 num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
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
        x = self.pos_dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


@register_model('transformer')
def create_transformer(img_size=32, patch_size=4, in_channels=3,
                       num_classes=10, embed_dim=128, num_layers=4,
                       num_heads=4, mlp_ratio=4.0, dropout=0.0):
    return SimpleViT(
        img_size=img_size, patch_size=patch_size,
        in_channels=in_channels, num_classes=num_classes,
        embed_dim=embed_dim, num_layers=num_layers,
        num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
    )
