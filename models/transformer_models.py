# ============================================================
# 4단계: Transformer 모델 정의
# ============================================================
#
# 이 파일은 Vision Transformer(ViT) 계열의 소형 모델을 정의합니다.
#
# Transformer란?
#   원래 자연어 처리(번역, 텍스트 생성)에 쓰이던 구조.
#   2020년대부터 이미지 분류에도 사용되기 시작 (Vision Transformer = ViT).
#
#   핵심 아이디어:
#   - 이미지를 작은 패치(조각)로 나눔 (예: 32×32 이미지를 4×4 패치 64개로)
#   - 각 패치를 단어처럼 취급하여 Transformer로 처리
#   - 패치들 사이의 관계를 "Attention(주의)" 메커니즘으로 학습
#
# 교수님 유의사항:
#   - ResNet/MobileNet 자제 → 직접 구현한 Transformer 사용
#   - 레이어 수, 임베딩 차원, Head 수 등 다양하게 구성
#   - 상세한 한국어 주석 작성
#
# 하드웨어: MacBook Air M1 (CPU + MPS)
# 작성자: 김홍근
# ============================================================

import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────────────────
# 핵심 구성 요소 1: Patch Embedding (패치 임베딩)
# ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    이미지를 패치(작은 조각)로 나누고 각 패치를 벡터로 변환하는 모듈

    예시:
      입력: 32×32×3 (CIFAR-10 이미지)
      patch_size=4로 설정 시:
        → 32/4 × 32/4 = 8×8 = 64개의 패치 생성
        → 각 패치는 4×4×3 = 48차원 벡터
        → Linear 레이어로 embed_dim 차원으로 변환

    Args:
        img_size (int): 입력 이미지의 가로/세로 크기 (정사각형 가정)
        patch_size (int): 패치의 가로/세로 크기
        in_channels (int): 입력 채널 수 (흑백=1, 컬러=3)
        embed_dim (int): 각 패치를 변환할 임베딩 차원
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size   = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim  = embed_dim

        # 이미지에서 패치 수 계산
        # 예: 32×32 이미지, 패치 크기 4 → 64개 패치
        assert img_size % patch_size == 0, \
            f"이미지 크기({img_size})는 패치 크기({patch_size})로 나누어 떨어져야 합니다"
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d를 이용해 패치 추출 + 임베딩을 한 번에 처리
        # kernel_size=patch_size, stride=patch_size → 겹치지 않는 패치 추출
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (배치 크기, 채널, 높이, 너비)
        x = self.projection(x)  # (배치, embed_dim, H/patch, W/patch)
        x = x.flatten(2)        # (배치, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (배치, num_patches, embed_dim)
        return x


# ──────────────────────────────────────────────────────────
# 핵심 구성 요소 2: Multi-Head Self-Attention
# ──────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (다중 헤드 자기 주의) 모듈

    Attention이란?
      패치들 사이의 관계를 학습하는 메커니즘.
      예: 하늘 패치가 있을 때, 구름 패치에 더 많이 "주의"를 기울이는 것.

    "Multi-Head"란?
      여러 개의 Attention을 병렬로 수행하여 다양한 관계를 학습.
      예: Head 1은 색상 관계, Head 2는 형태 관계를 학습할 수 있음.

    Args:
        embed_dim (int): 임베딩 차원
        num_heads (int): Attention Head 수 (embed_dim이 num_heads로 나누어져야 함)
        dropout (float): Dropout 비율
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim({embed_dim})은 num_heads({num_heads})로 나누어 떨어져야 합니다"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads  # 각 헤드의 차원

        # Q(Query), K(Key), V(Value) 생성을 위한 Linear 레이어
        # 입력 → Q, K, V 세 가지 벡터로 변환
        self.qkv        = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)
        self.dropout    = nn.Dropout(dropout)

        # Attention 스코어를 균일하게 만들기 위한 스케일 인자
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape  # (배치, 패치 수, 임베딩 차원)

        # Q, K, V 계산: (배치, 패치 수, 3 × 임베딩 차원)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, 배치, 헤드, 패치, 헤드차원)
        q, k, v = qkv.unbind(0)            # 각각 (배치, 헤드, 패치, 헤드차원)

        # Scaled Dot-Product Attention
        # 각 패치가 다른 패치들에 얼마나 주의를 기울이는지 계산
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (배치, 헤드, 패치, 패치)
        attn = attn.softmax(dim=-1)                     # 확률값으로 변환 (합=1)
        attn = self.dropout(attn)

        # 주의 가중치와 V를 곱하여 최종 출력 계산
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.out_proj(x)
        return x


# ──────────────────────────────────────────────────────────
# 핵심 구성 요소 3: Transformer Encoder Block
# ──────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Transformer의 기본 단위 블록

    구조:
      입력
       ↓  LayerNorm (정규화)
       ↓  Multi-Head Attention
       ↓  + 잔차 연결 (Residual Connection: 입력을 그대로 더함)
       ↓  LayerNorm (정규화)
       ↓  Feed-Forward Network (MLP: 두 개의 Linear 레이어)
       ↓  + 잔차 연결
      출력

    잔차 연결이란?
      출력 = 입력 + 레이어(입력)
      → 기울기 소실 문제 해결, 깊은 네트워크 학습 안정화

    Args:
        embed_dim (int): 임베딩 차원
        num_heads (int): Attention Head 수
        mlp_ratio (float): MLP 중간 차원 비율 (embed_dim × mlp_ratio)
        dropout (float): Dropout 비율
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        # 첫 번째 LayerNorm + Attention
        self.norm1   = nn.LayerNorm(embed_dim)
        self.attn    = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 두 번째 LayerNorm + MLP (Feed-Forward Network)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.mlp     = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),                              # 활성화 함수 (Transformer에서 주로 사용)
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention + 잔차 연결
        x = x + self.attn(self.norm1(x))
        # MLP + 잔차 연결
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────
# 전체 모델: SimpleViT (Simple Vision Transformer)
# ──────────────────────────────────────────────────────────

class SimpleViT(nn.Module):
    """
    직접 구현한 소형 Vision Transformer (이미지 분류용)

    교수님 유의사항에 따라 ResNet/MobileNet 대신 직접 구현.
    레이어 수, 임베딩 차원, Head 수를 다양하게 구성하여 데이터 수집.

    Args:
        img_size (int): 입력 이미지 크기 (기본 32 for CIFAR-10)
        patch_size (int): 패치 크기 (4 또는 8)
        in_channels (int): 입력 채널 수
        num_classes (int): 분류 클래스 수
        embed_dim (int): 임베딩 차원 (64, 128, 256)
        num_layers (int): Transformer Encoder Block 수 (2, 4, 6)
        num_heads (int): Attention Head 수 (4, 8)
        mlp_ratio (float): MLP 중간 차원 비율
        dropout (float): Dropout 비율
    """
    def __init__(
        self,
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=128, num_layers=4, num_heads=4,
        mlp_ratio=4.0, dropout=0.0
    ):
        super().__init__()

        self.img_size   = img_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.num_layers = num_layers
        self.num_heads  = num_heads

        # 1. Patch Embedding: 이미지 → 패치 벡터 시퀀스
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        # 2. CLS 토큰: 전체 이미지를 대표하는 특수 토큰
        #    (Transformer 출력에서 이 토큰만 꺼내서 분류에 사용)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Embedding: 각 패치의 위치 정보를 더해줌
        #    (Attention은 순서를 모르기 때문에 위치 정보를 별도로 추가)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder Blocks (num_layers개 반복)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # 5. 최종 정규화 + 분류 헤드
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 파라미터 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]  # 배치 크기

        # 패치 임베딩 (배치, num_patches, embed_dim)
        x = self.patch_embed(x)

        # CLS 토큰을 배치 크기에 맞게 복사 후 앞에 붙임
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (배치, num_patches+1, embed_dim)

        # 위치 임베딩 더하기
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer Encoder Blocks 통과
        x = self.blocks(x)
        x = self.norm(x)

        # CLS 토큰만 꺼내서 분류
        cls_output = x[:, 0]          # (배치, embed_dim)
        output     = self.head(cls_output)  # (배치, num_classes)
        return output

    def get_model_info(self):
        """모델 구조 정보 반환 (Feature 추출용)"""
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_patches      = self.patch_embed.num_patches

        return {
            'model_type': 'Transformer',
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_patches': num_patches,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


# ──────────────────────────────────────────────────────────
# 다양한 Transformer 조합 생성 함수
# ──────────────────────────────────────────────────────────

def create_transformer_variants(img_size=32, in_channels=3, num_classes=10):
    """
    교수님 유의사항: 레이어 수, 임베딩 차원, Head 수를 다양하게 구성

    측정할 조합:
    - embed_dim: 64, 128, 256
    - num_layers: 2, 4, 6
    - num_heads: 4, 8 (embed_dim이 num_heads로 나누어져야 함)
    - patch_size: 4, 8

    Returns:
        list: (모델 인스턴스, 모델 정보 딕셔너리) 튜플 리스트
    """
    # (embed_dim, num_layers, num_heads, patch_size) 조합
    # embed_dim >= num_heads × 8 인 조합만 사용 (최소 head_dim=8 확보)
    configs = [
        # 소형 (빠른 측정용)
        (64,  2, 4, 4),
        (64,  4, 4, 4),
        (64,  2, 4, 8),

        # 중형
        (128, 2, 4, 4),
        (128, 4, 4, 4),
        (128, 6, 4, 4),
        (128, 2, 8, 4),
        (128, 4, 8, 4),

        # 대형
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
        info = model.get_model_info()
        info['config_str'] = f"dim{embed_dim}_L{num_layers}_H{num_heads}_P{patch_size}"
        variants.append((model, info))

    return variants


# ──────────────────────────────────────────────────────────
# 직접 실행 시 모델 구조 확인
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    variants = create_transformer_variants()
    print(f"총 {len(variants)}가지 Transformer 조합 생성됨\n")
    for model, info in variants:
        print(f"  {info['config_str']:30s} | params: {info['total_params']:>8,} | "
              f"patches: {info['num_patches']}")
