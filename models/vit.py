"""ViT (Vision Transformer) 모델 정의

ViT는 Google Brain이 2020년에 발표한 표준 Vision Transformer.
이미지를 패치로 나눠 Transformer Encoder에 입력하는 구조.
timm 라이브러리에서 Small / Base / Large 변형을 가져와 사용.

── 핵심 아이디어 ──────────────────────────────────────────
이미지를 NLP의 토큰처럼 취급:
  - 이미지를 고정 크기 패치로 분할 (예: 16×16)
  - 각 패치를 1D 벡터로 변환 (Linear Projection)
  - [CLS] 토큰 추가 (분류용 특수 토큰)
  - Transformer Encoder로 패치 간 관계 학습

── Swin과 비교 ────────────────────────────────────────────
  ViT:  모든 패치 쌍 비교 (전역 Attention) O(N²), 단일 해상도
  Swin: 윈도우 내 비교 (지역 Attention)  O(N),  계층적 구조
  → ViT는 구조 단순, Swin은 효율적

── 입력 처리 흐름 ────────────────────────────────────────
  이미지 (B, C, H, W)
    → 패치 분할 + Linear Projection (B, num_patches, embed_dim)
    → [CLS] 토큰 추가 (B, num_patches+1, embed_dim)
    → Position Embedding 추가
    → Transformer Encoder × N (Multi-Head Attention + FFN)
    → [CLS] 토큰 추출
    → 분류 헤드 (num_classes)

── timm 라이브러리 사용 ──────────────────────────────────
  pretrained=False: 가중치 랜덤 초기화 (구조만 사용)
  img_size=224: ViT는 224×224 기준 설계

── 모델 변형 ─────────────────────────────────────────────
  vit_small_patch16_224: embed=384, heads=6,  layers=12 - 경량
  vit_base_patch16_224:  embed=768, heads=12, layers=12 - 기본
  vit_large_patch16_224: embed=1024,heads=16, layers=24 - 대형
"""

import timm


def create_vit_variants():
    """ViT 변형 모델 목록 반환

    CIFAR-10 기준 (224×224 리사이즈, 3채널, 10클래스).
    Small → Base → Large 순으로 크기 증가.

    Returns:
        list: (model, config_dict) 튜플 리스트
    """
    # (model_name, embed_dim, num_layers, num_heads, patch_size)
    configs = [
        ('vit_small_patch16_224', 384,  12, 6,  16),  # Small - 경량
        ('vit_base_patch16_224',  768,  12, 12, 16),  # Base  - 기본
        ('vit_large_patch16_224', 1024, 24, 16, 16),  # Large - 대형
    ]

    variants = []
    for model_name, embed_dim, num_layers, num_heads, patch_size in configs:
        # pretrained=False: 랜덤 가중치 (구조만 사용)
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=10,
            img_size=224,
        )

        config = {
            'model_name':             model_name,
            'embed_dim':              embed_dim,     # 패치 임베딩 차원
            'num_transformer_layers': num_layers,    # Transformer 블록 수
            'num_heads':              num_heads,     # Attention 헤드 수
            'patch_size':             patch_size,    # 패치 크기 (픽셀)
            'has_attention':          1,
            'has_cls_token':          1,             # ViT는 [CLS] 토큰 사용
            'has_residual':           1,             # 각 블록에 residual 있음
            'has_batchnorm':          0,
            'has_pooling':            0,
        }
        variants.append((model, config))

    return variants
