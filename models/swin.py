"""Swin Transformer 모델 정의

Swin Transformer는 Microsoft가 2021년에 발표한 계층적 Vision Transformer.
timm 라이브러리에서 Tiny/Small/Base 변형을 가져와 사용.

── 핵심 아이디어 ──────────────────────────────────────────
기존 ViT와 달리 두 가지 핵심 개념을 도입:

  1. Window-based Attention (윈도우 기반 어텐션)
     이미지를 고정 크기 윈도우(7×7)로 나눠 윈도우 내에서만 Attention 수행
     → 전체 패치 쌍 비교 O(N²) → 윈도우 내 비교 O(N) 선형 복잡도

  2. Shifted Window (이동 윈도우)
     레이어마다 윈도우를 절반씩 이동시켜 윈도우 간 정보 교환
     → 지역 정보와 전역 정보를 모두 포착 가능

── ViT와 비교 ────────────────────────────────────────────
  ViT:  모든 패치 쌍 비교 (전역 Attention), 고정 해상도
  Swin: 윈도우 내 비교 (지역 Attention), 계층적 구조
  → Swin이 더 효율적이고 고해상도 이미지에 유리

── 계층적 구조 ───────────────────────────────────────────
  Stage 1: 패치 분할 (H/4 × W/4)
  Stage 2: 패치 병합 (H/8 × W/8)
  Stage 3: 패치 병합 (H/16 × W/16)
  Stage 4: 패치 병합 (H/32 × W/32)
  → CNN처럼 계층적으로 해상도 줄이고 채널 늘림

── timm 라이브러리 사용 ──────────────────────────────────
  pretrained=False: 가중치 랜덤 초기화 (구조만 사용)
  img_size=32: CIFAR-10 입력 크기에 맞게 설정

── 모델 변형 ─────────────────────────────────────────────
  swin_tiny:  채널 96,  레이어 [2,2,6,2]   - 경량
  swin_small: 채널 96,  레이어 [2,2,18,2]  - 중간
  swin_base:  채널 128, 레이어 [2,2,18,2]  - 대형
"""

import timm


def create_swin_variants():
    """Swin Transformer 변형 모델 목록 반환

    CIFAR-10 기준 (32×32 입력, 3채널, 10클래스).
    Tiny → Small → Base 순으로 크기 증가.

    Returns:
        list: (model, config_dict) 튜플 리스트
    """
    # (model_name, embed_dim, num_layers, num_heads, window_size)
    configs = [
        ('swin_tiny_patch4_window7_224',  96,  2, 3, 7),   # Tiny  - 경량
        ('swin_small_patch4_window7_224', 96,  2, 3, 7),   # Small - 중간
        ('swin_base_patch4_window7_224',  128, 2, 4, 7),   # Base  - 대형
    ]

    variants = []
    for model_name, embed_dim, num_layers, num_heads, window_size in configs:
        # pretrained=False: 랜덤 가중치 (구조만 사용)
        # img_size=224: Swin은 224×224 기준 설계
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=10,
            img_size=224,
        )

        config = {
            'model_name':            model_name,
            'embed_dim':             embed_dim,       # 초기 채널 수
            'num_transformer_layers': num_layers,     # Swin Block 수 (Stage 1 기준)
            'num_heads':             num_heads,       # Attention 헤드 수
            'patch_size':            4,               # 초기 패치 크기
            'window_size':           window_size,     # Attention 윈도우 크기
            'has_attention':         1,
            'has_residual':          1,               # 각 블록에 residual 있음
            'has_batchnorm':         0,
            'has_pooling':           0,
        }
        variants.append((model, config))

    return variants
