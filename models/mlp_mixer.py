"""MLP-Mixer 모델 정의

MLP-Mixer는 Google Brain이 2021년에 발표한 모델로,
Transformer의 Attention 메커니즘 없이 MLP(다층 퍼셉트론)만으로
이미지 분류를 수행한다.

── 핵심 아이디어 ──────────────────────────────────────────
이미지를 패치로 나눈 뒤 두 종류의 MLP를 교대로 적용:

  1. Token Mixing MLP: 패치 간 정보를 섞음 (공간 정보)
     → 여러 위치의 패치들이 서로 정보를 교환
  2. Channel Mixing MLP: 채널 간 정보를 섞음 (채널 정보)
     → 같은 위치에서 다른 특징들끼리 정보를 교환

── Transformer와 비교 ────────────────────────────────────
  Transformer: Attention(패치 간) + FFN(채널 간)
  MLP-Mixer:   Token MLP(패치 간) + Channel MLP(채널 간)
  → Attention 없이도 비슷한 효과

── 입력 처리 흐름 ────────────────────────────────────────
  이미지 (B, C, H, W)
    → 패치 분할 (B, num_patches, patch_dim)
    → Linear Projection (B, num_patches, hidden_dim)
    → Mixer Layer × N
       ├─ Token Mixing: (B, hidden_dim, num_patches) → MLP → 원복
       └─ Channel Mixing: (B, num_patches, hidden_dim) → MLP → 원복
    → Global Average Pooling
    → 분류 (num_classes)

── timm 라이브러리 사용 ──────────────────────────────────
  timm(PyTorch Image Models)은 이미지 모델 라이브러리.
  pretrained=False: 가중치를 랜덤 초기화 (구조만 가져옴)
  → 실행 시간 측정 목적이라 학습된 가중치 불필요

── 모델 변형 ─────────────────────────────────────────────
  mixer_s16_224: Small,  patch=16, hidden=512,  layers=8
  mixer_b16_224: Base,   patch=16, hidden=768,  layers=12
  mixer_l16_224: Large,  patch=16, hidden=1024, layers=24
"""

import timm


def create_mlp_mixer_variants():
    """MLP-Mixer 변형 모델 목록 반환

    Small → Base → Large 순으로 크기가 증가.
    patch_size=16: 224×224 이미지를 16×16 패치로 나누면 14×14=196개 패치

    Returns:
        list: (model, config_dict) 튜플 리스트
    """
    configs = [
        # (model_name, num_layers, hidden_dim, patch_size)
        ('mixer_s16_224', 8,  512,  16),  # Small  - 경량 모델
        ('mixer_b16_224', 12, 768,  16),  # Base   - 기본 모델
        ('mixer_l16_224', 24, 1024, 16),  # Large  - 대형 모델
    ]

    variants = []
    for model_name, num_layers, hidden_dim, patch_size in configs:
        # pretrained=False: 랜덤 가중치로 초기화 (구조만 사용)
        model = timm.create_model(model_name, pretrained=False, num_classes=10)

        config = {
            'model_name':  model_name,
            'num_layers':  num_layers,   # Mixer Layer 수
            'hidden_dim':  hidden_dim,   # 패치 임베딩 차원
            'patch_size':  patch_size,   # 패치 크기 (픽셀)
            'has_residual': 1,           # Mixer Layer 내부에 residual connection 있음
            'has_batchnorm': 0,
            'has_pooling': 0,
        }
        variants.append((model, config))

    return variants
