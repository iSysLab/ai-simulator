import torch
import torch.nn as nn


class SimpleANN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128], num_classes=10):
        """
        단순한 ANN 모델

        Args:
            input_size: 입력 크기 (MNIST의 경우 28*28=784)
            hidden_sizes: 히든 레이어 크기 리스트 [128, 256] 등
            num_classes: 출력 클래스 수 (MNIST의 경우 10)
        """
        super(SimpleANN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # 레이어 구성
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 입력을 flatten
        x = x.view(-1, self.input_size)
        return self.network(x)

    def get_model_info(self):
        """
        모델 구조 정보 반환
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'num_layers': len(self.hidden_sizes) + 1,
            'hidden_sizes': self.hidden_sizes,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def create_model_variants():
    """
    다양한 구조의 모델 생성 (기존 1단계에서 사용한 14가지 조합)
    Layer 수와 width를 변경하면서 생성
    """
    variants = []

    # Layer 수: 2, 3, 4 / Width: 64, 128, 256, 512
    layer_configs = [
        [64],
        [128],
        [256],
        [512],
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512],
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
        [128, 256, 128],
        [64, 128, 64],
        [256, 512, 256],
    ]

    for config in layer_configs:
        model = SimpleANN(hidden_sizes=config)
        model_info = model.get_model_info()
        model_info['config'] = config
        variants.append((model, model_info))

    return variants


def create_extended_variants():
    """
    3단계 개선을 위한 추가 ANN 모델 조합 생성 (신규 12가지)

    추가 목적:
    - 교수님 유의사항: 레이어 수와 뉴런 수를 다양하게 구성할 것
    - Width 32 (매우 작은 모델): GPU 오버헤드가 더 극명하게 드러남
    - Width 1024 (매우 큰 모델): 파라미터 수 범위를 크게 확장
    - 5층 레이어 (4개 히든 레이어): 깊이 방향 다양성 확보
    - 피라미드/역피라미드: 실제 DNN 설계에서 자주 쓰이는 패턴

    Returns:
        list: (모델 인스턴스, 모델 정보 딕셔너리) 튜플 리스트
    """
    new_configs = [
        # ── Width 32 (매우 작은 히든 레이어) ─────────────────────
        [32],           # 히든 1층: 약 25K 파라미터
        [32, 32],       # 히든 2층: 약 26K 파라미터
        [32, 32, 32],   # 히든 3층: 약 27K 파라미터

        # ── Width 1024 (매우 큰 히든 레이어) ─────────────────────
        [1024],         # 히든 1층: 약 813K 파라미터
        [1024, 1024],   # 히든 2층: 약 1.8M 파라미터

        # ── 5층 레이어 (히든 레이어 4개) ─────────────────────────
        [64, 64, 64, 64],       # 히든 4층, 넓이 64
        [128, 128, 128, 128],   # 히든 4층, 넓이 128
        [256, 256, 256, 256],   # 히든 4층, 넓이 256

        # ── 피라미드/역피라미드 형태 ──────────────────────────────
        [512, 256, 128],          # 역피라미드: 점점 좁아짐 (히든 3개)
        [128, 256, 512],          # 피라미드: 점점 넓어짐 (히든 3개)
        [512, 512, 512],          # 히든 3층, 넓이 512 (크고 깊음)
        [64, 128, 256, 128, 64],  # 히든 5층, 다이아몬드 형태
    ]

    variants = []
    for config in new_configs:
        model = SimpleANN(hidden_sizes=config)
        model_info = model.get_model_info()
        model_info['config'] = config
        variants.append((model, model_info))

    return variants


def create_expanded_variants_for_92features():
    """
    데이터 확대용 추가 ANN 조합 (통합 92 Feature 스키마 대응).
    목표: ANN만 40~50개 구성 → 2 디바이스 = 80~100행.

    기존 create_model_variants + create_extended_variants 에 더해
    균일 구조·추가 너비·레이어 조합을 넣음.
    """
    extra_configs = [
        # 균일 구조 추가 (2~5층, 너비 128/256/512)
        [128, 128],
        [256, 256],
        [512, 512],
        [128, 128, 128],
        [256, 256, 256],
        [512, 512, 512],
        [128, 128, 128, 128],
        [256, 256, 256, 256],
        [64, 64, 64, 64, 64],
        [128, 128, 128, 128, 128],
        [256, 256, 256, 256, 256],
        # 추가 너비/깊이 조합
        [64, 256, 64],
        [256, 64, 256],
        [128, 512, 128],
        [512, 128, 512],
        [64, 256, 512, 256, 64],
        [128, 256, 512, 256, 128],
        [96, 192, 384],
        [384, 192, 96],
        [192, 384, 192],
        [320, 640],
        [640, 320],
        [320, 320, 320],
        [160, 320, 640, 320],
    ]
    variants = []
    for config in extra_configs:
        model = SimpleANN(hidden_sizes=config)
        model_info = model.get_model_info()
        model_info['config'] = config
        variants.append((model, model_info))
    return variants


def get_all_ann_variants():
    """기존 14 + 확장 12 + 데이터확대용 추가분을 모두 합친 ANN variant 리스트."""
    base = create_model_variants()
    extended = create_extended_variants()
    expanded = create_expanded_variants_for_92features()
    # config 중복 제거 (동일 config는 하나만)
    seen = set()
    out = []
    for model, info in base + extended + expanded:
        key = tuple(info['config'])
        if key in seen:
            continue
        seen.add(key)
        out.append((model, info))
    return out