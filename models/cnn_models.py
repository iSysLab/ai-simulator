"""
models/cnn_models.py - CNN 모델 정의 모음

[이 파일의 역할]
Stage 2 (CNN × CIFAR-10 실험)에서 사용하는 CNN 모델들을 정의합니다.
직접 구현한 SimpleCNN과 torchvision에서 불러온 ResNet18, MobileNetV2를 포함합니다.

[CNN (Convolutional Neural Network)이란?]
이미지처럼 공간 구조가 있는 데이터를 처리하기 위해 설계된 신경망입니다.
- Convolution(합성곱): 이미지를 작은 필터로 훑어 패턴(에지, 텍스처 등)을 추출
- Pooling(풀링): 특징 맵의 크기를 줄여 연산량 감소 및 위치 불변성 확보
- Fully Connected(FC): 추출된 특징을 바탕으로 최종 분류

[포함된 모델]
1. SimpleCNN: 직접 구현한 단순 CNN. 레이어 수와 채널 수를 조합해 9가지 변형 생성
2. ResNet18: 18개 레이어의 잔차 연결(Residual Connection) 기반 CNN
3. MobileNetV2: 모바일 환경에 최적화된 경량 CNN (Depthwise Separable Convolution 활용)
"""

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """
    직접 구현한 단순 CNN 모델.

    [구조]
    Conv 블록 × num_conv_layers → Flatten → FC(512) → Dropout → FC(num_classes)

    각 Conv 블록은 아래 4개 연산으로 구성됩니다:
    - Conv2d: 이미지에서 특징 추출 (edge, texture, pattern 등)
    - BatchNorm2d: 배치 정규화로 학습 안정화 및 속도 향상
    - ReLU: 비선형 활성화 함수 (음수를 0으로 만들어 비선형성 부여)
    - MaxPool2d: 2×2 영역 중 최댓값만 남겨 특징 맵 크기를 절반으로 축소
    """

    def __init__(self, num_classes=10, num_conv_layers=2, base_channels=32):
        """
        SimpleCNN 초기화.

        Args:
            num_classes (int): 출력 클래스 수. CIFAR-10은 10가지 클래스.
            num_conv_layers (int): Conv 블록의 수. 2, 3, 4 중 선택.
                - 2: 32×32 → 8×8 (MaxPool 2회)
                - 3: 32×32 → 4×4 (MaxPool 3회)
                - 4: 32×32 → 2×2 (MaxPool 4회)
            base_channels (int): 첫 번째 Conv 레이어의 출력 채널 수. 16, 32, 64 중 선택.
                각 Conv 블록마다 채널 수가 2배씩 증가합니다. (16→32→64→128...)
        """
        super(SimpleCNN, self).__init__()

        # 구조 정보를 인스턴스 변수로 저장 (get_model_info()에서 사용)
        self.num_conv_layers = num_conv_layers
        self.base_channels = base_channels

        # Conv 블록들을 순서대로 쌓을 리스트
        layers = []
        in_channels = 3          # CIFAR-10은 RGB 3채널 이미지
        out_channels = base_channels  # 첫 블록의 출력 채널 수

        # [핵심] num_conv_layers개의 Conv 블록 생성 (반복문으로 동적 구성)
        for i in range(num_conv_layers):
            # Conv2d: 합성곱 레이어
            #   - kernel_size=3: 3×3 필터로 주변 픽셀 패턴을 학습
            #   - padding=1: 입력 테두리에 0을 추가하여 출력 크기를 유지
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            # BatchNorm2d: 배치 정규화
            #   각 채널의 출력값을 평균 0, 분산 1로 정규화
            #   → 학습 속도 향상, gradient vanishing 완화
            layers.append(nn.BatchNorm2d(out_channels))

            # ReLU: 비선형 활성화 함수
            #   f(x) = max(0, x) → 음수는 0, 양수는 그대로
            #   신경망에 비선형성을 부여하여 복잡한 패턴 학습 가능
            layers.append(nn.ReLU())

            # MaxPool2d: 최대 풀링
            #   2×2 영역 중 최댓값만 남김 → 특징 맵 크기가 절반으로 줄어듦
            #   → 연산량 감소, 위치 변화에 대한 불변성 획득
            layers.append(nn.MaxPool2d(2, 2))

            # 다음 Conv 블록의 입력 채널 = 이번 블록의 출력 채널
            in_channels = out_channels
            # 다음 블록의 출력 채널은 2배 (채널 수가 점점 증가)
            out_channels = out_channels * 2

        # nn.Sequential: 레이어들을 순서대로 실행하는 컨테이너
        self.features = nn.Sequential(*layers)

        # [FC 입력 크기 계산]
        # CIFAR-10 이미지: 32×32 → MaxPool을 num_conv_layers번 적용
        # 각 MaxPool마다 절반: 32 / (2^num_conv_layers)
        # ex) num_conv_layers=2 → 32/4 = 8 → 8×8
        # ex) num_conv_layers=3 → 32/8 = 4 → 4×4
        final_size = 32 // (2 ** num_conv_layers)

        # 마지막 Conv 블록의 출력 채널 수
        # ex) base=32, layers=2 → 32 * 2^(2-1) = 64
        final_channels = base_channels * (2 ** (num_conv_layers - 1))

        # [분류기 (Classifier)]
        # Conv 블록으로 추출한 특징을 최종 클래스 확률로 변환
        self.classifier = nn.Sequential(
            # Flatten: (batch, channels, H, W) → (batch, channels*H*W) 로 펼침
            nn.Flatten(),

            # FC 레이어 1: 추출된 특징을 512차원으로 압축
            nn.Linear(final_channels * final_size * final_size, 512),
            nn.ReLU(),

            # Dropout: 학습 중 50%의 뉴런을 무작위로 비활성화
            # → 과적합(overfitting) 방지 효과
            # 추론 시에는 자동으로 비활성화됨
            nn.Dropout(0.5),

            # FC 레이어 2 (출력층): 512 → num_classes 차원
            # 각 클래스에 대한 점수(logit)를 출력
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        순전파(Forward Pass): 입력 이미지 → 클래스 점수(logit) 계산.

        Args:
            x (Tensor): 입력 이미지 배치, shape = (batch_size, 3, 32, 32)

        Returns:
            Tensor: 각 클래스에 대한 점수, shape = (batch_size, num_classes)
        """
        # 특징 추출 (Conv 블록들 순차 적용)
        x = self.features(x)
        # 분류 (FC 레이어들 순차 적용)
        x = self.classifier(x)
        return x

    def get_model_info(self):
        """
        모델 구조 정보를 딕셔너리로 반환.

        실험 결과 CSV에 모델 메타데이터를 저장할 때 사용됩니다.

        Returns:
            dict: 모델 타입, Conv 레이어 수, 채널 수, 전체/학습 가능 파라미터 수
        """
        # 전체 파라미터 수: 모델의 모든 텐서 원소 수의 합
        total_params = sum(p.numel() for p in self.parameters())

        # 학습 가능 파라미터 수: requires_grad=True인 파라미터만
        # (일반적으로 전체 파라미터와 동일, pretrained 레이어를 freeze하면 달라짐)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'SimpleCNN',
            'num_conv_layers': self.num_conv_layers,
            'base_channels': self.base_channels,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def get_resnet18(num_classes=10, pretrained=False):
    """
    ResNet18 모델을 CIFAR-10에 맞게 수정하여 반환합니다.

    [ResNet18이란?]
    2015년 Microsoft가 발표한 18개 레이어의 CNN 모델.
    핵심 아이디어: "Residual Connection (잔차 연결)"
    - 레이어가 깊어질수록 gradient가 사라지는 문제(vanishing gradient)를 해결
    - 각 블록의 입력을 출력에 더해줌: output = F(x) + x
    - ImageNet에서 top-5 오류율 3.57% 달성 (당시 SOTA)
    - 파라미터 수: 약 1,100만개

    [CIFAR-10 적응 수정]
    원래 ResNet18은 224×224 ImageNet 이미지용으로 설계됨.
    CIFAR-10의 32×32 이미지에 맞게 두 곳을 수정합니다:
    1. 첫 번째 Conv: 7×7 stride 2 → 3×3 stride 1 (작은 이미지에서 정보 손실 방지)
    2. MaxPool 제거: 32×32 이미지에서 MaxPool을 적용하면 너무 작아짐

    Args:
        num_classes (int): 출력 클래스 수 (CIFAR-10 → 10)
        pretrained (bool): ImageNet 사전 학습 가중치 사용 여부.
                           이 프로젝트에서는 False (처음부터 학습)

    Returns:
        nn.Module: CIFAR-10에 맞게 수정된 ResNet18 모델
    """
    # torchvision에서 ResNet18 구조 불러오기
    model = models.resnet18(pretrained=pretrained)

    # [수정 1] 첫 번째 Conv 레이어 교체
    # 원본: kernel_size=7, stride=2 (이미지를 빠르게 축소)
    # 수정: kernel_size=3, stride=1 (32×32에서 정보 보존)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # [수정 2] MaxPool 제거 (nn.Identity = 아무것도 하지 않고 입력을 그대로 전달)
    # 원본 MaxPool은 feature map을 절반으로 줄이는데, 32×32에서는 불필요
    model.maxpool = nn.Identity()

    # [수정 3] 출력 레이어를 CIFAR-10 클래스 수(10)에 맞게 교체
    # 원본 ResNet18의 fc: 512 → 1000 (ImageNet 클래스 수)
    # 수정 후: 512 → num_classes (10)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_mobilenetv2(num_classes=10, pretrained=False):
    """
    MobileNetV2 모델을 CIFAR-10에 맞게 수정하여 반환합니다.

    [MobileNetV2란?]
    2018년 Google이 발표한 모바일/엣지 디바이스용 경량 CNN.
    핵심 아이디어: "Depthwise Separable Convolution (깊이별 분리 합성곱)"
    - 일반 Convolution을 두 단계로 분리:
      1. Depthwise Conv: 각 채널을 독립적으로 합성곱 (채널간 정보 교환 없음)
      2. Pointwise Conv (1×1): 채널 간 정보 결합
    - 일반 Conv 대비 연산량을 약 8~9배 줄이면서 유사한 성능 유지
    - 파라미터 수: 약 350만개 (ResNet18의 1/3 수준)

    [CIFAR-10 적응 수정]
    출력 레이어만 CIFAR-10 클래스 수에 맞게 교체합니다.
    (MobileNetV2는 이미 소형 이미지에도 적합한 구조)

    Args:
        num_classes (int): 출력 클래스 수 (CIFAR-10 → 10)
        pretrained (bool): ImageNet 사전 학습 가중치 사용 여부.
                           이 프로젝트에서는 False (처음부터 학습)

    Returns:
        nn.Module: CIFAR-10에 맞게 수정된 MobileNetV2 모델
    """
    # torchvision에서 MobileNetV2 구조 불러오기
    model = models.mobilenet_v2(pretrained=pretrained)

    # 출력 레이어를 CIFAR-10 클래스 수(10)에 맞게 교체
    # 원본: classifier[1] = Linear(1280, 1000) (ImageNet용)
    # 수정: classifier[1] = Linear(1280, num_classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def create_cnn_variants():
    """
    실험에 사용할 다양한 CNN 모델 변형(Variant)들을 생성하여 리스트로 반환합니다.

    [생성되는 모델 목록]
    SimpleCNN: num_conv_layers (2, 3, 4, 5) × base_channels (16, 32, 64, 128, 96) = 20가지
    ResNet18: 1가지
    MobileNetV2: 1가지
    총 22가지 모델 변형 (데이터 확대: 25~35 목표에 맞춤)

    [왜 다양한 구조를 실험하는가?]
    예측 모델 학습에 충분한 학습 데이터(다양한 입력-출력 쌍)가 필요합니다.
    모델의 파라미터 수, 레이어 수 등이 다를수록 실행 시간이 다르게 측정되고,
    이 다양성이 예측 모델의 일반화 성능을 높여줍니다.

    Returns:
        list of tuple: [(model, model_info), ...]
            - model: PyTorch 모델 인스턴스
            - model_info (dict): 모델 메타데이터 (이름, 타입, 파라미터 수 등)
    """
    variants = []

    # ─────────────────────────────────────────────────────────────
    # SimpleCNN: 4 레이어 수 × 5 채널 수 = 20가지 조합 (데이터 확대)
    # num_conv_layers: 2, 3, 4, 5 / base_channels: 16, 32, 64, 96, 128
    # ─────────────────────────────────────────────────────────────
    for num_layers in [2, 3, 4, 5]:
        for base_ch in [16, 32, 64, 96, 128]:
            # 모델 인스턴스 생성
            model = SimpleCNN(num_classes=10, num_conv_layers=num_layers, base_channels=base_ch)

            # 모델 정보 딕셔너리 가져오기 (파라미터 수 등 포함)
            model_info = model.get_model_info()

            # 실험 결과에 기록될 이름: 예) "SimpleCNN_L2_C32"
            # L = num_conv_Layers, C = base_Channels
            model_info['name'] = f'SimpleCNN_L{num_layers}_C{base_ch}'
            variants.append((model, model_info))

    # ─────────────────────────────────────────────────────────────
    # ResNet18
    # ─────────────────────────────────────────────────────────────
    resnet = get_resnet18(num_classes=10, pretrained=False)
    resnet_info = {
        'model_type': 'ResNet18',
        'name': 'ResNet18',
        'total_params': sum(p.numel() for p in resnet.parameters()),
        'trainable_params': sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    }
    variants.append((resnet, resnet_info))

    # ─────────────────────────────────────────────────────────────
    # MobileNetV2
    # ─────────────────────────────────────────────────────────────
    mobilenet = get_mobilenetv2(num_classes=10, pretrained=False)
    mobilenet_info = {
        'model_type': 'MobileNetV2',
        'name': 'MobileNetV2',
        'total_params': sum(p.numel() for p in mobilenet.parameters()),
        'trainable_params': sum(p.numel() for p in mobilenet.parameters() if p.requires_grad)
    }
    variants.append((mobilenet, mobilenet_info))

    return variants
