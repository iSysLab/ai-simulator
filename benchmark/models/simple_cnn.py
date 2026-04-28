import torch.nn as nn
from .registry import register_model


class SimpleCNN(nn.Module):
    """수정된 CNN: AdaptiveAvgPool2d로 공간 차원 정규화

    기존 문제: MaxPool 횟수가 레이어 수에 의존 → 깊은 모델이
    오히려 파라미터가 적어지는 역전 현상 발생

    수정: AdaptiveAvgPool2d(1)로 항상 1x1로 축소하여
    classifier 파라미터가 채널 수에만 비례하도록 변경
    """

    def __init__(self, num_filters=32, num_conv_layers=2,
                 use_batchnorm=False, num_classes=10):
        super().__init__()

        conv_layers = []
        in_channels = 1  # MNIST 그레이스케일

        for i in range(num_conv_layers):
            out_channels = num_filters * (2 ** min(i, 2))  # 최대 4배
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            if (i + 1) % 2 == 0:
                conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        # 핵심 수정: 공간 차원을 항상 1x1로 만듦
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # flat_size = last out_channels (공간 크기 무관)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)              # (B, C, 1, 1)
        x = x.view(x.size(0), -1)     # (B, C)
        x = self.classifier(x)
        return x


@register_model('simple_cnn')
def create_simple_cnn(num_filters=32, num_conv_layers=2,
                      use_batchnorm=False, **kwargs):
    return SimpleCNN(
        num_filters=num_filters,
        num_conv_layers=num_conv_layers,
        use_batchnorm=use_batchnorm)
