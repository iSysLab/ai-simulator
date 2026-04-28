import torch.nn as nn
from .registry import register_model


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block: depthwise separable conv + 잔차"""

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = in_channels * expand_ratio

        layers = []
        # Pointwise expansion (1x1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise conv (3x3)
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])

        # Pointwise projection (1x1, linear)
        layers.extend([
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetMNIST(nn.Module):
    """MNIST용 MobileNetV2: depthwise separable conv 기반 경량 모델

    width_mult로 채널 수 스케일링, num_blocks로 깊이 조절
    """

    def __init__(self, width_mult=1.0, num_blocks=5, num_classes=10):
        super().__init__()

        def _ch(c):
            return max(8, int(c * width_mult))

        # 초기 conv
        init_channels = _ch(32)
        self.stem = nn.Sequential(
            nn.Conv2d(1, init_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True),
        )

        # InvertedResidual 블록 생성
        # 채널 설정: 점진적으로 증가
        channel_schedule = [_ch(16), _ch(24), _ch(32), _ch(64),
                            _ch(96), _ch(128), _ch(160), _ch(192),
                            _ch(224), _ch(256)]

        blocks = []
        in_ch = init_channels
        for i in range(num_blocks):
            out_ch = channel_schedule[min(i, len(channel_schedule) - 1)]
            stride = 2 if i in (1, 3) else 1  # 특정 블록에서 다운샘플
            expand_ratio = 1 if i == 0 else 6
            blocks.append(InvertedResidual(in_ch, out_ch, stride, expand_ratio))
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        # 마지막 1x1 conv
        last_channels = _ch(256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@register_model('mobilenet_mnist')
def create_mobilenet_mnist(width_mult=1.0, num_blocks=5, **kwargs):
    return MobileNetMNIST(width_mult=width_mult, num_blocks=num_blocks)
