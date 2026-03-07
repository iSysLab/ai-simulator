import torch.nn as nn
from .registry import register_model


class BasicBlock(nn.Module):
    """ResNet BasicBlock: 3x3 conv 2개 + 잔차 연결"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetMNIST(nn.Module):
    """MNIST용 ResNet: 1채널 28x28 입력에 맞게 수정

    변경점 vs 원본 ResNet:
    1. conv1: 7x7 stride=2 → 3x3 stride=1 (28x28이 너무 작아서)
    2. maxpool 제거 (초기 다운샘플링 불필요)
    3. 입력 채널: 3 → 1 (그레이스케일)
    """

    def __init__(self, layers=(2, 2, 2, 2), base_width=64, num_classes=10):
        super().__init__()
        self.in_planes = base_width

        # 수정된 스템: 작은 커널, stride=1, maxpool 없음
        self.conv1 = nn.Conv2d(
            1, base_width, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@register_model('resnet_mnist')
def create_resnet_mnist(layers=(2, 2, 2, 2), base_width=64, **kwargs):
    return ResNetMNIST(layers=tuple(layers), base_width=base_width)
