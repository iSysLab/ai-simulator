import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_conv_layers=2, base_channels=32):
        """
        단순한 CNN 모델

        Args:
            num_classes: 출력 클래스 수
            num_conv_layers: Convolution 레이어 수 (2, 3, 4)
            base_channels: 첫 번째 conv 레이어의 채널 수 (16, 32, 64)
        """
        super(SimpleCNN, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.base_channels = base_channels

        layers = []
        in_channels = 3  # CIFAR-10은 RGB
        out_channels = base_channels

        # Convolution 레이어 구성
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))

            in_channels = out_channels
            out_channels = out_channels * 2

        self.features = nn.Sequential(*layers)

        # Feature map 크기 계산 (CIFAR-10: 32x32 입력)
        # 각 MaxPool2d마다 크기가 반으로 줄어듦
        final_size = 32 // (2 ** num_conv_layers)
        final_channels = base_channels * (2 ** (num_conv_layers - 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * final_size * final_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_model_info(self):
        """
        모델 구조 정보 반환
        """
        total_params = sum(p.numel() for p in self.parameters())
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
    ResNet18 모델 (CIFAR-10용으로 수정)
    """
    model = models.resnet18(pretrained=pretrained)

    # CIFAR-10은 작은 이미지이므로 첫 conv 레이어 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # MaxPool 제거

    # 마지막 FC 레이어를 CIFAR-10 클래스 수에 맞게 수정
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_mobilenetv2(num_classes=10, pretrained=False):
    """
    MobileNetV2 모델 (CIFAR-10용으로 수정)
    """
    model = models.mobilenet_v2(pretrained=pretrained)

    # 마지막 classifier를 CIFAR-10 클래스 수에 맞게 수정
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def create_cnn_variants():
    """
    다양한 CNN 모델 생성
    """
    variants = []

    # SimpleCNN variants
    # num_conv_layers: 2, 3, 4
    # base_channels: 16, 32, 64
    for num_layers in [2, 3, 4]:
        for base_ch in [16, 32, 64]:
            model = SimpleCNN(num_classes=10, num_conv_layers=num_layers, base_channels=base_ch)
            model_info = model.get_model_info()
            model_info['name'] = f'SimpleCNN_L{num_layers}_C{base_ch}'
            variants.append((model, model_info))

    # ResNet18
    resnet = get_resnet18(num_classes=10, pretrained=False)
    resnet_info = {
        'model_type': 'ResNet18',
        'name': 'ResNet18',
        'total_params': sum(p.numel() for p in resnet.parameters()),
        'trainable_params': sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    }
    variants.append((resnet, resnet_info))

    # MobileNetV2
    mobilenet = get_mobilenetv2(num_classes=10, pretrained=False)
    mobilenet_info = {
        'model_type': 'MobileNetV2',
        'name': 'MobileNetV2',
        'total_params': sum(p.numel() for p in mobilenet.parameters()),
        'trainable_params': sum(p.numel() for p in mobilenet.parameters() if p.requires_grad)
    }
    variants.append((mobilenet, mobilenet_info))

    return variants