import torch.nn as nn


class SimpleCNN(nn.Module):
    """단순 CNN 모델

    Args:
        input_channels (int): 입력 채널 수 (MNIST: 1)
        num_filters (int): Conv 레이어의 필터(채널) 수
        num_conv_layers (int): Conv 레이어 수
        use_batchnorm (bool): BatchNorm 사용 여부
        num_classes (int): 분류 클래스 수 (MNIST: 10)
    """

    def __init__(self, input_channels=1, num_filters=32, num_conv_layers=3,
                 use_batchnorm=False, num_classes=10):
        super().__init__()

        layers = []
        in_channels = input_channels

        for _ in range(num_conv_layers):
            # Conv → (BN) → ReLU → MaxPool
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters

        self.features = nn.Sequential(*layers)

        # MNIST 28×28 기준: MaxPool 한 번에 절반 → num_conv_layers번 후 크기 계산
        # 28 → 14 → 7 → 3 → 1 (최소 1 보장)
        feature_size = max(1, 28 // (2 ** num_conv_layers))
        fc_input_size = num_filters * feature_size * feature_size

        # Fully Connected 출력층
        self.classifier = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)
