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
                 use_batchnorm=False, num_classes=10, input_size=28):
        super().__init__()

        layers = []
        in_channels = input_channels
        spatial = input_size

        for _ in range(num_conv_layers):
            # Conv → (BN) → ReLU → MaxPool (공간 크기가 충분할 때만)
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            if spatial >= 2:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                spatial = spatial // 2
            in_channels = num_filters

        self.features = nn.Sequential(*layers)

        fc_input_size = num_filters * spatial * spatial

        # Fully Connected 출력층
        self.classifier = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)
