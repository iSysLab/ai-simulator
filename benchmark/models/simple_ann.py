import torch.nn as nn
from .registry import register_model


class SimpleANN(nn.Module):
    """단순 완전연결 신경망 (MNIST용)

    입력(784) → [Linear(hidden_size) + ReLU] × num_layers → Linear(10)
    """

    def __init__(self, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        layers = []
        in_features = 784  # 28 × 28

        # 첫 번째 히든 레이어
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.ReLU())

        # 나머지 히든 레이어
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 1, 28, 28) → (B, 784)
        x = self.features(x)
        x = self.classifier(x)
        return x


@register_model('simple_ann')
def create_simple_ann(hidden_size=64, num_layers=2, **kwargs):
    return SimpleANN(hidden_size=hidden_size, num_layers=num_layers)
