import torch.nn as nn


class SimpleANN(nn.Module):
    """단순 다층 퍼셉트론 (ANN)

    Args:
        input_size (int): 입력 크기 (MNIST: 28*28 = 784)
        hidden_size (int): 히든 레이어 뉴런 수
        num_hidden_layers (int): 히든 레이어 수
        output_size (int): 출력 크기 (MNIST 클래스 수: 10)
    """

    def __init__(self, input_size=784, hidden_size=128, num_hidden_layers=2, output_size=10):
        super().__init__()

        layers = []

        # 입력층 → 첫 번째 히든층
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # 히든층 반복 (num_hidden_layers - 1 번 추가)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # 출력층
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 이미지를 1차원 벡터로 변환 (배치 크기 유지)
        x = x.view(x.size(0), -1)
        return self.network(x)
