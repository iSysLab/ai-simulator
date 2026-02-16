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
    다양한 구조의 모델 생성
    Layer 수와 width를 변경하면서 생성
    """
    variants = []

    # Layer 수: 2, 3, 4, 5
    # Width: 64, 128, 256, 512
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