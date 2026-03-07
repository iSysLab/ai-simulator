from itertools import product


def generate_all_configs():
    """모든 모델 유형의 설정 조합 생성

    Returns:
        list of dict: 각각 model_type, model_name, config 포함
    """
    configs = []
    configs.extend(_ann_configs())
    configs.extend(_cnn_configs())
    configs.extend(_resnet_configs())
    configs.extend(_mobilenet_configs())
    return configs


def generate_configs(model_type=None):
    """특정 모델만 필터링하여 설정 생성"""
    all_configs = generate_all_configs()
    if model_type:
        return [c for c in all_configs if c['model_type'] == model_type]
    return all_configs


def _ann_configs():
    """SimpleANN 설정: 42개"""
    configs = []
    for hidden_size, num_layers in product(
        [16, 32, 64, 128, 256, 512],  # 6개
        [1, 2, 3, 4, 5, 6, 8],        # 7개
    ):
        configs.append({
            'model_type': 'simple_ann',
            'model_name': f'ANN_h{hidden_size}_l{num_layers}',
            'config': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
            }
        })
    return configs


def _cnn_configs():
    """SimpleCNN 설정: 60개"""
    configs = []
    for num_filters, num_conv_layers, use_batchnorm in product(
        [8, 16, 32, 64, 128],   # 5개
        [1, 2, 3, 4, 5, 6],     # 6개
        [False, True],           # 2개
    ):
        bn_tag = 'bn1' if use_batchnorm else 'bn0'
        configs.append({
            'model_type': 'simple_cnn',
            'model_name': f'CNN_f{num_filters}_l{num_conv_layers}_{bn_tag}',
            'config': {
                'num_filters': num_filters,
                'num_conv_layers': num_conv_layers,
                'use_batchnorm': use_batchnorm,
            }
        })
    return configs


def _resnet_configs():
    """ResNet-MNIST 설정: 18개"""
    configs = []
    layer_configs = [
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 2, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ]
    for layers, base_width in product(
        layer_configs,           # 6개
        [16, 32, 64],            # 3개
    ):
        layer_tag = ''.join(map(str, layers))
        configs.append({
            'model_type': 'resnet_mnist',
            'model_name': f'ResNet_{layer_tag}_w{base_width}',
            'config': {
                'layers': layers,
                'base_width': base_width,
            }
        })
    return configs


def _mobilenet_configs():
    """MobileNet-MNIST 설정: 20개"""
    configs = []
    for width_mult, num_blocks in product(
        [0.25, 0.5, 0.75, 1.0, 1.5],  # 5개
        [3, 5, 7, 10],                  # 4개
    ):
        configs.append({
            'model_type': 'mobilenet_mnist',
            'model_name': f'MobileNet_w{width_mult}_b{num_blocks}',
            'config': {
                'width_mult': width_mult,
                'num_blocks': num_blocks,
            }
        })
    return configs
