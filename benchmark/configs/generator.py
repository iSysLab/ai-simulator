"""설정 조합 생성기 — 6개 모델 타입 지원

ANN(42) + CNN(60) + ResNet(18) + MobileNet(20) + Transformer(12) + GAN(8) = 160개
"""
from itertools import product


def generate_all_configs():
    """모든 모델 유형의 설정 조합 생성"""
    configs = []
    configs.extend(_ann_configs())
    configs.extend(_cnn_configs())
    configs.extend(_resnet_configs())
    configs.extend(_mobilenet_configs())
    configs.extend(_transformer_configs())
    configs.extend(_gan_configs())
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
        [16, 32, 64, 128, 256, 512],
        [1, 2, 3, 4, 5, 6, 8],
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
        [8, 16, 32, 64, 128],
        [1, 2, 3, 4, 5, 6],
        [False, True],
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
    for layers, base_width in product(layer_configs, [16, 32, 64]):
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
        [0.25, 0.5, 0.75, 1.0, 1.5],
        [3, 5, 7, 10],
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


def _transformer_configs():
    """Vision Transformer 설정: 12개 (khg9859에서 병합)

    CIFAR-10 기준 (32x32, 3ch).
    embed_dim >= num_heads × 8 조건 충족하는 조합만.
    """
    # (embed_dim, num_layers, num_heads, patch_size)
    combos = [
        # 소형
        (64,  2, 4, 4),
        (64,  4, 4, 4),
        (64,  2, 4, 8),
        # 중형
        (128, 2, 4, 4),
        (128, 4, 4, 4),
        (128, 6, 4, 4),
        (128, 2, 8, 4),
        (128, 4, 8, 4),
        # 대형
        (256, 2, 8, 4),
        (256, 4, 8, 4),
        (256, 6, 8, 4),
        (256, 4, 8, 8),
    ]
    configs = []
    for embed_dim, num_layers, num_heads, patch_size in combos:
        configs.append({
            'model_type': 'transformer',
            'model_name': f'ViT_d{embed_dim}_l{num_layers}_h{num_heads}_p{patch_size}',
            'config': {
                'img_size': 32,
                'patch_size': patch_size,
                'in_channels': 3,
                'num_classes': 10,
                'embed_dim': embed_dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
            },
            'dataset': 'cifar10',
        })
    return configs


def _gan_configs():
    """GAN 설정: 8개 (khg9859에서 병합)

    CIFAR-10 기준 (32x32, 3ch).
    Discriminator는 Generator의 역순 구조.
    """
    combos = [
        # 소형
        (64,  [128, 256]),
        (64,  [128, 256, 512]),
        (128, [256, 512]),
        # 중형
        (128, [256, 512, 1024]),
        (128, [128, 256, 512, 256]),
        (256, [256, 512, 1024]),
        # 대형
        (256, [512, 1024, 512]),
        (256, [256, 512, 1024, 512]),
    ]
    configs = []
    for latent_dim, g_hidden_dims in combos:
        d_hidden_dims = list(reversed(g_hidden_dims))
        g_tag = '_'.join(str(d) for d in g_hidden_dims)
        configs.append({
            'model_type': 'gan',
            'model_name': f'GAN_z{latent_dim}_G{g_tag}',
            'config': {
                'latent_dim': latent_dim,
                'img_size': 32,
                'img_channels': 3,
                'g_hidden_dims': g_hidden_dims,
                'd_hidden_dims': d_hidden_dims,
            },
            'dataset': 'cifar10',
        })
    return configs
