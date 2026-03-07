import torch
import torch.nn as nn


def extract_features(model, model_type, input_shape=(1, 1, 28, 28)):
    """모델 구조에서 ML 입력 피처 추출

    Args:
        model: PyTorch 모델
        model_type: 모델 유형 문자열 (simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist)
        input_shape: 입력 텐서 shape (배치 포함)

    Returns:
        dict: 23개 피처
    """
    features = {}

    # === 파라미터 수 (종류별 분리) ===
    conv_params = 0
    linear_params = 0
    bn_params = 0
    other_params = 0
    total_params = 0
    trainable_params = 0

    num_conv_layers = 0
    num_linear_layers = 0
    num_bn_layers = 0
    num_pool_layers = 0
    num_activation_layers = 0
    total_layers = 0

    has_residual = 0
    has_depthwise = 0
    has_attention = 0
    max_channel_width = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            p = sum(param.numel() for param in module.parameters())
            conv_params += p
            num_conv_layers += 1
            max_channel_width = max(max_channel_width, module.out_channels)
            # depthwise conv 감지
            if module.groups > 1 and module.groups == module.in_channels:
                has_depthwise = 1

        elif isinstance(module, nn.Linear):
            p = sum(param.numel() for param in module.parameters())
            linear_params += p
            num_linear_layers += 1
            max_channel_width = max(max_channel_width, module.out_features)

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            p = sum(param.numel() for param in module.parameters())
            bn_params += p
            num_bn_layers += 1

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            num_pool_layers += 1

        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU)):
            num_activation_layers += 1

        elif isinstance(module, nn.MultiheadAttention):
            has_attention = 1

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    other_params = total_params - conv_params - linear_params - bn_params
    total_layers = num_conv_layers + num_linear_layers + num_bn_layers

    # 잔차 연결 감지 (모델 타입 기반)
    if model_type in ('resnet_mnist', 'mobilenet_mnist'):
        has_residual = 1

    # === FLOPs 추정 (forward hook 기반) ===
    flops = _estimate_flops(model, input_shape)

    # === 메모리 크기 ===
    memory_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    # === 모델 깊이 (sequential layers) ===
    depth = num_conv_layers + num_linear_layers

    # === 모델 유형 원핫 ===
    model_types = ['simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist']

    features = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'bn_params': bn_params,
        'other_params': other_params,
        'num_conv_layers': num_conv_layers,
        'num_linear_layers': num_linear_layers,
        'num_bn_layers': num_bn_layers,
        'num_pool_layers': num_pool_layers,
        'num_activation_layers': num_activation_layers,
        'total_layers': total_layers,
        'flops': flops,
        'memory_bytes': memory_bytes,
        'depth': depth,
        'max_channel_width': max_channel_width,
        'has_residual': has_residual,
        'has_depthwise': has_depthwise,
        'has_attention': has_attention,
    }

    # 모델 유형 원핫
    for mt in model_types:
        features[f'model_type_{mt}'] = 1 if model_type == mt else 0

    return features


def _estimate_flops(model, input_shape):
    """Forward hook으로 FLOPs 추정"""
    flops_count = [0]
    hooks = []

    def conv_hook(module, input, output):
        batch = input[0].size(0)
        out_h, out_w = output.size(2), output.size(3)
        # FLOPs = 2 * Cin * Cout * K * K * Hout * Wout / groups
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        flops_count[0] += (
            2 * module.in_channels * module.out_channels *
            kernel_ops * out_h * out_w // module.groups
        )

    def linear_hook(module, input, output):
        # FLOPs = 2 * in_features * out_features
        flops_count[0] += 2 * module.in_features * module.out_features

    def bn_hook(module, input, output):
        # BN: ~2 ops per element (scale + shift)
        flops_count[0] += 2 * input[0].numel()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(bn_hook))

    # 더미 forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        model(dummy)

    # hook 정리
    for h in hooks:
        h.remove()

    return flops_count[0]
