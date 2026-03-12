"""모델 구조 피처 추출기

train_predictor.py의 FEATURE_COLUMNS (44개)를 모두 직접 생성.
extractor가 피처의 단일 소스(single source of truth)가 되도록 통합.
"""
import torch
import torch.nn as nn


# 모델 계열 인코딩 (train_predictor.py와 동일)
MODEL_FAMILY_MAP = {
    'simple_ann': 0,
    'simple_cnn': 1,
    'resnet_mnist': 2,
    'mobilenet_mnist': 3,
    'transformer': 4,
    'gan': 5,
}

# 장치 인코딩
DEVICE_TYPE_MAP = {
    'cpu': 0,
    'cuda': 1,
    'mps': 2,
}

# 데이터셋 → 입력 정보 매핑
DATASET_INFO = {
    'simple_ann':       {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10},
    'simple_cnn':       {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10},
    'resnet_mnist':     {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10},
    'mobilenet_mnist':  {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10},
    'transformer':      {'input_height': 32, 'input_width': 32, 'input_channels': 3, 'num_classes': 10},
    'gan':              {'input_height': 32, 'input_width': 32, 'input_channels': 3, 'num_classes': 10},
}


def get_hardware_info(device_str='cpu'):
    """실행 환경 하드웨어 정보 수집

    Returns:
        dict: cpu_cores, cpu_freq_ghz, ram_total_gb, gpu_memory_gb
    """
    hw = {
        'cpu_cores': 0,
        'cpu_freq_ghz': 0.0,
        'ram_total_gb': 0.0,
        'gpu_memory_gb': 0.0,
    }

    try:
        import psutil
        hw['cpu_cores'] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            hw['cpu_freq_ghz'] = round(freq.max / 1000, 2)
        hw['ram_total_gb'] = round(
            psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    if device_str == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hw['gpu_memory_gb'] = round(props.total_memory / (1024 ** 3), 1)

    return hw


def extract_features(model, model_type, input_shape=(1, 1, 28, 28),
                     device_str='cpu', config=None, batch_size=64):
    """모델 구조에서 FEATURE_COLUMNS 44개 피처를 모두 추출

    Args:
        model: PyTorch 모델
        model_type: 모델 유형 문자열
        input_shape: 입력 텐서 shape (배치 포함)
        device_str: 장치 문자열 ('cpu', 'cuda', 'mps')
        config: 모델 생성 config dict (모델 전용 피처용)
        batch_size: 배치 크기

    Returns:
        dict: FEATURE_COLUMNS에 맞는 44개 피처
    """
    if config is None:
        config = {}

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
    has_pooling = 0
    has_batch_norm = 0
    has_layer_norm = 0
    has_dropout = 0
    max_channel_width = 0

    # GAN generator/discriminator 파라미터 분리
    generator_params = 0
    discriminator_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            p = sum(param.numel() for param in module.parameters())
            conv_params += p
            num_conv_layers += 1
            max_channel_width = max(max_channel_width, module.out_channels)
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
            has_batch_norm = 1

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            num_pool_layers += 1
            has_pooling = 1

        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.LeakyReLU)):
            num_activation_layers += 1

        elif isinstance(module, nn.MultiheadAttention):
            has_attention = 1

        elif isinstance(module, nn.LayerNorm):
            has_layer_norm = 1

        elif isinstance(module, nn.Dropout):
            has_dropout = 1

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    other_params = total_params - conv_params - linear_params - bn_params
    total_layers = num_conv_layers + num_linear_layers + num_bn_layers

    # 잔차 연결 감지
    if model_type in ('resnet_mnist', 'mobilenet_mnist', 'transformer'):
        has_residual = 1

    # === GAN generator/discriminator 파라미터 실제 분리 ===
    if model_type == 'gan' and hasattr(model, 'generator') and hasattr(model, 'discriminator'):
        generator_params = sum(p.numel() for p in model.generator.parameters())
        discriminator_params = sum(p.numel() for p in model.discriminator.parameters())

    # === num_hidden_layers ===
    num_hidden_layers = num_conv_layers + num_linear_layers

    # === 폭(Width) 관련 (config 기반) ===
    widths = []
    if model_type == 'simple_ann':
        hs = config.get('hidden_size', 0)
        nl = config.get('num_layers', 1)
        widths = [hs] * nl if hs else []
    elif model_type == 'simple_cnn':
        nf = config.get('num_filters', 0)
        nl = config.get('num_conv_layers', 1)
        widths = [min(nf * (2 ** i), nf * 4) for i in range(nl)] if nf else []
    elif model_type in ('resnet_mnist', 'mobilenet_mnist'):
        widths = [max_channel_width] if max_channel_width else []
    elif model_type == 'transformer':
        widths = [config.get('embed_dim', 0)]
    elif model_type == 'gan':
        widths = config.get('g_hidden_dims', [])

    max_width = max(widths) if widths else max_channel_width
    min_width = min(widths) if widths else 0
    avg_width = (sum(widths) / len(widths)) if widths else 0

    # === FLOPs 추정 ===
    flops = _estimate_flops(model, input_shape)
    flops_per_sample = flops  # batch=1이므로 동일
    params_per_flop = total_params / flops if flops > 0 else 0

    # === 메모리 크기 ===
    memory_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = round(total_params * 4 / (1024 ** 2), 4)

    # === 하드웨어 피처 ===
    hw = get_hardware_info(device_str)

    # === 입력 데이터 피처 ===
    ds_info = DATASET_INFO.get(model_type, {})

    # === 모델 전용 피처 ===
    hidden_size = config.get('hidden_size', 0)
    num_filters = config.get('num_filters', 0)
    use_batchnorm = 1 if config.get('use_batchnorm', False) else 0
    embed_dim = config.get('embed_dim', 0)
    num_heads = config.get('num_heads', 0)
    patch_size = config.get('patch_size', 0)
    latent_dim = config.get('latent_dim', 0)

    features = {
        # 3-1. 파라미터 관련
        'total_params': total_params,
        'trainable_params': trainable_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'bn_params': bn_params,
        'other_params': other_params,
        # 3-2. 레이어 수
        'total_layers': total_layers,
        'num_hidden_layers': num_hidden_layers,
        'num_conv_layers': num_conv_layers,
        'num_linear_layers': num_linear_layers,
        'num_bn_layers': num_bn_layers,
        'num_pool_layers': num_pool_layers,
        'num_activation_layers': num_activation_layers,
        # 3-3. 폭(Width)
        'max_width': max_width,
        'min_width': min_width,
        'avg_width': avg_width,
        'max_channel_width': max_channel_width,
        # 3-4. 연산량
        'flops': flops,
        'flops_per_sample': flops_per_sample,
        'params_per_flop': params_per_flop,
        'model_size_mb': model_size_mb,
        'memory_bytes': memory_bytes,
        # 3-5. 구조 플래그
        'has_residual': has_residual,
        'has_depthwise': has_depthwise,
        'has_attention': has_attention,
        'has_pooling': has_pooling,
        'has_batch_norm': has_batch_norm,
        'has_layer_norm': has_layer_norm,
        'has_dropout': has_dropout,
        # 3-6. 모델 분류
        'model_family_encoded': MODEL_FAMILY_MAP.get(model_type, -1),
        # 4. 모델 전용 피처
        'hidden_size': hidden_size,
        'num_filters': num_filters,
        'use_batchnorm': use_batchnorm,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'patch_size': patch_size,
        'latent_dim': latent_dim,
        'generator_params': generator_params,
        'discriminator_params': discriminator_params,
        # 5. 입력 데이터 피처
        'batch_size': batch_size,
        'input_height': ds_info.get('input_height', input_shape[2] if len(input_shape) >= 4 else 0),
        'input_width': ds_info.get('input_width', input_shape[3] if len(input_shape) >= 4 else 0),
        'input_channels': ds_info.get('input_channels', input_shape[1] if len(input_shape) >= 4 else 0),
        'num_classes': ds_info.get('num_classes', 10),
        # 6. 하드웨어 피처
        'device_type_encoded': DEVICE_TYPE_MAP.get(device_str, 0),
        'cpu_cores': hw['cpu_cores'],
        'cpu_freq_ghz': hw['cpu_freq_ghz'],
        'ram_total_gb': hw['ram_total_gb'],
        'gpu_memory_gb': hw['gpu_memory_gb'],
    }

    return features


def _estimate_flops(model, input_shape):
    """Forward hook으로 FLOPs 추정

    Transformer Attention의 Q@K, attn@V matmul도 포함.
    """
    flops_count = [0]
    hooks = []
    attention_info = []

    def conv_hook(module, input, output):
        out_h, out_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        flops_count[0] += (
            2 * module.in_channels * module.out_channels *
            kernel_ops * out_h * out_w // module.groups
        )

    def linear_hook(module, input, output):
        flops_count[0] += 2 * module.in_features * module.out_features

    def bn_hook(module, input, output):
        flops_count[0] += 2 * input[0].numel()

    def layernorm_hook(module, input, output):
        flops_count[0] += 5 * input[0].numel()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(bn_hook))
        elif isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(layernorm_hook))

    for name, module in model.named_modules():
        if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
            attention_info.append({
                'num_heads': module.num_heads,
                'head_dim': module.head_dim,
            })

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    # Transformer Attention matmul FLOPs 보정
    if attention_info:
        B = input_shape[0]
        if len(input_shape) == 4:
            for name, module in model.named_modules():
                if hasattr(module, 'patch_embed'):
                    N = module.patch_embed.num_patches + 1
                    break
            else:
                N = 65  # 기본값 (32/4)^2 + 1

        for attn in attention_info:
            H = attn['num_heads']
            d = attn['head_dim']
            attn_flops = 4 * N * N * d * H + 5 * N * N * H
            flops_count[0] += attn_flops

    return flops_count[0]
