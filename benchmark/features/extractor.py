"""모델 구조 피처 추출기

ijunsoo 기본 23개 피처 + dal-merge 하드웨어 피처 + Transformer/GAN 피처 통합.
"""
import torch
import torch.nn as nn
import platform


def get_hardware_info(device_str='cpu'):
    """실행 환경 하드웨어 정보 수집 (dal-merge에서 병합)

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
                     device_str='cpu'):
    """모델 구조에서 ML 입력 피처 추출

    Args:
        model: PyTorch 모델
        model_type: 모델 유형 문자열
            (simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist,
             transformer, gan)
        input_shape: 입력 텐서 shape (배치 포함)
        device_str: 장치 문자열 ('cpu', 'cuda', 'mps')

    Returns:
        dict: 구조 피처 + 하드웨어 피처
    """
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

        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.LeakyReLU)):
            num_activation_layers += 1

        elif isinstance(module, (nn.MultiheadAttention, nn.LayerNorm)):
            if isinstance(module, nn.MultiheadAttention):
                has_attention = 1

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    other_params = total_params - conv_params - linear_params - bn_params
    total_layers = num_conv_layers + num_linear_layers + num_bn_layers

    # 잔차 연결 감지
    if model_type in ('resnet_mnist', 'mobilenet_mnist', 'transformer'):
        has_residual = 1

    # Transformer는 attention 있음
    if model_type == 'transformer':
        has_attention = 1

    # === FLOPs 추정 ===
    flops = _estimate_flops(model, input_shape)

    # === 메모리 크기 ===
    memory_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = round(total_params * 4 / (1024 ** 2), 4)

    # === 모델 깊이 ===
    depth = num_conv_layers + num_linear_layers

    # === 하드웨어 피처 (dal-merge에서 병합) ===
    hw = get_hardware_info(device_str)

    # === 모델 유형 원핫 (6종) ===
    model_types = [
        'simple_ann', 'simple_cnn', 'resnet_mnist',
        'mobilenet_mnist', 'transformer', 'gan',
    ]

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
        'model_size_mb': model_size_mb,
        'depth': depth,
        'max_channel_width': max_channel_width,
        'has_residual': has_residual,
        'has_depthwise': has_depthwise,
        'has_attention': has_attention,
        # 하드웨어 피처
        'cpu_cores': hw['cpu_cores'],
        'cpu_freq_ghz': hw['cpu_freq_ghz'],
        'ram_total_gb': hw['ram_total_gb'],
        'gpu_memory_gb': hw['gpu_memory_gb'],
    }

    # 모델 유형 원핫
    for mt in model_types:
        features[f'model_type_{mt}'] = 1 if model_type == mt else 0

    return features


def _estimate_flops(model, input_shape):
    """Forward hook으로 FLOPs 추정

    Transformer Attention의 Q@K, attn@V matmul도 포함:
    - nn.Linear hook으로 QKV projection, out_proj FLOPs 자동 포착
    - LayerNorm hook 추가 (5 ops per element)
    - Attention의 수동 matmul (Q@K, softmax, attn@V)은
      모델의 num_heads, embed_dim, seq_len으로 별도 추정
    """
    flops_count = [0]
    hooks = []
    attention_info = []  # Transformer attention 정보 수집

    def conv_hook(module, input, output):
        out_h, out_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        flops_count[0] += (
            2 * module.in_channels * module.out_channels *
            kernel_ops * out_h * out_w // module.groups
        )

    def linear_hook(module, input, output):
        # QKV projection, MLP 등 모든 Linear FLOPs
        flops_count[0] += 2 * module.in_features * module.out_features

    def bn_hook(module, input, output):
        flops_count[0] += 2 * input[0].numel()

    def layernorm_hook(module, input, output):
        # LayerNorm: mean + var + normalize + scale + shift ≈ 5 ops
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

    # Attention matmul FLOPs 추정 (Transformer 모델용)
    # Q@K: (B, H, N, d) @ (B, H, d, N) → 2*B*H*N*N*d
    # attn@V: (B, H, N, N) @ (B, H, N, d) → 2*B*H*N*N*d
    # softmax: ~5*N*N*H
    # 총 Attention FLOPs ≈ 4*N*N*d*H + 5*N*N*H (per block)
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
    # (Linear hook은 QKV projection만 잡고, Q@K, attn@V는 못 잡음)
    if attention_info:
        # 입력에서 패치 수 추정
        B = input_shape[0]
        if len(input_shape) == 4:
            # 이미지 입력 → ViT
            for name, module in model.named_modules():
                if hasattr(module, 'patch_embed'):
                    N = module.patch_embed.num_patches + 1  # +1 for CLS
                    break
            else:
                N = 65  # 기본값 (32/4)^2 + 1

        for attn in attention_info:
            H = attn['num_heads']
            d = attn['head_dim']
            # Q@K: 2*N*N*d*H, attn@V: 2*N*N*d*H, softmax: 5*N*N*H
            attn_flops = 4 * N * N * d * H + 5 * N * N * H
            flops_count[0] += attn_flops

    return flops_count[0]
