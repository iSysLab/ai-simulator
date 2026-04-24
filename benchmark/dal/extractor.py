import math

import numpy as np
import torch
import torch.nn as nn

from benchmark.dal.op_profiler import get_op_level_features
from benchmark.support.hardware_info import get_hardware_info


# hong-0311 기준 model_family_encoded 값
MODEL_FAMILY_MAP = {'ann': 0, 'cnn': 1, 'transformer': 4, 'gan': 5}


def extract_features(model, model_type, model_config, device_str, input_config):
    """모델 구조에서 feature 추출 (hong-0311 호환 92개 + dal 고유 19개 = 111개)

    Args:
        model: PyTorch 모델 인스턴스
        model_type (str): 'ann', 'cnn', 'transformer', 'gan'
        model_config (dict): 모델 구조 설정
            - ANN: {'hidden_size': 128, 'num_hidden_layers': 2}
            - CNN: {'num_filters': 32, 'num_conv_layers': 3, 'has_batchnorm': 1,
                    'has_pooling': 1, 'kernel_size': 3, 'num_fc_layers': 1}
            - Transformer: {'embed_dim': 128, 'num_transformer_layers': 4,
                            'num_heads': 4, 'patch_size': 4}
            - GAN: {'latent_dim': 128, 'g_hidden_dims': [...]}
        device_str (str): 'cpu' 또는 'cuda'
        input_config (dict): 입력 데이터 설정
            {'batch_size', 'input_channels', 'input_height', 'input_width',
             'num_classes', 'dataset_encoded'}

    Returns:
        dict: feature 딕셔너리
    """
    # ── 파라미터 수 집계 (타입별) ─────────────────────────
    total_params = 0 # 모델의 총 파라미터
    trainable_params = 0 # 모델에서 학습시킨 파라미터
    conv_params = 0
    linear_params = 0
    bn_params = 0
    other_params = 0

    for module in model.modules():
        if len(list(module.children())) > 0:
            continue  # 리프 모듈만 처리
        p_count = sum(p.numel() for p in module.parameters())
        if isinstance(module, nn.Linear): # 일반 연결층 (ANN의 히든층)
            linear_params += p_count
        elif isinstance(module, nn.Conv2d): # 이미지 처리층 (CNN에서 사용)
            conv_params += p_count
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)): # 정규화층
            bn_params += p_count
        else:
            other_params += p_count

    for p in model.parameters(): # 모델의 모든 파라미터를 하나씩 꺼냄
        total_params += p.numel() # 꺼낸 파라미터의 숫자 개수를 전체에 누적
        if p.requires_grad:
            trainable_params += p.numel() # 학습 가능한 파라미터 수 

    model_size_mb = round(total_params * 4 / (1024 ** 2), 4) # 모델의 크기

    # ── 레이어 수 ─────────────────────────────────────────
    total_layers = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)) # 실질적으로 의미 있는 층만 셈
    )

    # ── FLOPs 추정 + op-level feature ─────────────────────
    input_shape = (
        1,
        input_config['input_channels'],
        input_config['input_height'],
        input_config['input_width'],
    )
    flops    = _estimate_flops(model, input_shape) # 입력이 모델을 통과할 때 총 연산량
    op_feats = get_op_level_features(model, input_shape) # 층별 통계

    # ── 하드웨어 정보 ─────────────────────────────────────
    hw = get_hardware_info(device_str)

    # ── model_family_encoded (hong 기준) ──────────────────
    model_family_encoded = MODEL_FAMILY_MAP.get(model_type, 0)

    # ── 구조적 특성 플래그 (모델 검사) ───────────────────
    has_depthwise  = 1 if any(  # 특수한 Conv층이 있냐 (예 MobileNet)
        isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.in_channels > 1
        for m in model.modules()
    ) else 0
    has_attention  = 1 if any(  # 어텐션 구조가 있냐 (Transformer 계열)
        isinstance(m, (nn.MultiheadAttention,))
        for m in model.modules()
    ) else 0
    has_cls_token  = 1 if hasattr(model, 'cls_token') else 0 # 분류용 특수 토큰이 있냐 (ViT 계열)
    has_batch_norm = model_config.get('has_batchnorm', 0) # 배치정규화층이 있냐
    has_pooling    = model_config.get('has_pooling', 0) # 풀링층이 있냐
    has_residual   = model_config.get('has_residual', 0) # 잔차연결이 있냐 (ResNet 계열)
    has_skip_connection = has_residual
    is_sequential  = 1  # 현재 모든 모델은 순차 구조

    # ── 너비 통계 (모델 타입별) ───────────────────────────
    # hong 기준: hidden layer 출력 크기들의 통계
    if model_type == 'ann':
        hidden_size       = model_config.get('hidden_size', 0)
        num_hidden_layers = model_config.get('num_hidden_layers', 0)
        widths = [hidden_size] * num_hidden_layers if num_hidden_layers > 0 else [hidden_size]
        max_width         = hidden_size
        min_width         = hidden_size
        avg_width         = float(hidden_size)
        first_layer_width = hidden_size
        last_layer_width  = hidden_size
        base_channels     = 0
        num_blocks        = 0
        max_channels      = 0
        min_channels      = 0

    elif model_type == 'cnn':
        num_filters = model_config.get('num_filters', 0)
        max_width         = num_filters
        min_width         = num_filters
        avg_width         = float(num_filters)
        first_layer_width = num_filters
        last_layer_width  = num_filters
        base_channels     = num_filters
        num_blocks        = 0
        max_channels      = num_filters
        min_channels      = num_filters

    elif model_type == 'transformer':
        embed_dim_val = model_config.get('embed_dim', 0)
        max_width         = embed_dim_val
        min_width         = embed_dim_val
        avg_width         = float(embed_dim_val)
        first_layer_width = embed_dim_val
        last_layer_width  = embed_dim_val
        base_channels     = 0
        num_blocks        = model_config.get('num_transformer_layers', 0)
        max_channels      = 0
        min_channels      = 0

    elif model_type == 'gan':
        g_dims = model_config.get('g_hidden_dims', [0])
        max_width         = max(g_dims) if g_dims else 0
        min_width         = min(g_dims) if g_dims else 0
        avg_width         = float(np.mean(g_dims)) if g_dims else 0.0
        first_layer_width = g_dims[0]  if g_dims else 0
        last_layer_width  = g_dims[-1] if g_dims else 0
        base_channels     = 0
        num_blocks        = 0
        max_channels      = 0
        min_channels      = 0
    else:
        max_width = min_width = avg_width = 0.0
        first_layer_width = last_layer_width = base_channels = 0
        num_blocks = max_channels = min_channels = 0

    log_max_width = round(math.log1p(max_width), 6)

    # ── ANN 전용 ──────────────────────────────────────────
    hidden_size       = model_config.get('hidden_size', 0)
    num_hidden_layers = model_config.get('num_hidden_layers', 0)
    ann_max_hidden    = hidden_size if model_type == 'ann' else 0
    ann_min_hidden    = hidden_size if model_type == 'ann' else 0
    ann_avg_hidden    = float(hidden_size) if model_type == 'ann' else 0.0
    ann_num_layers    = num_hidden_layers if model_type == 'ann' else 0

    # ── CNN 전용 ──────────────────────────────────────────
    num_conv_layers   = model_config.get('num_conv_layers', 0)
    num_filters       = model_config.get('num_filters', 0)
    kernel_size       = model_config.get('kernel_size', 0)
    num_fc_layers     = model_config.get('num_fc_layers', 0)
    cnn_num_filters   = num_filters if model_type == 'cnn' else 0
    cnn_max_channels  = num_filters if model_type == 'cnn' else 0
    cnn_stem_channels = num_filters if model_type == 'cnn' else 0
    cnn_has_residual  = has_residual if model_type == 'cnn' else 0
    cnn_has_depthwise = has_depthwise if model_type == 'cnn' else 0

    # ── Transformer 전용 ──────────────────────────────────
    embed_dim              = model_config.get('embed_dim', 0)
    num_transformer_layers = model_config.get('num_transformer_layers', 0)
    num_heads              = model_config.get('num_heads', 0)
    patch_size             = model_config.get('patch_size', 0)
    ffn_dim                = embed_dim * 4 if model_type == 'transformer' else 0
    vit_has_cls_token      = has_cls_token
    vit_num_encoder_layers = num_transformer_layers if model_type == 'transformer' else 0

    # ── GAN 전용 ──────────────────────────────────────────
    latent_dim = model_config.get('latent_dim', 0)
    if model_type == 'gan':
        # Generator / Discriminator 파라미터 분리 (모델 구조에서 직접 계산)
        if hasattr(model, 'generator') and hasattr(model, 'discriminator'):
            generator_params     = sum(p.numel() for p in model.generator.parameters())
            discriminator_params = sum(p.numel() for p in model.discriminator.parameters())
        else:
            generator_params     = total_params // 2
            discriminator_params = total_params - generator_params
    else:
        generator_params     = 0
        discriminator_params = 0

    # ── 입력 데이터 파생 ──────────────────────────────────
    input_channels  = input_config['input_channels']
    input_height    = input_config['input_height']
    input_width     = input_config['input_width']
    dataset_encoded = input_config.get('dataset_encoded', 0)
    input_pixels    = input_height * input_width * input_channels
    seq_length      = (input_height // patch_size) ** 2 if model_type == 'transformer' and patch_size > 0 else 0

    return {
        # ── 공통 모델 구조 (hong 33개) ────────────────────
        'total_params':          total_params,
        'log_total_params':      round(math.log1p(total_params), 6),
        'trainable_params':      trainable_params,
        'model_size_mb':         model_size_mb,
        'log_model_size_mb':     round(math.log1p(model_size_mb), 6),
        'total_layers':          total_layers,
        'num_hidden_layers':     num_hidden_layers,
        'num_linear_layers':     num_fc_layers,
        'num_conv_layers':       num_conv_layers,
        'max_width':             max_width,
        'log_max_width':         log_max_width,
        'min_width':             min_width,
        'avg_width':             round(avg_width, 4),
        'base_channels':         base_channels,
        'model_family_encoded':  model_family_encoded,
        'has_pooling':           has_pooling,
        'has_batch_norm':        has_batch_norm,
        'cnn_num_fc_layers':     num_fc_layers,
        'cnn_kernel_size':       kernel_size,
        'flops':                 flops,
        'has_residual':          has_residual,
        'has_depthwise':         has_depthwise,
        'has_attention':         has_attention,
        'has_cls_token':         has_cls_token,
        'num_blocks':            num_blocks,
        'num_mult_adds':         flops // 2,
        'activation_memory_mb':  0,
        'first_layer_width':     first_layer_width,
        'last_layer_width':      last_layer_width,
        'is_sequential':         is_sequential,
        'has_skip_connection':   has_skip_connection,
        'max_channels':          max_channels,
        'min_channels':          min_channels,
        # ── 모델 전용 (hong 18개) ─────────────────────────
        'ann_max_hidden':        ann_max_hidden,
        'ann_min_hidden':        ann_min_hidden,
        'ann_avg_hidden':        round(ann_avg_hidden, 4),
        'cnn_num_filters':       cnn_num_filters,
        'cnn_max_channels':      cnn_max_channels,
        'cnn_has_residual':      cnn_has_residual,
        'cnn_has_depthwise':     cnn_has_depthwise,
        'embed_dim':             embed_dim,
        'num_heads':             num_heads,
        'patch_size':            patch_size,
        'ffn_dim':               ffn_dim,
        'vit_has_cls_token':     vit_has_cls_token,
        'latent_dim':            latent_dim,
        'generator_params':      generator_params,
        'discriminator_params':  discriminator_params,
        'ann_num_layers':        ann_num_layers,
        'cnn_stem_channels':     cnn_stem_channels,
        'vit_num_encoder_layers': vit_num_encoder_layers,
        # ── 입력 데이터 (hong 8개) ────────────────────────
        'input_height':          input_height,
        'input_width':           input_width,
        'input_channels':        input_channels,
        'num_classes':           input_config['num_classes'],
        'batch_size':            input_config['batch_size'],
        'dataset_encoded':       dataset_encoded,
        'input_pixels':          input_pixels,
        'seq_length':            seq_length,
        # ── 하드웨어 (hong 33개) ──────────────────────────
        'device':                device_str,
        **hw,
        # ── dal 고유 피처 (op-level 15개 + param types 4개) ─
        'conv_params':           conv_params,
        'linear_params':         linear_params,
        'bn_params':             bn_params,
        'other_params':          other_params,
        **op_feats,
    }


def _estimate_flops(model, input_shape):
    """Forward hook으로 FLOPs(모델이 계산해야 하는 총 연산 횟수) 추정

    Conv2d: 2 * Cin * Cout * K * K * Hout * Wout
    Linear: 2 * in_features * out_features
    """
    flops_count = [0]
    hooks = []

    def conv_hook(module, input, output):
        out_h, out_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        flops_count[0] += (
            2 * module.in_channels * module.out_channels
            * kernel_ops * out_h * out_w // module.groups
        )

    def linear_hook(module, input, output):
        flops_count[0] += 2 * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        model(dummy)

    for h in hooks:
        h.remove()

    return flops_count[0]
