"""ONNX 모델 파싱 → 피처 추출 유틸리티

ONNX 파일을 읽어서 Feature Schema v1.0 전체 피처를 생성.
extractor.py와 동일한 피처 스키마를 따름.
"""
from .extractor import (
    get_hardware_info, MODEL_FAMILY_MAP, MODEL_ARCH_MAP,
    DEVICE_TYPE_MAP, DATASET_INFO, DATASET_TYPE_MAP,
)


# ONNX 모델 이름 → 내부 model_type 매핑
MODEL_TYPE_MAP = {
    'ANN': 'simple_ann',
    'SimpleCNN': 'simple_cnn',
    'ResNet': 'resnet_mnist',
    'MobileNet': 'mobilenet_mnist',
    'Transformer': 'transformer',
    'GAN': 'gan',
}


def extract_features_from_onnx(onnx_path, device='cpu',
                                embed_dim=0, num_heads=0,
                                patch_size=0, latent_dim=0,
                                batch_size=64, hidden_size=0,
                                num_filters=0, use_batchnorm=0):
    """ONNX 파일에서 Feature Schema v1.0 전체 피처 추출

    Args:
        onnx_path: ONNX 파일 경로
        device: 예측 대상 장치
        embed_dim, num_heads, patch_size, latent_dim: 모델 전용 힌트
        batch_size, hidden_size, num_filters, use_batchnorm: 설정 힌트

    Returns:
        dict: FEATURE_COLUMNS에 맞는 전체 피처
    """
    try:
        import onnx
        import onnx.numpy_helper as onp
    except ImportError:
        raise ImportError("onnx 패키지 필요: pip install onnx")

    model = onnx.load(onnx_path)

    # 1. 총 파라미터 수
    total_params = 0
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        total_params += arr.size

    # 2. 연산 노드 분석
    op_counter = {}
    for node in model.graph.node:
        op_counter[node.op_type] = op_counter.get(node.op_type, 0) + 1

    num_conv_layers = op_counter.get('Conv', 0)
    num_linear_layers = op_counter.get('Gemm', 0) + op_counter.get('MatMul', 0)
    num_bn_layers = op_counter.get('BatchNormalization', 0)
    num_pool_layers = sum(op_counter.get(k, 0)
                          for k in ['MaxPool', 'GlobalAveragePool', 'AveragePool'])
    num_activation_layers = sum(op_counter.get(k, 0)
                                 for k in ['Relu', 'LeakyRelu', 'Gelu', 'Sigmoid', 'Tanh'])
    num_attention_layers_val = op_counter.get('MultiHeadAttention', 0) + op_counter.get('Attention', 0)

    # 3. 모델 타입 자동 판별
    if latent_dim > 0:
        model_type_name = 'GAN'
    elif embed_dim > 0 and num_heads > 0:
        model_type_name = 'Transformer'
    elif num_conv_layers > 0:
        if total_params > 2_000_000 and num_conv_layers > 15:
            model_type_name = 'ResNet'
        elif total_params > 2_000_000:
            model_type_name = 'MobileNet'
        else:
            model_type_name = 'SimpleCNN'
    else:
        model_type_name = 'ANN'

    model_type = MODEL_TYPE_MAP.get(model_type_name, 'simple_ann')

    # 4. 레이어 너비 추정
    layer_widths = []
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if arr.ndim >= 1:
            layer_widths.append(arr.shape[0])

    max_channel_width = int(max(layer_widths)) if layer_widths else 0

    # 5. 입력 텐서 정보
    ds_info = DATASET_INFO.get(model_type, {})
    input_channels = ds_info.get('input_channels', 1)
    input_height = ds_info.get('input_height', 28)
    input_width = ds_info.get('input_width', 28)
    num_classes = ds_info.get('num_classes', 10)
    dataset_type = ds_info.get('dataset_type', 'mnist')

    try:
        input_tensor = model.graph.input[0]
        shape = input_tensor.type.tensor_type.shape
        dims = [d.dim_value for d in shape.dim]
        if len(dims) >= 4:
            input_channels = dims[1] if dims[1] > 0 else input_channels
            input_height = dims[2] if dims[2] > 0 else input_height
            input_width = dims[3] if dims[3] > 0 else input_width
    except Exception:
        pass

    # 6. Conv/Linear 파라미터 분리
    conv_params = 0
    linear_params = 0
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if arr.ndim == 4:
            conv_params += arr.size
        elif arr.ndim == 2:
            linear_params += arr.size

    bn_params = max(0, total_params - conv_params - linear_params)
    other_params = 0

    # 7. 구조 플래그
    has_batch_norm = 1 if num_bn_layers > 0 else 0
    has_pooling = 1 if num_pool_layers > 0 else 0
    has_residual = 1 if model_type in ('resnet_mnist', 'mobilenet_mnist', 'transformer') else 0
    has_depthwise = 1 if model_type == 'mobilenet_mnist' else 0
    has_attention = 1 if model_type == 'transformer' else 0
    has_layer_norm = 1 if op_counter.get('LayerNormalization', 0) > 0 else 0
    has_dropout = 1 if op_counter.get('Dropout', 0) > 0 else 0
    has_skip_connection = has_residual

    total_layers = num_conv_layers + num_linear_layers + num_bn_layers
    num_hidden_layers = num_conv_layers + num_linear_layers
    memory_bytes = total_params * 4
    model_size_mb = round(total_params * 4 / (1024 * 1024), 4)

    # 8. FLOPs
    flops = _estimate_onnx_flops(model, op_counter,
                                  input_height, input_width, input_channels)
    flops_per_sample = flops
    params_per_flop = total_params / flops if flops > 0 else 0

    # 9. 폭(Width) 추정
    widths = layer_widths if layer_widths else []
    max_width = max(widths) if widths else max_channel_width
    min_width = min(widths) if widths else 0
    avg_width = (sum(widths) / len(widths)) if widths else 0

    # 10. GAN
    generator_params = total_params if model_type == 'gan' else 0
    discriminator_params = 0

    # Transformer 전용
    ffn_dim = embed_dim * 4 if embed_dim > 0 else 0
    if model_type == 'transformer' and patch_size > 0:
        sequence_length = (input_height // patch_size) ** 2 + 1
        has_cls_token = 1
    else:
        sequence_length = 0
        has_cls_token = 0

    # 하드웨어 피처
    hw = get_hardware_info(device)

    features = {
        # === 1차 핵심 (기존 호환) ===
        'total_params': total_params,
        'trainable_params': total_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'bn_params': bn_params,
        'other_params': other_params,
        'total_layers': total_layers,
        'num_hidden_layers': num_hidden_layers,
        'num_conv_layers': num_conv_layers,
        'num_linear_layers': num_linear_layers,
        'num_bn_layers': num_bn_layers,
        'num_pool_layers': num_pool_layers,
        'num_activation_layers': num_activation_layers,
        'max_width': max_width,
        'min_width': min_width,
        'avg_width': avg_width,
        'max_channel_width': max_channel_width,
        'flops': flops,
        'flops_per_sample': flops_per_sample,
        'params_per_flop': params_per_flop,
        'model_size_mb': model_size_mb,
        'memory_bytes': memory_bytes,
        'has_residual': has_residual,
        'has_depthwise': has_depthwise,
        'has_attention': has_attention,
        'has_pooling': has_pooling,
        'has_batch_norm': has_batch_norm,
        'has_layer_norm': has_layer_norm,
        'has_dropout': has_dropout,
        'model_family_encoded': MODEL_FAMILY_MAP.get(model_type, -1),
        'hidden_size': hidden_size,
        'num_filters': num_filters,
        'use_batchnorm': use_batchnorm,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'patch_size': patch_size,
        'latent_dim': latent_dim,
        'generator_params': generator_params,
        'discriminator_params': discriminator_params,
        'batch_size': batch_size,
        'input_height': input_height,
        'input_width': input_width,
        'input_channels': input_channels,
        'num_classes': num_classes,
        'device_type_encoded': DEVICE_TYPE_MAP.get(device, 0),
        'cpu_cores': hw['cpu_cores'],
        'cpu_freq_ghz': hw['cpu_freq_ghz'],
        'ram_total_gb': hw['ram_total_gb'],
        'gpu_cores': hw['gpu_cores'],
        'gpu_memory_gb': hw['gpu_memory_gb'],

        # === 2차 확장 (신규) ===
        'has_skip_connection': has_skip_connection,
        'model_family': MODEL_FAMILY_MAP.get(model_type, -1),
        'model_arch': MODEL_ARCH_MAP.get(model_type, 'unknown'),
        'kernel_size': 3 if num_conv_layers > 0 else 0,
        'stride': 1,
        'padding': 1 if num_conv_layers > 0 else 0,
        'max_channels': max_channel_width,
        'ffn_dim': ffn_dim,
        'num_attention_layers': num_attention_layers_val,
        'sequence_length': sequence_length,
        'has_cls_token': has_cls_token,
        'generator_layers': 0,
        'discriminator_layers': 0,
        'dataset_type': dataset_type,
        'input_dtype': 'float32',
        'input_elements': input_height * input_width * input_channels,
        'os_type': hw['os_type'],
        'accelerator_brand': hw['accelerator_brand'],
        'accelerator_name': hw['accelerator_name'],
        'cpu_cores_physical': hw['cpu_cores_physical'],
        'cpu_cores_logical': hw['cpu_cores_logical'],
        'cpu_perf_cores': hw['cpu_perf_cores'],
        'cpu_efficiency_cores': hw['cpu_efficiency_cores'],
        'cpu_freq_base_ghz': hw['cpu_freq_base_ghz'],
        'cpu_freq_boost_ghz': hw['cpu_freq_boost_ghz'],
        'cpu_cache_l2_mb': hw['cpu_cache_l2_mb'],
        'cpu_cache_l3_mb': hw['cpu_cache_l3_mb'],
        'memory_type': hw['memory_type'],
        'memory_bandwidth_gbs': hw['memory_bandwidth_gbs'],
        'is_unified_memory': hw['is_unified_memory'],
        'shared_memory_gb': hw['shared_memory_gb'],
        'dedicated_vram_gb': hw['dedicated_vram_gb'],
        'gpu_count': hw['gpu_count'],
        'gpu_core_count': hw['gpu_core_count'],
        'gpu_tensor_core_count': hw['gpu_tensor_core_count'],
        'gpu_compute_capability': hw['gpu_compute_capability'],
        'gpu_clock_ghz': hw['gpu_clock_ghz'],
        'peak_bandwidth_gbs': hw['peak_bandwidth_gbs'],
        'tflops_fp32': hw['tflops_fp32'],
        'tflops_fp16': hw['tflops_fp16'],
        'fp16_support': hw['fp16_support'],
        'bf16_support': hw['bf16_support'],
        'interconnect_type': hw['interconnect_type'],
        'host_to_device_bandwidth_gbs': hw['host_to_device_bandwidth_gbs'],
        'is_discrete_gpu': hw['is_discrete_gpu'],
        'is_integrated_gpu': hw['is_integrated_gpu'],
    }

    print(f"  ONNX 피처 추출 완료: {model_type_name}, "
          f"params={total_params:,}, layers={total_layers}, flops={flops:,}")

    return features


def _estimate_onnx_flops(model, op_counter, input_h, input_w, input_ch):
    """ONNX 그래프의 가중치 shape에서 FLOPs 추정"""
    import onnx.numpy_helper as onp

    flops = 0
    spatial_h, spatial_w = input_h, input_w

    for node in model.graph.node:
        if node.op_type == 'Conv':
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name:
                for init in model.graph.initializer:
                    if init.name == weight_name:
                        arr = onp.to_array(init)
                        if arr.ndim == 4:
                            cout, cin_per_g, kh, kw = arr.shape
                            groups = 1
                            for attr in node.attribute:
                                if attr.name == 'group':
                                    groups = attr.i
                            stride = 1
                            for attr in node.attribute:
                                if attr.name == 'strides' and len(attr.ints) > 0:
                                    stride = attr.ints[0]
                            out_h = spatial_h // stride
                            out_w = spatial_w // stride
                            flops += 2 * cin_per_g * groups * cout * kh * kw * out_h * out_w // groups
                            spatial_h, spatial_w = out_h, out_w
                        break

        elif node.op_type in ('Gemm', 'MatMul'):
            for inp_name in node.input:
                for init in model.graph.initializer:
                    if init.name == inp_name:
                        arr = onp.to_array(init)
                        if arr.ndim == 2:
                            flops += 2 * arr.shape[0] * arr.shape[1]
                        break

        elif node.op_type == 'BatchNormalization':
            if len(node.input) > 1:
                for init in model.graph.initializer:
                    if init.name == node.input[1]:
                        arr = onp.to_array(init)
                        flops += 2 * arr.size * spatial_h * spatial_w
                        break

        elif node.op_type in ('MaxPool', 'AveragePool'):
            for attr in node.attribute:
                if attr.name == 'strides' and len(attr.ints) > 0:
                    spatial_h = max(1, spatial_h // attr.ints[0])
                    spatial_w = max(1, spatial_w // attr.ints[-1])

        elif node.op_type == 'GlobalAveragePool':
            spatial_h, spatial_w = 1, 1

    return flops
