"""ONNX 모델 파싱 → 피처 추출 유틸리티

ONNX 파일을 읽어서 예측 모델에 필요한 피처 딕셔너리로 변환.
khg9859 브랜치에서 포팅 + dal-merge 하드웨어 피처 통합.
ONNX 그래프에서 FLOPs 직접 계산 추가.
"""
import numpy as np
from .extractor import get_hardware_info


# 모델 타입 인코딩
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
                                batch_size=64):
    """ONNX 파일에서 피처 추출

    Args:
        onnx_path: ONNX 파일 경로
        device: 예측 대상 장치 ('cpu', 'cuda', 'mps')
        embed_dim: Transformer 임베딩 차원 (0이면 비-Transformer)
        num_heads: Transformer Attention Head 수
        patch_size: ViT 패치 크기
        latent_dim: GAN 노이즈 벡터 차원 (0이면 비-GAN)
        batch_size: 배치 크기

    Returns:
        dict: 피처 딕셔너리
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
    op_types = [node.op_type for node in model.graph.node]
    op_counter = {}
    for op in op_types:
        op_counter[op] = op_counter.get(op, 0) + 1

    num_conv_layers = op_counter.get('Conv', 0)
    num_gemm = op_counter.get('Gemm', 0) + op_counter.get('MatMul', 0)

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

    model_type_key = MODEL_TYPE_MAP.get(model_type_name, 'simple_ann')

    # 4. 레이어 너비 추정
    layer_widths = []
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if arr.ndim >= 1:
            layer_widths.append(arr.shape[0])

    max_channel_width = int(max(layer_widths)) if layer_widths else 0

    # 5. 입력 텐서 정보
    input_channels = 1
    input_height = 28
    input_width = 28
    num_classes = 10

    try:
        input_tensor = model.graph.input[0]
        shape = input_tensor.type.tensor_type.shape
        dims = [d.dim_value for d in shape.dim]
        if len(dims) >= 4:
            input_channels = dims[1] if dims[1] > 0 else 1
            input_height = dims[2] if dims[2] > 0 else 28
            input_width = dims[3] if dims[3] > 0 else 28
    except Exception:
        pass

    # 6. CNN 관련 피처
    has_batchnorm = 1 if 'BatchNormalization' in op_counter else 0
    has_pooling = 1 if any(k in op_counter for k in
                           ['MaxPool', 'GlobalAveragePool', 'AveragePool']) else 0

    num_layers = num_conv_layers + num_gemm
    memory_bytes = total_params * 4
    model_size_mb = round(total_params * 4 / (1024 * 1024), 4)

    # FLOPs 계산 (ONNX 그래프의 가중치 shape 기반 추정)
    flops = _estimate_onnx_flops(model, op_counter,
                                  input_height, input_width, input_channels)

    # 잔차/depthwise/attention 추정
    has_residual = 1 if model_type_name in ('ResNet', 'MobileNet', 'Transformer') else 0
    has_depthwise = 1 if model_type_name == 'MobileNet' else 0
    has_attention = 1 if model_type_name == 'Transformer' else 0

    # Conv/Linear 파라미터 분리 (근사)
    conv_params = 0
    linear_params = 0
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if arr.ndim == 4:  # Conv weight
            conv_params += arr.size
        elif arr.ndim == 2:  # Linear weight
            linear_params += arr.size

    bn_params = total_params - conv_params - linear_params
    if bn_params < 0:
        bn_params = 0

    # 하드웨어 피처
    hw = get_hardware_info(device)

    # 모델 유형 원핫 (6종)
    all_types = [
        'simple_ann', 'simple_cnn', 'resnet_mnist',
        'mobilenet_mnist', 'transformer', 'gan',
    ]

    features = {
        'total_params': total_params,
        'trainable_params': total_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'bn_params': bn_params,
        'other_params': max(0, total_params - conv_params - linear_params - bn_params),
        'num_conv_layers': num_conv_layers,
        'num_linear_layers': num_gemm,
        'num_bn_layers': op_counter.get('BatchNormalization', 0),
        'num_pool_layers': sum(op_counter.get(k, 0)
                               for k in ['MaxPool', 'GlobalAveragePool', 'AveragePool']),
        'num_activation_layers': sum(op_counter.get(k, 0)
                                     for k in ['Relu', 'LeakyRelu', 'Gelu', 'Sigmoid', 'Tanh']),
        'total_layers': num_layers,
        'flops': flops,
        'memory_bytes': memory_bytes,
        'model_size_mb': model_size_mb,
        'depth': num_layers,
        'max_channel_width': max_channel_width,
        'has_residual': has_residual,
        'has_depthwise': has_depthwise,
        'has_attention': has_attention,
        'cpu_cores': hw['cpu_cores'],
        'cpu_freq_ghz': hw['cpu_freq_ghz'],
        'ram_total_gb': hw['ram_total_gb'],
        'gpu_memory_gb': hw['gpu_memory_gb'],
    }

    for mt in all_types:
        features[f'model_type_{mt}'] = 1 if model_type_key == mt else 0

    print(f"  ONNX 피처 추출 완료: {model_type_name}, "
          f"params={total_params:,}, layers={num_layers}, flops={flops:,}")

    return features


def _estimate_onnx_flops(model, op_counter, input_h, input_w, input_ch):
    """ONNX 그래프의 가중치 shape에서 FLOPs 추정

    Conv: 가중치 shape = (Cout, Cin/g, Kh, Kw) → 2*Cin*Cout*K²*Hout*Wout/g
    Gemm/MatMul: 가중치 shape = (out, in) 또는 (in, out) → 2*in*out
    BatchNorm: 2 * feature_size (scale + shift)
    """
    import onnx.numpy_helper as onp

    flops = 0
    spatial_h, spatial_w = input_h, input_w

    for node in model.graph.node:
        if node.op_type == 'Conv':
            # Conv 가중치 찾기
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name:
                for init in model.graph.initializer:
                    if init.name == weight_name:
                        arr = onp.to_array(init)
                        if arr.ndim == 4:
                            cout, cin_per_g, kh, kw = arr.shape
                            # stride, groups 추정 (기본값)
                            groups = 1
                            for attr in node.attribute:
                                if attr.name == 'group':
                                    groups = attr.i
                            # 출력 공간 크기 (stride 고려)
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
            # Linear 가중치 찾기
            for inp_name in node.input:
                for init in model.graph.initializer:
                    if init.name == inp_name:
                        arr = onp.to_array(init)
                        if arr.ndim == 2:
                            flops += 2 * arr.shape[0] * arr.shape[1]
                        break

        elif node.op_type == 'BatchNormalization':
            # BN scale 가중치에서 feature 크기 추출
            if len(node.input) > 1:
                for init in model.graph.initializer:
                    if init.name == node.input[1]:
                        arr = onp.to_array(init)
                        flops += 2 * arr.size * spatial_h * spatial_w
                        break

        elif node.op_type in ('MaxPool', 'AveragePool'):
            # Pooling 후 공간 크기 축소
            for attr in node.attribute:
                if attr.name == 'strides' and len(attr.ints) > 0:
                    spatial_h = max(1, spatial_h // attr.ints[0])
                    spatial_w = max(1, spatial_w // attr.ints[-1])

        elif node.op_type == 'GlobalAveragePool':
            spatial_h, spatial_w = 1, 1

    return flops
