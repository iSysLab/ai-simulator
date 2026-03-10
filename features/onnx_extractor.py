"""ONNX 모델 파싱 → Feature 추출

.onnx 파일을 읽어서 예측 모델에 필요한 feature 딕셔너리로 변환.
출처: khg9859/present utils/onnx_feature_extractor.py 포팅
수정: dal-merge feature 스키마(33개)에 맞게 컬럼명/구조 변경

사용법:
    from features.onnx_extractor import extract_features_from_onnx
    features = extract_features_from_onnx('model.onnx', device='cpu')
"""

import numpy as np


# model_type 인코딩 (extractor.py와 동일하게 맞춤)
MODEL_TYPE_MAP = {'ann': 0, 'cnn': 1, 'transformer': 2, 'gan': 3}

DEFAULT_BATCH_SIZE = 64


def extract_features_from_onnx(
    onnx_path: str,
    device: str = 'cpu',
    # Transformer 전용 (ONNX에서 자동 추출 어려워 직접 입력)
    embed_dim: int = 0,
    num_transformer_layers: int = 0,
    num_heads: int = 0,
    patch_size: int = 0,
    # GAN 전용
    latent_dim: int = 0,
    g_hidden_max: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """ONNX 파일에서 dal-merge feature 스키마에 맞는 딕셔너리 반환

    Args:
        onnx_path: ONNX 파일 경로
        device: 예측 디바이스 ('cpu' 또는 'cuda')
        embed_dim: Transformer 임베딩 차원 (Transformer가 아니면 0)
        num_transformer_layers: Transformer 인코더 블록 수
        num_heads: Attention 헤드 수
        patch_size: ViT 패치 크기
        latent_dim: GAN 노이즈 차원 (GAN이 아니면 0)
        g_hidden_max: GAN Generator 최대 hidden dim
        batch_size: 배치 크기

    Returns:
        dict: extractor.py의 extract_features()와 동일한 키 구조
    """
    try:
        import onnx
        import onnx.numpy_helper as onp
    except ImportError:
        raise ImportError("pip install onnx 로 설치 필요")

    model = onnx.load(onnx_path)

    # ── 파라미터 수 ──────────────────────────────────────
    total_params     = 0
    linear_params    = 0
    conv_params      = 0
    trainable_params = 0

    for init in model.graph.initializer:
        arr   = onp.to_array(init)
        count = int(arr.size)
        total_params     += count
        trainable_params += count

    # 노드 분석으로 conv/linear 파라미터 분리
    conv_weight_names  = set()
    linear_weight_names = set()
    for node in model.graph.node:
        if node.op_type == 'Conv':
            for inp in node.input:
                conv_weight_names.add(inp)
        elif node.op_type in ('Gemm', 'MatMul'):
            for inp in node.input:
                linear_weight_names.add(inp)

    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if init.name in conv_weight_names:
            conv_params += int(arr.size)
        elif init.name in linear_weight_names:
            linear_params += int(arr.size)

    # ── 노드 분포 ────────────────────────────────────────
    op_counter = {}
    for node in model.graph.node:
        op_counter[node.op_type] = op_counter.get(node.op_type, 0) + 1

    num_conv_layers = op_counter.get('Conv', 0)
    num_gemm        = op_counter.get('Gemm', 0) + op_counter.get('MatMul', 0)
    num_layers      = num_conv_layers + num_gemm

    # ── 모델 타입 자동 판별 ──────────────────────────────
    if latent_dim > 0:
        model_type_id = MODEL_TYPE_MAP['gan']
    elif embed_dim > 0:
        model_type_id = MODEL_TYPE_MAP['transformer']
    elif num_conv_layers > 0:
        model_type_id = MODEL_TYPE_MAP['cnn']
    else:
        model_type_id = MODEL_TYPE_MAP['ann']

    # ── 모델 크기 ────────────────────────────────────────
    model_size_mb = round(total_params * 4 / (1024 ** 2), 4)

    # ── 입력 텐서 정보 ───────────────────────────────────
    input_channels = 1
    input_height   = 28
    input_width    = 28
    num_classes    = 10
    try:
        shape = model.graph.input[0].type.tensor_type.shape
        dims  = [d.dim_value for d in shape.dim]
        if len(dims) >= 4:
            input_channels = dims[1] if dims[1] > 0 else 1
            input_height   = dims[2] if dims[2] > 0 else 28
            input_width    = dims[3] if dims[3] > 0 else 28
    except Exception:
        pass

    # ── CNN feature ──────────────────────────────────────
    has_batchnorm = 1 if 'BatchNormalization' in op_counter else 0
    has_pooling   = 1 if any(k in op_counter for k in
                             ('MaxPool', 'GlobalAveragePool', 'AveragePool')) else 0
    kernel_size   = 3
    num_fc_layers = num_gemm

    # ── ANN feature ──────────────────────────────────────
    hidden_size       = 0
    num_hidden_layers = num_layers if model_type_id == MODEL_TYPE_MAP['ann'] else 0

    # ── 필터 수 추정 (CNN) ───────────────────────────────
    num_filters = 0
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        if init.name in conv_weight_names and arr.ndim == 4:
            num_filters = max(num_filters, arr.shape[0])

    # ── 하드웨어 정보 (실행 환경에서 직접 수집) ─────────
    hw = _get_hardware_info(device)

    return {
        # 모델 구조 (공통)
        'total_params':     total_params,
        'trainable_params': trainable_params,
        'linear_params':    linear_params,
        'flops':            0,   # ONNX에서 정확한 FLOPs 계산 어려움
        'model_size_mb':    model_size_mb,
        'num_layers':       num_layers,
        'model_type':       model_type_id,
        # ANN 전용
        'hidden_size':       hidden_size,
        'num_hidden_layers': num_hidden_layers,
        # CNN 전용
        'conv_params':     conv_params,
        'num_conv_layers': num_conv_layers,
        'num_filters':     num_filters,
        'has_batchnorm':   has_batchnorm,
        'has_pooling':     has_pooling,
        'kernel_size':     kernel_size,
        'num_fc_layers':   num_fc_layers,
        # Transformer 전용
        'embed_dim':              embed_dim,
        'num_transformer_layers': num_transformer_layers,
        'num_heads':              num_heads,
        'patch_size':             patch_size,
        # GAN 전용
        'latent_dim':   latent_dim,
        'g_hidden_max': g_hidden_max,
        # 하드웨어
        'device': device,
        **hw,
        # 입력 데이터
        'batch_size':      batch_size,
        'input_channels':  input_channels,
        'input_height':    input_height,
        'input_width':     input_width,
        'num_classes':     num_classes,
    }


def _get_hardware_info(device_str):
    """실행 환경 하드웨어 정보 수집"""
    hw = {'cpu_cores': 0, 'cpu_freq_ghz': 0.0, 'cpu_cache_l2_mb': 0.0,
          'ram_total_gb': 0.0, 'gpu_memory_gb': 0.0}
    try:
        import psutil
        hw['cpu_cores']    = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            hw['cpu_freq_ghz'] = round(freq.max / 1000, 2)
        hw['ram_total_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    try:
        import torch
        if device_str == 'cuda' and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw['gpu_memory_gb'] = round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(['wmic', 'cpu', 'get', 'L2CacheSize'],
                                capture_output=True, text=True, timeout=3)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            hw['cpu_cache_l2_mb'] = round(int(lines[1]) / 1024, 1)
    except Exception:
        pass

    return hw
