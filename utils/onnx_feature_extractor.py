# ============================================================
# Stage 5: ONNX 모델 파싱 → Feature 추출 유틸리티
# ============================================================
#
# 역할:
#   ONNX 파일을 읽어서 train_predictor.py에서 사용하는
#   33개 Feature 딕셔너리로 변환합니다.
#
# ONNX란?
#   Open Neural Network Exchange의 약자.
#   PyTorch, TensorFlow 등 다양한 프레임워크의 모델을
#   하나의 표준 파일 형식(.onnx)으로 저장할 수 있음.
#   모델 구조(레이어 연결 관계)와 파라미터(가중치)를 모두 포함.
#
# 파싱 전략:
#   ONNX 그래프의 노드(Node) 타입을 보고 모델 종류를 자동 판별
#   - Conv 노드가 많으면 → CNN
#   - Gemm(행렬 곱) 위주이면 → ANN
#   - Attention 패턴이 있으면 → Transformer (완전 자동 어려움)
#
# 작성자: 김홍근 / 하드웨어: MacBook Air M1
# ============================================================

import numpy as np


# ──────────────────────────────────────────────────────────
# 하드웨어 고정 Feature (MacBook Air M1 기준)
# ──────────────────────────────────────────────────────────
# 교수님 유의사항: 하드웨어 사양을 반드시 기록할 것
HARDWARE_INFO = {
    'cpu_cores':      8,      # M1 CPU 코어 수 (성능 4 + 효율 4)
    'cpu_freq_ghz':   3.2,    # 최대 클럭 속도 (GHz)
    'cpu_cache_l2_mb': 12.0,  # L2 캐시 크기 (MB)
    'ram_total_gb':   8.0,    # 통합 메모리 (GB)
    'gpu_memory_gb':  8.0,    # M1 GPU 메모리 (CPU와 Unified Memory 공유)
}

# ──────────────────────────────────────────────────────────
# 모델 타입 인코딩 (train_predictor.py와 동일하게 맞춰야 함)
# ──────────────────────────────────────────────────────────
MODEL_TYPE = {
    'ANN': 0,
    'SimpleCNN': 1,
    'ResNet': 2,
    'MobileNet': 3,
    'Transformer': 4,
    'GAN': 5,
}

# 배치 크기 기본값 (학습/추론 시 사용한 배치 크기)
DEFAULT_BATCH_SIZE = 64


def extract_features_from_onnx(
    onnx_path: str,
    device: str = 'cpu',
    # Transformer 전용 인자 (ONNX에서 자동 추출이 어려움 → 사용자가 직접 입력)
    embed_dim: int = 0,
    num_heads: int = 0,
    patch_size: int = 0,
    # GAN 전용 인자
    latent_dim: int = 0,
    # 배치 크기 (기본값 64)
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    ONNX 파일을 파싱하여 예측 모델에 필요한 33개 Feature 딕셔너리를 반환합니다.

    ONNX 구조 설명:
        model.graph.node         → 레이어들의 연산 목록 (예: Conv, Relu, Gemm...)
        model.graph.initializer  → 학습된 가중치(파라미터) 목록
        model.graph.input        → 모델 입력 텐서 정보 (배치, 채널, 높이, 너비)

    Args:
        onnx_path  (str): ONNX 파일 경로
        device     (str): 예측할 디바이스 ('cpu' 또는 'mps')
        embed_dim  (int): Transformer 임베딩 차원 (0이면 Transformer 아님으로 처리)
        num_heads  (int): Transformer Attention 헤드 수
        patch_size (int): Vision Transformer 패치 크기
        latent_dim (int): GAN 노이즈 벡터 차원 (0이면 GAN 아님)
        batch_size (int): 예측에 사용한 배치 크기 (기본 64)

    Returns:
        dict: Feature 이름 → 값 딕셔너리 (feature_columns.pkl의 순서에 맞춤)
    """
    try:
        import onnx
        import onnx.numpy_helper as onp
    except ImportError:
        raise ImportError(
            "onnx 패키지가 설치되지 않았습니다.\n"
            "설치 방법: pip install onnx"
        )

    # ONNX 파일 로드
    print(f"  ONNX 파일 로드: {onnx_path}")
    model = onnx.load(onnx_path)

    # ── 1. 총 파라미터 수 계산 ─────────────────────────────
    # initializer = 가중치, 바이어스 등 학습된 값들의 집합
    # 각 텐서(다차원 배열)의 원소 수를 합산하면 총 파라미터 수가 됨
    total_params = 0
    for initializer in model.graph.initializer:
        arr = onp.to_array(initializer)  # numpy 배열로 변환
        total_params += arr.size         # 원소 수 합산
    print(f"  총 파라미터 수: {total_params:,}")

    # ── 2. 레이어 유형 분석 ────────────────────────────────
    # ONNX 그래프의 각 노드(연산)의 op_type을 모아서 분포 파악
    op_types = [node.op_type for node in model.graph.node]
    op_counter = {}
    for op in op_types:
        op_counter[op] = op_counter.get(op, 0) + 1

    print(f"  연산 종류 분포: {op_counter}")

    # Conv 레이어 수 (CNN의 핵심 연산)
    num_conv_layers = op_counter.get('Conv', 0)

    # Gemm = 행렬 곱 연산 (완전 연결 레이어, ANN의 핵심)
    # MatMul = Transformer 등에서도 사용
    num_gemm = op_counter.get('Gemm', 0) + op_counter.get('MatMul', 0)

    # ── 3. 모델 타입 자동 판별 ─────────────────────────────
    # 노드 분포를 보고 모델 종류를 자동으로 추정합니다.
    if latent_dim > 0:
        # GAN은 latent_dim이 명시적으로 주어질 때만 GAN으로 처리
        model_type_encoded = MODEL_TYPE['GAN']
        model_type_name    = 'GAN'
    elif embed_dim > 0 and num_heads > 0:
        # Transformer는 embed_dim, num_heads가 명시적으로 주어질 때 처리
        model_type_encoded = MODEL_TYPE['Transformer']
        model_type_name    = 'Transformer'
    elif num_conv_layers > 0:
        # Conv 레이어가 있으면 CNN으로 분류
        # ResNet/MobileNet 구분은 파라미터 수 + 레이어 수로 휴리스틱 추정
        if total_params > 2_000_000 and num_conv_layers > 15:
            model_type_encoded = MODEL_TYPE['ResNet']
            model_type_name    = 'ResNet'
        elif total_params > 2_000_000:
            model_type_encoded = MODEL_TYPE['MobileNet']
            model_type_name    = 'MobileNet'
        else:
            model_type_encoded = MODEL_TYPE['SimpleCNN']
            model_type_name    = 'SimpleCNN'
    else:
        # Conv 없이 Gemm만 있으면 ANN으로 분류
        model_type_encoded = MODEL_TYPE['ANN']
        model_type_name    = 'ANN'

    print(f"  모델 타입 추정: {model_type_name} (encoded={model_type_encoded})")

    # ── 4. 레이어 너비(뉴런/채널 수) 추정 ─────────────────
    # 각 레이어의 출력 크기를 보고 max/min/avg_width를 계산
    layer_widths = []
    for init in model.graph.initializer:
        arr = onp.to_array(init)
        # 2D 이상 텐서의 첫 번째 차원이 레이어 출력 크기에 해당
        if arr.ndim >= 1:
            layer_widths.append(arr.shape[0])

    if layer_widths:
        max_width = int(max(layer_widths))
        min_width = int(min(layer_widths))
        avg_width = float(np.mean(layer_widths))
    else:
        max_width = min_width = avg_width = 0

    # ── 5. 입력 텐서 정보 파싱 ────────────────────────────
    # ONNX 그래프의 입력 노드에서 배치, 채널, 높이, 너비를 읽음
    input_channels = 1
    input_height   = 28
    input_width    = 28
    num_classes    = 10
    dataset_encoded = 0  # 기본값: MNIST

    try:
        input_tensor = model.graph.input[0]
        shape = input_tensor.type.tensor_type.shape
        dims = [d.dim_value for d in shape.dim]
        # dims = [배치 크기, 채널 수, 높이, 너비] (4D 이미지 입력 기준)
        if len(dims) >= 4:
            # 배치 크기는 dims[0] (보통 동적이라 0 또는 -1일 수 있음)
            input_channels = dims[1] if dims[1] > 0 else 1
            input_height   = dims[2] if dims[2] > 0 else 28
            input_width    = dims[3] if dims[3] > 0 else 28
        # CIFAR-10 판별: 32×32 컬러 이미지
        if input_channels == 3 and input_height == 32:
            dataset_encoded = 1  # CIFAR-10 = 1
            num_classes     = 10
    except Exception:
        pass  # 파싱 실패 시 기본값 유지

    # ── 6. CNN 관련 Feature 추출 ──────────────────────────
    # BatchNorm 유무: BatchNormalization 노드가 있으면 1
    cnn_has_batchnorm = 1 if 'BatchNormalization' in op_counter else 0

    # Pooling 유무: MaxPool, GlobalAveragePool 등이 있으면 1
    cnn_has_pooling = 1 if (
        'MaxPool' in op_counter or
        'GlobalAveragePool' in op_counter or
        'AveragePool' in op_counter
    ) else 0

    # Conv 레이어의 커널 크기 추정 (대부분 3×3 또는 7×7)
    cnn_kernel_size = 3  # 기본값

    # Gemm 노드 수 = FC 레이어 수로 근사
    cnn_num_fc_layers = num_gemm

    # 총 레이어 수 = Conv + Gemm 수
    num_layers = num_conv_layers + num_gemm
    num_hidden_layers = num_layers

    # ── 7. 모델 크기 (MB) 계산 ────────────────────────────
    # 파라미터 하나당 float32 = 4 bytes
    model_size_mb = total_params * 4 / (1024 * 1024)

    # ── 8. 로그 변환 Feature ──────────────────────────────
    # train_predictor.py에서 로그 변환한 Feature와 동일하게 변환해야
    # 예측 모델이 올바르게 작동함
    log_total_params  = np.log1p(total_params)
    log_model_size_mb = np.log1p(model_size_mb)
    log_max_width     = np.log1p(max_width)

    # ── 9. 디바이스 인코딩 ────────────────────────────────
    device_encoded = 0 if device.lower() == 'cpu' else 1  # cpu=0, mps=1

    # ── 10. base_channels (CNN 기본 채널 수) ──────────────
    # CNN이면 max_width를 base_channels로 사용 (단순화)
    base_channels = max_width if model_type_encoded in [1, 2, 3] else 0

    # ── 11. 최종 Feature 딕셔너리 구성 ────────────────────
    # train_predictor.py의 FEATURE_COLUMNS 순서와 반드시 일치해야 함
    features = {
        # 모델 구조 Feature
        'total_params':      total_params,
        'log_total_params':  log_total_params,
        'model_size_mb':     model_size_mb,
        'log_model_size_mb': log_model_size_mb,
        'num_layers':        num_layers,
        'num_hidden_layers': num_hidden_layers,
        'num_conv_layers':   num_conv_layers,
        'max_width':         max_width,
        'log_max_width':     log_max_width,
        'min_width':         min_width,
        'avg_width':         avg_width,
        'base_channels':     base_channels,
        'model_type_encoded': model_type_encoded,
        # CNN 전용 Feature
        'cnn_has_pooling':   cnn_has_pooling,
        'cnn_has_batchnorm': cnn_has_batchnorm,
        'cnn_num_fc_layers': cnn_num_fc_layers,
        'cnn_kernel_size':   cnn_kernel_size,
        # 하드웨어 Feature (M1 Mac 고정)
        'device_encoded':    device_encoded,
        'cpu_cores':         HARDWARE_INFO['cpu_cores'],
        'cpu_freq_ghz':      HARDWARE_INFO['cpu_freq_ghz'],
        'cpu_cache_l2_mb':   HARDWARE_INFO['cpu_cache_l2_mb'],
        'ram_total_gb':      HARDWARE_INFO['ram_total_gb'],
        'gpu_memory_gb':     HARDWARE_INFO['gpu_memory_gb'],
        # 입력 데이터 Feature
        'input_channels':    input_channels,
        'input_height':      input_height,
        'input_width':       input_width,
        'num_classes':       num_classes,
        'batch_size':        batch_size,
        'dataset_encoded':   dataset_encoded,
        # Transformer 전용 Feature (다른 모델은 0)
        'embed_dim':         embed_dim,
        'num_heads':         num_heads,
        'patch_size':        patch_size,
        # GAN 전용 Feature (다른 모델은 0)
        'latent_dim':        latent_dim,
    }

    print(f"\n  [Feature 추출 완료]")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  모델 타입  : {model_type_name}")
    print(f"  레이어 수  : {num_layers} (Conv: {num_conv_layers}, FC: {cnn_num_fc_layers})")
    print(f"  모델 크기  : {model_size_mb:.2f} MB")
    print(f"  입력 크기  : {input_channels}×{input_height}×{input_width}")
    print(f"  디바이스   : {device} (encoded={device_encoded})")

    return features
