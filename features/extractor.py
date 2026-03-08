import torch
import torch.nn as nn
import psutil


def get_hardware_info(device_str):
    """실행 환경의 하드웨어 정보 수집

    Args:
        device_str (str): 'cpu' 또는 'cuda'

    Returns:
        dict: 하드웨어 관련 feature
    """
    # CPU 코어 수 (논리 코어)
    cpu_cores = psutil.cpu_count(logical=True)

    # CPU 클럭 속도 (MHz → GHz 변환)
    freq = psutil.cpu_freq()
    cpu_freq_ghz = round(freq.max / 1000, 2) if freq else 0.0

    # L2 캐시 크기
    cpu_cache_l2_mb = _get_l2_cache_mb()

    # 전체 RAM (바이트 → GB 변환)
    ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    # GPU 메모리 (CUDA인 경우에만)
    gpu_memory_gb = 0.0
    if device_str == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)

    return {
        'cpu_cores': cpu_cores,
        'cpu_freq_ghz': cpu_freq_ghz,
        'cpu_cache_l2_mb': cpu_cache_l2_mb,
        'ram_total_gb': ram_total_gb,
        'gpu_memory_gb': gpu_memory_gb,
    }


def _get_l2_cache_mb():
    """L2 캐시 크기 추정 (Windows: wmic, 실패 시 0 반환)"""
    try:
        import subprocess
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'L2CacheSize'],
            capture_output=True, text=True, timeout=3
        )
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            return round(int(lines[1]) / 1024, 1)  # KB → MB
    except Exception:
        pass
    return 0.0


def extract_features(model, model_type, model_config, device_str, input_config):
    """모델 구조에서 27개 feature 추출

    Args:
        model: PyTorch 모델 인스턴스
        model_type (str): 'ann' 또는 'cnn'
        model_config (dict): 모델 구조 설정
            - ANN: {'hidden_size': 128, 'num_hidden_layers': 2}
            - CNN: {'num_filters': 32, 'num_conv_layers': 3, 'has_batchnorm': 1,
                    'has_pooling': 1, 'kernel_size': 3, 'num_fc_layers': 1}
        device_str (str): 'cpu' 또는 'cuda'
        input_config (dict): 입력 데이터 설정
            {'batch_size', 'input_channels', 'input_height', 'input_width', 'num_classes'}

    Returns:
        dict: 27개 feature
    """
    # ── 파라미터 수 집계 ──────────────────────────────────
    total_params = 0
    trainable_params = 0
    linear_params = 0
    conv_params = 0

    # 레이어 타입별 파라미터 분리: Linear/Conv를 구분하려면 module 단위로 순회해야 함
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters())

    # 전체/학습가능 파라미터는 타입 구분 없이 모든 텐서를 봐야 하므로 별도 순회
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    # 모델 크기 (파라미터 수 × 4바이트 → MB)
    model_size_mb = round(total_params * 4 / (1024 ** 2), 4)

    # ── 레이어 수 (Linear + Conv + BN 합산) ──────────────
    num_layers = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d))
    )

    # ── FLOPs 추정 ────────────────────────────────────────
    input_shape = (
        1,
        input_config['input_channels'],
        input_config['input_height'],
        input_config['input_width'],
    )
    flops = _estimate_flops(model, input_shape)

    # ── 하드웨어 정보 ─────────────────────────────────────
    hw = get_hardware_info(device_str)

    # ── ANN 전용 feature (CNN이면 0) ──────────────────────
    hidden_size       = model_config.get('hidden_size', 0)
    num_hidden_layers = model_config.get('num_hidden_layers', 0)

    # ── CNN 전용 feature (ANN이면 0) ──────────────────────
    num_conv_layers = model_config.get('num_conv_layers', 0)
    num_filters     = model_config.get('num_filters', 0)
    has_batchnorm   = model_config.get('has_batchnorm', 0)
    has_pooling     = model_config.get('has_pooling', 0)
    kernel_size     = model_config.get('kernel_size', 0)
    num_fc_layers   = model_config.get('num_fc_layers', 0)

    return {
        # 모델 구조 (공통)
        'total_params':     total_params,
        'trainable_params': trainable_params,
        'linear_params':    linear_params,
        'flops':            flops,
        'model_size_mb':    model_size_mb,
        'num_layers':       num_layers,
        'model_type':       0 if model_type == 'ann' else 1,
        # ANN 전용
        'hidden_size':       hidden_size,
        'num_hidden_layers': num_hidden_layers,
        # CNN 전용 (ANN이면 0)
        'conv_params':     conv_params,
        'num_conv_layers': num_conv_layers,
        'num_filters':     num_filters,
        'has_batchnorm':   has_batchnorm,
        'has_pooling':     has_pooling,
        'kernel_size':     kernel_size,
        'num_fc_layers':   num_fc_layers,
        # 하드웨어
        'device':          device_str,
        **hw,
        # 입력 데이터
        'batch_size':      input_config['batch_size'],
        'input_channels':  input_config['input_channels'],
        'input_height':    input_config['input_height'],
        'input_width':     input_config['input_width'],
        'num_classes':     input_config['num_classes'],
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
