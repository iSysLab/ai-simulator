# ============================================================
# 하드웨어 정보 동적 감지 유틸리티
# ============================================================
#
# 출처: dal-merge (features/extractor.py) → hong-0311 포팅 → dal-merge 재포팅
# 역할: 실행 환경의 CPU/RAM/GPU 정보를 자동 수집 (하드웨어 feature 33개 반환)
#
# 의존성: pip install psutil
# ============================================================

import platform
import subprocess

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


HARDWARE_FEATURE_KEYS = [
    'device_type', 'os_type', 'accelerator_brand', 'accelerator_name',
    'cpu_cores_physical', 'cpu_cores_logical', 'cpu_perf_cores', 'cpu_efficiency_cores',
    'cpu_freq_base_ghz', 'cpu_freq_boost_ghz', 'cpu_cache_l2_mb', 'cpu_cache_l3_mb',
    'ram_total_gb', 'memory_type', 'memory_bandwidth_gbs', 'is_unified_memory',
    'shared_memory_gb', 'dedicated_vram_gb', 'gpu_count', 'gpu_memory_gb',
    'gpu_core_count', 'peak_bandwidth_gbs', 'tflops_fp32', 'tflops_fp16',
    'fp16_support', 'bf16_support', 'interconnect_type', 'host_to_device_bandwidth_gbs',
    'is_discrete_gpu', 'is_integrated_gpu', 'device_encoded', 'cpu_freq_ghz', 'memory_channels',
]


def get_hardware_info(device_str='cpu'):
    """실행 환경의 하드웨어 정보를 동적으로 수집합니다.

    Args:
        device_str (str): 'cpu', 'cuda', 'mps' 중 하나

    Returns:
        dict: 하드웨어 feature 33개
    """
    if psutil is None:
        return _fallback_hardware_info(device_str)

    system = platform.system()
    os_type = 1 if system == 'Windows' else (2 if system == 'Darwin' else (3 if system == 'Linux' else 0))

    # CPU
    cpu_physical = psutil.cpu_count(logical=False) or 0
    cpu_logical  = psutil.cpu_count(logical=True)  or 0
    cpu_perf_cores = 0
    cpu_efficiency_cores = 0
    if system == 'Darwin' and cpu_logical == 8 and cpu_physical == 8:
        cpu_perf_cores = 4
        cpu_efficiency_cores = 4

    freq = psutil.cpu_freq()
    cpu_freq_max = round(freq.max / 1000, 2) if freq and freq.max else 0.0
    cpu_freq_min = round(freq.min / 1000, 2) if freq and freq.min else 0.0

    cpu_cache_l2_mb = _get_l2_cache_mb()
    cpu_cache_l3_mb = _get_l3_cache_mb()

    # RAM
    ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    # GPU (기본값)
    gpu_count = 0
    gpu_memory_gb = 0.0
    dedicated_vram_gb = 0.0
    shared_memory_gb = 0.0
    is_unified_memory = 0
    accelerator_brand = 0
    accelerator_name  = 0
    gpu_core_count = 0
    peak_bandwidth_gbs = 0.0
    tflops_fp32 = 0.0
    tflops_fp16 = 0.0
    fp16_support = 0
    bf16_support = 0
    interconnect_type = 0
    host_to_device_bandwidth_gbs = 0.0
    is_discrete_gpu = 0
    is_integrated_gpu = 0

    if device_str == 'cuda' and torch and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count() or 1
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb    = round(props.total_memory / (1024 ** 3), 1)
        dedicated_vram_gb = gpu_memory_gb
        is_discrete_gpu  = 1
        accelerator_brand = 1
        accelerator_name  = 1
        gpu_core_count   = getattr(props, 'multi_processor_count', 0) or 0
        fp16_support     = 1
        bf16_support     = 1 if getattr(props, 'major', 0) >= 8 else 0
        interconnect_type = 1  # PCIe
    elif device_str == 'mps' and torch and torch.backends.mps.is_available():
        gpu_count         = 1
        gpu_memory_gb     = ram_total_gb
        shared_memory_gb  = ram_total_gb
        is_unified_memory = 1
        is_integrated_gpu = 1
        accelerator_brand = 2  # Apple
        accelerator_name  = 2
        interconnect_type = 2  # Unified Memory

    device_type    = 1 if device_str in ('cuda', 'mps') else 0
    device_encoded = 1 if device_str in ('cuda', 'mps') else 0

    return {
        'device_type':                  device_type,
        'os_type':                      os_type,
        'accelerator_brand':            accelerator_brand,
        'accelerator_name':             accelerator_name,
        'cpu_cores_physical':           cpu_physical,
        'cpu_cores_logical':            cpu_logical,
        'cpu_perf_cores':               cpu_perf_cores,
        'cpu_efficiency_cores':         cpu_efficiency_cores,
        'cpu_freq_base_ghz':            cpu_freq_min,
        'cpu_freq_boost_ghz':           cpu_freq_max,
        'cpu_cache_l2_mb':              cpu_cache_l2_mb,
        'cpu_cache_l3_mb':              cpu_cache_l3_mb,
        'ram_total_gb':                 ram_total_gb,
        'memory_type':                  0,
        'memory_bandwidth_gbs':         0.0,
        'is_unified_memory':            is_unified_memory,
        'shared_memory_gb':             shared_memory_gb,
        'dedicated_vram_gb':            dedicated_vram_gb,
        'gpu_count':                    gpu_count,
        'gpu_memory_gb':                gpu_memory_gb,
        'gpu_core_count':               gpu_core_count,
        'peak_bandwidth_gbs':           peak_bandwidth_gbs,
        'tflops_fp32':                  tflops_fp32,
        'tflops_fp16':                  tflops_fp16,
        'fp16_support':                 fp16_support,
        'bf16_support':                 bf16_support,
        'interconnect_type':            interconnect_type,
        'host_to_device_bandwidth_gbs': host_to_device_bandwidth_gbs,
        'is_discrete_gpu':              is_discrete_gpu,
        'is_integrated_gpu':            is_integrated_gpu,
        'device_encoded':               device_encoded,
        'cpu_freq_ghz':                 cpu_freq_max,
        'memory_channels':              0,
    }


def _get_l2_cache_mb():
    """L2 캐시 크기 추정 (플랫폼별)"""
    system = platform.system()
    if system == 'Windows':
        try:
            result = subprocess.run(['wmic', 'cpu', 'get', 'L2CacheSize'],
                                    capture_output=True, text=True, timeout=3)
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return round(int(lines[1]) / 1024, 1)
        except Exception:
            pass
    elif system == 'Darwin':
        for key in ['hw.l2cachesize', 'hw.perflevel0.l2cachesize']:
            try:
                result = subprocess.run(['sysctl', '-n', key],
                                        capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    return round(int(result.stdout.strip()) / (1024 ** 2), 1)
            except Exception:
                pass
    elif system == 'Linux':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index2/size', 'r') as f:
                s = f.read().strip().lower()
            if s.endswith('k'):
                return round(int(s[:-1]) / 1024, 1)
            elif s.endswith('m'):
                return float(s[:-1])
        except Exception:
            pass
    return 0.0


def _get_l3_cache_mb():
    """L3 캐시 크기 추정"""
    system = platform.system()
    if system == 'Darwin':
        try:
            result = subprocess.run(['sysctl', '-n', 'hw.l3cachesize'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return round(int(result.stdout.strip()) / (1024 ** 2), 1)
        except Exception:
            pass
    elif system == 'Linux':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index3/size', 'r') as f:
                s = f.read().strip().lower()
            if s.endswith('k'):
                return round(int(s[:-1]) / 1024, 1)
            elif s.endswith('m'):
                return float(s[:-1])
        except Exception:
            pass
    return 0.0


def _fallback_hardware_info(device_str):
    """psutil 미설치 시 기본값 반환"""
    return {k: 0 for k in HARDWARE_FEATURE_KEYS}
