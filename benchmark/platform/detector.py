"""플랫폼 자동 감지 오케스트레이터

Feature Schema v1.0 섹션 6 전체를 하나의 detect() 호출로 통합.
macOS / Windows / Linux + CPU / CUDA / MPS 자동 분기.
"""
import platform

import torch

from .cpu_info import detect_cpu_info
from .gpu_info import detect_gpu_info
from .memory_info import detect_memory_info


# 가속기 브랜드 매핑
_BRAND_MAP = {
    'cuda': 'nvidia',
    'mps': 'apple',
    'cpu': 'none',
}

# OS 이름 정규화
_OS_MAP = {
    'Darwin': 'macos',
    'Windows': 'windows',
    'Linux': 'linux',
}


def detect_platform(device_str='cpu'):
    """전체 플랫폼 정보 자동 감지

    Args:
        device_str: 'cpu', 'cuda', 'mps'

    Returns:
        dict: Feature Schema v1.0 하드웨어 피처 33개 전체
    """
    os_name = platform.system()

    # 1. CPU 감지
    cpu = detect_cpu_info()

    # 2. GPU 감지
    gpu = detect_gpu_info(device_str)

    # 3. 메모리 구조 감지
    mem = detect_memory_info(device_str, gpu['gpu_memory_gb'])

    # 4. 디바이스 일반 정보
    accelerator_brand = _BRAND_MAP.get(device_str, 'none')
    accelerator_name = _get_accelerator_name(device_str)

    # 5. 통합
    result = {
        # 6-1. 디바이스 일반
        'device_type': device_str,
        'os_type': _OS_MAP.get(os_name, 'linux'),
        'accelerator_brand': accelerator_brand,
        'accelerator_name': accelerator_name,

        # 6-2. CPU
        'cpu_cores_physical': cpu['cpu_cores_physical'],
        'cpu_cores_logical': cpu['cpu_cores_logical'],
        'cpu_perf_cores': cpu['cpu_perf_cores'],
        'cpu_efficiency_cores': cpu['cpu_efficiency_cores'],
        'cpu_freq_base_ghz': cpu['cpu_freq_base_ghz'],
        'cpu_freq_boost_ghz': cpu['cpu_freq_boost_ghz'],
        'cpu_cache_l2_mb': cpu['cpu_cache_l2_mb'],
        'cpu_cache_l3_mb': cpu['cpu_cache_l3_mb'],

        # 6-3. 메모리 구조
        'ram_total_gb': mem['ram_total_gb'],
        'memory_type': mem['memory_type'],
        'memory_bandwidth_gbs': mem['memory_bandwidth_gbs'],
        'is_unified_memory': mem['is_unified_memory'],
        'shared_memory_gb': mem['shared_memory_gb'],
        'dedicated_vram_gb': mem['dedicated_vram_gb'],

        # 6-4. GPU
        'gpu_count': gpu['gpu_count'],
        'gpu_memory_gb': gpu['gpu_memory_gb'],
        'gpu_core_count': gpu['gpu_core_count'],
        'gpu_tensor_core_count': gpu['gpu_tensor_core_count'],
        'gpu_compute_capability': gpu['gpu_compute_capability'],
        'gpu_clock_ghz': gpu['gpu_clock_ghz'],
        'peak_bandwidth_gbs': gpu['peak_bandwidth_gbs'],
        'tflops_fp32': gpu['tflops_fp32'],
        'tflops_fp16': gpu['tflops_fp16'],
        'fp16_support': gpu['fp16_support'],
        'bf16_support': gpu['bf16_support'],

        # 6-5. 인터커넥트
        'interconnect_type': gpu['interconnect_type'],
        'host_to_device_bandwidth_gbs': gpu['host_to_device_bandwidth_gbs'],
        'is_discrete_gpu': gpu['is_discrete_gpu'],
        'is_integrated_gpu': gpu['is_integrated_gpu'],
    }

    return result


def _get_accelerator_name(device_str):
    """가속기 구체적 모델명 감지"""
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    if device_str == 'mps':
        import subprocess
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brand = result.stdout.strip()
                # "Apple M4" 추출
                for chip in ['M4 Ultra', 'M4 Max', 'M4 Pro', 'M4',
                             'M3 Ultra', 'M3 Max', 'M3 Pro', 'M3',
                             'M2 Ultra', 'M2 Max', 'M2 Pro', 'M2',
                             'M1 Ultra', 'M1 Max', 'M1 Pro', 'M1']:
                    if chip in brand:
                        return f'Apple {chip}'
                return brand
        except Exception:
            pass

    return 'none'
