# ============================================================
# 하드웨어 정보 동적 감지 유틸리티
# ============================================================
#
# 출처: dal-merge 브랜치 (features/extractor.py) 내 get_hardware_info 포팅
# 수정: M1 Mac / MPS 지원, macOS L2 캐시 감지, 한국어 주석 추가
#
# 역할: psutil을 사용하여 실행 환경의 CPU, RAM, GPU 정보를 자동 수집합니다.
#       train_predictor.py의 HARDWARE_INFO는 고정값(M1)을 쓰지만,
#       다른 PC에서 실험할 때는 이 모듈로 동적 감지가 가능합니다.
#
# 의존성: pip install psutil
#
# 작성자: 달현 (원본) / 김홍근 (MPS/macOS 지원 포팅)
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


def get_hardware_info(device_str='cpu'):
    """실행 환경의 하드웨어 정보를 동적으로 수집합니다.

    device_str이 'mps'일 때 M1 Mac의 통합 메모리를 GPU 메모리로 간주합니다.
    (M1은 CPU/GPU가 메모리를 공유하므로 ram_total_gb와 동일한 값 사용)

    Args:
        device_str (str): 'cpu', 'cuda', 'mps' 중 하나

    Returns:
        dict: 하드웨어 관련 feature
            - cpu_cores: CPU 논리 코어 수
            - cpu_freq_ghz: CPU 최대 클럭 (GHz)
            - cpu_cache_l2_mb: L2 캐시 크기 (MB), 추정 실패 시 0
            - ram_total_gb: 전체 RAM (GB)
            - gpu_memory_gb: GPU 메모리 (GB), CUDA/MPS 미사용 시 0
    """
    if psutil is None:
        return _fallback_hardware_info(device_str)

    # CPU 코어 수 (논리 코어, 하이퍼스레딩 포함)
    cpu_cores = psutil.cpu_count(logical=True) or 0

    # CPU 클럭 속도 (MHz → GHz 변환)
    freq = psutil.cpu_freq()
    cpu_freq_ghz = round(freq.max / 1000, 2) if freq and freq.max else 0.0

    # L2 캐시 크기 (플랫폼별로 다른 방법 사용)
    cpu_cache_l2_mb = _get_l2_cache_mb()

    # 전체 RAM (바이트 → GB 변환)
    ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    # GPU 메모리
    gpu_memory_gb = 0.0
    if device_str == 'cuda' and torch and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)
    elif device_str == 'mps' and torch and torch.backends.mps.is_available():
        # M1 Mac: CPU와 GPU가 통합 메모리 공유 → RAM과 동일하게 간주
        gpu_memory_gb = ram_total_gb

    return {
        'cpu_cores': cpu_cores,
        'cpu_freq_ghz': cpu_freq_ghz,
        'cpu_cache_l2_mb': cpu_cache_l2_mb,
        'ram_total_gb': ram_total_gb,
        'gpu_memory_gb': gpu_memory_gb,
    }


def _get_l2_cache_mb():
    """L2 캐시 크기 추정 (플랫폼별)

    - Windows: wmic cpu get L2CacheSize
    - macOS (Intel): sysctl hw.l2cachesize
    - macOS (Apple Silicon M1): sysctl hw.l2cachesize (또는 0 반환)
    - Linux: /sys/devices/system/cpu/cpu0/cache/ 에서 읽기
    - 추정 실패 시 0 반환
    """
    system = platform.system()

    # Windows
    if system == 'Windows':
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'L2CacheSize'],
                capture_output=True, text=True, timeout=3
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) > 1 and lines[1].isdigit():
                return round(int(lines[1]) / 1024, 1)  # KB → MB
        except Exception:
            pass

    # macOS (Intel, Apple Silicon)
    elif system == 'Darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.l2cachesize'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                # 바이트 → MB
                return round(int(result.stdout.strip()) / (1024 ** 2), 1)
        except Exception:
            pass
        # M1 Mac: hw.l2cachesize가 없거나 0일 수 있음. 대략 12MB (성능코어 4×3MB)
        # sysctl hw.perflevel0.l2cachesize 등으로 시도할 수 있으나 단순화
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.perflevel0.l2cachesize'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return round(int(result.stdout.strip()) / (1024 ** 2), 1)
        except Exception:
            pass

    # Linux
    elif system == 'Linux':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index2/size', 'r') as f:
                size_str = f.read().strip().lower()
            if size_str.endswith('k'):
                return round(int(size_str[:-1]) / 1024, 1)
            elif size_str.endswith('m'):
                return round(int(size_str[:-1]), 1)
        except Exception:
            pass

    return 0.0


def _fallback_hardware_info(device_str):
    """psutil 미설치 시 MacBook Air M1 고정값 반환"""
    return {
        'cpu_cores': 8,
        'cpu_freq_ghz': 3.2,
        'cpu_cache_l2_mb': 12.0,
        'ram_total_gb': 8.0,
        'gpu_memory_gb': 8.0 if device_str == 'mps' else 0.0,
    }


# ──────────────────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────────────────
# from utils.hardware_info import get_hardware_info
# hw = get_hardware_info('mps')
# print(hw)  # {'cpu_cores': 8, 'cpu_freq_ghz': 3.2, ...}
