# ============================================================
# 하드웨어 정보 동적 감지 유틸리티
# ============================================================
#
# 출처: dal-merge 브랜치 (features/extractor.py) 내 get_hardware_info 포팅
# 수정: M1 Mac / MPS 지원, macOS L2 캐시 감지, 한국어 주석 추가
# 확장: 통합 스키마 6장 하드웨어 피처 33개 전부 반환 (train_predictor 92개 feature 대응)
#
# 역할: psutil/torch를 사용하여 실행 환경의 CPU, RAM, GPU 정보를 자동 수집합니다.
#
# 의존성: pip install psutil
#
# 작성자: 달현 (원본) / 김홍근 (MPS/macOS 지원 포팅, 92 feature 확장)
# ============================================================

import json
import platform
import subprocess
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except (ImportError, OSError):
    # Windows Smart App Control / WDAC 등으로 torch DLL 로드가 막히면 OSError 발생
    torch = None


# 통합 스키마 6장 하드웨어 피처 키 목록 (33개). 반환 dict는 이 키를 모두 가짐.
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

    통합 스키마 6장 하드웨어 피처 33개를 반환합니다.
    device_str이 'mps'일 때 M1 Mac의 통합 메모리를 GPU 메모리로 간주합니다.
    (M1은 CPU/GPU가 메모리를 공유하므로 ram_total_gb와 동일한 값 사용)

    Args:
        device_str (str): 'cpu', 'cuda', 'mps' 중 하나

    Returns:
        dict: 하드웨어 관련 feature 33개 키
            - device_type, os_type, accelerator_brand, accelerator_name
            - cpu_cores_physical, cpu_cores_logical, cpu_perf_cores, cpu_efficiency_cores
            - cpu_freq_base_ghz, cpu_freq_boost_ghz, cpu_cache_l2_mb, cpu_cache_l3_mb
            - ram_total_gb, memory_type, memory_bandwidth_gbs, is_unified_memory
            - shared_memory_gb, dedicated_vram_gb, gpu_count, gpu_memory_gb
            - gpu_core_count, peak_bandwidth_gbs, tflops_fp32, tflops_fp16
            - fp16_support, bf16_support, interconnect_type, host_to_device_bandwidth_gbs
            - is_discrete_gpu, is_integrated_gpu, device_encoded, cpu_freq_ghz, memory_channels
    """
    if psutil is None:
        return _fallback_hardware_info(device_str)

    system = platform.system()
    # os_type: 0=unknown, 1=Windows, 2=macOS, 3=Linux
    os_type = 1 if system == 'Windows' else (2 if system == 'Darwin' else (3 if system == 'Linux' else 0))

    # CPU 코어 수
    cpu_physical = psutil.cpu_count(logical=False) or 0
    cpu_logical = psutil.cpu_count(logical=True) or 0
    cpu_perf_cores = 0
    cpu_efficiency_cores = 0
    # Apple Silicon: 성능/효율 코어 구분 가능 시 채움 (선택)
    if system == 'Darwin' and cpu_logical == 8 and cpu_physical == 8:
        cpu_perf_cores = 4
        cpu_efficiency_cores = 4

    # CPU 클럭 (MHz → GHz)
    freq = psutil.cpu_freq()
    cpu_freq_max = round(freq.max / 1000, 2) if freq and freq.max else 0.0
    cpu_freq_min = round(freq.min / 1000, 2) if freq and freq.min else 0.0

    cpu_cache_l2_mb = _get_l2_cache_mb()
    cpu_cache_l3_mb = _get_l3_cache_mb()

    ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    memory_type = 0
    memory_bandwidth_gbs = 0.0
    memory_channels = 0

    # GPU 정보
    gpu_count = 0
    gpu_memory_gb = 0.0
    dedicated_vram_gb = 0.0
    shared_memory_gb = 0.0
    is_unified_memory = 0
    accelerator_brand = 0
    accelerator_name = 0
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
        gpu_memory_gb = round(props.total_memory / (1024 ** 3), 1)
        dedicated_vram_gb = gpu_memory_gb
        is_discrete_gpu = 1
        accelerator_brand = 1   # NVIDIA
        accelerator_name = 1    # NVIDIA
        gpu_core_count = getattr(props, 'multi_processor_count', 0) or 0
        fp16_support = 1
        bf16_support = 1 if getattr(props, 'major', 0) >= 8 else 0
        interconnect_type = 1  # PCIe
    elif device_str == 'mps' and torch and torch.backends.mps.is_available():
        gpu_count = 1
        gpu_memory_gb = ram_total_gb
        shared_memory_gb = ram_total_gb
        is_unified_memory = 1
        is_integrated_gpu = 1
        accelerator_brand = 2  # Apple
        accelerator_name = 2   # Apple Silicon
        interconnect_type = 2  # Unified

    # device_type: 0=cpu, 1=gpu
    device_type = 1 if device_str in ('cuda', 'mps') else 0
    # device_encoded: 0=cpu, 1=cuda/mps (기존 호환)
    device_encoded = 1 if device_str in ('cuda', 'mps') else 0

    return {
        'device_type': device_type,
        'os_type': os_type,
        'accelerator_brand': accelerator_brand,
        'accelerator_name': accelerator_name,
        'cpu_cores_physical': cpu_physical,
        'cpu_cores_logical': cpu_logical,
        'cpu_perf_cores': cpu_perf_cores,
        'cpu_efficiency_cores': cpu_efficiency_cores,
        'cpu_freq_base_ghz': cpu_freq_min,
        'cpu_freq_boost_ghz': cpu_freq_max,
        'cpu_cache_l2_mb': cpu_cache_l2_mb,
        'cpu_cache_l3_mb': cpu_cache_l3_mb,
        'ram_total_gb': ram_total_gb,
        'memory_type': memory_type,
        'memory_bandwidth_gbs': memory_bandwidth_gbs,
        'is_unified_memory': is_unified_memory,
        'shared_memory_gb': shared_memory_gb,
        'dedicated_vram_gb': dedicated_vram_gb,
        'gpu_count': gpu_count,
        'gpu_memory_gb': gpu_memory_gb,
        'gpu_core_count': gpu_core_count,
        'peak_bandwidth_gbs': peak_bandwidth_gbs,
        'tflops_fp32': tflops_fp32,
        'tflops_fp16': tflops_fp16,
        'fp16_support': fp16_support,
        'bf16_support': bf16_support,
        'interconnect_type': interconnect_type,
        'host_to_device_bandwidth_gbs': host_to_device_bandwidth_gbs,
        'is_discrete_gpu': is_discrete_gpu,
        'is_integrated_gpu': is_integrated_gpu,
        'device_encoded': device_encoded,
        'cpu_freq_ghz': cpu_freq_max,
        'memory_channels': memory_channels,
    }


def _get_l3_cache_mb():
    """L3 캐시 크기 추정. 실패 시 0."""
    system = platform.system()
    if system == 'Darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.l3cachesize'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return round(int(result.stdout.strip()) / (1024 ** 2), 1)
        except Exception:
            pass
    elif system == 'Linux':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index3/size', 'r') as f:
                size_str = f.read().strip().lower()
            if size_str.endswith('k'):
                return round(int(size_str[:-1]) / 1024, 1)
            elif size_str.endswith('m'):
                return round(int(size_str[:-1]), 1)
        except Exception:
            pass
    return 0.0


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
    """psutil 미설치 시 MacBook Air M1 고정값 반환 (33개 키)."""
    base = {
        'device_type': 1 if device_str in ('cuda', 'mps') else 0,
        'os_type': 2,
        'accelerator_brand': 2 if device_str == 'mps' else 0,
        'accelerator_name': 2 if device_str == 'mps' else 0,
        'cpu_cores_physical': 8,
        'cpu_cores_logical': 8,
        'cpu_perf_cores': 4,
        'cpu_efficiency_cores': 4,
        'cpu_freq_base_ghz': 0.6,
        'cpu_freq_boost_ghz': 3.2,
        'cpu_cache_l2_mb': 12.0,
        'cpu_cache_l3_mb': 0.0,
        'ram_total_gb': 8.0,
        'memory_type': 0,
        'memory_bandwidth_gbs': 0.0,
        'is_unified_memory': 1 if device_str == 'mps' else 0,
        'shared_memory_gb': 8.0 if device_str == 'mps' else 0.0,
        'dedicated_vram_gb': 0.0,
        'gpu_count': 1 if device_str in ('cuda', 'mps') else 0,
        'gpu_memory_gb': 8.0 if device_str == 'mps' else 0.0,
        'gpu_core_count': 0,
        'peak_bandwidth_gbs': 0.0,
        'tflops_fp32': 0.0,
        'tflops_fp16': 0.0,
        'fp16_support': 0,
        'bf16_support': 0,
        'interconnect_type': 2 if device_str == 'mps' else 0,
        'host_to_device_bandwidth_gbs': 0.0,
        'is_discrete_gpu': 0,
        'is_integrated_gpu': 1 if device_str == 'mps' else 0,
        'device_encoded': 1 if device_str in ('cuda', 'mps') else 0,
        'cpu_freq_ghz': 3.2,
        'memory_channels': 0,
    }
    return base


# ──────────────────────────────────────────────────────────
# 실험 환경 캡션 (시각화·보고서용)
# ──────────────────────────────────────────────────────────

def load_benchmark_gpu_hardware_snapshot(benchmark_json_path):
    """benchmark_results.json 에서 GPU(CUDA) 첫 행의 하드웨어 스냅샷.

    벤치 실행 시 피처로 기록된 cpu_cores, ram_total_gb, gpu_memory_gb 등.
    GPU **모델 문자열**은 JSON에 없을 수 있음 → 런타임 torch 로 보완.
    """
    path = Path(benchmark_json_path)
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    for r in data:
        if r.get("device") == "GPU(CUDA)":
            return {
                "cpu_cores": int(r.get("cpu_cores") or 0),
                "cpu_freq_ghz": float(r.get("cpu_freq_ghz") or 0.0),
                "ram_total_gb": float(r.get("ram_total_gb") or 0.0),
                "gpu_memory_gb": float(r.get("gpu_memory_gb") or 0.0),
            }
    return {}


def collect_experiment_environment_ko(benchmark_json_path=None):
    """실험 PC 환경 dict (한글 보고·그림 캡션용).

    - GPU 이름 / SM 수 / VRAM: torch.cuda 사용 가능 시 우선.
    - CPU·RAM·벤치 시점 VRAM: benchmark JSON 스냅샷 + psutil 보완.
    """
    snap = load_benchmark_gpu_hardware_snapshot(benchmark_json_path) if benchmark_json_path else {}

    system = platform.system()
    rel = platform.release()
    os_line = f"{system} {rel}".strip()

    cpu_cores = int(snap.get("cpu_cores") or 0)
    cpu_ghz = float(snap.get("cpu_freq_ghz") or 0.0)
    ram_gb = float(snap.get("ram_total_gb") or 0.0)
    vram_snap = float(snap.get("gpu_memory_gb") or 0.0)

    if psutil:
        if cpu_cores <= 0:
            cpu_cores = int(psutil.cpu_count(logical=True) or 0)
        if cpu_ghz <= 0:
            freq = psutil.cpu_freq()
            if freq and freq.max:
                cpu_ghz = round(freq.max / 1000, 2)
        if ram_gb <= 0:
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    gpu_name = "알 수 없음"
    gpu_sm = 0
    vram_gb = vram_snap

    if torch and getattr(torch, "cuda", None) and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_sm = int(getattr(props, "multi_processor_count", 0) or 0)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
    elif vram_snap > 0:
        gpu_name = "GPU(CUDA) — 벤치마크 JSON 기준 (모델명은 torch 필요)"
    else:
        gpu_name = "GPU(CUDA) 미기록 또는 torch/CUDA 사용 불가"

    return {
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram_gb,
        "gpu_sm_count": gpu_sm,
        "cpu_cores": cpu_cores,
        "cpu_freq_ghz": cpu_ghz,
        "ram_total_gb": ram_gb,
        "os_line": os_line,
    }


def format_experiment_environment_caption_ko(info: dict) -> str:
    """그림 하단 2줄 캡션 (짧은 형식)."""
    if not info:
        return ""
    L1 = (
        f"GPU: {info['gpu_name']} · VRAM {info['gpu_vram_gb']}GB · "
        f"SM {info['gpu_sm_count']}"
    )
    L2 = (
        f"CPU {info['cpu_cores']}코어 · {info['cpu_freq_ghz']}GHz · "
        f"RAM {info['ram_total_gb']}GB · {info['os_line']}"
    )
    return f"{L1}\n{L2}"


def format_experiment_environment_report_ko(info: dict) -> str:
    """보고서용 조금 긴 설명 (마크다운 아님, 순수 텍스트)."""
    if not info:
        return ""
    lines = [
        "=== 실험 환경 (자동 요약) ===",
        f"GPU 이름: {info['gpu_name']}",
        f"GPU VRAM: {info['gpu_vram_gb']} GB",
        f"GPU SM(멀티프로세서) 수: {info['gpu_sm_count']}",
        f"CPU: {info['cpu_cores']} 논리 코어, 최대 약 {info['cpu_freq_ghz']} GHz",
        f"시스템 RAM: {info['ram_total_gb']} GB",
        f"OS: {info['os_line']}",
        "",
        "※ GPU 모델 문자열은 torch.cuda.get_device_name(0) 으로 읽습니다.",
        "※ CPU/RAM 일부는 benchmark_results.json 의 GPU 행 스냅샷과 병합됩니다.",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────────────────
# from utils.hardware_info import get_hardware_info
# hw = get_hardware_info('mps')
# print(hw)  # {'cpu_cores': 8, 'cpu_freq_ghz': 3.2, ...}
