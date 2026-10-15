"""GPU 상세 정보 감지 — CUDA / MPS / ROCm / 미사용

Feature Schema v1.0 섹션 6-4, 6-5 구현:
  gpu_count, gpu_memory_gb, gpu_core_count, gpu_tensor_core_count,
  gpu_compute_capability, gpu_clock_ghz, peak_bandwidth_gbs,
  tflops_fp32, tflops_fp16, fp16_support, bf16_support,
  interconnect_type, host_to_device_bandwidth_gbs,
  is_discrete_gpu, is_integrated_gpu
"""
import platform
import subprocess

import torch


# NVIDIA Compute Capability → 텐서코어/SM 비율 + fp16/bf16 지원 매핑
_NVIDIA_CC_INFO = {
    # CC: (tensor_cores_per_sm, fp16, bf16, description)
    7.0: (8, 1, 0, 'Volta'),       # V100
    7.5: (8, 1, 0, 'Turing'),      # RTX 2080
    8.0: (4, 1, 1, 'Ampere'),      # A100
    8.6: (4, 1, 1, 'Ampere'),      # RTX 3090
    8.9: (4, 1, 1, 'Ada Lovelace'),  # RTX 4090
    9.0: (4, 1, 1, 'Hopper'),      # H100
}


def detect_gpu_info(device_str='cpu'):
    """GPU 상세 정보 감지

    Returns:
        dict: Feature Schema 6-4, 6-5 피처
    """
    info = {
        'gpu_count': 0,
        'gpu_memory_gb': 0.0,
        'gpu_core_count': 0,
        'gpu_tensor_core_count': 0,
        'gpu_compute_capability': 0.0,
        'gpu_clock_ghz': 0.0,
        'peak_bandwidth_gbs': 0.0,
        'tflops_fp32': 0.0,
        'tflops_fp16': 0.0,
        'fp16_support': 0,
        'bf16_support': 0,
        'interconnect_type': 'none',
        'host_to_device_bandwidth_gbs': 0.0,
        'is_discrete_gpu': 0,
        'is_integrated_gpu': 0,
    }

    if device_str == 'cuda' and torch.cuda.is_available():
        _detect_cuda_gpu(info)
    elif device_str == 'mps' and torch.backends.mps.is_available():
        _detect_mps_gpu(info)
    elif device_str == 'cpu':
        info['interconnect_type'] = 'none'

    return info


def _cuda_cores_per_sm(cc):
    """NVIDIA Compute Capability → SM당 CUDA(shader) 코어 수."""
    if cc >= 8.9:      # Ada Lovelace (RTX 40xx)
        return 128
    if cc >= 8.0:      # Ampere (RTX 30xx / A100)
        return 128
    if cc >= 7.5:      # Turing (RTX 20xx)
        return 64
    if cc >= 7.0:      # Volta (V100)
        return 64
    if cc >= 6.0:      # Pascal (GTX 10xx)
        return 128 if cc == 6.1 else 64
    return 128         # Maxwell 등 근사


def _detect_cuda_gpu(info):
    """NVIDIA CUDA GPU 감지 — torch.cuda API + nvidia-smi fallback"""
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count  # SM(멀티프로세서) 수

    info['gpu_count'] = torch.cuda.device_count()
    info['gpu_memory_gb'] = round(props.total_memory / (1024 ** 3), 1)
    info['is_discrete_gpu'] = 1
    info['is_integrated_gpu'] = 0

    # Compute capability
    cc = float(f"{props.major}.{props.minor}")
    info['gpu_compute_capability'] = cc

    # 실제 CUDA(shader) 코어 수 = SM 수 × SM당 코어 수
    # (예: RTX 4060 Ti = 34 SM × 128 = 4352 CUDA 코어)
    # Apple GPU 코어 수와 의미를 맞추기 위해 SM 수가 아닌 실제 코어 수를 기록.
    info['gpu_core_count'] = sm_count * _cuda_cores_per_sm(cc)

    # fp16/bf16 지원 + 텐서코어(=SM 기준)
    cc_key = cc
    if cc_key not in _NVIDIA_CC_INFO:
        cc_key = max((k for k in _NVIDIA_CC_INFO if k <= cc), default=0)

    if cc_key > 0:
        tc_per_sm, fp16, bf16, _ = _NVIDIA_CC_INFO[cc_key]
        info['gpu_tensor_core_count'] = sm_count * tc_per_sm
        info['fp16_support'] = fp16
        info['bf16_support'] = bf16
    else:
        # CC < 7.0: 텐서코어 없음, fp16은 일부 지원
        info['fp16_support'] = 1 if cc >= 5.3 else 0
        info['bf16_support'] = 0

    # GPU 클럭 (nvidia-smi)
    clock_out = _run_cmd([
        'nvidia-smi', '--query-gpu=clocks.max.graphics',
        '--format=csv,noheader,nounits'
    ])
    if clock_out:
        try:
            info['gpu_clock_ghz'] = round(int(clock_out.strip().split('\n')[0]) / 1000, 2)
        except ValueError:
            pass

    # 메모리 대역폭 (nvidia-smi)
    bw_out = _run_cmd([
        'nvidia-smi', '--query-gpu=memory.bus_width',
        '--format=csv,noheader,nounits'
    ])
    mem_clock = _run_cmd([
        'nvidia-smi', '--query-gpu=clocks.max.memory',
        '--format=csv,noheader,nounits'
    ])
    if bw_out and mem_clock:
        try:
            bus_width = int(bw_out.strip().split('\n')[0])
            mem_mhz = int(mem_clock.strip().split('\n')[0])
            # GDDR6: effective rate = clock × 2 (double data rate)
            info['peak_bandwidth_gbs'] = round(
                bus_width * mem_mhz * 2 / 8 / 1000, 1)
        except ValueError:
            pass

    # TFLOPS 추정: CUDA 코어 × clock × 2 (FMA) / 1e3
    # (gpu_core_count가 이미 실제 CUDA 코어 수이므로 그대로 사용)
    if info['gpu_clock_ghz'] > 0 and info['gpu_core_count'] > 0:
        info['tflops_fp32'] = round(
            info['gpu_core_count'] * info['gpu_clock_ghz'] * 2 / 1000, 2)
        if info['fp16_support']:
            info['tflops_fp16'] = round(info['tflops_fp32'] * 2, 2)

    # PCIe 인터커넥트
    pcie_gen = _run_cmd([
        'nvidia-smi', '--query-gpu=pcie.link.gen.current',
        '--format=csv,noheader,nounits'
    ])
    if pcie_gen:
        try:
            gen = int(pcie_gen.strip().split('\n')[0])
            info['interconnect_type'] = f'pcie{gen}'
            # 대역폭: PCIe gen × width × encoding
            pcie_bw = {3: 8, 4: 16, 5: 32}  # GB/s per x16
            info['host_to_device_bandwidth_gbs'] = pcie_bw.get(gen, 8)
        except ValueError:
            info['interconnect_type'] = 'pcie3'


def _detect_mps_gpu(info):
    """Apple Silicon MPS GPU 감지"""
    os_name = platform.system()
    if os_name != 'Darwin':
        return

    info['gpu_count'] = 1
    info['is_discrete_gpu'] = 0
    info['is_integrated_gpu'] = 1
    info['interconnect_type'] = 'unified'
    info['fp16_support'] = 1
    info['bf16_support'] = 0  # Apple GPU는 bf16 미지원 (fp16만)

    # GPU 코어 수 (system_profiler)
    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Total Number of Cores' in line:
                    info['gpu_core_count'] = int(line.split(':')[1].strip())
                    break
    except Exception:
        pass

    # Apple Silicon 통합 메모리: GPU 메모리 ≈ RAM의 75%
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        info['gpu_memory_gb'] = round(ram_gb * 0.75, 1)
    except ImportError:
        ram_out = _run_cmd(['sysctl', '-n', 'hw.memsize'])
        if ram_out:
            ram_gb = int(ram_out) / (1024 ** 3)
            info['gpu_memory_gb'] = round(ram_gb * 0.75, 1)

    # Apple Silicon TFLOPS/대역폭 추정 (칩 감지)
    brand = _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string']).lower()
    if 'apple' in brand:
        # 코어 수 기반 추정
        cores = info['gpu_core_count']
        if cores > 0:
            # Apple GPU: ~0.5 TFLOPS/core (FP32), ~1.0 TFLOPS/core (FP16)
            info['tflops_fp32'] = round(cores * 0.5, 2)
            info['tflops_fp16'] = round(cores * 1.0, 2)

        # 대역폭 추정 (칩별)
        chip = _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if 'M4' in chip:
            info['peak_bandwidth_gbs'] = 120.0  # M4 base
        elif 'M3' in chip:
            info['peak_bandwidth_gbs'] = 100.0
        elif 'M2' in chip:
            info['peak_bandwidth_gbs'] = 100.0
        elif 'M1' in chip:
            info['peak_bandwidth_gbs'] = 68.25
        else:
            info['peak_bandwidth_gbs'] = 68.25  # conservative default

        # 통합 메모리: host↔device 대역폭 = 메모리 대역폭 (전송 불필요)
        info['host_to_device_bandwidth_gbs'] = info['peak_bandwidth_gbs']


def _run_cmd(cmd, timeout=5):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ''
