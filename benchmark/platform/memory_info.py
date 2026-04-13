"""메모리 구조 감지 — Mac/Windows 핵심 차이

Feature Schema v1.0 섹션 6-3 구현:
  ram_total_gb, memory_type, memory_bandwidth_gbs,
  is_unified_memory, shared_memory_gb, dedicated_vram_gb
"""
import platform
import subprocess


def detect_memory_info(device_str='cpu', gpu_memory_gb=0.0):
    """메모리 구조 감지

    Args:
        device_str: 디바이스 타입 ('cpu', 'cuda', 'mps')
        gpu_memory_gb: GPU 메모리 (이미 감지된 경우)

    Returns:
        dict: Feature Schema 6-3 피처
    """
    info = {
        'ram_total_gb': 0.0,
        'memory_type': 'unknown',
        'memory_bandwidth_gbs': 0.0,
        'is_unified_memory': 0,
        'shared_memory_gb': 0.0,
        'dedicated_vram_gb': 0.0,
    }

    os_name = platform.system()

    # RAM 총 용량
    try:
        import psutil
        info['ram_total_gb'] = round(
            psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        if os_name == 'Darwin':
            val = _run_cmd(['sysctl', '-n', 'hw.memsize'])
            if val:
                info['ram_total_gb'] = round(int(val) / (1024 ** 3), 1)

    # OS별 메모리 구조 감지
    if os_name == 'Darwin':
        _detect_macos_memory(info)
    elif os_name == 'Windows':
        _detect_windows_memory(info, device_str, gpu_memory_gb)
    elif os_name == 'Linux':
        _detect_linux_memory(info, device_str, gpu_memory_gb)

    return info


def _detect_macos_memory(info):
    """macOS: Apple Silicon = unified memory"""
    brand = _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string']).lower()

    if 'apple' in brand:
        info['memory_type'] = 'unified'
        info['is_unified_memory'] = 1
        info['shared_memory_gb'] = info['ram_total_gb']
        info['dedicated_vram_gb'] = 0.0

        # Apple Silicon 대역폭 추정
        chip = _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if 'M4 Pro' in chip:
            info['memory_bandwidth_gbs'] = 273.0
        elif 'M4 Max' in chip:
            info['memory_bandwidth_gbs'] = 546.0
        elif 'M4' in chip:
            info['memory_bandwidth_gbs'] = 120.0
        elif 'M3 Pro' in chip:
            info['memory_bandwidth_gbs'] = 150.0
        elif 'M3 Max' in chip:
            info['memory_bandwidth_gbs'] = 400.0
        elif 'M3' in chip:
            info['memory_bandwidth_gbs'] = 100.0
        elif 'M2 Pro' in chip:
            info['memory_bandwidth_gbs'] = 200.0
        elif 'M2 Max' in chip:
            info['memory_bandwidth_gbs'] = 400.0
        elif 'M2' in chip:
            info['memory_bandwidth_gbs'] = 100.0
        elif 'M1 Pro' in chip:
            info['memory_bandwidth_gbs'] = 200.0
        elif 'M1 Max' in chip:
            info['memory_bandwidth_gbs'] = 400.0
        elif 'M1' in chip:
            info['memory_bandwidth_gbs'] = 68.25
        else:
            info['memory_bandwidth_gbs'] = 68.25
    else:
        # Intel Mac
        info['memory_type'] = 'ddr4'
        info['is_unified_memory'] = 0
        info['shared_memory_gb'] = 0.0
        info['dedicated_vram_gb'] = 0.0


def _detect_windows_memory(info, device_str, gpu_memory_gb):
    """Windows: discrete memory"""
    info['is_unified_memory'] = 0

    # 메모리 타입 감지
    mem_type = _run_cmd([
        'wmic', 'memorychip', 'get', 'SMBIOSMemoryType', '/value'])
    if 'SMBIOSMemoryType' in mem_type:
        for line in mem_type.split('\n'):
            if 'SMBIOSMemoryType' in line:
                smbios = line.split('=')[1].strip()
                type_map = {'26': 'ddr4', '34': 'ddr5', '24': 'ddr3'}
                info['memory_type'] = type_map.get(smbios, 'ddr4')
                break

    # PowerShell fallback
    if info['memory_type'] == 'unknown':
        ps_out = _run_cmd([
            'powershell', '-Command',
            '(Get-CimInstance Win32_PhysicalMemory | Select -First 1).SMBIOSMemoryType'
        ])
        if ps_out:
            type_map = {'26': 'ddr4', '34': 'ddr5', '24': 'ddr3'}
            info['memory_type'] = type_map.get(ps_out.strip(), 'ddr4')

    # 메모리 속도 → 대역폭 추정
    speed = _run_cmd(['wmic', 'memorychip', 'get', 'Speed', '/value'])
    if 'Speed' in speed:
        for line in speed.split('\n'):
            if 'Speed' in line:
                try:
                    mhz = int(line.split('=')[1].strip())
                    channels = 2  # 대부분 듀얼채널
                    # 대역폭 = speed × bus_width × channels / 8
                    info['memory_bandwidth_gbs'] = round(
                        mhz * 8 * channels / 1000, 1)  # DDR = ×2 이미 반영
                except ValueError:
                    pass
                break

    # VRAM
    if device_str == 'cuda' and gpu_memory_gb > 0:
        info['dedicated_vram_gb'] = gpu_memory_gb
        info['shared_memory_gb'] = 0.0
    else:
        info['dedicated_vram_gb'] = 0.0
        info['shared_memory_gb'] = 0.0


def _detect_linux_memory(info, device_str, gpu_memory_gb):
    """Linux: discrete or unified"""
    info['is_unified_memory'] = 0  # 대부분 discrete

    # 메모리 타입 감지 (dmidecode 필요, root 권한)
    dmi = _run_cmd(['sudo', 'dmidecode', '-t', 'memory'])
    if 'DDR5' in dmi:
        info['memory_type'] = 'ddr5'
    elif 'DDR4' in dmi:
        info['memory_type'] = 'ddr4'
    elif 'DDR3' in dmi:
        info['memory_type'] = 'ddr3'

    # 메모리 속도
    if 'Speed:' in dmi:
        for line in dmi.split('\n'):
            if 'Configured Memory Speed:' in line or 'Speed:' in line:
                parts = line.split(':')[1].strip().split()
                if parts and parts[0].isdigit():
                    mhz = int(parts[0])
                    channels = 2
                    info['memory_bandwidth_gbs'] = round(
                        mhz * 8 * channels / 1000, 1)
                    break

    if device_str == 'cuda' and gpu_memory_gb > 0:
        info['dedicated_vram_gb'] = gpu_memory_gb
    else:
        info['dedicated_vram_gb'] = 0.0


def _run_cmd(cmd, timeout=5):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ''
