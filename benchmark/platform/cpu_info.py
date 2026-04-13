"""CPU 상세 정보 감지 — 크로스 플랫폼 (macOS/Windows/Linux)

Feature Schema v1.0 섹션 6-2 구현:
  cpu_cores_physical, cpu_cores_logical, cpu_perf_cores, cpu_efficiency_cores,
  cpu_freq_base_ghz, cpu_freq_boost_ghz, cpu_cache_l2_mb, cpu_cache_l3_mb
"""
import platform
import subprocess


def detect_cpu_info():
    """CPU 상세 정보 감지

    Returns:
        dict: cpu_cores_physical, cpu_cores_logical, cpu_perf_cores,
              cpu_efficiency_cores, cpu_freq_base_ghz, cpu_freq_boost_ghz,
              cpu_cache_l2_mb, cpu_cache_l3_mb
    """
    info = {
        'cpu_cores_physical': 0,
        'cpu_cores_logical': 0,
        'cpu_perf_cores': 0,
        'cpu_efficiency_cores': 0,
        'cpu_freq_base_ghz': 0.0,
        'cpu_freq_boost_ghz': 0.0,
        'cpu_cache_l2_mb': 0.0,
        'cpu_cache_l3_mb': 0.0,
    }

    os_name = platform.system()

    # psutil 우선 (크로스 플랫폼)
    try:
        import psutil
        info['cpu_cores_physical'] = psutil.cpu_count(logical=False) or 0
        info['cpu_cores_logical'] = psutil.cpu_count(logical=True) or 0
        freq = psutil.cpu_freq()
        if freq:
            if freq.current > 0:
                info['cpu_freq_base_ghz'] = round(freq.current / 1000, 2)
            if freq.max > 0:
                info['cpu_freq_boost_ghz'] = round(freq.max / 1000, 2)
    except ImportError:
        pass

    # OS별 상세 감지
    if os_name == 'Darwin':
        _detect_macos_cpu(info)
    elif os_name == 'Windows':
        _detect_windows_cpu(info)
    elif os_name == 'Linux':
        _detect_linux_cpu(info)

    # boost가 비어있으면 base로 fallback
    if info['cpu_freq_boost_ghz'] == 0.0 and info['cpu_freq_base_ghz'] > 0:
        info['cpu_freq_boost_ghz'] = info['cpu_freq_base_ghz']

    return info


def _run_cmd(cmd, timeout=5):
    """시스템 명령 실행 헬퍼"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ''


def _detect_macos_cpu(info):
    """macOS: sysctl + system_profiler"""
    # 물리/논리 코어
    phys = _run_cmd(['sysctl', '-n', 'hw.physicalcpu'])
    if phys:
        info['cpu_cores_physical'] = int(phys)
    logical = _run_cmd(['sysctl', '-n', 'hw.logicalcpu'])
    if logical:
        info['cpu_cores_logical'] = int(logical)

    # P/E 코어 (Apple Silicon)
    perf = _run_cmd(['sysctl', '-n', 'hw.perflevel0.physicalcpu'])
    if perf:
        info['cpu_perf_cores'] = int(perf)
    eff = _run_cmd(['sysctl', '-n', 'hw.perflevel1.physicalcpu'])
    if eff:
        info['cpu_efficiency_cores'] = int(eff)

    # 주파수 (Apple Silicon은 hw.cpufrequency가 없을 수 있음)
    for key in ['hw.cpufrequency_max', 'hw.cpufrequency']:
        val = _run_cmd(['sysctl', '-n', key])
        if val:
            ghz = round(int(val) / 1e9, 2)
            if info['cpu_freq_boost_ghz'] == 0:
                info['cpu_freq_boost_ghz'] = ghz
            if info['cpu_freq_base_ghz'] == 0:
                info['cpu_freq_base_ghz'] = ghz
            break

    # Apple Silicon 칩별 알려진 주파수
    if info['cpu_freq_boost_ghz'] == 0:
        brand = _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if 'apple' in brand.lower():
            info['cpu_freq_boost_ghz'] = 3.5  # P-core 평균
            if info['cpu_freq_base_ghz'] == 0:
                info['cpu_freq_base_ghz'] = 2.0  # E-core 평균

    # L2 캐시
    for key in ['hw.perflevel0.l2cachesize', 'hw.l2cachesize']:
        val = _run_cmd(['sysctl', '-n', key])
        if val:
            info['cpu_cache_l2_mb'] = round(int(val) / (1024 * 1024), 2)
            break

    # L3 캐시
    val = _run_cmd(['sysctl', '-n', 'hw.l3cachesize'])
    if val:
        info['cpu_cache_l3_mb'] = round(int(val) / (1024 * 1024), 2)


def _detect_windows_cpu(info):
    """Windows: wmic → PowerShell fallback"""
    # wmic cpu (deprecated Win11 이후, PowerShell fallback)
    max_clock = _run_cmd(['wmic', 'cpu', 'get', 'MaxClockSpeed', '/value'])
    if 'MaxClockSpeed' in max_clock:
        for line in max_clock.split('\n'):
            if 'MaxClockSpeed' in line:
                mhz = int(line.split('=')[1].strip())
                info['cpu_freq_boost_ghz'] = round(mhz / 1000, 2)
                break

    # PowerShell fallback for frequency
    if info['cpu_freq_boost_ghz'] == 0:
        ps_out = _run_cmd([
            'powershell', '-Command',
            '(Get-CimInstance Win32_Processor).MaxClockSpeed'
        ])
        if ps_out:
            try:
                info['cpu_freq_boost_ghz'] = round(int(ps_out) / 1000, 2)
            except ValueError:
                pass

    # base frequency
    base_clock = _run_cmd(['wmic', 'cpu', 'get', 'CurrentClockSpeed', '/value'])
    if 'CurrentClockSpeed' in base_clock:
        for line in base_clock.split('\n'):
            if 'CurrentClockSpeed' in line:
                mhz = int(line.split('=')[1].strip())
                info['cpu_freq_base_ghz'] = round(mhz / 1000, 2)
                break

    # L2 캐시
    l2_out = _run_cmd(['wmic', 'cpu', 'get', 'L2CacheSize', '/value'])
    if 'L2CacheSize' in l2_out:
        for line in l2_out.split('\n'):
            if 'L2CacheSize' in line:
                kb = int(line.split('=')[1].strip())
                info['cpu_cache_l2_mb'] = round(kb / 1024, 2)
                break

    # L3 캐시
    l3_out = _run_cmd(['wmic', 'cpu', 'get', 'L3CacheSize', '/value'])
    if 'L3CacheSize' in l3_out:
        for line in l3_out.split('\n'):
            if 'L3CacheSize' in line:
                kb = int(line.split('=')[1].strip())
                info['cpu_cache_l3_mb'] = round(kb / 1024, 2)
                break

    # physical cores (wmic)
    if info['cpu_cores_physical'] == 0:
        cores_out = _run_cmd(['wmic', 'cpu', 'get', 'NumberOfCores', '/value'])
        if 'NumberOfCores' in cores_out:
            for line in cores_out.split('\n'):
                if 'NumberOfCores' in line:
                    info['cpu_cores_physical'] = int(line.split('=')[1].strip())
                    break

    if info['cpu_cores_logical'] == 0:
        lp_out = _run_cmd([
            'wmic', 'cpu', 'get', 'NumberOfLogicalProcessors', '/value'])
        if 'NumberOfLogicalProcessors' in lp_out:
            for line in lp_out.split('\n'):
                if 'NumberOfLogicalProcessors' in line:
                    info['cpu_cores_logical'] = int(line.split('=')[1].strip())
                    break


def _detect_linux_cpu(info):
    """Linux: /proc/cpuinfo + /sys/devices/"""
    # 코어 수
    if info['cpu_cores_physical'] == 0:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cores = set()
                for line in f:
                    if line.startswith('core id'):
                        cores.add(line.split(':')[1].strip())
                if cores:
                    info['cpu_cores_physical'] = len(cores)
        except Exception:
            pass

    if info['cpu_cores_logical'] == 0:
        nproc = _run_cmd(['nproc'])
        if nproc:
            info['cpu_cores_logical'] = int(nproc)

    # 주파수
    if info['cpu_freq_boost_ghz'] == 0:
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq') as f:
                khz = int(f.read().strip())
                info['cpu_freq_boost_ghz'] = round(khz / 1e6, 2)
        except Exception:
            pass

    if info['cpu_freq_base_ghz'] == 0:
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/base_frequency') as f:
                khz = int(f.read().strip())
                info['cpu_freq_base_ghz'] = round(khz / 1e6, 2)
        except Exception:
            pass

    # L2 캐시
    if info['cpu_cache_l2_mb'] == 0:
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index2/size') as f:
                size_str = f.read().strip().lower()
                if size_str.endswith('k'):
                    info['cpu_cache_l2_mb'] = round(int(size_str[:-1]) / 1024, 2)
                elif size_str.endswith('m'):
                    info['cpu_cache_l2_mb'] = float(size_str[:-1])
        except Exception:
            pass

    # L3 캐시
    if info['cpu_cache_l3_mb'] == 0:
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index3/size') as f:
                size_str = f.read().strip().lower()
                if size_str.endswith('k'):
                    info['cpu_cache_l3_mb'] = round(int(size_str[:-1]) / 1024, 2)
                elif size_str.endswith('m'):
                    info['cpu_cache_l3_mb'] = float(size_str[:-1])
        except Exception:
            pass
