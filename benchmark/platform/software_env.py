"""소프트웨어 환경·시스템 부하 수집 — 측정 재현성을 위해 결과 행에 기록한다.

심사 대응 R1-12(라이브러리 버전 미기록)와 R1-11(측정 잡음)의 근거 자료.

  collect_software_env(): Python/OS/PyTorch/CUDA/cuDNN/oneDNN/BLAS/GPU 드라이버 버전
                          → 결과 행의 문자열 필드 sw_* (학습 입력에서는 제외됨)
  measure_system_load():  측정 직전 CPU 사용률·GPU 사용률 → load_* 필드 (유휴 상태 증빙)
  is_idle():              유휴 판정 (--require-idle 옵션이 사용)
"""
import platform
import re
import subprocess


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _nvidia_query(fields):
    """nvidia-smi --query-gpu 결과의 첫 GPU 행을 리스트로 반환 (실패 시 [])."""
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    return [v.strip() for v in out.splitlines()[0].split(",")]


def collect_software_env():
    """실행 환경의 버전 정보를 평평한 dict로 반환한다 (값은 문자열/정수)."""
    env = {
        "sw_python": platform.python_version(),
        "sw_os": platform.platform(),
        "sw_machine": platform.machine(),
    }
    if platform.system() == "Darwin":
        env["sw_macos"] = platform.mac_ver()[0]

    try:
        import torch
        env["sw_torch"] = torch.__version__
        env["sw_cuda_runtime"] = torch.version.cuda or ""
        try:
            env["sw_cudnn"] = (str(torch.backends.cudnn.version() or "")
                               if torch.backends.cudnn.is_available() else "")
        except Exception:
            env["sw_cudnn"] = ""
        env["sw_mps_available"] = int(bool(torch.backends.mps.is_available()))
        env["sw_mkldnn_available"] = int(bool(torch.backends.mkldnn.is_available()))
        cfg = torch.__config__.show()
        m = re.search(r"(?:MKL-DNN|oneDNN) v([\d.]+)", cfg)
        env["sw_onednn"] = m.group(1) if m else ""
        m = re.search(r"BLAS_INFO=(\w+)", cfg)
        env["sw_blas"] = m.group(1) if m else ""
        if torch.cuda.is_available():
            env["sw_gpu_name"] = torch.cuda.get_device_name(0)
            drv = _nvidia_query("driver_version")
            env["sw_gpu_driver"] = drv[0] if drv else ""
    except ImportError:
        env["sw_torch"] = ""

    try:
        import torchvision
        env["sw_torchvision"] = torchvision.__version__
    except ImportError:
        env["sw_torchvision"] = ""
    return env


def measure_system_load(device_type="cpu", interval=1.0):
    """측정 직전 시스템 부하.

    CPU: psutil.cpu_percent(interval 초 평균, 전체 코어). GPU: nvidia-smi(CUDA만).
    측정 불가 항목은 -1.0.
    """
    load = {"load_cpu_pct": -1.0, "load_gpu_util_pct": -1.0, "load_gpu_mem_used_mb": -1.0}
    try:
        import psutil
        load["load_cpu_pct"] = float(psutil.cpu_percent(interval=interval))
    except Exception:
        pass
    if device_type == "cuda":
        q = _nvidia_query("utilization.gpu,memory.used")
        if len(q) == 2:
            try:
                load["load_gpu_util_pct"] = float(q[0])
                load["load_gpu_mem_used_mb"] = float(q[1])
            except ValueError:
                pass
    return load


def is_idle(load, cpu_max=20.0, gpu_max=10.0):
    """유휴 판정: CPU·GPU 사용률이 기준 이하이면 True (측정 불가(-1)는 통과)."""
    cpu = load.get("load_cpu_pct", -1.0)
    gpu = load.get("load_gpu_util_pct", -1.0)
    cpu_ok = cpu < 0 or cpu <= cpu_max
    gpu_ok = gpu < 0 or gpu <= gpu_max
    return cpu_ok and gpu_ok
