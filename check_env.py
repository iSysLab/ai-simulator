"""환경 검증 스크립트 — 실행 전 점검

사용법:
    python check_env.py            # 사람이 읽는 점검 결과
    python check_env.py --json     # 소프트웨어 환경을 JSON으로 출력 (논문 §3.1 기재·재현용)

시스템 요구사항, 패키지 설치 상태, 디바이스 가용성을 확인하고
문제가 있으면 해결 방법을 안내합니다.
"""
import json
import sys
import platform


def check_python():
    """Python 버전 확인"""
    v = sys.version_info
    ok = v >= (3, 8)
    ver = f"{v.major}.{v.minor}.{v.micro}"
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python: {ver}", end='')
    if not ok:
        print(" (3.8 이상 필요)")
    else:
        print()
    return ok


def check_package(name, import_name=None):
    """패키지 설치 확인"""
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', '?')
        print(f"  [OK]   {name}: {ver}")
        return True
    except ImportError:
        print(f"  [MISS] {name}: 미설치 → pip install {name}")
        return False


def check_torch_device():
    """PyTorch 디바이스 확인"""
    try:
        import torch
        print(f"\n  PyTorch: {torch.__version__}")

        devices = ['CPU']
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            devices.append(f"CUDA ({gpu_name}, {gpu_mem}GB)")
        if torch.backends.mps.is_available():
            devices.append("MPS (Apple Silicon)")

        print(f"  디바이스: {', '.join(devices)}")
        return True
    except ImportError:
        print(f"\n  [FAIL] PyTorch 미설치!")
        print(f"         pip install torch torchvision")
        return False


def check_disk_space():
    """디스크 여유 공간 확인"""
    import shutil
    free_gb = shutil.disk_usage('.').free / (1024 ** 3)
    ok = free_gb >= 1.0
    status = "OK" if ok else "WARN"
    print(f"  [{status}] 디스크 여유: {free_gb:.1f}GB", end='')
    if not ok:
        print(" (최소 1GB 권장)")
    else:
        print()
    return ok


def check_platform_detection():
    """플랫폼 자동 감지 테스트"""
    try:
        from benchmark.platform import PlatformInfo
        hw = PlatformInfo.detect('cpu')
        d = hw.to_dict()
        print(f"\n  OS: {d['os_type']}")
        print(f"  CPU: {d['cpu_cores_physical']}P + {d['cpu_efficiency_cores']}E 코어, "
              f"{d['cpu_freq_boost_ghz']}GHz")
        print(f"  RAM: {d['ram_total_gb']}GB ({d['memory_type']})")
        print(f"  대역폭: {d['memory_bandwidth_gbs']}GB/s")
        print(f"  통합 메모리: {'Yes' if d['is_unified_memory'] else 'No'}")

        # GPU 감지
        import torch
        if torch.cuda.is_available():
            hw_gpu = PlatformInfo.detect('cuda')
            g = hw_gpu.to_dict()
            print(f"  GPU: {g['accelerator_name']}, {g['gpu_core_count']}코어, "
                  f"{g['gpu_memory_gb']}GB, CC {g['gpu_compute_capability']}")
        elif torch.backends.mps.is_available():
            hw_gpu = PlatformInfo.detect('mps')
            g = hw_gpu.to_dict()
            print(f"  GPU: {g['accelerator_name']}, {g['gpu_core_count']}코어, "
                  f"{g['gpu_memory_gb']}GB, {g['tflops_fp32']} TFLOPS")

        return True
    except Exception as e:
        print(f"\n  [FAIL] 플랫폼 감지 실패: {e}")
        return False


def env_json():
    """소프트웨어 환경 + 감지된 하드웨어 요약을 JSON으로 출력 (--json)"""
    from benchmark.platform.software_env import collect_software_env
    out = {"software": collect_software_env(), "hardware": {}}
    try:
        from benchmark.platform import PlatformInfo
        out["hardware"]["cpu"] = PlatformInfo.detect('cpu').to_dict()
        import torch
        if torch.cuda.is_available():
            out["hardware"]["cuda"] = PlatformInfo.detect('cuda').to_dict()
        elif torch.backends.mps.is_available():
            out["hardware"]["mps"] = PlatformInfo.detect('mps').to_dict()
    except Exception as e:
        out["hardware"]["error"] = str(e)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def main():
    if '--json' in sys.argv[1:]:
        env_json()
        return

    os_name = platform.system()
    machine = platform.machine()

    print("=" * 60)
    print("  DNN 벤치마크 프레임워크 환경 검증")
    print(f"  OS: {os_name} ({machine})")
    print("=" * 60)

    all_ok = True

    # 1. Python
    print("\n[1] Python 버전")
    if not check_python():
        all_ok = False

    # 2. 필수 패키지
    print("\n[2] 필수 패키지")
    required = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('psutil', 'psutil'),
        ('joblib', 'joblib'),
    ]
    for name, imp in required:
        if not check_package(name, imp):
            all_ok = False

    # 3. 선택적 패키지
    print("\n[3] 선택적 패키지")
    optional = [
        ('xgboost', 'xgboost'),
        ('matplotlib', 'matplotlib'),
        ('onnx', 'onnx'),
        ('pandas', 'pandas'),
    ]
    for name, imp in optional:
        check_package(name, imp)

    # 4. 디바이스
    print("\n[4] PyTorch 디바이스")
    if not check_torch_device():
        all_ok = False

    # 5. 디스크
    print("\n[5] 디스크 공간")
    check_disk_space()

    # 6. 플랫폼 자동 감지
    print("\n[6] 플랫폼 자동 감지")
    check_platform_detection()

    # 결과
    print(f"\n{'='*60}")
    if all_ok:
        print("  모든 검사 통과! 벤치마크를 실행할 수 있습니다.")
        print(f"\n  빠른 시작:")
        print(f"    python run_benchmark.py --quick          # 빠른 테스트")
        print(f"    python run_benchmark.py --full-pipeline  # 전체 파이프라인")
    else:
        print("  일부 검사 실패. 위의 안내를 따라 문제를 해결하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
