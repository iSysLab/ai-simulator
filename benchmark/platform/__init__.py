"""플랫폼 자동 감지 모듈

사용법:
    from benchmark.platform import PlatformInfo

    # 디바이스별 하드웨어 정보 감지
    hw = PlatformInfo.detect('cpu')
    hw = PlatformInfo.detect('cuda')
    hw = PlatformInfo.detect('mps')

    # dict로 변환
    features = hw.to_dict()

    # 요약 출력
    hw.print_summary()
"""
from .detector import detect_platform


class PlatformInfo:
    """플랫폼 하드웨어 정보 통합 클래스

    Feature Schema v1.0 섹션 6 (하드웨어 피처 33개) 전체 구현.
    """

    def __init__(self, info_dict):
        self._info = info_dict

    @classmethod
    def detect(cls, device_str='cpu'):
        """자동 감지하여 PlatformInfo 생성"""
        return cls(detect_platform(device_str))

    def to_dict(self):
        """dict 반환 (피처 추출기에서 사용)"""
        return dict(self._info)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._info[name]
        except KeyError:
            raise AttributeError(f"PlatformInfo has no attribute '{name}'")

    def print_summary(self):
        """하드웨어 정보 요약 출력"""
        i = self._info
        print(f"\n{'='*50}")
        print(f"  플랫폼 정보 요약")
        print(f"{'='*50}")
        print(f"  OS: {i['os_type']}")
        print(f"  디바이스: {i['device_type']}")
        print(f"  가속기: {i['accelerator_brand']} ({i['accelerator_name']})")
        print(f"\n  [CPU]")
        print(f"    물리 코어: {i['cpu_cores_physical']} "
              f"(P: {i['cpu_perf_cores']}, E: {i['cpu_efficiency_cores']})")
        print(f"    논리 코어: {i['cpu_cores_logical']}")
        print(f"    주파수: {i['cpu_freq_base_ghz']}GHz (base) / "
              f"{i['cpu_freq_boost_ghz']}GHz (boost)")
        print(f"    캐시: L2={i['cpu_cache_l2_mb']}MB, L3={i['cpu_cache_l3_mb']}MB")
        print(f"\n  [메모리]")
        print(f"    RAM: {i['ram_total_gb']}GB ({i['memory_type']})")
        print(f"    대역폭: {i['memory_bandwidth_gbs']}GB/s")
        print(f"    통합 메모리: {'Yes' if i['is_unified_memory'] else 'No'}")
        if i['is_unified_memory']:
            print(f"    공유 메모리: {i['shared_memory_gb']}GB")
        if i['dedicated_vram_gb'] > 0:
            print(f"    전용 VRAM: {i['dedicated_vram_gb']}GB")
        print(f"\n  [GPU]")
        print(f"    GPU 수: {i['gpu_count']}")
        print(f"    GPU 메모리: {i['gpu_memory_gb']}GB")
        print(f"    코어: {i['gpu_core_count']}")
        if i['gpu_tensor_core_count'] > 0:
            print(f"    텐서코어: {i['gpu_tensor_core_count']}")
        if i['gpu_compute_capability'] > 0:
            print(f"    Compute Capability: {i['gpu_compute_capability']}")
        print(f"    FP32: {i['tflops_fp32']} TFLOPS | "
              f"FP16: {i['tflops_fp16']} TFLOPS")
        print(f"    fp16: {'Yes' if i['fp16_support'] else 'No'} | "
              f"bf16: {'Yes' if i['bf16_support'] else 'No'}")
        print(f"\n  [인터커넥트]")
        print(f"    타입: {i['interconnect_type']}")
        print(f"    대역폭: {i['host_to_device_bandwidth_gbs']}GB/s")
        print(f"    외장GPU: {'Yes' if i['is_discrete_gpu'] else 'No'} | "
              f"내장GPU: {'Yes' if i['is_integrated_gpu'] else 'No'}")
        print(f"{'='*50}\n")
