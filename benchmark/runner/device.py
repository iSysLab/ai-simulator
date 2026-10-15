import torch


class DeviceManager:
    """장치 관리: 감지, 동기화, 캐시 정리, 피크 메모리 스냅샷"""

    def __init__(self):
        self.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            self.devices.append(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            self.devices.append(torch.device("mps"))

    def sync(self, device):
        """GPU 비동기 연산 완료 대기"""
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    def clear_cache(self, device):
        """GPU 메모리 캐시 정리"""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    def reset_peak_memory(self, device):
        """피크 메모리 통계 초기화 (CUDA만 지원. MPS는 누적 피크 API가 없음)"""
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    def memory_snapshot_mb(self, device):
        """장치 메모리 사용량(MB)과 측정 방법을 반환한다.

        CUDA: reset_peak_memory 이후의 최대 할당량(max_memory_allocated).
        MPS:  드라이버 할당량(driver_allocated_memory). 피크 API가 없어 호출 시점 값이므로
              학습 직후·추론 직후에 호출해 최댓값을 취한다.
        CPU:  0 (측정하지 않음)
        """
        if device.type == 'cuda':
            return torch.cuda.max_memory_allocated(device) / 2 ** 20, 'cuda_max_memory_allocated'
        if device.type == 'mps':
            try:
                return torch.mps.driver_allocated_memory() / 2 ** 20, 'mps_driver_allocated_memory'
            except Exception:
                return 0.0, 'none'
        return 0.0, 'none'

    def warmup(self, device, model_fn, input_shape=None):
        """(구 프로토콜, v1) GPU에서 배치 1 더미 순전파 1회.

        2026-09 재측정부터는 ExperimentRunner.warmup()/warmup_gan()이 실제 학습 스텝으로
        모든 백엔드·계열을 동일하게 워밍업한다. 이 함수는 과거 결과 재현용으로만 남긴다.
        """
        if device.type in ('cuda', 'mps'):
            shape = input_shape or (1, 1, 28, 28)
            dummy = torch.randn(*shape, device=device)
            dummy_model = model_fn().to(device)
            with torch.no_grad():
                _ = dummy_model(dummy)
            self.sync(device)
            del dummy, dummy_model
            self.clear_cache(device)

    def label(self, device):
        """장치 표시 이름"""
        if device.type == 'cuda':
            return "GPU(CUDA)"
        elif device.type == 'mps':
            return "GPU(MPS)"
        return "CPU"
