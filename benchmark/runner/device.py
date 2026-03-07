import torch


class DeviceManager:
    """장치 관리: 감지, 동기화, 캐시 정리, 워밍업"""

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

    def warmup(self, device, model_fn):
        """GPU 워밍업: 더미 연산으로 커널 초기화"""
        if device.type in ('cuda', 'mps'):
            dummy = torch.randn(1, 1, 28, 28, device=device)
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
