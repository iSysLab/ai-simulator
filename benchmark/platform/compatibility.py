"""호환성 매트릭스 — 환경에 따라 실행 가능한 설정 자동 필터링

MobileNet ARM CPU 14x 느림, GPU 메모리 부족 등 자동 처리.
"""
import torch

from .detector import detect_platform


class CompatibilityMatrix:
    """환경 호환성 자동 판단"""

    def __init__(self):
        self._platform_cache = {}

    def get_available_devices(self):
        """사용 가능한 디바이스 목록"""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if torch.backends.mps.is_available():
            devices.append('mps')
        return devices

    def get_platform(self, device_str='cpu'):
        """캐시된 플랫폼 정보"""
        if device_str not in self._platform_cache:
            self._platform_cache[device_str] = detect_platform(device_str)
        return self._platform_cache[device_str]

    def get_optimal_batch_size(self, device_str='cpu'):
        """GPU 메모리에 따라 배치 크기 자동 조정"""
        if device_str == 'cpu':
            return 64

        hw = self.get_platform(device_str)
        gpu_mem = hw['gpu_memory_gb']

        if gpu_mem >= 16:
            return 128
        elif gpu_mem >= 8:
            return 64
        elif gpu_mem >= 4:
            return 32
        else:
            return 16

    def should_skip_config(self, config, device_str='cpu'):
        """설정을 건너뛸지 판단

        Returns:
            tuple: (skip: bool, reason: str)
        """
        model_type = config.get('model_type', '')
        cfg = config.get('config', {})

        hw = self.get_platform(device_str)

        # MPS에서 MobileNet 경고 (14x 느림)
        if device_str == 'mps' and model_type == 'mobilenet_mnist':
            if hw.get('os_type') == 'macos':
                # 스킵은 안 하지만 경고
                pass

        # GPU 메모리 기반 대형 모델 스킵
        if device_str in ('cuda', 'mps'):
            gpu_mem = hw['gpu_memory_gb']
            total_params_estimate = _estimate_params(model_type, cfg)
            # 모델 크기 (MB) = params × 4 bytes × 3 (모델 + 그래디언트 + 옵티마이저)
            required_mb = total_params_estimate * 4 * 3 / (1024 * 1024)
            available_mb = gpu_mem * 1024 * 0.8  # 80% 사용 가능

            if required_mb > available_mb:
                return True, f"GPU 메모리 부족 (필요: {required_mb:.0f}MB, 가용: {available_mb:.0f}MB)"

        return False, ''

    def get_recommended_repeats(self, device_str='cpu'):
        """디바이스별 권장 반복 횟수"""
        if device_str == 'cpu':
            return 10
        return 10  # GPU도 동일

    def print_summary(self):
        """호환성 요약 출력"""
        devices = self.get_available_devices()
        print(f"\n사용 가능 디바이스: {devices}")
        for dev in devices:
            hw = self.get_platform(dev)
            batch = self.get_optimal_batch_size(dev)
            print(f"  [{dev}] 배치: {batch}, "
                  f"GPU메모리: {hw['gpu_memory_gb']}GB, "
                  f"통합메모리: {'Yes' if hw['is_unified_memory'] else 'No'}")


def _estimate_params(model_type, config):
    """설정에서 대략적 파라미터 수 추정"""
    if model_type == 'simple_ann':
        hs = config.get('hidden_size', 64)
        nl = config.get('num_layers', 1)
        return 784 * hs + hs * hs * (nl - 1) + hs * 10

    elif model_type == 'simple_cnn':
        nf = config.get('num_filters', 16)
        nl = config.get('num_conv_layers', 2)
        return nf * 9 * nl * nf + nf * 4 * 128 + 128 * 10

    elif model_type == 'transformer':
        ed = config.get('embed_dim', 128)
        nl = config.get('num_layers', 4)
        return ed * ed * 4 * nl + ed * 10

    elif model_type == 'gan':
        ld = config.get('latent_dim', 128)
        dims = config.get('g_hidden_dims', [256, 512])
        return ld * sum(dims) + sum(dims) * 3072

    return 1_000_000  # conservative default
