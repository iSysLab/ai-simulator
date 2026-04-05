"""
utils/device_utils.py - 플랫폼별 디바이스 자동 선택 유틸리티

Mac(MPS), Windows(CUDA), Linux(CUDA) 등에서 사용 가능한 최적의
디바이스를 자동으로 선택합니다.

사용법:
    from utils.device_utils import get_best_device, device_to_encoded

    device = get_best_device()           # 'cuda', 'mps', 'cpu' 중 하나
    device_encoded = device_to_encoded(device)  # 예측 모델용 (0=CPU, 1=GPU)
"""

import torch


def get_best_device(prefer: str = None) -> str:
    """사용 가능한 최적의 디바이스를 반환합니다.

    우선순위: CUDA > MPS > CPU

    Args:
        prefer (str, optional): 선호 디바이스 ('cuda', 'mps', 'cpu').
            지정 시 해당 디바이스가 사용 가능하면 반환, 아니면 CPU.

    Returns:
        str: 'cuda', 'mps', 'cpu' 중 하나
    """
    if prefer:
        if prefer == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if prefer == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        if prefer == 'cpu':
            return 'cpu'
        return 'cpu'

    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def device_to_encoded(device: str) -> int:
    """디바이스 문자열을 예측 모델의 인코딩 값으로 변환합니다.

    Args:
        device (str): 'cpu', 'cuda', 'mps' 중 하나

    Returns:
        int: 0 (CPU) 또는 1 (GPU)
    """
    return 0 if (device or '').lower() == 'cpu' else 1


def synchronize_device(device) -> None:
    """GPU 연산 완료 대기. 시간 측정 정확도를 위해 사용.

    Args:
        device: torch.device 또는 'cuda', 'mps', 'cpu' 문자열
    """
    device_type = device.lower() if isinstance(device, str) else getattr(device, 'type', 'cpu')

    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()
