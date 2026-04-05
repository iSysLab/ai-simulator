"""
utils/timer.py - DNN 모델 실행 시간 측정 유틸리티

PyTorch 모델의 추론(Inference) 시간과 학습(Training) 시간을 측정하는
TimeEstimator 클래스를 제공합니다.

핵심 아이디어:
    1. Warmup: 처음 몇 번 실행은 캐시/JIT/GPU 초기화로 비정상적으로 느림
               → 워밍업으로 초기화 비용 제거 후 안정 구간만 측정
    2. 반복 측정: OS 스케줄링 등 간섭 요인으로 1회 측정은 오차가 큼
               → 반복 측정 후 평균/표준편차 계산
    3. GPU 동기화: CUDA/MPS는 비동기 실행이라 synchronize() 없이 측정하면 부정확

사용법:
    estimator = TimeEstimator(device='cuda', warmup_runs=1, measure_runs=1)
    mean_ms, std_ms = estimator.measure_inference_time(model, input_shape=(64, 1, 28, 28))
    mean_sec, std_sec = estimator.measure_training_time(model, train_loader, epochs=1)
"""

import time
import torch
import numpy as np

from utils.device_utils import synchronize_device


class TimeEstimator:
    """PyTorch 모델의 실행 시간을 정밀하게 측정하는 클래스."""

    def __init__(self, device='cpu', warmup_runs=1, measure_runs=1):
        """
        Args:
            device (str): 'cpu', 'cuda', 'mps' 중 하나
            warmup_runs (int): 워밍업 실행 횟수
            measure_runs (int): 실제 측정 반복 횟수
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

        if device == 'mps' and not torch.backends.mps.is_available():
            print("MPS 사용 불가 → CPU로 대체")
            self.device = 'cpu'

    def measure_inference_time(self, model, input_shape):
        """추론 시간 측정.

        Args:
            model: PyTorch 모델 (nn.Module)
            input_shape (tuple): 입력 텐서 shape. 예: (64, 1, 28, 28)

        Returns:
            tuple: (평균 ms, 표준편차 ms)
        """
        model = model.to(self.device).eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        with torch.no_grad():
            for _ in range(self.warmup_runs):
                model(dummy_input)
                synchronize_device(self.device)

        times = []
        with torch.no_grad():
            for _ in range(self.measure_runs):
                start = time.perf_counter()
                model(dummy_input)
                synchronize_device(self.device)
                times.append((time.perf_counter() - start) * 1000)

        return float(np.mean(times)), float(np.std(times))

    def measure_training_time(self, model, train_loader, epochs=1):
        """학습 시간 측정.

        Args:
            model: PyTorch 모델 (nn.Module)
            train_loader: DataLoader
            epochs (int): 측정할 에포크 수

        Returns:
            tuple: (에포크당 평균 초, 표준편차 초)
        """
        model = model.to(self.device).train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        epoch_times = []

        for _ in range(epochs):
            start = time.perf_counter()

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                synchronize_device(self.device)

            epoch_times.append(time.perf_counter() - start)

        return float(np.mean(epoch_times)), float(np.std(epoch_times))
