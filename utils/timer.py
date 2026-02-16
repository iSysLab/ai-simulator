import torch
import time
import numpy as np


class TimeEstimator:
    def __init__(self, device='mps', warmup_runs=5, measure_runs=10):
        """
        시간 측정 클래스

        Args:
            device: 'mps'(M1 GPU), 'cpu', 'cuda'
            warmup_runs: 워밍업 실행 횟수
            measure_runs: 실제 측정 횟수
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

        # M1 Mac에서 MPS 사용 가능 여부 확인
        if device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, using CPU")
            self.device = 'cpu'

    def measure_inference_time(self, model, input_shape):
        """
        모델 추론 시간 측정

        Args:
            model: PyTorch 모델
            input_shape: 입력 shape (batch_size, features)

        Returns:
            평균 추론 시간 (ms)
        """
        model = model.to(self.device)
        model.eval()

        # 더미 입력 생성
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
                if self.device == 'mps':
                    torch.mps.synchronize()

        # 실제 측정
        times = []
        with torch.no_grad():
            for _ in range(self.measure_runs):
                start = time.time()
                _ = model(dummy_input)
                if self.device == 'mps':
                    torch.mps.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # ms 단위

        return np.mean(times), np.std(times)

    def measure_training_time(self, model, train_loader, epochs=1):
        """
        모델 학습 시간 측정

        Args:
            model: PyTorch 모델
            train_loader: DataLoader
            epochs: 학습 에포크 수

        Returns:
            에포크당 평균 학습 시간 (초)
        """
        model = model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        epoch_times = []

        for epoch in range(epochs):
            start = time.time()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if self.device == 'mps':
                    torch.mps.synchronize()

            end = time.time()
            epoch_times.append(end - start)

        return np.mean(epoch_times), np.std(epoch_times)