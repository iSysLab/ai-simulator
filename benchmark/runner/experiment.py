import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .device import DeviceManager


class ExperimentRunner:
    """단일 모델 설정에 대한 벤치마크 실험 실행

    기존 ann.py/cnn.py의 시간 측정 패턴을 일반화:
    - sync_device → perf_counter → 학습/추론 → sync_device → perf_counter
    - 10회 반복 후 평균/표준편차 계산
    """

    def __init__(self, device_manager, epochs=1, repeats=10, lr=0.01):
        self.dm = device_manager
        self.epochs = epochs
        self.repeats = repeats
        self.lr = lr

    def run(self, model_fn, device, train_batches, test_batches,
            num_test_samples, config_name=""):
        """분류 모델 벤치마크 실행

        Args:
            model_fn: 모델 생성 함수 (호출 시 새 모델 반환)
            device: torch.device
            train_batches: 사전 로딩된 학습 배치 리스트
            test_batches: 사전 로딩된 테스트 배치 리스트
            num_test_samples: 테스트 샘플 수 (정확도 계산용)
            config_name: 설정 이름 (출력용)

        Returns:
            dict: avg/std train/infer time + accuracy
        """
        train_times = []
        infer_times = []
        accuracies = []

        for i in range(self.repeats):
            model = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self.lr)

            # --- 학습 시간 측정 ---
            model.train()
            self.dm.sync(device)
            start = time.perf_counter()

            for _ in range(self.epochs):
                for data, target in train_batches:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            self.dm.sync(device)
            train_time = time.perf_counter() - start

            # --- 추론 시간 측정 ---
            model.eval()
            self.dm.sync(device)
            start = time.perf_counter()

            correct = 0
            with torch.no_grad():
                for data, target in test_batches:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            self.dm.sync(device)
            infer_time = time.perf_counter() - start
            accuracy = 100. * correct / num_test_samples

            train_times.append(train_time)
            infer_times.append(infer_time)
            accuracies.append(accuracy)

            print(f"    [{i+1}/{self.repeats}] "
                  f"학습: {train_time:.4f}s | "
                  f"추론: {infer_time:.4f}s | "
                  f"정확도: {accuracy:.2f}%")

            del model
            self.dm.clear_cache(device)

        return {
            'avg_train': round(float(np.mean(train_times)), 5),
            'std_train': round(float(np.std(train_times)), 5),
            'avg_infer': round(float(np.mean(infer_times)), 5),
            'std_infer': round(float(np.std(infer_times)), 5),
            'avg_accuracy': round(float(np.mean(accuracies)), 2),
        }
