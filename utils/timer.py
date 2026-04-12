"""
utils/timer.py - DNN 모델 실행 시간 측정 유틸리티

[이 파일의 역할]
PyTorch 모델의 추론(Inference) 시간과 학습(Training) 시간을 정확하게 측정하는
TimeEstimator 클래스를 제공합니다.

[핵심 아이디어]
1. Warmup (워밍업): 처음 몇 번의 실행은 캐시 미스, JIT 컴파일, GPU 초기화 등
   외부 요인으로 인해 비정상적으로 느릴 수 있습니다.
   → Warmup을 통해 이런 초기화 비용을 "버리고" 안정적인 구간만 측정합니다.

2. 반복 측정 후 평균: 운영체제의 스케줄링, 다른 프로세스의 간섭 등으로 인해
   한 번 측정한 결과는 오차가 클 수 있습니다.
   → 10회 이상 반복 측정 후 평균(mean)과 표준편차(std)를 계산합니다.

3. MPS 동기화 (torch.mps.synchronize()): Apple Silicon(M1/M2)의 MPS 백엔드는
   CPU와 GPU가 비동기적으로 실행됩니다. 즉, model(input) 호출이 끝나도
   GPU 연산이 아직 완료되지 않았을 수 있습니다.
   → synchronize()를 호출하여 GPU 연산이 완전히 끝날 때까지 기다린 후 시간을 기록합니다.
   → 이를 생략하면 실제보다 훨씬 짧은 시간이 측정됩니다 (정확하지 않음).

[사용 방법]
    estimator = TimeEstimator(device='cpu', warmup_runs=5, measure_runs=10)
    mean_ms, std_ms = estimator.measure_inference_time(model, input_shape=(64, 1, 28, 28))
    mean_sec, std_sec = estimator.measure_training_time(model, train_loader, epochs=1)
"""

import torch
import time
import numpy as np


class TimeEstimator:
    """
    PyTorch 모델의 실행 시간을 정밀하게 측정하는 클래스.

    추론(Inference)과 학습(Training) 시간을 별도로 측정하며,
    Warmup + 반복 측정 전략으로 신뢰성 높은 결과를 제공합니다.
    """

    def __init__(self, device='mps', warmup_runs=5, measure_runs=10):
        """
        TimeEstimator 초기화.

        Args:
            device (str): 측정할 디바이스.
                - 'cpu'  : 일반 CPU (M1 포함)
                - 'mps'  : Apple Silicon GPU (M1/M2의 Metal Performance Shaders)
                - 'cuda' : NVIDIA GPU (이 프로젝트에서는 미사용)
            warmup_runs (int): 워밍업 실행 횟수.
                측정 전에 모델을 몇 번 "빈 실행"할지 결정합니다.
                캐시, JIT 컴파일, GPU 초기화 등의 초기 오버헤드를 제거하기 위해 필요합니다.
                권장값: 3~10회
            measure_runs (int): 실제 측정 횟수.
                이 횟수만큼 반복 측정한 평균과 표준편차를 반환합니다.
                값이 클수록 정확하지만 총 소요 시간이 늘어납니다.
                권장값: 10~30회
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

        # M1 Mac에서 MPS 사용 불가 시 자동으로 CPU로 전환
        # (예: macOS 버전이 낮거나, Intel Mac인 경우)
        if device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, using CPU")
            self.device = 'cpu'

    def measure_inference_time(self, model, input_shape):
        """
        모델의 추론(Inference) 시간을 측정합니다.

        추론이란? 학습이 완료된 모델에 새로운 데이터를 넣어 예측 결과를 얻는 과정.
        학습(backpropagation 포함)보다 훨씬 빠르며, 실제 서비스에서 사용되는 과정입니다.

        [측정 절차]
        1. 모델을 지정한 device로 이동 (CPU 또는 MPS)
        2. eval() 모드 전환 → Dropout, BatchNorm이 추론 모드로 동작
        3. 더미(가짜) 입력 텐서 생성 (실제 데이터와 같은 shape, 값은 랜덤)
        4. Warmup: warmup_runs번 실행하여 초기화 비용 제거
        5. 측정: measure_runs번 반복 측정, MPS의 경우 synchronize() 호출
        6. 평균과 표준편차 반환 (단위: ms)

        Args:
            model: 측정할 PyTorch 모델 (nn.Module)
            input_shape (tuple): 입력 텐서의 형태. 예:
                - ANN의 경우: (batch_size, input_features) = (64, 784)
                - CNN의 경우: (batch_size, channels, height, width) = (64, 3, 32, 32)

        Returns:
            tuple: (평균 추론 시간 [ms], 표준편차 [ms])
                   표준편차가 작을수록 측정이 안정적임을 의미합니다.
        """
        # 모델을 지정 디바이스로 이동
        model = model.to(self.device)

        # 추론 모드 전환: Dropout은 비활성화, BatchNorm은 학습된 통계값 사용
        model.eval()

        # 더미 입력 생성: 실제 데이터 없이도 시간 측정 가능
        # torch.randn: 평균 0, 분산 1의 정규분포 난수로 채워진 텐서
        dummy_input = torch.randn(input_shape).to(self.device)

        # [Warmup] 초기 실행: GPU 캐시, JIT 컴파일, 메모리 할당 등 초기화
        # torch.no_grad(): 역전파(gradient) 계산을 비활성화하여 추론 속도 향상
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
                # MPS 동기화: GPU 연산이 완전히 끝날 때까지 대기
                # 이것 없이 시간을 재면 GPU가 아직 연산 중인데 time.time()을 찍게 됨
                if self.device == 'mps':
                    torch.mps.synchronize()

        # [실제 측정] measure_runs번 반복하여 시간 기록
        times = []
        with torch.no_grad():
            for _ in range(self.measure_runs):
                start = time.time()          # 시작 시간 기록
                _ = model(dummy_input)       # 추론 실행
                if self.device == 'mps':
                    torch.mps.synchronize()  # GPU 연산 완료 대기
                end = time.time()            # 종료 시간 기록

                # 초(sec) → 밀리초(ms)로 변환하여 저장
                times.append((end - start) * 1000)

        # 평균(mean)과 표준편차(std) 반환
        return np.mean(times), np.std(times)

    def measure_training_time(self, model, train_loader, epochs=1):
        """
        모델의 학습(Training) 시간을 측정합니다.

        학습이란? 데이터를 보고 모델의 가중치(파라미터)를 업데이트하는 과정.
        순전파(forward) → 손실 계산(loss) → 역전파(backward) → 가중치 업데이트(optimizer.step())
        로 이루어지며, 추론보다 훨씬 많은 연산이 필요합니다.

        [측정 절차]
        1. 모델을 지정한 device로 이동
        2. train() 모드 전환 → Dropout, BatchNorm이 학습 모드로 동작
        3. 손실함수(CrossEntropyLoss)와 옵티마이저(Adam) 설정
        4. epochs 수만큼 전체 데이터셋을 순회하며 학습 시간 측정
        5. 에포크당 평균 학습 시간과 표준편차 반환 (단위: 초)

        [왜 에포크당 시간인가?]
        총 학습 시간은 에포크 수에 비례하므로, 에포크당 시간을 기준으로 비교하면
        서로 다른 에포크 수로 실험한 결과도 일관되게 비교 가능합니다.

        Args:
            model: 측정할 PyTorch 모델 (nn.Module)
            train_loader: 학습 데이터를 배치 단위로 공급하는 DataLoader
            epochs (int): 측정할 에포크(전체 데이터 1회 순회) 수.
                          여러 에포크를 측정하면 에포크간 변동을 확인할 수 있습니다.

        Returns:
            tuple: (에포크당 평균 학습 시간 [초], 표준편차 [초])
        """
        # 모델을 지정 디바이스로 이동
        model = model.to(self.device)

        # 학습 모드 전환: Dropout 활성화, BatchNorm이 배치 통계값 사용
        model.train()

        # 손실 함수: CrossEntropyLoss = 분류 문제에 표준적으로 사용
        # 내부적으로 Softmax + Log + NLLLoss를 합친 것
        criterion = torch.nn.CrossEntropyLoss()

        # 옵티마이저: Adam (Adaptive Moment Estimation)
        # SGD보다 학습률 조정이 자동적이어서 빠른 수렴에 유리
        optimizer = torch.optim.Adam(model.parameters())

        epoch_times = []

        for epoch in range(epochs):
            start = time.time()  # 에포크 시작 시간

            # 배치 단위로 전체 학습 데이터 순회
            for batch_idx, (data, target) in enumerate(train_loader):
                # 데이터와 레이블을 지정 디바이스로 이동
                data, target = data.to(self.device), target.to(self.device)

                # 이전 배치의 gradient 초기화 (안 하면 누적되어 학습 오류 발생)
                optimizer.zero_grad()

                # 순전파: 모델에 입력을 넣어 예측값 계산
                output = model(data)

                # 손실 계산: 예측값과 실제 레이블의 차이
                loss = criterion(output, target)

                # 역전파: 손실에 대한 각 파라미터의 gradient 계산
                loss.backward()

                # 가중치 업데이트: gradient를 이용해 파라미터 갱신
                optimizer.step()

                # MPS 동기화: GPU 연산 완료 대기
                if self.device == 'mps':
                    torch.mps.synchronize()

            end = time.time()  # 에포크 종료 시간
            epoch_times.append(end - start)

        # 에포크당 평균 학습 시간과 표준편차 반환 (단위: 초)
        return np.mean(epoch_times), np.std(epoch_times)
