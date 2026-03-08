# DNN 실행 시간 예측 시스템 보고서

## 1. 프로젝트 개요

딥러닝 모델(ANN, CNN)의 실행 시간을 머신러닝 메타모델로 예측하는 시스템을 구현하였다.
모델 구조 및 하드웨어 정보를 feature로 추출하고, 이를 학습 데이터로 삼아 실행 시간을 예측한다.

### 파일 구조

```
ai-simulator/
├── models/
│   ├── ann.py              # ANN 모델 정의 (SimpleANN)
│   └── cnn.py              # CNN 모델 정의 (SimpleCNN)
├── features/
│   └── extractor.py        # feature 추출 및 하드웨어 정보 수집
├── collect/
│   ├── ann_collector.py    # ANN 실행 시간 측정 및 CSV 저장
│   └── cnn_collector.py    # CNN 실행 시간 측정 및 CSV 저장
├── predictor/
│   └── train.py            # 예측 모델 학습 및 성능 평가
├── data/
│   ├── ann_results.csv                 # ANN 측정 결과
│   ├── cnn_results.csv                 # CNN 측정 결과
│   ├── predictor_results.csv           # 예측 모델 성능 결과
│   └── feature_importance_results.csv  # feature 중요도 결과
├── reports/
│   ├── report.md               # 프로젝트 설명 및 feature 선택 이유
│   ├── ann_result.md           # ANN 실험 결과 분석
│   ├── cnn_result.md           # CNN 실험 결과 분석
│   └── predictor_result.md     # 예측 모델 성능 분석
└── requirements.txt
```

### 실행 순서

```bash
python collect/ann_collector.py   # ANN 데이터 수집
python collect/cnn_collector.py   # CNN 데이터 수집
python predictor/train.py         # 예측 모델 학습
```

---

## 2. 모델 구조

### ANN (SimpleANN)

입력층 → 히든층(N개) → 출력층으로 구성되는 다층 퍼셉트론이다.

```
입력 (784) → Linear → ReLU → Linear → ReLU → ... → Linear → 출력 (10)
              ←────────── num_hidden_layers개 ──────────→
```

- 입력: MNIST 이미지 28×28을 1차원 벡터(784)로 flatten
- 히든층: `hidden_size` 크기의 Linear + ReLU를 `num_hidden_layers`번 반복
- 출력층: Linear(hidden_size → 10)
- 모든 히든층의 뉴런 수는 동일한 `hidden_size`로 고정

| 파라미터 | 설명 | 실험 범위 |
|----------|------|----------|
| hidden_size | 히든층 뉴런 수 | 16, 32, 64, 128, 256, 512, 1024 |
| num_hidden_layers | 히든층 수 | 1 ~ 7 |

### CNN (SimpleCNN)

Conv 블록을 N번 쌓은 뒤 Fully Connected 출력층으로 구성된다.

```
입력 (1×28×28)
  → [Conv2d → (BatchNorm) → ReLU → MaxPool] × num_conv_layers
  → Flatten
  → Linear → 출력 (10)
```

- Conv 블록마다 MaxPool(2×2)이 적용되어 특성 맵 크기가 절반으로 줄어듦
- MNIST 기준 크기 변화: 28 → 14 → 7 → 3 → 1 (최소 1 보장)
- FC 입력 크기 = num_filters × feature_size × feature_size 로 자동 계산
- kernel_size=3, padding=1로 고정 (크기 유지형 conv)

| 파라미터 | 설명 | 실험 범위 |
|----------|------|----------|
| num_filters | Conv 필터 수 (출력 채널) | 8, 16, 32, 64, 128 |
| num_conv_layers | Conv 블록 수 | 1 ~ 6 |
| use_batchnorm | BatchNorm 사용 여부 | False, True |

---

## 3. 실험 설계

### 측정 대상 모델

| 모델 | 변수 | 범위 |
|------|------|------|
| ANN (SimpleANN) | hidden_size | 16, 32, 64, 128, 256, 512, 1024 |
| ANN | num_hidden_layers | 1 ~ 7 |
| CNN (SimpleCNN) | num_filters | 8, 16, 32, 64, 128 |
| CNN | num_conv_layers | 1 ~ 6 |
| CNN | use_batchnorm | False, True |

- 데이터셋: MNIST
- 배치 크기: 64
- 워밍업: 3회
- 측정 반복: 10회 (평균/표준편차 계산)
- 측정 장치: CPU, CUDA (사용 가능한 경우 자동 감지)

### 시간 측정 방법론

단순히 시작/종료 시간을 기록하는 방식은 부정확하다. 두 가지 문제가 있다.

**1. 첫 실행 오염 문제 (워밍업 필요 이유)**

모델을 처음 실행할 때는 다음과 같은 초기화 비용이 포함된다.
- CPU/GPU 캐시가 비어 있어 메모리 접근 지연 발생 (캐시 미스)
- CUDA 커널 최초 로딩 시 컴파일 지연 발생

이로 인해 첫 실행은 이후 실행보다 항상 느리게 측정된다.
이를 제거하기 위해 측정 전 워밍업 3회를 수행하고, 이후 10회 측정값만 사용한다.

**2. GPU 비동기 실행 문제 (synchronize 필요 이유)**

CUDA는 연산을 비동기로 실행한다. 즉, `model(x)` 호출 직후 시간을 기록하면 GPU 연산이 끝나기 전에 측정이 종료된다. 실제보다 짧게 측정되는 문제가 생긴다.

```python
# 잘못된 측정 (GPU 연산 완료 전에 시간 기록)
start = time.perf_counter()
model(x)
end = time.perf_counter()  # GPU가 아직 연산 중일 수 있음

# 올바른 측정 (GPU 연산 완료 후 시간 기록)
start = time.perf_counter()
model(x)
torch.cuda.synchronize()   # GPU 연산 완료 대기
end = time.perf_counter()
```

이 프로젝트에서는 CUDA 장치인 경우 매 측정마다 `torch.cuda.synchronize()`를 호출하여 정확한 시간을 측정한다.

### 예측 모델

총 3가지 모델을 사용하여 성능을 비교한다. 모든 모델은 feature 행렬 X를 입력받아 log1p 변환된 실행 시간 예측값을 반환하며, 평가 시 expm1으로 역변환하여 원래 단위(초)로 복원한다.

#### LinearRegression (베이스라인)

feature들의 선형 결합으로 실행 시간을 예측한다. 구조가 단순하여 학습이 빠르고, 다른 복잡한 모델의 성능을 비교하는 기준선으로 사용한다. 하이퍼파라미터가 없으므로 GridSearchCV 없이 K-Fold 교차검증만 수행한다.

- 입력: feature 행렬 X, log1p 변환된 실행 시간 y
- 반환: 학습된 모델 객체, K-Fold 교차검증 예측값 배열

#### RandomForest (GridSearchCV + K-Fold 5)

여러 개의 결정 트리를 독립적으로 학습한 뒤 예측값을 평균내는 앙상블 모델이다. 각 트리가 데이터의 일부만 보고 학습하므로 과적합에 강하다. feature 중요도를 직접 계산할 수 있어 어떤 feature가 예측에 영향을 많이 미치는지 분석 가능하다.

- GridSearchCV 탐색 파라미터: n_estimators, max_depth, min_samples_split
- 입력: feature 행렬 X, log1p 변환된 실행 시간 y
- 반환: 최적 파라미터로 학습된 모델 객체, K-Fold 교차검증 예측값 배열

#### XGBoost (GridSearchCV + K-Fold 5)

이전 트리의 오차를 다음 트리가 보정하는 방식으로 순차적으로 학습하는 그래디언트 부스팅 모델이다. RandomForest보다 정확도가 높은 경우가 많으나 하이퍼파라미터 튜닝이 중요하다. 마찬가지로 feature 중요도 분석이 가능하다.

- GridSearchCV 탐색 파라미터: n_estimators, learning_rate, max_depth
- 입력: feature 행렬 X, log1p 변환된 실행 시간 y
- 반환: 최적 파라미터로 학습된 모델 객체, K-Fold 교차검증 예측값 배열

#### 공통 설정

- 타겟 변환: log1p 적용 (값 범위 균일화), 평가 시 expm1으로 역변환
- 장치별 분리 학습: CPU / CUDA 데이터를 각각 별도 모델로 학습
- 평가 지표: R², R²(log), RMSE, MAE
- 출력: 각 모델별 성능 수치 + feature 중요도 상위 10개 (RF, XGBoost만)

---

## 3. 입력 Feature 선택 이유

### 3-1. 모델 구조 feature (공통)

| feature | 설명 | 선택 이유 |
|---------|------|----------|
| total_params | 전체 파라미터 수 | 파라미터가 많을수록 연산량 증가 → 실행 시간과 직접적 상관관계 |
| trainable_params | 학습 가능한 파라미터 수 | 역전파 대상 파라미터 수가 학습 시간에 영향 |
| linear_params | Linear 레이어 파라미터 수 | FC 레이어의 연산 비중 파악 |
| flops | 부동소수점 연산 수 | 실행 시간의 이론적 하한을 나타내는 핵심 지표 |
| model_size_mb | 모델 크기 (MB) | 메모리 대역폭 사용량과 관련, 캐시 적중률에 영향 |
| num_layers | 레이어 수 | 레이어가 깊을수록 순차 연산이 늘어 실행 시간 증가 |
| model_type | ANN=0, CNN=1 | ANN과 CNN의 연산 패턴이 근본적으로 다름 |

### 3-2. ANN 전용 feature

| feature | 설명 | 선택 이유 |
|---------|------|----------|
| hidden_size | 은닉층 노드 수 | 행렬 크기 결정 → 연산량과 직결 |
| num_hidden_layers | 은닉층 수 | 레이어 수가 늘면 순차적으로 연산이 추가됨 |

### 3-3. CNN 전용 feature

| feature | 설명 | 선택 이유 |
|---------|------|----------|
| conv_params | Conv 레이어 파라미터 수 | CNN 연산의 주요 병목 구간 |
| num_conv_layers | Conv 레이어 수 | 합성곱 연산 반복 횟수 |
| num_filters | 필터 수 | 출력 채널 수 → FLOPs에 직접 비례 |
| has_batchnorm | BatchNorm 사용 여부 | BatchNorm은 추가 정규화 연산을 포함하여 실행 시간에 영향 |

### 3-4. 하드웨어 feature

| feature | 설명 | 선택 이유 |
|---------|------|----------|
| cpu_cores | CPU 코어 수 | 병렬 연산 가능 스레드 수 결정 |
| cpu_freq_ghz | CPU 클럭 속도 | 클럭이 높을수록 단위 연산 속도 빠름 |
| cpu_cache_l2_mb | L2 캐시 크기 | 캐시가 클수록 메모리 접근 지연 감소 |
| ram_total_gb | RAM 용량 | 대용량 모델 처리 시 메모리 병목 가능성 반영 |
| gpu_memory_gb | GPU 메모리 용량 | CUDA 실행 시 GPU 메모리 한계가 실행 시간에 영향 |

### 3-5. 입력 데이터 feature

| feature | 설명 | 선택 이유 |
|---------|------|----------|
| batch_size | 배치 크기 | 한 번에 처리하는 데이터 양 → 실행 시간에 선형적 영향 |
| input_channels | 입력 채널 수 | 첫 번째 레이어 연산량 결정 |
| input_height | 입력 이미지 높이 | 공간 해상도 → Conv 연산량 결정 |
| input_width | 입력 이미지 너비 | 공간 해상도 → Conv 연산량 결정 |
| num_classes | 출력 클래스 수 | 마지막 Linear 레이어 크기 결정 |

---

## 4. 이 접근법이 우수한 이유

- **모듈 분리**: 모델 정의 / feature 추출 / 데이터 수집 / 예측 학습을 독립 모듈로 분리하여 유지보수 용이
- **ANN + CNN 통합**: 단일 예측 모델이 두 아키텍처를 모두 처리, ANN 전용 feature는 CNN에서 0으로 패딩
- **장치별 분리 학습**: CPU와 CUDA의 실행 특성이 다르므로 별도 모델로 학습하여 정확도 향상
- **3가지 모델 비교**: LinearRegression(베이스라인) → RandomForest → XGBoost 순으로 복잡도를 높이며 성능 비교 가능
- **log1p 변환**: 실행 시간의 분포가 편향되어 있어 변환 후 학습하면 예측 안정성 향상

---

## 5. 사용한 생성형 AI

- **Claude (Anthropic)**: 전체 코드 설계 및 구현에 활용
  - 파일 구조 설계
  - 각 모듈 코드 작성 (models, features, collect, predictor)
  - feature 선택 및 설계 방향 논의
  - 다른 브랜치 코드 분석 및 통합 전략 수립
