# DNN 실행 시간/공간 예측 시뮬레이션 프레임워크

PyTorch 기반 DNN 모델의 **실행 시간(학습/추론)** 및 **메모리 요구량**을 예측하는 벤치마크 프레임워크입니다.
MNIST / CIFAR-10 데이터셋에서 6종의 모델 아키텍처를 벤치마킹하고, 모델 구조 피처로부터 실행 시간을 예측하는 ML 모델을 학습합니다.

## 팀원별 기여 (3개 브랜치 병합)

### ijunsoo — 기본 프레임워크 설계 + 모듈형 구조

- **벤치마크 프레임워크 재설계**: 단일 스크립트(`ann.py`, `cnn.py`) 구조에서 `benchmark/` 패키지 기반 모듈형 구조로 전환
- **모델 레지스트리** (`benchmark/models/registry.py`): 데코레이터 기반 모델 팩토리 패턴으로 모델 등록/생성 통합
- **CNN 파라미터 역전 수정**: `AdaptiveAvgPool2d((1,1))` 도입으로 레이어 증가 시 파라미터가 오히려 줄어드는 문제 해결
- **4종 기본 모델**: SimpleANN, SimpleCNN, ResNet-MNIST, MobileNet-MNIST 구현
- **설정 자동 생성** (`benchmark/configs/generator.py`): 모델별 하이퍼파라미터 조합 자동 생성 (160개 config)
- **장치 관리 / 데이터 로딩 / 실험 루프**: `benchmark/runner/` 패키지로 분리하여 재사용 가능한 구조
- **피처 추출기** (`benchmark/features/extractor.py`): PyTorch 모델에서 30개 구조 피처 자동 추출 (Forward hook 기반 FLOPs 계산 포함)

### dal-merge (달현) — XGBoost + GridSearchCV + 하드웨어 피처

- **XGBoost 회귀 모델 도입**: RandomForest/GradientBoosting 외에 XGBoost + GridSearchCV로 최적 하이퍼파라미터 자동 탐색
- **log1p 변환**: 실행 시간의 넓은 범위(μs~s)를 `log1p`/`expm1`으로 안정화 → 예측 정확도 향상
- **하드웨어 피처 통합**: CPU 코어 수, 클럭 주파수, RAM, GPU 메모리를 피처에 추가 (`psutil` 활용)
- **장치별 별도 모델 학습**: CPU와 GPU의 실행 시간 패턴이 다르므로 장치별 분리 학습
- **R²(log) 메트릭 추가**: log 공간에서의 결정계수를 별도 평가하여 전 구간 예측 성능 확인

### khg9859 (홍근) — Transformer/GAN + ONNX + Op-Level 프로파일링

- **Transformer (Vision Transformer)** 모델 추가 (`benchmark/models/transformer.py`): PatchEmbedding, MultiHeadAttention, TransformerEncoderBlock 직접 구현, CIFAR-10 대응
- **GAN (Generator + Discriminator)** 모델 추가 (`benchmark/models/gan.py`): 적대적 학습 벤치마크 지원
- **ONNX 파이프라인**: `export_onnx.py`(PyTorch→ONNX 변환) + `predict_from_onnx.py`(ONNX→피처추출→시간예측)
- **ONNX 피처 추출기** (`benchmark/features/onnx_extractor.py`): ONNX 그래프에서 가중치 shape 기반 FLOPs 직접 계산
- **joblib 모델 저장/로딩**: 학습된 예측 모델을 저장하여 재사용 (`--save-models`)
- **Op-Level 프로파일러** (`benchmark/features/op_profiler.py`): 모델을 개별 연산 단위로 분해하여 시간/메모리 측정 → 합산 시뮬레이션
- **Transformer Attention FLOPs 보정**: `nn.Linear` hook으로 잡히지 않는 Q@K, attn@V matmul FLOPs를 별도 추정

## 교수님 연구 방향 대응

| 연구 요구사항 | 구현 | 파일 |
|---|---|---|
| DNN 추론 시간/공간 예측 시뮬레이션 | 학습시간 + 추론시간 + 메모리 3가지 타겟 예측 | `train_predictor.py` |
| DNN 시간 추정 모델 타당성 검증 | 4개 ML 모델 × K-Fold CV × GridSearchCV | `train_predictor.py` |
| 다양한 모델에 적용 | 6종 (ANN, CNN, ResNet, MobileNet, Transformer, GAN) | `benchmark/models/` |
| 모델을 작은 연산 단위로 분해하여 예측 | Op-level 분해 → 개별 시간 측정 → 합산 시뮬레이션 | `benchmark/features/op_profiler.py` |
| ONNX 표준 형식 모델 실행 시간 추론 | ONNX export → 피처 추출 → 학습된 모델로 예측 | `export_onnx.py`, `predict_from_onnx.py` |

## 프로젝트 구조

```
ann/
├── benchmark/                          # 메인 패키지
│   ├── models/                         # 모델 정의
│   │   ├── registry.py                 # 모델 팩토리 레지스트리
│   │   ├── simple_ann.py               # SimpleANN (MNIST)
│   │   ├── simple_cnn.py              # SimpleCNN (MNIST, AdaptiveAvgPool2d 적용)
│   │   ├── resnet_mnist.py            # ResNet (MNIST)
│   │   ├── mobilenet_mnist.py         # MobileNet (MNIST)
│   │   ├── transformer.py            # Vision Transformer (CIFAR-10)
│   │   └── gan.py                     # GAN Generator+Discriminator (CIFAR-10)
│   ├── features/                       # 피처 추출
│   │   ├── extractor.py               # PyTorch 모델 → 30개 구조 피처
│   │   ├── onnx_extractor.py          # ONNX 파일 → 30개 구조 피처
│   │   └── op_profiler.py             # Op-level 분해/시간 측정/시뮬레이션
│   ├── runner/                         # 실험 실행
│   │   ├── device.py                  # 장치 감지/동기화/워밍업
│   │   ├── data.py                    # MNIST + CIFAR-10 데이터 로딩
│   │   └── experiment.py              # 학습/추론/GAN 시간 측정 루프
│   ├── configs/
│   │   └── generator.py               # 160개 설정 조합 자동 생성
│   └── results/
│       └── io.py                       # JSON 증분 저장/CSV 변환
├── run_benchmark.py                    # 통합 벤치마크 실행 진입점
├── train_predictor.py                  # 예측 모델 학습 (LR/RF/GB/XGBoost)
├── export_onnx.py                      # PyTorch → ONNX 변환
├── predict_from_onnx.py                # ONNX → 실행 시간 예측
├── ann.py / cnn.py / cnn_remaining.py  # 기존 단독 실행 스크립트
└── requirements.txt
```

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 의존성

```
torch, torchvision, numpy, scikit-learn, xgboost, onnx, joblib, psutil
```

## 사용법

### 1. 벤치마크 실행

```bash
# 전체 실행 (6종 모델 × 160개 설정 × 10회 반복)
python run_benchmark.py

# 특정 모델만 실행
python run_benchmark.py --model simple_ann
python run_benchmark.py --model transformer
python run_benchmark.py --model gan

# 빠른 테스트 (3회 반복, CPU만)
python run_benchmark.py --repeats 3 --device cpu

# Op-level 프로파일링 포함
python run_benchmark.py --profile-ops

# 중단 후 이어서 실행
python run_benchmark.py --resume
```

### 2. 예측 모델 학습

```bash
# 기본 실행 (5-fold CV)
python train_predictor.py

# 10-fold CV + 모델 저장
python train_predictor.py --cv 10 --save-models
```

**예측 타겟 3가지**: 학습 시간, 추론 시간, 메모리 요구량
**ML 모델 4가지**: LinearRegression, RandomForest+GridSearchCV, GradientBoosting, XGBoost+GridSearchCV

### 3. ONNX 예측 파이프라인

```bash
# 대표 모델을 ONNX로 변환
python export_onnx.py

# ONNX 파일에서 실행 시간 예측
python predict_from_onnx.py --onnx model.onnx --device cpu
python predict_from_onnx.py --demo  # 전체 샘플 예측
```

### 4. Op-Level 프로파일링 (단독 사용)

```python
from benchmark.features.op_profiler import measure_op_times, simulate_total_time, print_op_profile
from benchmark.models.simple_cnn import SimpleCNN

model = SimpleCNN(num_filters=32, num_conv_layers=3)
ops = measure_op_times(model, input_shape=(1, 1, 28, 28), device='cpu')
print_op_profile(ops)

sim = simulate_total_time(ops)
print(f"시뮬레이션 총 시간: {sim['total_time_ms']:.4f}ms")
```

## 피처 목록 (30개)

| 카테고리 | 피처 |
|---|---|
| 파라미터 수 | total_params, trainable_params, conv_params, linear_params, bn_params, other_params |
| 레이어 수 | num_conv_layers, num_linear_layers, num_bn_layers, num_pool_layers, num_activation_layers, total_layers |
| 연산량/크기 | flops, memory_bytes, model_size_mb, depth, max_channel_width |
| 구조 플래그 | has_residual, has_depthwise, has_attention |
| 하드웨어 | cpu_cores, cpu_freq_ghz, ram_total_gb, gpu_memory_gb |
| 모델 유형 (원핫) | model_type_simple_ann, model_type_simple_cnn, model_type_resnet_mnist, model_type_mobilenet_mnist, model_type_transformer, model_type_gan |
