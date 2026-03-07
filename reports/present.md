# DNN 실행 시간 예측 시스템 - 팀 공유 문서

> 작성자: 김홍근 (학부연구생)  
> 최종 업데이트: 2026년 3월  
> 연구 주제: AI 프로세서 SW 프레임워크 연구 / 4세부 통합 시뮬레이터

---

## 한 줄 요약

**"AI 모델의 구조 정보만 입력하면, 실제로 돌려보지 않아도 실행 시간을 예측하는 시스템"**

예를 들어, `"이 모델은 레이어 4개에 파라미터 10만 개짜리야"` 라고 알려주면,
CPU에서 추론하는 데 몇 ms가 걸리는지, 학습에 몇 초가 걸리는지 예측합니다.

---

## 프로젝트 배경

교수님 연구계획 중 DNN 파트:

> DNN 모델 추론 시간/공간 요구량 예측  
> → 하드웨어 비의존적 성능 예측 모델  
> → 모델 구조 분석 및 연산 특징(연산 수, 메모리 접근 패턴 등) 추출  
> → 대상 모델: **Transformer, CNN, GAN**

방법 1 (이번 연구에서 채택):
```
입력: 모델 구조 정보 (파라미터 수, 레이어 수, 채널 수, ...) 
  → 기계학습(Random Forest / XGBoost) 예측 모델
  → 출력: 추론 시간(ms), 학습 시간(sec)
```

---

## 완료된 작업

### ✅ Stage 1: ANN 모델 × MNIST 실행 시간 측정

**무슨 작업?**  
가장 단순한 신경망(ANN = 선형 레이어를 쌓은 것)으로 손글씨 숫자(MNIST) 분류를 학습시키면서, 레이어 수와 뉴런 수를 바꿔가며 실행 시간을 측정했습니다.

**실험 설계:**
- 레이어 수: 2, 3, 4, 5층
- 뉴런 수(width): 32, 64, 128, 256, 512, 1024
- 다양한 형태: 피라미드형(넓다가 좁아짐), 역피라미드형(좁다가 넓어짐)
- 디바이스: CPU, MPS (M1 GPU) 각각 측정
- **총 52가지 모델 × 2 디바이스 = 52개 데이터 포인트**

**주요 발견:**
- 파라미터가 많을수록 추론/학습 시간 증가 (양의 상관관계 확인)
- 작은 모델에서는 MPS가 CPU보다 오히려 느림 (GPU 오버헤드 때문)
- CPU 평균 추론 시간: 0.08ms / MPS: 0.60ms

**결과 파일:** `data/stage1/ann_mnist_results.csv`

---

### ✅ Stage 2: CNN 모델 × CIFAR-10 실행 시간 측정

**무슨 작업?**  
이미지 분류에 특화된 CNN(합성곱 신경망)으로 컬러 이미지(CIFAR-10)를 학습시키면서 실행 시간을 측정했습니다.

**실험 설계:**
- SimpleCNN: Conv 레이어 수 (2, 3, 4) × 채널 수 (16, 32, 64) = 9가지
- ResNet18: 1종 (잔차 연결 기반 18레이어 CNN)
- MobileNetV2: 1종 (모바일용 경량 CNN)
- **총 11가지 모델 × 2 디바이스 = 22개 데이터 포인트**

**주요 발견:**
- ResNet18: CPU 799ms / MPS 80ms → 약 10배 차이
- 큰 모델일수록 MPS 가속 효과가 더 커짐
- 레이어 증가 → 정확도 향상 (L2: 39~49%, L4: 58~61%)

**결과 파일:** `data/stage2/cnn_cifar10_results.csv`

---

### ✅ Stage 3: 예측 모델 개발 (Random Forest + XGBoost)

**무슨 작업?**  
Stage 1~2에서 수집한 74개 데이터를 학습 데이터로 삼아,
"모델 구조 → 실행 시간" 예측 모델(Random Forest, XGBoost)을 개발했습니다.

**핵심 아이디어 3가지:**

| 기법 | 이유 |
|------|------|
| **로그(log) 변환** | 실행 시간 범위가 0.026ms ~ 799ms로 매우 넓음 → log를 씌워서 범위를 균등하게 |
| **K-Fold 교차검증 (K=5)** | 데이터가 74개뿐 → 5번 나눠서 평균 성능을 보면 더 신뢰할 수 있음 |
| **GridSearchCV** | 최적 하이퍼파라미터 자동 탐색 (n_estimators, max_depth 등) |

**성능 결과 (최종 v2):**

| 모델 | 예측 대상 | R² (log 공간) | R² (원래 단위) |
|------|-----------|--------------|----------------|
| XGBoost | 추론 시간 | **0.947** | 0.743 |
| Random Forest | 추론 시간 | 0.886 | 0.209 |
| XGBoost | 학습 시간 | 0.779 | 0.017 |
| Random Forest | 학습 시간 | 0.787 | 0.013 |

> R²가 1에 가까울수록 좋음. 추론 시간 예측은 잘 되고 있으며(XGBoost R²=0.74),  
> 학습 시간 예측은 데이터 다양성 부족으로 아직 낮음 → Transformer/GAN 데이터 추가 시 개선 기대

**결과 파일:**
- `data/stage3/v2/stage3_v2_prediction_results.csv`
- `data/stage3/v2/stage3_v2_merged_features.csv`
- `data/stage3/v2/*.png` (Feature 중요도, 예측 vs 실제 산점도 그래프)

---

### ✅ Stage 4 (일부): Transformer 실행 시간 측정

**무슨 작업?**  
최신 AI에서 핵심적으로 사용되는 Transformer 구조(ViT: Vision Transformer)를 직접 구현하고, 다양한 크기 조합으로 CIFAR-10 학습 시간을 측정했습니다.

**실험 설계:**

| 하이퍼파라미터 | 실험 값 |
|--------------|---------|
| embed_dim (벡터 차원) | 64, 128, 256 |
| num_layers (인코더 블록 수) | 2, 4, 6 |
| num_heads (어텐션 헤드 수) | 4, 8 |
| patch_size (이미지 패치 크기) | 4, 8 |

- **총 12가지 조합 × 2 디바이스 = 24개 데이터 포인트** ✅ 완료

**결과 파일:** `data/stage4/transformer_results.csv`

---

## 미완료 작업

### ⏳ Stage 4 (미완료): GAN 실행 시간 측정

GAN(Generative Adversarial Network, 생성적 적대 신경망) 구조를 구현하고 측정 예정.
- Generator와 Discriminator를 함께 학습시키는 독특한 구조
- 실험 스크립트 준비 완료: `experiments/gan_experiment.py`
- 데이터 저장 예정: `data/stage4/gan_results.csv`

### ⏳ Stage 3 업데이트: Transformer/GAN 데이터 통합

`experiments/train_predictor.py`가 현재 ANN+CNN 데이터(74개)만 사용.  
Transformer/GAN 데이터를 통합하면 예측 성능이 향상될 것으로 기대.

### ⏳ Stage 5: ONNX 기반 시스템 구축 (미시작)

최종 목표:
```
ONNX 모델 파일 입력 → CPU/GPU에서 학습/추론 시간 예측
```
- ONNX: 다양한 AI 프레임워크 간 표준 모델 포맷 (PyTorch, TensorFlow 등)
- 모델을 ONNX로 저장하면 구조 정보를 자동으로 파싱 가능
- 현재 연구에서 구축한 예측 모델을 ONNX 파이프라인에 통합할 예정

---

## 코드 파일 구조

```
dnn/
├── models/                         # 신경망 모델 정의
│   ├── ann_models.py               # ANN 모델 (SimpleANN, 52가지 변형 생성)
│   ├── cnn_models.py               # CNN 모델 (SimpleCNN, ResNet18, MobileNetV2)
│   ├── transformer_models.py       # Transformer 모델 (SimpleViT, 12가지 변형)
│   └── gan_models.py               # GAN 모델 (Generator, Discriminator, SimpleGAN)
│
├── experiments/                    # 실험 스크립트
│   ├── ann_mnist_experiment.py     # Stage 1: ANN 시간 측정 (MNIST)
│   ├── cnn_cifar10_experiment.py   # Stage 2: CNN 시간 측정 (CIFAR-10)
│   ├── transformer_experiment.py   # Stage 4: Transformer 시간 측정 (CIFAR-10)
│   ├── gan_experiment.py           # Stage 4: GAN 시간 측정 (미완료)
│   └── train_predictor.py          # Stage 3: 예측 모델 학습 (RF + XGBoost)
│
├── utils/
│   └── timer.py                    # 핵심: 시간 측정 유틸리티 (Warmup + 반복측정)
│
├── data/                           # 실험 데이터 (Git에 포함 권장)
│   ├── stage1/ann_mnist_results.csv        # ANN 실험 결과 (52행)
│   ├── stage2/cnn_cifar10_results.csv      # CNN 실험 결과 (22행)
│   ├── stage3/v2/stage3_v2_*.csv/png       # 예측 모델 결과 및 그래프
│   └── stage4/transformer_results.csv      # Transformer 실험 결과 (24행)
│
└── reports/
    ├── research_plan.md            # 전체 연구 계획 (항상 먼저 읽을 것)
    ├── stage1_2_report.md          # Stage 1~2 완료 보고서
    ├── stage3_report.md            # Stage 3 완료 보고서 (입문자 친화적)
    └── present.md                  # 이 파일 (팀 공유용)
```

---

## 팀과 공유하면 좋은 코드 & 아이디어

### 1. 정확한 시간 측정 방법론 → `utils/timer.py`

단순히 `time.time()`으로 측정하면 부정확합니다.

```python
# ❌ 나쁜 예 (부정확)
start = time.time()
output = model(input)
end = time.time()

# ✅ 좋은 예 (이 프로젝트 방식)
# 1) Warmup: 캐시, JIT 컴파일 등 초기화 비용 제거
for _ in range(5):
    _ = model(dummy_input)
    torch.mps.synchronize()  # ← GPU 비동기 실행 완료 대기

# 2) 반복 측정 후 평균
times = []
for _ in range(10):
    start = time.time()
    _ = model(dummy_input)
    torch.mps.synchronize()
    end = time.time()
    times.append((end - start) * 1000)  # ms 변환

mean_time = np.mean(times)  # 평균
std_time = np.std(times)    # 표준편차 (신뢰성 확인용)
```

**핵심 포인트:**
- `torch.mps.synchronize()`: MPS(M1 GPU)에서는 GPU가 비동기로 실행되기 때문에 반드시 필요
- Warmup: 첫 실행은 항상 느림 (GPU 초기화, 캐시 미스) → 제외해야 함

---

### 2. 29가지 Feature Engineering → `experiments/train_predictor.py`

모델 구조를 수치로 표현하는 방법:

```python
# 모델 구조 → 예측에 사용할 숫자들 (Feature)
features = {
    # 모델 크기 관련
    'total_params': 1_234_567,      # 전체 파라미터 수
    'model_size_mb': 4.71,          # 모델 파일 크기 (MB)
    'num_layers': 4,                # 레이어 수
    'max_width': 512,               # 가장 넓은 레이어 뉴런 수

    # CNN 구조 관련
    'num_conv_layers': 3,           # Conv 레이어 수
    'max_channels': 256,            # 최대 채널 수
    'kernel_size': 3,               # Conv 필터 크기
    'has_batch_norm': 1,            # BatchNorm 사용 여부

    # Transformer 구조 관련
    'embed_dim': 128,               # 임베딩 차원
    'num_heads': 8,                 # Attention Head 수
    'patch_size': 4,                # 이미지 패치 크기

    # 하드웨어 관련
    'cpu_cores': 8,                 # CPU 코어 수
    'ram_gb': 8,                    # RAM 크기
    'device_type': 0,               # 0=CPU, 1=MPS/GPU

    # 로그 변환 Feature (범위 정규화용)
    'log_total_params': np.log1p(1_234_567),
    'log_model_size_mb': np.log1p(4.71),
    ...
}
```

---

### 3. 로그 변환 + K-Fold + GridSearchCV 파이프라인

데이터가 적을 때 (74개) 믿을 만한 성능을 얻는 방법:

```python
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np

# Step 1: 로그 변환 (실행 시간은 범위가 넓어서 그대로 쓰면 ML이 힘들어함)
y_log = np.log1p(y)  # log(1+x): 0도 안전하게 처리

# Step 2: K-Fold 교차검증 (데이터를 5번 나눠서 평균 성능 계산)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: GridSearchCV (하이퍼파라미터 자동 탐색)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, None],
    'learning_rate': [0.05, 0.1, 0.2],
}
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='r2')
grid_search.fit(X, y_log)

# Step 4: 예측 후 역변환
y_pred_log = grid_search.predict(X_test)
y_pred = np.expm1(y_pred_log)  # log 역변환
```

---

### 4. Transformer 직접 구현 코드 → `models/transformer_models.py`

GPT, BERT 등에 쓰이는 Transformer를 PyTorch로 직접 구현했습니다:

```python
class SimpleViT(nn.Module):
    """
    Vision Transformer (ViT):
    이미지를 패치로 나눠 → 각 패치를 임베딩 → Transformer Encoder 통과 → 분류
    """
    # PatchEmbedding → MultiHeadAttention → TransformerEncoderBlock → 분류
```

---

## 코드 합치기 (팀 코드 통합) 가이드

> 팀원들이 다른 브랜치에서 작업한 코드를 이 프로젝트와 합치는 방법

### Git 브랜치 전략

```bash
# 브랜치 확인
git branch -a

# 원격 저장소 연결 (처음 1회만)
git remote add origin https://github.com/iSysLab/ai-simulator.git

# 내 브랜치: khg9859-xxx 형태로 생성
git checkout -b khg9859-dnn-predictor

# 팀원 브랜치 내려받기
git fetch origin
git checkout -b teammate-branch origin/teammate-branch

# 내 브랜치에 팀원 작업 병합
git checkout khg9859-dnn-predictor
git merge teammate-branch
```

### 파일 충돌 없이 통합하는 팁

각자 역할에 따라 파일을 분리하면 충돌을 최소화할 수 있습니다:

| 역할 | 담당 파일 |
|------|----------|
| DNN 시간 예측 (이 프로젝트) | `models/`, `experiments/`, `utils/timer.py` |
| SNN 시뮬레이터 | `snn/` 폴더 별도 생성 |
| 통합 인터페이스 | `main.py`, `api/` 폴더 |
| 공통 유틸리티 | `utils/` (timer.py 외) |

---

## 환경 세팅 및 실행 방법

### 1. 환경 준비

```bash
# Python 3.13 가상환경 (이미 있으면 활성화만)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# 패키지 설치
pip install torch torchvision pandas numpy scikit-learn xgboost matplotlib joblib

# macOS에서 XGBoost 필수 의존성
brew install libomp
```

### 2. 실험 실행 순서

```bash
# Stage 1: ANN 실험 (M1 MacBook 기준 약 2~3시간)
python experiments/ann_mnist_experiment.py

# Stage 2: CNN 실험 (약 4~6시간)
python experiments/cnn_cifar10_experiment.py

# Stage 3: 예측 모델 학습 (약 1~2분, 데이터 있어야 함)
python experiments/train_predictor.py

# Stage 4-a: Transformer 실험 (약 3~5시간)
python experiments/transformer_experiment.py

# Stage 4-b: GAN 실험 (미완료)
python experiments/gan_experiment.py
```

> **참고:** Stage 1~2, 4 실험은 수집한 데이터가 이미 CSV로 저장되어 있으면 재실행하지 않아도 됩니다.  
> `data/stage1/ann_mnist_results.csv` 등이 있으면 Stage 3만 바로 실행 가능합니다.

### 3. 결과 확인

```bash
# 예측 성능 결과 보기
cat data/stage3/v2/stage3_v2_prediction_results.csv

# 수집된 데이터 현황
wc -l data/stage1/*.csv data/stage2/*.csv data/stage4/*.csv
```

---

## 연구 진행 현황 요약

| 단계 | 내용 | 상태 | 데이터 수 |
|------|------|------|----------|
| Stage 1 | ANN × MNIST 시간 측정 | ✅ 완료 | 52행 |
| Stage 2 | CNN × CIFAR-10 시간 측정 | ✅ 완료 | 22행 |
| Stage 3 | RF/XGBoost 예측 모델 (v1) | ✅ 완료 | - |
| Stage 3 | RF/XGBoost 예측 모델 (v2, 개선) | ✅ 완료 | 74행 학습 |
| Stage 4-a | Transformer × CIFAR-10 시간 측정 | ✅ 완료 | 24행 |
| Stage 4-b | GAN × MNIST 시간 측정 | ⏳ 미완료 | - |
| Stage 3 업 | Transformer+GAN 데이터 통합 | ⏳ 미완료 | - |
| Stage 5 | ONNX 기반 예측 시스템 | ⏳ 미시작 | - |

---

## 참고 자료

- **전체 연구 계획:** `reports/research_plan.md`
- **Stage 1~2 보고서:** `reports/stage1_2_report.md`  
- **Stage 3 상세 보고서:** `reports/stage3_report.md` (입문자용 설명 포함)
- **SNN 튜토리얼:** https://snntorch.readthedocs.io/en/latest/tutorials/index.html
- **GitHub:** https://github.com/iSysLab/ai-simulator.git
