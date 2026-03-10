# dal-merge 브랜치 통합 보고서

dal-merge 브랜치를 베이스로, khg9859/present 브랜치의 주요 기능을 포팅하여 통합하였다.

---

## 1. dal-merge 원본 (통합 전)

### 구현 내용
- ANN(SimpleANN), CNN(SimpleCNN) 두 가지 모델
- 워밍업 + 반복 측정 기반 실행 시간 수집 (`collect/ann_collector.py`, `collect/cnn_collector.py`)
- 27개 feature 추출 (`features/extractor.py`)
- LinearRegression / RandomForest / XGBoost 예측 모델 학습 (`predictor/train.py`)
  - GridSearchCV + K-Fold 교차검증
  - log1p 변환으로 타겟 안정화
  - 장치별(CPU/CUDA) 분리 학습
- 하드웨어 feature 수집 (`cpu_cores`, `cpu_freq_ghz`, `ram_gb`, `gpu_memory_gb` 등)

### 데이터
- ANN: 49가지 조합 × 2 device = 98개
- CNN: 60가지 조합 × 2 device = 120개
- 합계: 218개

---

## 2.  가져온 기능
**출처**: khg9859/present 브랜치

### 2-1. Transformer 모델 (`models/transformer.py`)

**개념**

원래 자연어 처리(번역, 텍스트 생성)에 쓰이던 구조로, 2020년대부터 이미지 분류에도 적용되기 시작했다(Vision Transformer, ViT).

핵심 아이디어는 이미지를 작은 패치(조각)로 나눠 각 패치를 단어처럼 취급하고, Attention 메커니즘으로 패치 간의 관계를 학습하는 것이다.

```
입력 이미지 (32×32)
    ↓ PatchEmbedding (4×4 패치 64개로 분할)
패치 시퀀스 (64개 벡터)
    ↓ MultiHeadAttention × num_layers
패치 간 관계 학습
    ↓ CLS 토큰 → Linear
분류 출력
```

ANN/CNN과 달리 `embed_dim`, `num_heads`, `patch_size` 등 고유한 구조 파라미터를 가지며, GELU 활성화 함수와 LayerNorm을 사용하는 것이 특징이다.

**가져온 이유**
- ANN/CNN만으로는 모델 아키텍처 다양성이 부족하여 예측 모델의 일반화 한계가 있음
- Transformer는 현재 가장 널리 쓰이는 아키텍처로, 실험 범위 확장에 필수적
- `embed_dim`, `num_heads`, `num_transformer_layers`, `patch_size` 등 ANN/CNN과 전혀 다른 구조 feature를 추가로 수집 가능

**구현 내용**
- `SimpleViT`: PatchEmbedding + MultiHeadAttention + TransformerEncoderBlock
- CIFAR-10(32×32 컬러) 기준 12가지 조합 생성
- `collect/transformer_collector.py`: CIFAR-10 기반 실행 시간 측정

---

### 2-2. GAN 모델 (`models/gan.py`)

**출처**: khg9859/present 브랜치

**개념**

GAN(Generative Adversarial Network)은 두 신경망이 서로 경쟁하며 학습하는 구조다.

- **Generator**: 무작위 노이즈 벡터를 입력받아 가짜 이미지를 생성
- **Discriminator**: 이미지를 보고 진짜/가짜를 판별

```
노이즈 벡터 (latent_dim 차원)
    ↓ Generator (Linear → BN → ReLU 반복)
가짜 이미지
    ↓ Discriminator (Linear → LeakyReLU 반복)
진짜/가짜 확률
```

일반 분류 모델과 달리 Generator와 Discriminator를 번갈아 학습하기 때문에 학습 시간 패턴이 근본적으로 다르다. Generator의 노이즈 차원(`latent_dim`)과 히든 레이어 크기가 실행 시간에 직접적인 영향을 준다.

**가져온 이유**
- GAN은 Generator/Discriminator 두 네트워크가 적대적으로 학습하는 구조로, 일반 분류 모델과 실행 시간 패턴이 근본적으로 다름
- `latent_dim`, `g_hidden_max` 등 GAN 전용 feature를 통해 예측 모델이 다양한 학습 패턴을 학습할 수 있음
- 8가지 조합으로 데이터 다양성 확보

**구현 내용**
- `SimpleGAN`: Generator(Linear + BN + ReLU) + Discriminator(Linear + LeakyReLU)
- MNIST(28×28 흑백) 기준 8가지 조합 생성
- `collect/gan_collector.py`: 적대적 학습 시간 + Generator 추론 시간 측정

---

### 2-3. Op-Level 프로파일러 (`features/op_profiler.py`)

**출처**: ijunsoo 브랜치 → khg9859가 MPS 지원 추가하여 포팅 → dal-merge에서 CUDA/CPU 환경으로 재포팅

**개념**

모델을 실행할 때 내부에서 일어나는 연산(Conv, Linear, BN 등)을 하나씩 분해하여 각각의 FLOPs와 메모리 사용량을 측정하는 기법이다.

기존 end-to-end 방식은 모델 전체 실행 시간만 측정하므로 "같은 FLOPs라도 어떤 연산으로 구성되어 있느냐"에 따라 달라지는 실행 시간 차이를 설명하지 못한다. Op-Level 프로파일링은 이를 보완하여 연산 구성 비율을 feature로 제공한다.

예시:
```
ANN:         flops_ratio_Linear ≈ 1.0  (행렬 곱만 있음)
CNN:         flops_ratio_Conv2d ≈ 0.9  (합성곱 위주)
Transformer: flops_ratio_GELU  ≈ 0.1  (Attention + GELU 혼합)
```

**가져온 이유**
- 모델 전체를 하나의 블랙박스로 보는 end-to-end 방식으로는 "같은 FLOPs라도 연산 구성에 따라 실행 시간이 다름"을 설명 불가
- 연산 단위(Conv, Linear, BN, Pooling 등)로 분해하면 예측 모델이 연산 특성을 더 세밀하게 학습 가능
- Transformer의 Attention 비율(`flops_ratio_GELU`, `flops_ratio_LayerNorm`)과 CNN의 Conv 비율(`flops_ratio_Conv2d`)을 구분하여 아키텍처 특성 반영 가능

**추가된 feature (15개)**
| feature | 설명 |
|---------|------|
| `num_ops` | 모델 내 연산 노드 수 |
| `total_op_flops` | op 단위로 계산한 전체 FLOPs |
| `total_op_memory_read` | 전체 메모리 읽기량 (bytes) |
| `total_op_memory_write` | 전체 메모리 쓰기량 (bytes) |
| `memory_bytes` | 가중치 메모리 크기 (bytes) |
| `flops_ratio_Conv2d` | Conv 연산이 차지하는 FLOPs 비율 |
| `flops_ratio_Linear` | Linear 연산 FLOPs 비율 |
| `flops_ratio_BatchNorm2d` | BatchNorm 연산 FLOPs 비율 |
| `flops_ratio_LayerNorm` | LayerNorm 연산 FLOPs 비율 (Transformer) |
| `flops_ratio_MaxPool2d` | MaxPool 연산 FLOPs 비율 |
| `flops_ratio_ReLU` | ReLU 연산 FLOPs 비율 |
| `flops_ratio_GELU` | GELU 연산 FLOPs 비율 (Transformer) |
| `max_op_flops` | 단일 op 중 최대 FLOPs |
| `avg_op_flops` | 단일 op 평균 FLOPs |
| `std_op_flops` | 단일 op FLOPs 표준편차 |

---

### 2-4. ONNX Feature 추출기 (`features/onnx_extractor.py`)

**출처**: khg9859/present 브랜치 → dal-merge feature 스키마에 맞게 재포팅

**개념**

ONNX(Open Neural Network Exchange)는 PyTorch, TensorFlow 등 서로 다른 프레임워크로 만든 모델을 하나의 표준 파일 형식(`.onnx`)으로 저장하는 규격이다.

`.onnx` 파일 안에는 모델의 레이어 연결 구조와 학습된 가중치가 모두 포함되어 있어, PyTorch 코드 없이도 모델 구조를 파싱할 수 있다.

활용 흐름:
```
PyTorch 모델
    ↓ torch.onnx.export()
model.onnx 파일
    ↓ onnx_extractor.py 파싱
feature 딕셔너리 (파라미터 수, 레이어 수 등)
    ↓ 학습된 예측 모델 적용
실행 시간 예측 출력
```

**가져온 이유**
- PyTorch 모델 객체 없이 `.onnx` 파일만으로도 feature를 추출하여 실행 시간을 예측하는 것이 최종 목표
- 다른 프레임워크(TensorFlow 등)로 만든 모델도 ONNX로 변환하면 동일한 예측 파이프라인 적용 가능
- khg9859가 Stage 5로 검증한 구조를 dal-merge feature 스키마에 맞게 포팅

**활용 시점**: 현재는 데이터 수집 단계라 미사용. 예측 파이프라인 완성 시 사용

