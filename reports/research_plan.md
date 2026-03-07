# DNN 시간 추정 모델 개발 - 연구 계획 및 진행 현황

**작성자:** 김홍근  
**하드웨어:** MacBook Air M1 (CPU + MPS Apple Silicon GPU)  
**최종 수정:** 2026년 3월

---

## 1. 연구 개요

### 1.1 소속 및 주제
- **소속:** 학부연구생
- **연구 주제:** AI 프로세서 SW 프레임워크 연구 — 4세부 통합 시뮬레이터
- **담당 파트:** DNN 모델 추론 시간 / 공간(메모리) 요구량 예측

### 1.2 연구 목표
DNN 모델의 **구조 정보(레이어 수, 파라미터 수, FLOPs 등)** 만으로  
CPU / GPU에서의 **학습 및 추론 시간을 사전에 예측**하는 시스템 개발

```
모델 구조 정보 (Feature)
       ↓
   예측 모델 (기계학습)
       ↓
CPU / GPU 추론·학습 시간 예측
```

### 1.3 최종 목표
```
ONNX 모델 파일 입력
       ↓
   모델 구조 파싱
       ↓
   Feature 추출
       ↓
   예측 모델 적용
       ↓
CPU / GPU 학습·추론 시간 출력
```

---

## 2. 연도별 연구 로드맵

| 연도 | 내용 |
|------|------|
| **1차년도** | BindsNet / snntorch / Brian2 비교 분석, 메타모델 가능성 검증, 시뮬레이션 코드 변환 |
| **2차년도** | SpikeJelly 추가, SNN + DNN 통합 메타모델 정의, ONNX 기반 시간 예측 시스템 완성 |

### 2.1 접근 방법 (DNN 시간 예측)
- **방법 1:** 기계학습 메타모델 — `모델 구조 (input)` → `실행 시간 (target)` 학습
- **방법 2:** 모델을 작은 연산 단위로 분해 → 각 단위 시간 측정 → 합산으로 전체 예측
- **방법 3:** ONNX 표준 포맷 모델 입력 → CPU/GPU 추론 시간 예측

---

## 3. 완료된 작업

### 3.1 단계 1 — 환경 설정 및 ANN 데이터 수집 ✅

#### 완료 내용
| 항목 | 내용 |
|------|------|
| 환경 | Python 3.13, PyTorch, PyCharm |
| 데이터셋 | MNIST (손글씨 숫자 인식) |
| 모델 | ANN (단순 다층 퍼셉트론) |
| 측정 방식 | Warmup 후 10회 이상 반복 측정 → 평균값 사용 |
| 수집 데이터 | 14 모델 × 2 디바이스(CPU/MPS) = **28개 데이터 포인트** |

#### ANN 모델 구성
- **레이어 수:** 2층, 3층, 4층
- **Width(히든 뉴런 수):** 64, 128, 256, 512
- **조합:** 14가지 (일부 조합 제외)

#### 주요 발견 사항
- **파라미터 수 ↑ → 추론·학습 시간 ↑** (강한 양의 상관관계)
- **작은 모델에서는 MPS가 CPU보다 느림** (GPU 초기화·데이터 전송 오버헤드)
- **추론 시간:** CPU 평균 0.08ms / MPS 평균 0.60ms
- **학습 시간:** CPU 평균 3.1초 / MPS 평균 5.4초

---

### 3.2 단계 2 — CNN 모델로 데이터 확장 ✅

#### 완료 내용
| 항목 | 내용 |
|------|------|
| 데이터셋 | CIFAR-10 (컬러 이미지 분류, 32×32) |
| 모델 | SimpleCNN, ResNet18, MobileNetV2 |
| 수집 데이터 | 11 모델 × 2 디바이스(CPU/MPS) = **22개 데이터 포인트** |

#### CNN 모델 구성
- **SimpleCNN:** 레이어 수(2, 3, 4) × 채널 수(16, 32, 64) = 9개
- **ResNet18:** CIFAR-10용으로 수정
- **MobileNetV2:** CIFAR-10용으로 수정

#### 주요 발견 사항

| 모델 | 파라미터 | CPU 추론 | MPS 추론 | 비고 |
|------|----------|----------|----------|------|
| SimpleCNN_L2_C16 | 1.05M | 6.1ms | 2.0ms | 가장 단순 |
| SimpleCNN_L4_C64 | 2.60M | 95.8ms | 9.9ms | 깊고 넓음 |
| ResNet18 | 11.1M | 799ms | 80ms | 잔차 연결 구조 |
| MobileNetV2 | 2.23M | 508ms | 16ms | 경량화 설계 (MPS 최고 효율) |

- **MobileNetV2:** 파라미터는 적지만 CPU에서 ResNet18보다 느림 (Depthwise Conv CPU 비효율)
- **MPS 가속:** MobileNetV2 기준 약 **31배**, ResNet18 기준 약 **10배**

---

### 3.3 통합 현황 (1+2단계)
- **총 데이터 포인트:** 28 + 22 = **50개**
- **파라미터 범위:** ~100K (작은 ANN) ~ 11.1M (ResNet18)

---

## 4. 교수님 유의사항

> 아래 사항을 반드시 지켜서 실험 및 코드 작성

1. **자신의 H/W에서 실험 수행**  
   - 사용 환경: MacBook Air M1 (CPU + MPS)
   - 하드웨어 사양, 모델 특징, 실행 시간 모두 기록

2. **Feature를 최대한 많이 선정**  
   - 모델 구조 Feature + 하드웨어 Feature + 입력 데이터 Feature (아래 섹션 참고)

3. **ResNet / MobileNet 가급적 사용 자제**  
   - 이미 수행한 데이터는 유지하되, 새로운 실험에서는 가급적 피할 것

4. **ANN은 레이어 수, 히든 뉴런 수를 다양하게**  
   - 더 많은 조합으로 데이터 확장 권장

5. **소스코드에 주석 상세히 작성**  
   - 교수님이 코드를 직접 검토하심
   - 초보자도 이해할 수 있는 수준으로 설명

6. **수집한 데이터로 입/출력 정리 후 기계학습 수행**  
   - 학습(train) / 테스트(test) 분리하여 성능 평가

---

## 5. Feature 목록 (Feature Engineering)

> 예측 모델의 입력(Input)으로 사용할 Feature들. **최대한 많이** 정의하는 것이 목표.

### 5.1 모델 구조 Feature (공통)
| Feature | 설명 |
|---------|------|
| `num_layers` | 전체 레이어 수 |
| `num_params` | 전체 파라미터 수 |
| `model_type` | 모델 종류 (ANN / CNN / Transformer 등) |
| `flops` | 전체 FLOPs (부동소수점 연산 수) |
| `model_size_mb` | 모델 파일 크기 (MB) |

### 5.2 ANN 전용 Feature
| Feature | 설명 |
|---------|------|
| `ann_hidden_layers` | 히든 레이어 수 |
| `ann_neurons_per_layer` | 각 레이어별 뉴런 수 (리스트) |
| `ann_max_width` | 가장 넓은 레이어의 뉴런 수 |
| `ann_min_width` | 가장 좁은 레이어의 뉴런 수 |
| `ann_activation` | 활성화 함수 종류 (ReLU, Sigmoid 등) |

### 5.3 CNN 전용 Feature
| Feature | 설명 |
|---------|------|
| `cnn_num_conv_layers` | Conv 레이어 수 |
| `cnn_channels` | 각 Conv 레이어별 채널 수 (리스트) |
| `cnn_kernel_sizes` | 각 레이어별 커널 크기 |
| `cnn_stride` | Stride 값 |
| `cnn_has_pooling` | 풀링 레이어 유무 (True/False) |
| `cnn_pooling_type` | 풀링 종류 (MaxPool / AvgPool 등) |
| `cnn_has_batchnorm` | BatchNorm 유무 |
| `cnn_num_fc_layers` | Fully Connected 레이어 수 |

### 5.4 하드웨어 Feature
| Feature | 설명 |
|---------|------|
| `device_type` | CPU / MPS / CUDA |
| `cpu_cores` | CPU 코어 수 (논리) |
| `cpu_freq_ghz` | CPU 클럭 속도 (GHz) |
| `cpu_cache_l2_mb` | L2 캐시 크기 (MB) |
| `cpu_cache_l3_mb` | L3 캐시 크기 (MB) |
| `ram_total_gb` | 전체 RAM 용량 (GB) |
| `ram_available_gb` | 실험 시점 가용 RAM (GB) |
| `gpu_memory_gb` | GPU 메모리 (VRAM) — M1은 Unified Memory |

### 5.5 입력 데이터 Feature
| Feature | 설명 |
|---------|------|
| `batch_size` | 배치 크기 |
| `input_channels` | 입력 채널 수 (흑백=1, 컬러=3) |
| `input_height` | 입력 이미지 높이 |
| `input_width` | 입력 이미지 너비 |
| `dataset_name` | 데이터셋 이름 (MNIST, CIFAR-10 등) |
| `num_classes` | 분류 클래스 수 |

---

## 6. 앞으로 해야 할 것

### 단계 3 — 예측 모델 개발 ✅ 완료

- ANN 데이터 12종 추가 (Width 32/1024, 5층 레이어, 피라미드형) → 총 74개 데이터
- 로그 변환(log1p) + K-Fold(5) 교차검증 + GridSearchCV 하이퍼파라미터 튜닝 적용
- **XGBoost 추론 시간 R² = 0.74** (원래 단위) / **0.95** (log 공간) 달성
- 결과: `data/stage3/v2/` 폴더에 CSV + PNG 8개 저장
- 보고서: `reports/stage3_report.md`

### 단계 4 — Transformer / GAN 데이터 수집 (현재 단계)

**목표:** ANN / CNN 에만 편중된 데이터를 Transformer, GAN으로 다양화하여 예측 모델 일반화 성능 향상

#### 4.1 측정할 모델

| 모델 종류 | 구현 방식 | 데이터셋 | 예상 실험 시간 |
|-----------|-----------|---------|--------------|
| 소형 Transformer | 직접 구현 (Patch Embedding + Multi-Head Attention + MLP) | CIFAR-10 | 2~4시간 |
| 소형 GAN | 직접 구현 (Generator + Discriminator) | MNIST 또는 CIFAR-10 | 3~5시간 |

#### 4.2 Transformer 모델 구조 (직접 설계)

교수님 유의사항: ResNet/MobileNet 자제 → 직접 구현하는 소형 Transformer 사용

```
입력 이미지 (32×32×3)
  ↓  Patch Embedding  (이미지를 작은 조각으로 나눔)
  ↓  Transformer Encoder Block × N층
      - Multi-Head Self-Attention  (패치들 간 관계 학습)
      - Feed-Forward Network (MLP)
      - LayerNorm + Residual Connection
  ↓  Classification Head (MLP)
  ↓  출력 (10개 클래스)
```

다양한 구조 조합 측정:
- 레이어(Encoder Block) 수: 2, 4, 6
- 임베딩 차원: 64, 128, 256
- Attention Head 수: 4, 8
- 패치 크기: 4×4, 8×8

#### 4.3 GAN 모델 구조 (직접 설계)

```
Generator (생성자):
  노이즈 벡터 (latent_dim)
    ↓  FC → BatchNorm → ReLU (반복)
    ↓  최종 이미지 생성

Discriminator (판별자):
  이미지 입력
    ↓  FC → LeakyReLU (반복)
    ↓  진짜/가짜 판별
```

다양한 구조 조합 측정:
- latent_dim: 64, 128, 256
- Generator 레이어 수: 3, 4, 5
- Discriminator 레이어 수: 3, 4, 5

#### 4.4 추가 Feature (Transformer / GAN 전용)

| Feature | 설명 |
|---------|------|
| `num_attention_heads` | Multi-Head Attention의 Head 수 |
| `embedding_dim` | Transformer 임베딩 차원 |
| `num_encoder_layers` | Transformer Encoder 반복 수 |
| `patch_size` | Vision Transformer의 패치 크기 |
| `latent_dim` | GAN의 노이즈 벡터 크기 |
| `is_gan` | GAN 여부 (1 = GAN, 0 = 아님) |

#### 4.5 생성할 파일

| 파일 | 역할 |
|------|------|
| `models/transformer_models.py` | Transformer 모델 정의 |
| `models/gan_models.py` | GAN Generator / Discriminator 정의 |
| `experiments/transformer_experiment.py` | Transformer 실행 시간 측정 실험 |
| `experiments/gan_experiment.py` | GAN 실행 시간 측정 실험 |
| `data/stage4/transformer_results.csv` | Transformer 실험 결과 |
| `data/stage4/gan_results.csv` | GAN 실험 결과 |

#### 4.6 4단계 완료 후 train_predictor.py 재실행

- ANN 74개 + CNN 22개 + Transformer N개 + GAN N개 → 총 100개+ 데이터로 예측 모델 재학습
- Transformer / GAN Feature를 Feature 목록에 추가
- 학습 시간 예측 R²가 개선될 것으로 기대

### 단계 5 — ONNX 기반 최종 시스템

- ONNX 형식 모델 파일 입력 → 모델 구조 자동 파싱 (onnx 라이브러리 활용)
- Feature 자동 추출 → 예측 모델 적용 → 시간 출력
- M1 Mac vs 학교 GPU 성능 비교 검증
- 최종 연구 보고서 작성

---

## 7. 파일 구조 (3단계 기준)

```
dnn/
├── data/
│   ├── raw/                           # 원본 데이터셋
│   │   ├── MNIST/
│   │   └── cifar-10-batches-py/
│   ├── stage1/                        # 1단계: ANN 실험 결과
│   │   ├── ann_mnist_results.csv          ← 52개 데이터 (기존 28 + 신규 24)
│   │   └── ann_results_visualization.png
│   ├── stage2/                        # 2단계: CNN 실험 결과
│   │   ├── cnn_cifar10_results.csv        ← 22개 데이터
│   │   └── cnn_results_visualization.png
│   ├── stage3/                        # 3단계: 예측 모델 결과
│   │   ├── v1/                            ← 구버전 (50개 데이터)
│   │   └── v2/                            ← 개선판 (74개 데이터, 최종)
│   └── stage4/                        # 4단계: Transformer/GAN 결과 (예정)
│       ├── transformer_results.csv
│       └── gan_results.csv
├── models/
│   ├── ann_models.py                  ← ANN (create_extended_variants 포함)
│   ├── cnn_models.py                  ← CNN (SimpleCNN, ResNet18, MobileNetV2)
│   ├── transformer_models.py          ← ★ 4단계 신규: SimpleViT
│   └── gan_models.py                  ← ★ 4단계 신규: SimpleGAN
├── utils/
│   └── timer.py                       ← 시간 측정 프레임워크 (Warmup, 반복 측정)
├── experiments/
│   ├── ann_mnist_experiment.py        ← ANN 실험 (증분 수집 지원)
│   ├── cnn_cifar10_experiment.py      ← CNN 실험
│   ├── transformer_experiment.py      ← ★ 4단계 신규: Transformer 실험
│   ├── gan_experiment.py              ← ★ 4단계 신규: GAN 실험
│   └── train_predictor.py             ← 예측 모델 (v2 개선판)
└── reports/
    ├── research_plan.md               ← 이 파일
    ├── stage1_2_report.md             ← 1-2단계 보고서
    └── stage3_report.md               ← 3단계 보고서
```

---

## 8. 참고 자료
- [PyTorch Documentation](https://pytorch.org/docs/)
- [snntorch Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- ResNet 논문 (He et al., 2015)
- MobileNet 논문 (Howard et al., 2017)
