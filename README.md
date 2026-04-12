# DNN 모델 실행 시간 예측 시스템 — 통합 연구 보고서

**작성자**: 김홍근 (학부연구생)  
**소속**: iSysLab — AI 프로세서 SW 프레임워크 연구 / 4세부 통합 시뮬레이터  
**하드웨어**: MacBook Air M1 (CPU + MPS Apple Silicon GPU)  
**기간**: 2025-2026 겨울방학  
**최종 작성일**: 2026-03-10  

> **`final` 브랜치 (2026-04-12)**: `hong-0311` + `dal-merge` + `ijunsoo` 통합.  
> 통합 벤치마크·`train_predictor.py`는 **111차원 피처** 스키마(`features/extractor.py`, 루트)를 사용합니다.  
> 아래 본문의 “92개” 등은 Hong 단계별 보고 시점 기준이며, 최종 파이프라인 설명은 [MERGE_NOTES.md](MERGE_NOTES.md)를 함께 참고하세요.

---

## 목차

1. [연구 배경 및 목표](#1-연구-배경-및-목표)
2. [연도별 연구 로드맵](#2-연도별-연구-로드맵)
3. [교수님 유의사항](#3-교수님-유의사항)
4. [전체 연구 흐름 요약](#4-전체-연구-흐름-요약)
5. [코드 병합 시 본인 코드 선택 이유](#5-코드-병합-시-본인-코드-선택-이유)
6. [현재 진행 상황](#6-현재-진행-상황)
7. [Stage 1 — ANN + MNIST 데이터 수집](#7-stage-1--ann--mnist-데이터-수집)
8. [Stage 2 — CNN + CIFAR-10 데이터 수집](#8-stage-2--cnn--cifar-10-데이터-수집)
9. [Stage 3 — 예측 모델 개발 (v1 → v2 → v3)](#9-stage-3--예측-모델-개발-v1--v2--v3)
10. [Stage 4 — Transformer / GAN 데이터 수집](#10-stage-4--transformer--gan-데이터-수집)
11. [Stage 5 — ONNX 기반 실행 시간 예측 시스템](#11-stage-5--onnx-기반-실행-시간-예측-시스템)
12. [Feature 선정 이유 (92개)](#12-feature-선정-이유-92개)
13. [모델 비교·분석 (XGBoost vs Random Forest)](#13-모델-비교분석-xgboost-vs-random-forest)
14. [데이터 양 선정 이유](#14-데이터-양-선정-이유)
15. [최종 성능 결과 (Stage 3 v3)](#15-최종-성능-결과-stage-3-v3)
16. [성과 및 트러블슈팅](#16-성과-및-트러블슈팅)
17. [코드 파일 구조](#17-코드-파일-구조)
18. [팀원과 공유할 코드 추천](#18-팀원과-공유할-코드-추천)
19. [개발 환경 및 실행 방법](#19-개발-환경-및-실행-방법)
20. [Windows 환경 실행 가이드](#20-windows-환경-실행-가이드)
21. [브랜치 병합 (ijunsoo / dal-merge)](#21-브랜치-병합-ijunsoo--dal-merge)
22. [핵심 개념 용어 정리](#22-핵심-개념-용어-정리)
23. [참고 자료](#23-참고-자료)

---

## 1. 연구 배경 및 목표

### 왜 이 연구를 하는가?

AI 모델(딥러닝 신경망)을 실제로 돌려보려면 시간이 얼마나 걸릴까요?  
수백 가지 모델 구조 중에서 가장 빠른 것을 고르고 싶을 때, 일일이 돌려보는 건 너무 오래 걸립니다.

**이 연구의 목표**: 모델의 구조 정보(레이어 수, 파라미터 수 등)만 보고, 실제로 돌리지 않고도 **실행 시간을 미리 예측**하는 시스템을 만드는 것입니다.

```
모델 구조 정보 (Feature, 92개)
         ↓
   예측 모델 (XGBoost / Random Forest)
         ↓
CPU / GPU 추론·학습 시간 예측
```

### 최종 시스템 목표

```
ONNX 모델 파일 입력 (.onnx)
         ↓
   모델 구조 자동 파싱
         ↓
   92개 Feature 추출
         ↓
   학습된 예측 모델 적용
         ↓
CPU / GPU 학습·추론 시간 출력
```

### 접근 방법

- **방법 1 (이번 연구)**: 기계학습 메타모델 — `모델 구조 (input)` → `실행 시간 (target)` 학습
- **방법 2 (향후 연구)**: 모델을 작은 연산 단위로 분해 → 각 단위 시간 측정 → 합산하여 전체 예측

---

## 2. 연도별 연구 로드맵


| 연도       | 내용                                                           |
| -------- | ------------------------------------------------------------ |
| **1차년도** | BindsNet / snntorch / Brian2 비교 분석, 메타모델 가능성 검증, 시뮬레이션 코드 변환 |
| **2차년도** | SpikeJelly 추가, SNN + DNN 통합 메타모델 정의, ONNX 기반 시간 예측 시스템 완성    |


```
1차년도: bindsnet, snntorch, brian2 비교 → SNN 메타모델 검증
2차년도: spikingjelly 추가 → DNN + SNN 통합 메타모델 정의
         ↓
         [이 보고서의 내용]
         DNN 모델 추론 시간 / 공간 요구량 예측
         모델 구조 분석 → 실행 시간 예측 모델 개발
         최종: ONNX 표준 포맷 모델 입력 → CPU/GPU 학습/추론 시간 예측
```

---

## 3. 교수님 유의사항

> 아래 사항을 반드시 지켜서 실험 및 코드 작성

1. **자신의 H/W에서 실험 수행**
  - 사용 환경: MacBook Air M1 (CPU + MPS)  
  - 하드웨어 사양, 모델 특징, 실행 시간 모두 기록
2. **Feature를 최대한 많이 선정**
  - 모델 구조 Feature + 하드웨어 Feature + 입력 데이터 Feature (12번 섹션 참고)
3. **ResNet / MobileNet 가급적 사용 자제**
  - 이미 수행한 데이터는 유지하되, 새로운 실험에서는 가급적 피할 것
4. **ANN은 레이어 수, 히든 뉴런 수를 다양하게**
  - 더 많은 조합으로 데이터 확장 권장
5. **소스코드에 주석 상세히 작성**
  - 교수님이 코드를 직접 검토하심  
  - 초보자도 이해할 수 있는 수준으로 한국어 주석 작성
6. **수집한 데이터로 입/출력 정리 후 기계학습 수행**
  - 학습(train) / 테스트(test) 분리하여 성능 평가  
  - 반복 측정: 시간 측정 시 10회 이상 반복 후 평균 사용 (warmup 포함)

---

## 4. 전체 연구 흐름 요약

```
Stage 1: ANN 모델 14가지 구조 × 2 디바이스 = 28개 데이터 수집 (MNIST)
    ↓ (데이터 부족 → 데이터 확장)
    ANN 추가 24개 → 총 52개 데이터
    ↓
Stage 2: CNN 모델 11가지 구조 × 2 디바이스 = 22개 데이터 수집 (CIFAR-10)
    ↓
Stage 3 v1: ANN 28 + CNN 22 = 50개 데이터로 예측 모델 개발 → R² ≈ 0.35 (성능 낮음)
    ↓ (개선)
Stage 3 v2: ANN 52 + CNN 22 = 74개 데이터, 로그 변환 + K-Fold CV → R²_log ≈ 0.88
    ↓
Stage 4: Transformer 24개 + GAN 16개 데이터 추가 수집
    ↓
Stage 3 v3: ANN 52 + CNN 22 + Transformer 24 + GAN 16 = 114개 데이터
            → XGBoost R²_log ≈ 0.93 (추론 시간), 0.88 (학습 시간)
    ↓
Stage 5: 학습된 예측 모델 저장 + ONNX 파일 파싱 → 새 모델 실행 시간 예측 완성
```

---

## 5. 코드 병합 시 본인 코드 선택 이유

팀원(ijunsoo, dal-merge) 코드와 병합할 때 **본인 코드를 기준**으로 한 이유는 다음과 같습니다.

1. **Stage 1~5 전체 파이프라인**과 **ONNX 예측 시스템**이 이미 구현·검증되어 있었음
2. **experiments/, models/, utils/** 구조와 문서화가 정리되어 있었음
3. **교수님 유의사항**(Feature 다수 선정, 반복 측정 등)을 충족하고 있었음
4. **ijunsoo/dal-merge** 코드는 Op 프로파일러·하드웨어 감지 등 **보조 기능**에 해당 → 기존 구조에 **포팅**하는 방식이 적절했음

따라서 Git merge가 아닌, 필요한 모듈만 복사·수정하여 가져오는 포팅 방식으로 통합했습니다.

---

## 6. 현재 진행 상황

- **Windows 환경**에서 Stage 1~5 전체 파이프라인 실행·검증 완료
- **92개 Feature** 기반 예측 모델 학습·ONNX 예측 동작 확인
- **개발 환경**: MacBook Air M1 (CPU + MPS)
- **검증 환경**: Windows PC (Intel Core i7-11800H, CUDA 지원)
- **증분 데이터 수집**: CNN·Transformer·GAN 실험에 `(config, device)` 기준 증분 로직 적용 — Mac(MPS) 데이터에 Windows(CUDA) 데이터를 추가 가능
- **시각화**: MPS와 CUDA를 동시에 표시 (CPU vs MPS vs CUDA 막대 그래프)

---

## 7. Stage 1 — ANN + MNIST 데이터 수집

### ANN(인공 신경망)이란?

가장 기본적인 딥러닝 모델입니다. 뇌의 뉴런처럼 여러 층(레이어)으로 연결되어 있습니다.

```
입력 (MNIST 이미지 784픽셀)
    ↓  [Layer 1: 256개 뉴런]
    ↓  [Layer 2: 256개 뉴런]
    ↓  [Layer 3: 256개 뉴런]
출력 (숫자 0~9, 10가지 분류)
```

### 실험 설계

MNIST는 손으로 쓴 숫자(0~9) 이미지 데이터셋입니다.


| 변수             | 값                    |
| -------------- | -------------------- |
| 레이어 수          | 2, 3, 4층             |
| 각 층의 너비(Width) | 64, 128, 256, 512 뉴런 |
| 실험한 디바이스       | CPU, MPS(M1 GPU)     |
| 시간 측정 방법       | 10회 이상 반복 측정 후 평균    |


**확장 실험** (Stage 3 개선을 위해 추가):


| 추가 변수  | 값                                       |
| ------ | --------------------------------------- |
| 극단 너비  | 32, 1024 뉴런                             |
| 더 깊은 층 | 5층                                      |
| 특수 구조  | 피라미드형 (256→128→64), 역피라미드형 (64→128→256) |


### 수집한 데이터

- **초기**: 14개 모델 × 2 디바이스 = **28개 데이터 포인트**
- **확장 후**: 26개 모델 × 2 디바이스 = **52개 데이터 포인트**

### 주요 발견


| 항목       | CPU           | MPS (M1 GPU)  |
| -------- | ------------- | ------------- |
| 추론 시간 평균 | 0.08 ms       | 0.60 ms       |
| 학습 시간 평균 | 3.1 sec/epoch | 5.4 sec/epoch |


> **놀라운 발견**: 작은 ANN 모델에서는 GPU(MPS)가 오히려 더 **느립니다**.  
> 이유: 데이터를 GPU로 옮기고(메모리 전송) 다시 가져오는 오버헤드가  
> 실제 계산 시간보다 크기 때문입니다.

### 관련 파일

- `models/ann_models.py` — ANN 모델 구조 정의 (`SimpleANN`, `create_extended_variants`)
- `experiments/ann_mnist_experiment.py` — 시간 측정 및 데이터 수집 (증분 방식 지원)
- `utils/timer.py` — 반복 측정 유틸리티

---

## 8. Stage 2 — CNN + CIFAR-10 데이터 수집

### CNN(합성곱 신경망)이란?

이미지를 분류하는 데 특화된 모델입니다. ANN과 달리 이미지의 **지역적 패턴**(엣지, 텍스처 등)을 학습합니다.

```
입력 이미지 (CIFAR-10: 32×32 컬러 이미지)
    ↓  [Conv Layer: 이미지에서 패턴 추출]
    ↓  [MaxPool: 크기 줄이기]
    ↓  [Conv Layer: 더 복잡한 패턴 추출]
    ↓  [Fully Connected: 최종 분류]
출력 (비행기/자동차/새 등 10가지 분류)
```

### 실험한 모델 3종류


| 모델            | 설명                                | 파라미터 수    |
| ------------- | --------------------------------- | --------- |
| `SimpleCNN`   | 직접 만든 단순 CNN (L2~~L4, 채널 16~~128) | 0.1M ~ 5M |
| `ResNet18`    | 잔차 연결(Residual)이 있는 깊은 CNN        | 11M       |
| `MobileNetV2` | 모바일 기기용 경량 CNN                    | 2.2M      |


### 수집한 데이터

11개 모델 × 2 디바이스 = **22개 데이터 포인트**

### 주요 발견


| 모델               | CPU 추론 시간 | MPS 추론 시간 | 정확도    |
| ---------------- | --------- | --------- | ------ |
| SimpleCNN L2 C16 | 6.1 ms    | 2.0 ms    | 39~49% |
| SimpleCNN L4 C64 | 95.8 ms   | 9.9 ms    | 58~61% |
| ResNet18         | 799 ms    | **80 ms** | 69%    |
| MobileNetV2      | 508 ms    | 16 ms     | 65%    |


> **핵심 발견**: ResNet18에서 CPU vs MPS가 **10배 차이**!  
> MobileNetV2는 파라미터가 적지만 CPU에서 더 느림 (Depthwise Conv의 CPU 비효율)

### 관련 파일

- `models/cnn_models.py` — CNN 모델 정의 (SimpleCNN, ResNet18, MobileNetV2)
- `experiments/cnn_cifar10_experiment.py` — 시간 측정 및 데이터 수집 (증분 방식: `(model_name, device)` 기준, `data/stage2/cnn_cifar10_results.csv`에 append)

---

## 9. Stage 3 — 예측 모델 개발 (v1 → v2 → v3)

### 핵심 아이디어 (비유로 이해하기)

> "집 면적, 방 개수, 위치를 알면 집값을 예측하듯이,  
>  레이어 수, 파라미터 수, 채널 수를 알면 실행 시간을 예측한다."

### 사용한 예측 알고리즘 2가지

#### Random Forest (랜덤 포레스트)

- **비유**: 여러 명의 전문가에게 의견을 물어보고 **다수결**로 결정
- 수백 개의 결정 트리를 만들고 평균값을 예측값으로 사용
- 장점: 과적합에 강함, 해석이 쉬움

#### XGBoost (익스트림 그래디언트 부스팅)

- **비유**: 이전 전문가의 틀린 부분을 다음 전문가가 보완하며 연속으로 개선
- 결정 트리를 순차적으로 쌓아가며 오차를 줄임
- 장점: 정확도가 높음, 다양한 데이터에 강함

### 성능 측정 지표

#### R² (결정계수) — "예측이 얼마나 정확한가"

```
R² = 1.0  → 완벽한 예측 (100% 정확)
R² = 0.9  → 90% 설명 가능 (매우 좋음)
R² = 0.7  → 70% 설명 가능 (적당함)
R² = 0.0  → 평균값을 그냥 쓰는 것과 같음 (의미 없음)
R² < 0    → 평균값보다 나쁨 (최악)
```

#### R²_log (로그 스케일 R²)

실행 시간 범위가 0.001ms ~ 1000sec로 너무 넓어서 로그 변환 후 계산.  
모델 실제 품질 파악에 더 유용한 지표.

### v1 → v2 → v3 개선 과정

#### v1: 첫 번째 시도 (ANN 28개 + CNN 22개 = 50개)


| 모델            | 추론 R² | 학습 R² |
| ------------- | ----- | ----- |
| Random Forest | 0.35  | 0.20  |
| XGBoost       | 0.40  | 0.25  |


**문제점**: 데이터 50개로 부족, 실행 시간 범위가 너무 넓음, 단순 train/test 분리

#### v2: 개선 (ANN 52개 + CNN 22개 = 74개)

**적용한 개선 방법 3가지**:

**① 로그 변환 (Log Transformation)**

```python
# 전: [0.03ms, 0.08ms, 800ms, 1183000ms] → 범위가 너무 커서 학습 어려움
# 후: log(x+1)으로 변환 → 균일한 범위로 정규화
log_target = np.log1p(target)
# 예측 후 원래 값으로 복원:
prediction = np.expm1(log_prediction)
```

**② K-Fold Cross-Validation (교차 검증)**

```
데이터를 5개 구간으로 나눔:
  구간 1 테스트, 나머지 4개 학습 → R² 측정
  구간 2 테스트, 나머지 4개 학습 → R² 측정
  ...5번 반복 → 평균 R² 계산
```

**③ GridSearchCV (하이퍼파라미터 자동 탐색)**

```
탐색한 파라미터 조합:
  n_estimators: [100, 200, 500]     ← 트리 개수
  max_depth: [5, 10, None]          ← 트리 깊이
  learning_rate: [0.01, 0.1, 0.2]  ← 학습률 (XGBoost)
```


| 모델            | 추론 R²_log | 학습 R²_log |
| ------------- | --------- | --------- |
| Random Forest | 0.88      | 0.87      |
| XGBoost       | 0.93      | 0.88      |


#### v3: 최종 버전 (4가지 모델 114개 데이터)

Stage 4에서 수집한 Transformer, GAN 데이터를 추가하여 모델 다양성 확보.


| 모델 종류                  | 데이터 수    |
| ---------------------- | -------- |
| ANN (MNIST)            | 52개      |
| CNN (CIFAR-10)         | 22개      |
| Transformer (CIFAR-10) | 24개      |
| GAN (MNIST)            | 16개      |
| **합계**                 | **114개** |


### 관련 파일

- `experiments/train_predictor.py` — 전체 ML 파이프라인 (902줄)

---

## 10. Stage 4 — Transformer / GAN 데이터 수집

### Transformer란?

원래 자연어 처리를 위해 설계된 구조이지만, 이미지에도 적용한 것이 **Vision Transformer(ViT)**입니다.

```
이미지를 4×4 픽셀 패치로 나눔 → 각 패치를 벡터로 변환
    ↓
Self-Attention: 각 패치가 다른 패치들과 관계를 학습
    ↓
여러 Encoder Block 반복
    ↓
이미지 분류
```

**실험한 구성**:

- embed_dim: 64, 128, 256 (패치 벡터 차원)
- num_heads: 4, 8 (어텐션 헤드 수)
- num_layers: 2, 4, 6 (Encoder Block 수)
- patch_size: 4×4, 8×8
- 총 12개 구성 × 2 디바이스 = **24개 데이터 포인트**

### GAN이란?

**두 개의 신경망이 서로 경쟁하며** 진짜 같은 가짜 이미지를 생성하는 구조입니다.

```
Generator(생성자): 노이즈 → 가짜 이미지 생성
                           ↕ 경쟁
Discriminator(판별자): 진짜/가짜 이미지를 구분
```

**실험한 구성**:

- latent_dim: 64, 100 (노이즈 벡터 크기) -> ***dim = dimension(차원) = 벡터의 길이, 즉 숫자가 몇 개인지***
- Generator hidden layers: [128,256], [128,256,512], [256,512] 등
- 총 8개 구성 × 2 디바이스 = **16개 데이터 포인트**

### Transformer vs GAN 시간 특성


| 모델                       | CPU 추론 시간 | MPS 추론 시간 | 학습 시간/epoch    |
| ------------------------ | --------- | --------- | -------------- |
| Transformer (dim64, L2)  | 28.1 ms   | 8.2 ms    | 59 sec (CPU)   |
| Transformer (dim256, L6) | ~90 ms    | ~20 ms    | ~200 sec (CPU) |
| GAN (z64, h128,256)      | 0.31 ms   | 1.04 ms   | 0.004 sec      |


### 관련 파일

- `models/transformer_models.py` — SimpleViT 구현 (366줄)
- `models/gan_models.py` — Generator, Discriminator, SimpleGAN (326줄)
- `experiments/transformer_experiment.py` — Transformer 시간 측정 (298줄, 증분 방식: `(config_str, device)` 기준)
- `experiments/gan_experiment.py` — GAN 시간 측정 (417줄, 증분 방식: `(config_str, device)` 기준)

---

## 11. Stage 5 — ONNX 기반 실행 시간 예측 시스템

### ONNX란?

**Open Neural Network Exchange**의 약자.  
PyTorch, TensorFlow 등 다양한 딥러닝 프레임워크로 만든 모델을  
하나의 표준 파일 형식(`.onnx`)으로 저장하는 것입니다.

### Stage 5의 전체 흐름

```
① train_predictor.py 실행
   → 114개 데이터로 XGBoost, RandomForest 예측 모델 학습
   → models/trained/*.pkl 파일로 저장

② export_to_onnx.py 실행
   → PyTorch 모델들을 ONNX 형식으로 변환
   → data/stage5/onnx_samples/*.onnx 파일 생성 (10개)

③ predict_from_onnx.py 실행
   → .onnx 파일 파싱 → 92개 Feature 추출
   → 저장된 예측 모델 로드 → 실행 시간 예측
   → 결과 출력
```

### 생성된 ONNX 파일 10개


| 파일명                             | 모델 종류                    | 크기       |
| ------------------------------- | ------------------------ | -------- |
| `ann_l1_w128.onnx`              | ANN 1층, 너비 128           | 0.39 MB  |
| `ann_l2_w256.onnx`              | ANN 2층, 너비 256           | 0.90 MB  |
| `ann_l3_w512.onnx`              | ANN 3층, 너비 512           | 2.17 MB  |
| `cnn_l2_c32.onnx`               | CNN 2층, 채널 32            | 8.10 MB  |
| `cnn_l3_c64.onnx`               | CNN 3층, 채널 64            | 9.44 MB  |
| `cnn_l4_c128.onnx`              | CNN 4층, 채널 128           | 31.67 MB |
| `vit_dim64_heads4_l3.onnx`      | ViT dim=64, heads=4, 3층  | 0.63 MB  |
| `vit_dim128_heads4_l4.onnx`     | ViT dim=128, heads=4, 4층 | 3.12 MB  |
| `gan_gen_z100_h256_512.onnx`    | GAN Generator z=100      | 2.14 MB  |
| `gan_gen_z64_h128_256_512.onnx` | GAN Generator z=64       | 2.20 MB  |


### 예측 결과 예시

```
============================================================
  [예측 결과]
  파일: ann_l2_w256.onnx / 디바이스: cpu / 배치 크기: 64
------------------------------------------------------------
  추론 시간 예측 (XGBoost)     :       0.08 ms/batch
  추론 시간 예측 (RandomForest):       0.09 ms/batch
  학습 시간 예측 (XGBoost)     :       3.12 sec/epoch
  학습 시간 예측 (RandomForest):       2.98 sec/epoch
------------------------------------------------------------
  추론 시간 앙상블 평균         :       0.09 ms/batch
  학습 시간 앙상블 평균         :       3.05 sec/epoch
============================================================
```

### ONNX 파싱의 한계 및 해결

ONNX 파일에는 "레이어의 임베딩 차원" 같은 고수준 메타데이터가 없습니다.

```bash
# ANN, CNN → 자동 판별 가능
python predict_from_onnx.py ann_l2_w256.onnx --device cpu

# Transformer → embed_dim, num_heads, patch_size 직접 지정
python predict_from_onnx.py vit_dim128_heads4_l4.onnx \
    --embed-dim 128 --num-heads 4 --patch-size 4

# GAN → latent_dim 직접 지정
python predict_from_onnx.py gan_gen_z100_h256_512.onnx --latent-dim 100
```

### 관련 파일

- `experiments/export_to_onnx.py` — PyTorch → ONNX 변환 (275줄)
- `experiments/predict_from_onnx.py` — ONNX → 시간 예측 CLI (330줄)
- `utils/onnx_feature_extractor.py` — ONNX 파싱 + Feature 추출 (286줄)

---

## 15. 최종 성능 결과 (Stage 3 v3)

### 예측 모델 성능 요약


| 예측 대상      | 모델            | RMSE  | MAE  | R²    | **R²_log** | CV R²_log |
| ---------- | ------------- | ----- | ---- | ----- | ---------- | --------- |
| 추론 시간(ms)  | Random Forest | 82.9  | 16.6 | 0.15  | **0.88**   | 0.876     |
| 추론 시간(ms)  | **XGBoost**   | 72.2  | 13.2 | 0.35  | **0.93**   | 0.937     |
| 학습 시간(sec) | Random Forest | 602.4 | 86.5 | -0.01 | **0.86**   | 0.872     |
| 학습 시간(sec) | **XGBoost**   | 597.1 | 83.6 | 0.01  | **0.88**   | 0.891     |


### R²가 낮고 R²_log가 높은 이유

```
GAN 학습 시간 = 0.004 초
ResNet18 학습 시간 = 1183 초
비율 차이: 1183 / 0.004 = 295,750배 차이!

→ 원래 스케일에서 이 극단값 하나가 전체 R²를 망침
→ log 스케일에서는 범위가 균일해지므로 R²_log가 진짜 성능을 보여줌
```

### 최적 하이퍼파라미터 (GridSearchCV 탐색 결과)

**XGBoost (추론 시간)**:

```
learning_rate: 0.2 / max_depth: 3 / n_estimators: 200 / subsample: 0.8
```

**XGBoost (학습 시간)**:

```
learning_rate: 0.1 / max_depth: 4 / n_estimators: 50 / subsample: 0.8
```

**Random Forest (추론/학습 공통)**:

```
max_depth: 10 / min_samples_leaf: 1 / min_samples_split: 2 / n_estimators: 200
```

---

## 12. Feature 선정 이유 (92개)

> 예측 모델의 입력(Input)으로 사용하는 **92개 Feature**. 교수님 유의사항("Feature를 최대한 많이 선정") 및 통합 스키마 문서를 반영했습니다.

### 선정 근거

- **공통 모델 구조 33개**: `total_params`, `num_conv_layers`, `has_pooling`, `model_size_mb` 등 — 모델 복잡도·구조 특성 반영
- **모델 전용 18개**: ANN(`ann_max_hidden` 등), CNN(`cnn_num_filters`, `cnn_has_residual` 등), Transformer(`embed_dim`, `num_heads`, `patch_size` 등), GAN(`latent_dim`, `generator_params` 등)
- **입력 데이터 8개**: `batch_size`, `input_height`, `num_classes`, `dataset_encoded` 등 — 실행 조건 반영
- **하드웨어 33개**: `cpu_cores_physical`, `gpu_memory_gb`, `tflops_fp32`, `is_unified_memory` 등 — 실행 환경 반영

모델 타입별로 해당하지 않는 Feature는 0으로 채우며, merge 후 `fillna(0)` 처리합니다.

### Feature 구성 요약

| 그룹 | 개수 | 대표 예시 |
|------|------|-----------|
| 공통 모델 구조 | 33개 | total_params, num_conv_layers, has_pooling, flops |
| 모델 전용 | 18개 | embed_dim, num_heads, latent_dim, cnn_has_residual |
| 입력 데이터 | 8개 | batch_size, input_height, num_classes |
| 하드웨어 | 33개 | cpu_cores_physical, gpu_memory_gb, tflops_fp32 |
| **합계** | **92개** | |

---

## 13. 모델 비교·분석 (XGBoost vs Random Forest)

### 두 알고리즘 특성

| 항목 | Random Forest | XGBoost |
|------|---------------|---------|
| **방식** | 여러 결정 트리 병렬 학습 → 다수결 | 결정 트리 순차 보정 → 오차 점진 감소 |
| **장점** | 과적합에 강함, 해석 용이 | 정확도 우수, 다양한 데이터에 강함 |
| **추론 R²_log** | 0.88 | **0.93** |
| **학습 R²_log** | 0.86 | **0.88** |

### 선택 이유

GridSearchCV로 하이퍼파라미터 탐색 후 **XGBoost**가 추론·학습 모두 우수한 성능을 보였습니다.  
최종 예측 시에는 두 모델의 예측값 **앙상블 평균**을 사용하여 안정성을 높였습니다 (`predict_from_onnx.py`).

---

## 14. 데이터 양 선정 이유

### 과적합 방지 기준

- Feature 92개 사용 시, 일반적으로 **데이터 수 ≥ 피처 수 × 3~5** (약 276~460행)가 권장됩니다.
- 현재 **114개** 데이터: ANN 52 + CNN 22 + Transformer 24 + GAN 16 (각 디바이스당 ×2)

### 선정 이유

1. **4종 모델 다양성 확보**: ANN, CNN, Transformer, GAN을 모두 포함해 다양한 구조를 학습
2. **실험 시간·리소스 고려**: 전체 데이터 수집에 수 시간 소요, 확장 시 200~350행 목표 가능
3. **K-Fold CV 활용**: 소량 데이터에서도 5-Fold 교차 검증으로 안정적 성능 평가
4. **로그 변환**: 실행 시간 범위(0.004초~1183초)가 넓어 log 변환으로 학습 안정성 확보

---

## 15. 최종 성능 결과 (Stage 3 v3)

### 예측 모델 성능 요약


| 예측 대상      | 모델            | RMSE  | MAE  | R²    | **R²_log** | CV R²_log |
| ---------- | ------------- | ----- | ---- | ----- | ---------- | --------- |
| 추론 시간(ms)  | Random Forest | 82.9  | 16.6 | 0.15  | **0.88**   | 0.876     |
| 추론 시간(ms)  | **XGBoost**   | 72.2  | 13.2 | 0.35  | **0.93**   | 0.937     |
| 학습 시간(sec) | Random Forest | 602.4 | 86.5 | -0.01 | **0.86**   | 0.872     |
| 학습 시간(sec) | **XGBoost**   | 597.1 | 83.6 | 0.01  | **0.88**   | 0.891     |


### R²가 낮고 R²_log가 높은 이유

```
GAN 학습 시간 = 0.004 초
ResNet18 학습 시간 = 1183 초
비율 차이: 1183 / 0.004 = 295,750배 차이!

→ 원래 스케일에서 이 극단값 하나가 전체 R²를 망침
→ log 스케일에서는 범위가 균일해지므로 R²_log가 진짜 성능을 보여줌
```

### 최적 하이퍼파라미터 (GridSearchCV 탐색 결과)

**XGBoost (추론 시간)**: `learning_rate: 0.2 / max_depth: 3 / n_estimators: 200 / subsample: 0.8`

**XGBoost (학습 시간)**: `learning_rate: 0.1 / max_depth: 4 / n_estimators: 50 / subsample: 0.8`

**Random Forest (추론/학습 공통)**: `max_depth: 10 / min_samples_leaf: 1 / min_samples_split: 2 / n_estimators: 200`

---

## 16. 성과 및 트러블슈팅

### 성과 요약

- **예측 성능**: XGBoost R²_log 0.93 (추론), 0.88 (학습)
- **92개 Feature** 기반 ONNX 예측 파이프라인 완성
- **Windows 환경** 검증 완료 (CUDA 지원)
- Stage 1~5 전체 파이프라인 실행·검증 완료
- **증분 데이터 수집**: Mac(MPS)에서 수집한 CSV에 Windows(CUDA) 데이터를 추가 가능 (CNN·Transformer·GAN)
- **시각화**: `analyze_cnn_results.py` 등에서 MPS·CUDA 동시 표시 (한글 폰트, U+2212 경고 억제)

### 트러블슈팅 이력


| #   | 발생 문제                                                                                          | 원인                             | 해결법                                                                         |
| --- | ---------------------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| 1   | `ModuleNotFoundError: No module named 'pandas'`                                                | 가상환경에 패키지 미설치                  | `pip install scikit-learn xgboost matplotlib pandas numpy`                  |
| 2   | `XGBoostError: libxgboost.dylib could not be loaded`                                           | macOS에 libomp 미설치              | `brew install libomp`                                                       |
| 3   | `ImportError: cannot import name 'shape_base' from numpy._core`                                | numpy 설치 손상                    | `pip install --upgrade --force-reinstall numpy`                             |
| 4   | `ModuleNotFoundError: No module named 'torch.masked.maskedtensor.core'`                        | PyTorch 설치 손상                  | `pip uninstall torch torchvision -y && pip install torch torchvision`       |
| 5   | `FileNotFoundError: sklearn/externals/__init__.py`                                             | scikit-learn 설치 손상             | `pip install --upgrade --force-reinstall scikit-learn`                      |
| 6   | `import torch` 후 1~3분 멈춤 (M1 Mac)                                                              | 첫 실행 시 JIT 컴파일                 | 정상 현상 — 기다리면 됨                                                              |
| 7   | `git commit` 실패: `error: unknown option 'trailer'`                                             | git 버전(2.30)이 너무 오래됨           | `brew install git` (2.53 설치)                                                |
| 8   | `git push` 실패: `pack-objects died of signal 10 (SIGBUS)`                                       | 대용량 이미지/바이너리 파일 포함             | `.gitignore`에 PNG/raw data 추가, `pack.windowMemory=100m` 설정                  |
| 9   | `PermissionError: Operation not permitted` (joblib)                                            | 샌드박스 환경 병렬 처리 제한               | PyCharm에서는 발생 안 함 — 로컬 환경에서 실행                                              |
| 10  | Stage 3이 몇 초 만에 끝남 (Stage 1~2는 5시간)                                                            | Stage 3은 ML 모델 학습 (신경망 학습이 아님) | 정상 현상 — 개념 차이 이해                                                            |
| 11  | `TypeError: Generator.__init__() got an unexpected keyword argument 'output_dim'`              | ONNX 변환 시 잘못된 인자 전달            | `img_size=28, img_channels=1`로 수정                                           |
| 12  | `No module named 'onnxscript'` + dynamo 관련 경고                                                  | PyTorch 2.x 새 exporter 의존성     | `pip install onnxscript` + `torch.onnx.export(..., dynamo=False)`           |
| 13  | `UserWarning: X has feature names, but RandomForestRegressor was fitted without feature names` | 학습은 numpy, 예측은 DataFrame 사용    | `warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')` |
| 14  | `Font 'default' does not have a glyph for '\u2212' [U+2212]` (시각화 시) | 로그 스케일 틱 라벨에서 유니코드 마이너스 사용, 한글 폰트 미지원 | `visualize_all_stages.py`: savefig 시 stderr 리다이렉트, `mathtext.fontset='dejavusans'` |


### 새로운 문제 발생 시

1. 위 표에서 유사 문제 먼저 검색
2. 해결 후 위 표에 새 행으로 추가
3. 문제 원인, 해결법, 재현 가능한 명령어를 구체적으로 기록

---

## 17. 코드 파일 구조

```
dnn/
├── experiments/                        ← 실험 스크립트
│   ├── ann_mnist_experiment.py         Stage 1: ANN 시간 측정 (341줄)
│   ├── cnn_cifar10_experiment.py       Stage 2: CNN 시간 측정 (341줄)
│   ├── transformer_experiment.py       Stage 4: Transformer 시간 측정 (298줄)
│   ├── gan_experiment.py               Stage 4: GAN 시간 측정 (417줄)
│   ├── train_predictor.py              Stage 3: 예측 모델 학습 (902줄)
│   ├── export_to_onnx.py               Stage 5: PyTorch → ONNX 변환 (275줄)
│   ├── predict_from_onnx.py            Stage 5: ONNX → 시간 예측 CLI (330줄)
│   ├── analyze_results.py              Stage 1 ANN 결과 분석/시각화
│   ├── analyze_cnn_results.py          Stage 2 CNN 결과 분석/시각화 (CPU/MPS/CUDA 지원)
│   ├── analyze_transformer_results.py  Stage 4 Transformer 결과 분석/시각화
│   ├── analyze_gan_results.py          Stage 4 GAN 결과 분석/시각화
│   └── visualize_all_stages.py        전체 Stage 합본 시각화 (한글 폰트, U+2212 경고 억제)
│
├── models/                             ← 신경망 모델 정의
│   ├── ann_models.py                   SimpleANN (134줄)
│   ├── cnn_models.py                   SimpleCNN, ResNet18, MobileNetV2 (309줄)
│   ├── transformer_models.py           SimpleViT (366줄)
│   ├── gan_models.py                   Generator, Discriminator, SimpleGAN (326줄)
│   └── trained/                        ← 학습된 예측 모델 (joblib)
│       ├── xgb_inference.pkl           XGBoost 추론 시간 예측 모델
│       ├── xgb_training.pkl            XGBoost 학습 시간 예측 모델
│       ├── rf_inference.pkl            RandomForest 추론 시간 예측 모델
│       ├── rf_training.pkl             RandomForest 학습 시간 예측 모델
│       └── feature_columns.pkl         92개 Feature 컬럼 순서 정보
│
├── utils/                              ← 공통 유틸리티
│   ├── timer.py                        시간 측정 클래스 (206줄)
│   └── onnx_feature_extractor.py       ONNX 파싱 → Feature 추출 (286줄)
│
├── data/                               ← 수집한 데이터
│   ├── stage1/
│   │   ├── ann_mnist_results.csv       ANN 52개 데이터
│   │   └── ann_visualization.png       Stage 1 ANN 시각화
│   ├── stage2/
│   │   ├── cnn_cifar10_results.csv     CNN 22개 데이터
│   │   └── cnn_visualization.png       Stage 2 CNN 시각화
│   ├── stage3/
│   │   ├── v1/ (50개 첫 시도 결과)
│   │   ├── v2/ (74개 개선 결과)
│   │   └── v3/ (114개 최종 결과 + 시각화 PNG 8개)
│   ├── stage4/
│   │   ├── transformer_results.csv     Transformer 24개 데이터
│   │   ├── gan_results.csv             GAN 16개 데이터
│   │   ├── transformer_visualization.png  Stage 4 Transformer 시각화
│   │   └── gan_visualization.png       Stage 4 GAN 시각화
│   └── stage5/onnx_samples/            ONNX 샘플 파일 10개
│
└── reports/
    ├── visualizations/all_stages_summary.png  전체 합본 시각화
    └── final_report.md                 이 보고서 (통합 결과)
```

---

## 18. 팀원과 공유할 코드 추천

아래 파일들은 이 프로젝트에서 특히 재사용 가치가 높거나, 딥러닝/ML을 공부하는 팀원에게 참고가 될 만한 코드입니다.

### 즉시 재사용 가능한 도구

#### `utils/timer.py` — 시간 측정 프레임워크

```
[추천 이유]
- Warmup 후 반복 측정 → 평균/표준편차 계산을 자동화
- CPU와 MPS(Apple Silicon GPU) 모두 지원
- MPS 동기화(torch.mps.synchronize()) 처리 내장
- 어떤 실험에도 그대로 붙여서 사용 가능
```

사용법:

```python
from utils.timer import TimeEstimator
timer = TimeEstimator(device='mps', warmup=5, repeat=20)
mean_ms, std_ms = timer.measure_inference(model, input_tensor)
```

#### `experiments/predict_from_onnx.py` — ONNX 기반 실행 시간 예측 CLI

```
[추천 이유]
- .onnx 파일 하나를 주면 즉시 실행 시간 예측 결과 출력
- 완성된 도구이므로 데모나 발표 시 바로 시연 가능
- CLI 인자 파싱 패턴 (argparse) 참고 가능
```

#### `experiments/train_predictor.py` — 전체 ML 파이프라인

```
[추천 이유]
- 로그 변환 + K-Fold CV + GridSearchCV + 모델 저장까지 전체 흐름 포함
- 회귀 예측 문제라면 어디서든 이 구조를 그대로 적용 가능
- 데이터 병합, Feature 엔지니어링, 시각화까지 one-stop으로 담겨 있음
```

### 모델 구현 참고 코드

#### `models/transformer_models.py` — Vision Transformer 직접 구현

```
[추천 이유]
- PatchEmbedding, MultiHeadAttention, TransformerEncoderBlock을 처음부터 구현
- PyTorch로 Transformer를 어떻게 만드는지 단계별로 이해하기 좋음
- 외부 라이브러리 없이 순수 PyTorch만 사용
```

핵심 구성:

```
PatchEmbedding → MultiHeadAttention → TransformerEncoderBlock → SimpleViT
```

#### `models/gan_models.py` — GAN 직접 구현

```
[추천 이유]
- Generator / Discriminator의 최소 구현 예시
- GAN 학습 루프(Adversarial Training) 패턴 이해에 좋음
- MNIST 생성 기준으로 구현되어 있어 실행 결과 확인이 쉬움
```

### ONNX 연구자 참고 코드

#### `utils/onnx_feature_extractor.py` — ONNX 파싱 + Feature 추출

```
[추천 이유]
- ONNX 그래프 노드(Conv, Gemm, Relu 등)를 파싱하는 방법 예시
- 모델 타입 자동 판별 (Conv 수 → CNN, Gemm 위주 → ANN) 로직 참고
- ONNX 파일에서 파라미터 수를 계산하는 방법 포함
```

#### `experiments/export_to_onnx.py` — PyTorch → ONNX 변환

```
[추천 이유]
- torch.onnx.export 사용법 (dynamo=False로 레거시 exporter 지정)
- PyTorch 2.x에서 안정적인 ONNX 변환 방법 (dynamo exporter 경고 우회)
- ANN / CNN / Transformer / GAN 각각의 변환 패턴을 모두 포함
```

---

## 19. 개발 환경 및 실행 방법

### 환경 설정

```bash
# 가상환경 생성 및 활성화 (최초 1회)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 필수 패키지 설치
pip install torch torchvision
pip install scikit-learn xgboost matplotlib pandas numpy
pip install joblib onnx onnxscript

# macOS에서 XGBoost 실행 시 추가 필요
brew install libomp

# Windows 이전 시 참고 (Git clone 권장)
# reports/WINDOWS_SETUP.md — 이전 방법, 환경 설정, CUDA 지원
```

### 실행 순서 (Stage 1 ~ 5 전체)

```bash
# Stage 1: ANN 데이터 수집 (약 3~5시간)
python experiments/ann_mnist_experiment.py

# Stage 2: CNN 데이터 수집 (약 5~8시간)
python experiments/cnn_cifar10_experiment.py

# Stage 4: Transformer 데이터 수집 (약 3~4시간)
python experiments/transformer_experiment.py

# Stage 4: GAN 데이터 수집 (약 1~2시간)
python experiments/gan_experiment.py

# Stage 3: 예측 모델 학습 (수 분, 매우 빠름)
python experiments/train_predictor.py

# Stage 5: ONNX 변환 (수 분)
python experiments/export_to_onnx.py

# Stage 5: 예측 실행 (데모 모드 - 10개 ONNX 파일 모두 예측)
python experiments/predict_from_onnx.py

# Stage 5: 특정 파일 예측
python experiments/predict_from_onnx.py \
    data/stage5/onnx_samples/vit_dim128_heads4_l4.onnx \
    --device cpu --embed-dim 128 --num-heads 4 --patch-size 4
```

> **증분 방식**: `ann_mnist_experiment.py`, `cnn_cifar10_experiment.py`, `transformer_experiment.py`, `gan_experiment.py`는 이미 측정된 `(config, device)` 조합을 건너뜁니다. Mac(MPS)에서 수집한 CSV에 Windows(CUDA) 데이터를 추가할 수 있습니다.

### 시각화 실행 (Stage별 + 합본)

```bash
# Stage 1 ANN 시각화
python experiments/analyze_results.py

# Stage 2 CNN 시각화
python experiments/analyze_cnn_results.py

# Stage 4 Transformer 시각화
python experiments/analyze_transformer_results.py

# Stage 4 GAN 시각화
python experiments/analyze_gan_results.py

# 전체 합본 시각화 (Stage 1~4 요약 + Stage 3 성능)
python experiments/visualize_all_stages.py

# 또는 한 번에 모두 실행
python experiments/visualize_all_stages.py --all
```

- **Stage 3** 시각화(Feature Importance, Pred vs Actual 8개)는 `train_predictor.py` 실행 시 자동 생성됩니다.
- **합본** 결과는 `reports/visualizations/all_stages_summary.png`에 저장됩니다.

---

## 20. Windows 환경 실행 가이드

Mac에서 개발한 DNN 실행 시간 예측 시스템을 **Windows PC**에서 사용하는 방법입니다.

### 20.1 전체 그림 요약

- **연구 내용**: Stage 1~2 ANN/CNN, Stage 3 **92개 Feature** 예측 모델, Stage 4 Transformer/GAN, Stage 5 ONNX 파이프라인
- **Windows에서 할 일**: Python/가상환경/패키지 설치, (선택) 데이터 수집 재실행, 예측 모델 학습, ONNX 예측 데모

### 20.2 준비물

**반드시 있어야 할 것**: `experiments/`, `models/`, `utils/` 전체 코드, `data/stage1~4` CSV, `data/stage5/onnx_samples/*.onnx`, `models/trained/*.pkl`

**없어도 되는 것**: MNIST/CIFAR-10 원본 데이터, 가상환경 폴더, 캐시

### 20.3 Windows 환경 설정

1. [python.org](https://www.python.org/downloads/)에서 **Python 3.10 이상** 설치, "Add Python to PATH" 체크
2. `git clone` 또는 폴더 복사로 프로젝트 확보
3. 가상환경 및 패키지:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# 또는: pip install torch torchvision scikit-learn xgboost matplotlib pandas numpy joblib onnx onnxscript psutil
```

### 20.4 Mac vs Windows 디바이스 차이

| 환경 | GPU | `--device` 값 |
|------|-----|---------------|
| Mac (M1) | MPS | `mps` |
| Windows (NVIDIA) | CUDA | `cuda` |
| Windows (무 GPU) | — | `cpu` |

예측 모델은 `cpu` vs `GPU(=mps 또는 cuda)`만 구분합니다. Windows에서 `--device cuda`로 실행하면 기존 mps 데이터로 학습된 모델에도 "GPU"로 매핑됩니다.

### 20.5 Stage 3/5 필수 루틴

```powershell
# 예측 모델 학습
python experiments/train_predictor.py

# ONNX 변환 및 예측
python experiments/export_to_onnx.py
python experiments/predict_from_onnx.py data/stage5/onnx_samples/ann_l2_w256.onnx --device cpu
```

Transformer/GAN ONNX는 `--embed-dim`, `--num-heads`, `--patch-size`, `--latent-dim` 등 인자 추가.

### 20.6 Stage 1~4 데이터 수집 (선택)

```powershell
# CUDA 사용 가능 시 자동으로 ['cpu', 'cuda'] 선택
# 이미 측정된 (config, device) 조합은 건너뛰고, CUDA만 추가 측정 가능
python experiments/ann_mnist_experiment.py
python experiments/cnn_cifar10_experiment.py
python experiments/transformer_experiment.py
python experiments/gan_experiment.py
```

Mac(MPS)에서 수집한 CSV가 있으면 Windows에서 실행 시 **CUDA 데이터만 추가**됩니다. 시각화(`analyze_cnn_results.py`, `visualize_all_stages.py`)에서 CPU·MPS·CUDA를 동시에 표시할 수 있습니다.

### 20.7 Op-Level 프로파일링 (선택)

```powershell
python experiments/op_profile_demo.py --device cpu
```

### 20.8 권장 실행 순서 (체크리스트)

1. **B0** Python 3.10+, `.venv`, 패키지 설치
2. **B1** `ann_mnist_experiment.py`
3. **B2** `cnn_cifar10_experiment.py`
4. **B3** `transformer_experiment.py`, `gan_experiment.py`
5. **B4** `train_predictor.py` — 92개 Feature 학습
6. **B5** `export_to_onnx.py`, `predict_from_onnx.py` — ONNX 예측 확인

---

## 21. 브랜치 병합 (ijunsoo / dal-merge)

팀원(ijunsoo, 달현)이 각자 개발한 코드 중 유용한 부분을 **포팅**하여 통합했습니다.  
(Git merge가 아닌, 필요한 코드만 복사·수정하여 가져온 방식)

### 21.1 병합 개요

| 브랜치 | 가져온 내용 |
|--------|-------------|
| **ijunsoo** | Op-Level 프로파일러 (모델을 연산 단위로 분해 → 각 op 시간 측정 → 합산) |
| **dal-merge** (달현) | 하드웨어 동적 감지 (psutil로 CPU, RAM, GPU 자동 감지) |

현재 프로젝트 구조(experiments/, models/, utils/)를 유지하면서 필요한 모듈만 추가했습니다.

### 21.2 포팅한 내용

| 파일 | 역할 |
|------|------|
| `utils/op_profiler.py` | 모델을 Conv, Linear, ReLU 등 op 단위로 분해하고 각 op 실행 시간 측정 (보고서 "방법 2") |
| `utils/hardware_info.py` | CPU 코어, 클럭, RAM, GPU 메모리를 실행 시점에 자동 감지 (M1/MPS 지원) |
| `experiments/op_profile_demo.py` | Op 프로파일러를 ANN, CNN에 적용하는 데모 스크립트 |

### 21.3 실행 방법

```bash
# Op-Level 프로파일 데모
python experiments/op_profile_demo.py --device cpu
python experiments/op_profile_demo.py --device mps --model ann
```

```python
# 하드웨어 정보 확인 (hardware_info 사용 시 pip install psutil)
from utils.hardware_info import get_hardware_info
hw = get_hardware_info('mps')
```

---

## 22. 핵심 개념 용어 정리


| 용어                | 설명                                                   |
| ----------------- | ---------------------------------------------------- |
| **ANN**           | 인공 신경망. 뉴런을 모방한 층(Layer) 구조. 가장 기본적인 딥러닝 모델          |
| **CNN**           | 합성곱 신경망. 이미지 처리에 특화. 필터로 패턴을 추출                      |
| **Transformer**   | 어텐션 메커니즘 기반 모델. 언어모델(GPT, BERT)과 이미지 분류(ViT)에 사용     |
| **GAN**           | 생성적 적대 신경망. Generator와 Discriminator가 경쟁하며 이미지 생성    |
| **ONNX**          | 딥러닝 모델 교환 표준 포맷. 다양한 프레임워크 간 호환                      |
| **파라미터**          | 모델이 학습하는 가중치(weight)의 수. 모델 크기를 나타냄                  |
| **추론 시간**         | 학습된 모델로 새 데이터를 예측하는 시간 (inference time)              |
| **학습 시간**         | 모델을 데이터로 훈련시키는 시간 (training time per epoch)          |
| **MPS**           | Apple Silicon M1의 GPU 가속 (Metal Performance Shaders) |
| **Random Forest** | 여러 결정 트리의 평균으로 예측하는 앙상블 ML 모델                        |
| **XGBoost**       | 결정 트리를 순차적으로 보정하는 고성능 앙상블 ML 모델                      |
| **R²**            | 예측 정확도 지표. 1에 가까울수록 정확 (범위: -∞ ~ 1)                  |
| **RMSE**          | 예측값과 실제값의 평균 오차 (단위: 예측 대상과 동일)                      |
| **K-Fold CV**     | 데이터를 K개 구간으로 나누어 교차 검증하는 방법                          |
| **로그 변환**         | 넓은 범위의 값을 log(x+1)로 줄여 학습 안정성 향상                     |
| **Feature**       | 예측 모델의 입력 변수 (총 92개: 모델구조 + 하드웨어 + 데이터 정보)           |
| **GridSearchCV**  | 하이퍼파라미터 조합을 자동으로 탐색하여 최적값 찾기                         |
| **joblib**        | Python 객체(모델, 배열)를 파일로 저장/불러오는 라이브러리                 |
| **Epoch**         | 전체 학습 데이터를 한 번 순환하는 학습 단위                            |
| **Batch**         | 한 번에 모델에 넣는 데이터 묶음 크기                                |
| **Op-Level**      | 모델을 연산(op) 단위로 분해하여 각 op별 시간을 측정하는 방식 (방법 2) |
| **포팅 (Port)**   | 다른 브랜치의 코드를 복사해서 현재 구조에 맞게 수정하여 가져오는 것   |


---

## 23. 참고 자료

- [PyTorch Documentation](https://pytorch.org/docs/)
- [snntorch Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [ONNX 공식 문서](https://onnx.ai/onnx/)
- ResNet 논문: He et al., "Deep Residual Learning for Image Recognition", 2015
- MobileNet 논문: Howard et al., "MobileNets: Efficient Convolutional Neural Networks", 2017
- Vision Transformer 논문: Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020

---

*이 보고서는 Stage 1~5 전 과정 및 브랜치 병합(ijunsoo, dal-merge)을 포함한 시점을 기준으로 작성되었습니다.*  
*모든 실험은 MacBook Air M1 (CPU + MPS), Python 3.13, PyTorch 2.10.0 환경에서 수행되었습니다.*  
*Windows 검증 환경: Windows PC (Intel Core i7-11800H @ 2.30GHz, 16GB RAM, 64-bit)*