# 예측 모델 성능 결과 분석

- 학습 데이터: ANN 98행 + CNN 120행 = 총 218행 (CPU 109개 / CUDA 109개)
- 예측 모델: LinearRegression(베이스라인) / RandomForest / XGBoost
- 평가 방법: K-Fold(5) 교차검증, log1p 변환 후 학습 → expm1 역변환 후 평가
- 장치별 분리 학습 (CPU / CUDA)

---

## 1. 전체 성능 결과

### CPU (109개)

| 모델 | 예측 대상 | R² | R²(log) | RMSE | MAE | 최적 파라미터 |
|------|----------|-----|---------|------|-----|------------|
| LinearRegression | 학습 시간 | 0.859 | 0.963 | 11.180s | 4.579s | - |
| RandomForest | 학습 시간 | 0.974 | 0.984 | 4.773s | 2.354s | n_estimators=200, max_depth=None |
| **XGBoost** | **학습 시간** | **0.990** | **0.993** | **3.006s** | **1.586s** | lr=0.1, max_depth=3, n_estimators=200 |
| LinearRegression | 추론 시간 | 0.961 | 0.979 | 0.402s | 0.193s | - |
| RandomForest | 추론 시간 | 0.987 | 0.993 | 0.233s | 0.097s | n_estimators=200, max_depth=None |
| **XGBoost** | **추론 시간** | **0.994** | **0.997** | **0.151s** | **0.070s** | lr=0.05, max_depth=5, n_estimators=200 |

### CUDA (109개)

| 모델 | 예측 대상 | R² | R²(log) | RMSE | MAE | 최적 파라미터 |
|------|----------|-----|---------|------|-----|-----------
-|
| LinearRegression | 학습 시간 | 0.755 | 0.937 | 7.776s | 3.291s | - |
| RandomForest | 학습 시간 | 0.959 | 0.981 | 3.196s | 1.601s | n_estimators=200, max_depth=None |
| **XGBoost** | **학습 시간** | **0.975** | **0.988** | **2.461s** | **1.232s** | lr=0.1, max_depth=3, n_estimators=200 |
| LinearRegression | 추론 시간 | 0.951 | 0.970 | 0.189s | 0.091s | - |
| **RandomForest** | **추론 시간** | **0.970** | 0.985 | **0.148s** | 0.067s | n_estimators=200, max_depth=10 |
| XGBoost | 추론 시간 | 0.968 | **0.988** | 0.154s | **0.063s** | lr=0.1, max_depth=3, n_estimators=200 |

---

## 2. 모델별 비교 요약

| 장치 | 예측 대상 | 최고 모델 | R² |
|------|----------|----------|-----|
| CPU | 학습 시간 | XGBoost | 0.990 |
| CPU | 추론 시간 | XGBoost | 0.994 |
| CUDA | 학습 시간 | XGBoost | 0.975 |
| CUDA | 추론 시간 | RandomForest | 0.970 |

- XGBoost가 4개 중 3개 케이스에서 최고 성능
- CUDA 추론 시간에서 RF(0.970) vs XGBoost(0.968)는 사실상 동등
- **전반적으로 XGBoost 권장**

### LinearRegression 대비 XGBoost 개선폭

| 장치 | 예측 대상 | LR R² | XGBoost R² | 개선 |
|------|----------|--------|------------|------|
| CPU | 학습 시간 | 0.859 | 0.990 | **+0.131** |
| CPU | 추론 시간 | 0.961 | 0.994 | +0.033 |
| CUDA | 학습 시간 | 0.755 | 0.975 | **+0.220** |
| CUDA | 추론 시간 | 0.951 | 0.968 | +0.017 |

- CUDA 학습 시간 예측이 가장 어렵고(LR R²=0.755), XGBoost 개선폭도 가장 큼
- 추론 시간은 LR도 R²=0.95 이상으로 선형 패턴이 강함

---

## 3. Feature 중요도 분석

### 학습 시간 [CPU]

| 순위 | feature | RF | XGBoost |
|------|---------|-----|---------|
| 1 | flops | 0.551 | - |
| 1 | num_filters | - | 0.379 |
| 2 | num_filters | 0.315 | - |
| 2 | conv_params | - | 0.356 |
| 3 | conv_params | 0.062 | flops 0.199 |

### 학습 시간 [CUDA]

| 순위 | feature | RF | XGBoost |
|------|---------|-----|---------|
| 1 | flops | 0.695 | 0.366 |
| 2 | num_filters | 0.198 | 0.327 |
| 3 | conv_params | 0.048 | 0.257 |

### 추론 시간 [CPU]

| 순위 | feature | RF | XGBoost |
|------|---------|-----|---------|
| 1 | num_filters | 0.770 | **0.896** |
| 2 | flops | 0.151 | 0.024 |
| 3 | conv_params | 0.048 | 0.069 |

### 추론 시간 [CUDA]

| 순위 | feature | RF | XGBoost |
|------|---------|-----|---------|
| 1 | num_filters | 0.783 | **0.935** |
| 2 | flops | 0.155 | 0.014 |
| 3 | conv_params | 0.032 | 0.038 |

---

## 4. Feature 중요도 핵심 관찰

**공통적으로 상위 3개: `num_filters`, `flops`, `conv_params`**

- **추론 시간**: `num_filters`가 압도적 (XGBoost 기준 89~93%) → 필터 수가 추론 속도를 거의 단독으로 결정
- **학습 시간**: `flops`, `num_filters`, `conv_params` 3개가 고르게 기여
- **하드웨어 feature** (cpu_cores, ram, gpu_memory 등): 중요도 0.0 → 단일 환경 실험이라 변동 없어 기여 없음
- **ANN 전용 feature** (hidden_size, num_hidden_layers): 하위권 → CNN 데이터가 더 많고 영향력이 큰 feature들에 가려짐

---

## 5. CPU vs CUDA 예측 난이도

| 예측 대상 | CPU R² (XGBoost) | CUDA R² (XGBoost) |
|----------|-----------------|------------------|
| 학습 시간 | 0.990 | 0.975 |
| 추론 시간 | 0.994 | 0.968 |

CPU가 CUDA보다 일관되게 예측이 쉬움. CUDA는 GPU 스케줄링, 메모리 전송 등 비결정적 요소가 있어 측정값 변동이 크기 때문.

---

## 6. 분석 요약

| 항목 | 결과 |
|------|------|
| 가장 좋은 예측 | CPU 추론 시간 (XGBoost R²=0.994, MAE=0.070s) |
| 가장 어려운 예측 | CUDA 학습 시간 (LR R²=0.755) |
| 가장 중요한 feature | num_filters (추론 시간 89~93%) |
| 권장 예측 모델 | XGBoost |
