# DNN 실행 시간 예측 연구 — 현재 진행상황 보고

**작성자**: 김홍근 (학부연구생)  
**소속**: iSysLab — AI 프로세서 SW 프레임워크 연구 / 4세부 통합 시뮬레이터  
**보고일**: 2026-03-09  
**하드웨어**: MacBook Air M1 (CPU + MPS Apple Silicon GPU)

---

## 1. 현재 진행상황

### 1.1 완료 항목


| Stage       | 내용                          | 데이터 수       | 산출물                                            |
| ----------- | --------------------------- | ----------- | ---------------------------------------------- |
| **Stage 1** | ANN + MNIST 시간 측정           | 52개         | ann_mnist_results.csv                          |
| **Stage 2** | CNN + CIFAR-10 시간 측정        | 22개         | cnn_cifar10_results.csv                        |
| **Stage 3** | 예측 모델 학습 (v1→v2→v3)         | 114개 통합     | XGBoost/RF .pkl, R²_log ≈ 0.93                 |
| **Stage 4** | Transformer + GAN 시간 측정     | 40개 (24+16) | transformer_results.csv, gan_results.csv       |
| **Stage 5** | ONNX 기반 예측 시스템              | —           | predict_from_onnx.py, ONNX 샘플 10개              |
| **추가**      | Op-Level 프로파일러 포팅           | —           | utils/op_profiler.py                           |
| **추가**      | 브랜치 병합 (ijunsoo, dal-merge) | —           | utils/hardware_info.py, branch_merge_report.md |


### 1.2 진행 중

- 통합 Feature Schema 검토 및 문서화 (`reports/feature_schema.md`)
- DNN+SNN 통합 메타모델 준비

### 1.3 다음 단계 (2차년도)

- SpikeJelly 추가, SNN 데이터 수집
- DNN + SNN 통합 메타모델 정의
- ONNX 기반 시간 예측 시스템 확장

---

## 2. 코드 선택 이유 (왜, 어떻게)

### 2.1 프레임워크: PyTorch


| 선택          | 이유                                                       |
| ----------- | -------------------------------------------------------- |
| **PyTorch** | M1 Mac MPS(Apple Silicon GPU) 지원, ONNX 변환 공식 지원, 팀 표준 사용 |


### 2.2 예측 모델: XGBoost, Random Forest


| 선택                | 이유                                                     |
| ----------------- | ------------------------------------------------------ |
| **XGBoost**       | 회귀 문제에 강함, GridSearchCV로 하이퍼파라미터 자동 탐색, R²_log 0.93 달성 |
| **Random Forest** | 과적합에 강함, Feature 중요도 해석 가능, XGBoost와 앙상블로 활용           |


### 2.3 프로젝트 구조: experiments/, models/, utils/


| 선택           | 이유                                                                          |
| ------------ | --------------------------------------------------------------------------- |
| **단계별 분리**   | Stage 1~5 실험 스크립트를 experiments/에 분리하여 재실행·재현 용이                             |
| **모델 정의 분리** | models/에 ANN, CNN, Transformer, GAN 정의 — 재사용 및 확장 용이                        |
| **공통 유틸**    | utils/timer.py (시간 측정), onnx_feature_extractor.py (Feature 추출) — 여러 실험에서 공유 |


### 2.4 시간 측정: warmup 5회 + 10회 반복


| 선택                 | 이유                                                     |
| ------------------ | ------------------------------------------------------ |
| **utils/timer.py** | 교수님 유의사항 "10회 이상 반복 후 평균 사용 (warmup 포함)" 준수            |
| **MPS 동기화**        | `torch.mps.synchronize()` 호출로 GPU 연산 완료 대기 — 측정 정확도 향상 |


### 2.5 ONNX 표준 포맷


| 선택       | 이유                                                               |
| -------- | ---------------------------------------------------------------- |
| **ONNX** | PyTorch, TensorFlow 등 프레임워크 독립, 배포·공유 용이, 교수님 연구 방향(ONNX 기반)과 일치 |


---

## 3. Feature 선정 이유

### 3.1 그룹별 구성 (총 33개)


| 그룹                     | Feature 수 | 선정 이유                                                      |
| ---------------------- | --------- | ---------------------------------------------------------- |
| **모델 구조**              | 16개       | total_params, 레이어 수, 채널, 풀링/BN 유무 — 실행 시간과 직접적 상관          |
| **하드웨어**               | 6개        | M1 CPU/MPS 구조 반영, 교수님 유의사항(하드웨어 기록) 준수                     |
| **입력 데이터**             | 6개        | batch_size, 이미지 크기 — 연산량·메모리 영향                            |
| **Transformer/GAN 전용** | 5개        | embed_dim, num_heads, patch_size, latent_dim — 모델 구조 차이 반영 |


### 3.2 교수님 유의사항 반영

- **"Feature 최대한 많이"**: 모델 구조 + 하드웨어 + 입력 데이터를 빠짐없이 추출
- **로그 변환**: total_params, model_size_mb, max_width에 log1p 적용 — 값 범위 균일화로 예측 성능 향상

### 3.3 핵심 Feature

- `total_params`, `num_conv_layers`, `max_width`: 모델 크기·복잡도와 실행 시간 상관
- `device_encoded`, `cpu_cores`, `ram_total_gb`: 하드웨어별 실행 시간 차이 반영
- `batch_size`, `input_height`, `input_width`: 입력 크기에 따른 연산량 반영

---

## 4. 각 모델 비교분석

### 4.1 모델별 요약


| 모델              | 데이터 수 | 파라미터 범위   | CPU 추론       | MPS 추론       | 특징                          |
| --------------- | ----- | --------- | ------------ | ------------ | --------------------------- |
| **ANN**         | 52개   | 5만~180만   | 0.03~0.09 ms | 0.39~0.78 ms | 작은 모델은 CPU가 더 빠름 (GPU 오버헤드) |
| **CNN**         | 22개   | 36만~1100만 | 6~799 ms     | 2~80 ms      | ResNet18에서 MPS 10배 가속       |
| **Transformer** | 24개   | —         | 28~90 ms     | 8~20 ms      | Attention 구조, MPS 가속 효과     |
| **GAN**         | 16개   | —         | 0.31 ms      | 1.04 ms      | 학습 시간 극단적 짧음 (0.004 sec)    |


### 4.2 주요 발견


| 발견                 | 설명                                               |
| ------------------ | ------------------------------------------------ |
| **ANN CPU vs MPS** | 작은 ANN에서는 GPU(MPS)가 오히려 느림 — 데이터 전송 오버헤드가 계산보다 큼 |
| **CNN ResNet18**   | CPU 799ms vs MPS 80ms — 약 10배 차이, 대형 모델에서 GPU 이점 |
| **MobileNetV2**    | 파라미터 적지만 CPU에서 느림 — Depthwise Conv의 CPU 비효율      |
| **GAN 학습 시간**      | 0.004초 수준 — ResNet18(1183초)과 29만 배 차이, 로그 변환 필수  |


### 4.3 모델 다양성

- **ANN**: 레이어 2~~5층, 너비 32~~1024 — 교수님 유의사항 "레이어·뉴런 다양화" 준수
- **CNN**: SimpleCNN 변형 + ResNet18, MobileNetV2 (기존 데이터 유지)
- **Transformer, GAN**: 4종 모델로 다양성 확보 → R²_log 0.88→0.93 향상

---

## 5. 데이터량 선정 이유

### 5.1 단계별 데이터 수


| 단계                        | 데이터 수    | 선정 이유                                                       |
| ------------------------- | -------- | ----------------------------------------------------------- |
| **Stage 1 (ANN)**         | 52개      | 26개 모델 × CPU/MPS. 레이어·너비 다양화(교수님 유의사항), v2 개선을 위해 28→52로 확장 |
| **Stage 2 (CNN)**         | 22개      | 11개 모델 × 2. SimpleCNN 변형 9개 + ResNet18, MobileNetV2         |
| **Stage 4 (Transformer)** | 24개      | 12개 구성 × 2. embed_dim, num_heads, num_layers, patch_size 조합 |
| **Stage 4 (GAN)**         | 16개      | 8개 구성 × 2. latent_dim, hidden layers 조합                     |
| **총**                     | **114개** | 4종 모델 통합                                                    |


### 5.2 데이터 증가 효과


| 버전  | 데이터 수                     | 추론 R²_log             |
| --- | ------------------------- | --------------------- |
| v1  | 50개 (ANN 28 + CNN 22)     | 0.35 (원 스케일), 성능 낮음   |
| v2  | 74개 (ANN 52 + CNN 22)     | 0.88 (로그 변환 + K-Fold) |
| v3  | 114개 (+ Transformer, GAN) | **0.93**              |


### 5.3 과적합 방지

- **규칙**: 데이터 수 ≥ Feature 수 × 3
- 114개 ÷ 33개 ≈ 3.5 — 최소 기준 충족
- 92개 Feature 전체 적용 시 과적합 위험 → 1차 핵심 세트만 단계적 도입 (통합 스키마 참고)

---

## 6. 성과

### 6.1 예측 성능


| 지표            | 값                      | 의미                 |
| ------------- | ---------------------- | ------------------ |
| **추론 R²_log** | 0.93 (XGBoost)         | 실행 시간 변동의 약 93% 설명 |
| **학습 R²_log** | 0.88 (XGBoost)         | 학습 시간 추정 품질 양호     |
| **CV R²_log** | 0.937 (추론), 0.891 (학습) | K-Fold 교차검증으로 검증   |


### 6.2 산출물


| 산출물                   | 설명                                                  |
| --------------------- | --------------------------------------------------- |
| **ONNX 예측 시스템**       | .onnx 파일 입력 → 33개 Feature 추출 → 실행 시간 예측             |
| **학습된 모델**            | models/trained/*.pkl (XGBoost, Random Forest)       |
| **Op-Level 프로파일러**    | ijunsoo 브랜치 포팅, 방법 2(연산 단위 분해) 기반                   |
| **통합 Feature Schema** | reports/feature_schema.md — 향후 DNN+SNN 통합용          |
| **브랜치 병합**            | ijunsoo (op_profiler), dal-merge (hardware_info) 포팅 |


### 6.3 교수님 유의사항 준수 여부


| 항목                  | 준수                         |
| ------------------- | -------------------------- |
| 자신의 H/W에서 실험        | MacBook Air M1 (CPU + MPS) |
| Feature 최대한 많이      | 33개 (모델+하드웨어+입력+전용)        |
| ResNet/MobileNet 자제 | 기존 데이터 유지, 신규 실험에서 자제      |
| ANN 다양화             | 레이어 2~~5층, 너비 32~~1024     |
| 주석 상세 작성            | 한국어 주석, 초보자 이해 수준          |
| 반복 측정 10회 이상        | warmup 5회 + 10회 반복         |


---

## 참고 문서

- [final_report.md](final_report.md) — 전체 연구 보고서
- [feature_schema.md](feature_schema.md) — 통합 Feature Schema (향후 DNN+SNN용)
- [branch_merge_report.md](branch_merge_report.md) — 브랜치 병합 상세

