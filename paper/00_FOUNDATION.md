# 논문 기초 문서 (Foundation)

> KIISE 학술발표회 양식(`WDSC_example.md`) 기준 2~3쪽 논문 작성을 위한 기반 정리.
> 본 문서는 **작성 방향·구조·근거·자산**을 담는다. 실제 원고는 `10_draft_v0.md`.

---

## 1. 확정된 방향

**핵심 기여(초점): 이종 컴퓨팅 플랫폼을 아우르는 통합 DNN 실행시간·메모리 예측 메타모델**

- 하드웨어를 **입력 피처로 인코딩**한 단일 메타모델로 **CPU / CUDA GPU / Apple Silicon MPS** 세 백엔드의 학습·추론·메모리를 예측한다.
- 6개 아키텍처 계열(ANN/CNN/ResNet/MobileNet/ViT/GAN)을 포괄한다.
- 실측 피처 중요도로 **"실행시간은 FLOPs보다 메모리 트래픽에 지배된다"**는 통찰을 제시(roofline 이론의 실증).

### 한 문장 주제(Thesis)
> 모델 구조 피처와 하드웨어 피처를 입력으로 하는 단일 메타모델은 이종 플랫폼(CPU/CUDA/Apple MPS)에서 학습·추론·메모리를 예측할 수 있으며, 예측의 지배 요인은 연산량(FLOPs)이 아니라 메모리 트래픽이다.

---

## 2. 관련 연구와 차별점 (Positioning)

| 기존 연구 | 접근 | 대상 | 한계(=우리 기회) |
|---|---|---|---|
| **nn-Meter** (MobiSys 2021) | 커널 단위 분해 후 커널별 지연 예측 | 추론, CNN, 엣지/모바일 | 추론 한정, **디바이스별 개별 예측기**, CNN 위주 |
| **PreNeT** (2024) | 레이어별 연산·메모리 피처 회귀, 미지 가속기 일반화 | 학습시간, GPU 가속기 | Apple Silicon 통합메모리 미포함 |
| **Justus et al.** (IEEE BigData 2018) | 레이어별 시간 예측 후 합산 | 학습 스텝 시간 | FLOPs≠시간은 지적하나 합산식, 단일 환경 |
| **Roofline / 메모리바운드 이론** | FLOPs 하한, 실제로는 메모리 바운드 | 이론 | 다양한 아키텍처에서의 실증 부족 |

**우리 차별점 3가지**
1. **크로스플랫폼 통합**: 하드웨어=피처 → 디바이스별 예측기 불필요. **Apple Silicon(MPS, 통합메모리) 포함**이 핵심 novelty.
2. **아키텍처 다양성**: 6계열(생성모델 GAN·어텐션 ViT 포함) — nn-Meter(CNN)보다 넓음.
3. **메모리바운드 실증**: 피처 중요도 1위 `total_op_memory_write`(≈0.64) → 이론의 데이터 기반 확인. + 3개 타깃 동시 예측.

---

## 3. 데이터 자산 (이미 보유)

| 플랫폼 | 백엔드 | 행 수 | 파일 |
|---|---|---:|---|
| Windows PC | GPU(CUDA, RTX 4060 Ti) | 160 | `results/benchmark_results.json` |
| Windows PC | CPU | 218 | `results/benchmark_results.json` |
| MacBook M1 | GPU(MPS, 통합메모리) | 160 | `results/benchmark_results_mac.json` |
| MacBook M1 | CPU | 160 | `results/benchmark_results_mac.json` |

- 피처 스키마: **111차원** (구조 A33 + 모델전용 B18 + 입력 C8 + 하드웨어 D33 + Op-level E19). 단일 소스 `scripts/train_predictor.FEATURE_COLUMNS`.
- 측정 프로토콜: warmup 3회 + 10회 반복 평균, `perf_counter`, CUDA/MPS는 매 측정 `synchronize()`.
- 6계열 × config: ANN 42/CNN 60/ResNet 18/MobileNet 20/ViT 12/GAN 8 = 160 config.

### 이미 있는 결과물
- `results/report_meta_model_cv.csv` — **CPU + CUDA** 메타모델 5-Fold CV 성능(R²/R²_log/RMSE/MAE)
- `results/report_onnx_vs_benchmark.csv` — ONNX 예측 vs 벤치 비교
- 피처 중요도(GPU·학습·RF): `total_op_memory_write` 0.64, `total_op_memory_read` 0.11, `flops_ratio_Linear` 0.067 ...
- `docs/images/*.svg`, `results/figures/*` — 산점도·중요도·크로스플랫폼 그림

### 🔴 유일한 데이터 공백 (논문 전 반드시 채울 것)
- 현재 메타모델 CV 표는 **CPU + CUDA만**. **크로스플랫폼 주장을 완성하려면 Mac(MPS) 데이터로도 메타모델 학습·CV 필요.**
- 실행: `python scripts/train_predictor.py --input results/benchmark_results_mac.json --dedupe --filter-device "GPU(MPS)" --save-models --model-dir results/trained_models_mps`
- 이후 3플랫폼(CPU/CUDA/MPS) × 3타깃(학습/추론/메모리) 통합 표 재생성.

---

## 4. 논문 구조 (KIISE 2~3쪽)

1단(1-column): 제목·저자·소속·이메일(국/영문)·요약
2단(2-column): 본문 이하

```
요약 (국문 5~7줄)
1. 서론
   - 배경: 배포 전 실행시간 예측 필요 / 후보 아키텍처 수백 개 → 직접 실행 불가능
   - 한계: FLOPs는 시간과 불일치(메모리·병렬화), 기존 예측기는 단일 플랫폼·CNN 위주
   - 기여 3가지 (크로스플랫폼 통합 / 6계열 / 메모리바운드 실증)
2. 관련 연구  (nn-Meter, PreNeT, Justus, roofline — 각 1~2문장)
3. 제안 방법 (예측 프레임워크)
   3.1 벤치마크 데이터 수집 (6계열, 3플랫폼, 측정 프로토콜)
   3.2 피처 스키마 (구조+하드웨어+Op-level, 요약 표 1)
   3.3 메타모델 학습 (log1p 변환, XGBoost/RF/GB, 5-Fold CV, 하드웨어=피처 통합)
4. 실험 결과
   4.1 예측 정확도 (표 2: 플랫폼×타깃 R²_log)
   4.2 피처 중요도 → 메모리바운드 통찰 (그림/표)
   4.3 실측 vs 예측 산점도 (그림 1)
5. 결론 및 향후 과제 (ONNX 경로, SNN 확장)
참고문헌
```

### 그림·표 계획
- **표 1**: 피처 그룹 요약 (5그룹, 111차원)
- **표 2**: 메타모델 정확도 — 플랫폼(CPU/CUDA/MPS) × 타깃(학습/추론/메모리) × R²_log/RMSE
- **그림 1**: 실측 vs 예측 산점도 (플랫폼별 패널, y=x 기준선) — `docs/images/scatter_*.svg` 활용
- **그림 2**(여유 시): 피처 중요도 상위 (메모리 트래픽 상위) — `docs/images/feature_importance_gpu_train.svg`

---

## 5. 참고문헌 후보 (References)

1. S. Ni, Y. Tseng, Y. Chen, J. Sheu, "The Broadcast Storm Problem..." — (예시용, 실제 미사용)
2. **L. L. Zhang et al., "nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices," MobiSys 2021.**
3. **PreNeT: Leveraging Computational Features to Predict Deep Neural Network Training Time, arXiv:2412.15519, 2024.**
4. **D. Justus, J. Brennan, S. Bonner, A. S. McGough, "Predicting the Computational Cost of Deep Learning Models," IEEE BigData 2018.** (arXiv:1811.11880)
5. **S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," CACM 2009.**
6. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
7. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019.
8. (ONNX) J. Bai et al., ONNX: Open Neural Network Exchange, 2019.
9. (ViT) A. Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021.

> 최종 인용은 본문에서 실제 언급한 것만 번호순으로 정리.

---

## 6. 남은 작업 체크리스트

- [ ] **(데이터)** Mac MPS 메타모델 학습 → 3플랫폼 통합 CV 표 생성
- [ ] **(저자)** 국/영문 저자·소속·이메일 확정 (현재 placeholder)
- [ ] **(제목)** 국/영문 제목 확정
- [ ] **(원고)** `10_draft_v0.md` 본문 채우기 → 2~3쪽 분량 조정
- [ ] **(그림)** 크로스플랫폼 산점도 최종본 선택·캡션
- [ ] **(관련연구)** 각 논문 1~2문장 요약 + 정확한 서지정보
