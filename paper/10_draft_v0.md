<!-- KIISE 학술발표회 양식 스켈레톤 v0 — WDSC_example.md 형식 준수
     [1단] 1~8: 제목·저자·소속·이메일·요약  /  [2단] 9~11: 본문·참고문헌·부록
     그림 명칭은 하단, 표 명칭은 상단. 장/절 번호는 1., 1.1 형식. -->

# (국문 제목) 이종 컴퓨팅 플랫폼을 위한 딥러닝 모델 실행시간·메모리 예측 메타모델

**저자**: 〈저자1 국문이름〉O 〈저자2 국문이름〉 〈저자3 국문이름〉  <!-- O = 발표자 -->
**소속**: 〈소속 국문명〉
**이메일**: {id1, id2, id3}@〈도메인〉

**Title (영문)**: A Cross-Platform Meta-Model for Predicting Execution Time and Memory of Deep Learning Models

**Authors**: 〈Author1〉O 〈Author2〉 〈Author3〉
**Affiliation**: 〈Affiliation (English)〉

---

## 요 약

딥러닝 모델을 배포하기 전에 실행 시간과 메모리 사용량을 아는 것은 아키텍처 설계와 하드웨어 선택에 중요하다. 그러나 후보 구성이 수백 개에 이르면 모두 직접 실행해 보는 것은 비현실적이며, 연산량(FLOPs)은 실제 실행 시간과 잘 일치하지 않는다. 본 논문에서는 모델 **구조 피처와 하드웨어 피처를 함께 입력**으로 받는 **단일 메타모델**을 제안한다. 이 메타모델은 별도의 디바이스별 예측기 없이 **CPU, CUDA GPU, Apple Silicon(MPS) 통합메모리** 세 이종 플랫폼에서 학습·추론 시간과 메모리를 예측한다. ANN, CNN, ResNet, MobileNet, ViT, GAN 6개 계열 〈N〉개 구성을 세 플랫폼에서 벤치마크하여 학습한 결과, 로그 스케일 R²이 〈0.9x~0.9x〉에 도달하였다. 또한 피처 중요도 분석을 통해 실행 시간을 지배하는 요인이 연산량보다 **메모리 트래픽**임을 실측으로 확인하였다.

<!-- TODO: 벤치마크 config 총합 N, 최종 R²_log 범위 채우기 (3플랫폼 학습 후) -->

---

## 1. 서 론

딥러닝 모델을 실제 환경에 배포하기 전에 "이 구성이 너무 느리지 않은가?", "GPU 없이도 감당 가능한가?"와 같은 질문에 답하려면 보통 모델을 **직접 실행**해 본다. 그러나 신경망 구조 탐색(NAS)이나 하드웨어 선택 과정에서 후보 아키텍처가 수백 개에 이르면 모두 실행해 측정하는 방식은 사실상 불가능하다.

실행 시간의 이론적 하한으로 흔히 쓰이는 부동소수점 연산 수(FLOPs)는 실제 시간과 잘 일치하지 않는다. 동일한 FLOPs를 갖는 두 모델도 메모리 접근 패턴이나 병렬 실행 효율에 따라 실행 시간이 크게 달라지기 때문이다 [Justus, roofline]. 한편 기존의 실행 시간 예측 연구는 대체로 **단일 플랫폼**이나 특정 디바이스에 국한되거나(nn-Meter는 엣지 디바이스별 개별 예측기[nn-Meter]), **CNN 계열**에 집중되어 있어 다양한 아키텍처와 이종 하드웨어를 하나로 다루지 못한다.

본 논문은 모델 **구조 피처와 하드웨어 피처를 함께 입력**으로 받는 단일 메타모델을 제안한다. 하드웨어를 입력 피처로 인코딩함으로써 디바이스별 예측기를 따로 두지 않고도 **CPU / CUDA GPU / Apple Silicon MPS** 세 이종 플랫폼을 한 모델로 예측한다. 본 논문의 기여는 다음과 같다.

- **(1) 이종 플랫폼 통합 예측**: 하드웨어를 피처로 인코딩하여 CPU·CUDA·Apple MPS(통합메모리)를 단일 메타모델로 예측한다. 특히 기존 연구에서 잘 다루지 않은 Apple Silicon 통합메모리 구조를 포함한다.
- **(2) 아키텍처 다양성**: ANN·CNN·ResNet·MobileNet·ViT·GAN 6개 계열을 포괄하여 어텐션·생성 모델까지 다룬다.
- **(3) 메모리바운드 실증**: 피처 중요도 분석으로 실행 시간의 지배 요인이 FLOPs가 아니라 메모리 트래픽임을 데이터로 확인한다.

<!-- 논문 구성 안내 문단 -->
본 논문의 구성은 다음과 같다. 2장에서 관련 연구를 정리하고, 3장에서 제안하는 예측 프레임워크(데이터 수집·피처·메타모델)를 기술한다. 4장에서 실험 결과를 제시하고, 5장에서 결론을 맺는다.

## 2. 관련 연구

**nn-Meter** [nn-Meter]는 모델 추론을 커널 단위로 분해하고 커널별 지연을 예측하여 엣지 디바이스에서 높은 정확도를 얻었으나, 추론에 한정되고 디바이스마다 별도의 예측기를 학습한다. **PreNeT** [PreNeT]은 레이어별 연산·메모리 피처로 학습 시간을 예측하고 미지의 가속기로 일반화하지만 GPU 가속기 위주이며 Apple Silicon 통합메모리는 다루지 않는다. Justus 등 [Justus]은 레이어별 실행 시간을 예측해 합산하는 방식을 제안하며 FLOPs와 실행 시간의 불일치를 지적하였다. Roofline 모델 [Roofline]은 성능이 연산 능력과 메모리 대역폭 중 하나에 의해 제한됨을 보이며, 많은 경우 메모리 바운드임을 시사한다. 본 논문은 이들과 달리 **모델 전체 구조 피처 + 하드웨어 피처를 단일 회귀 메타모델**에 넣어 이종 플랫폼을 통합 예측하고, 메모리바운드 성질을 다양한 아키텍처에서 실증한다.

## 3. 제안 방법

### 3.1 벤치마크 데이터 수집

6개 모델 계열(ANN, CNN, ResNet, MobileNet, ViT, GAN)에 대해 하이퍼파라미터 조합으로 총 〈160〉개 구성을 생성하고, 이를 세 플랫폼에서 실행해 학습·추론 시간과 메모리를 측정하였다. 측정 신뢰도를 위해 워밍업 3회 후 **10회 반복 평균**을 사용하고, GPU(CUDA/MPS)는 매 측정마다 동기화(`synchronize()`)하여 비동기 실행으로 인한 오차를 제거하였다. 데이터는 배치 단위로 디바이스 메모리에 사전 적재해 전송 오버헤드를 분리하였다.

<!-- 표: 플랫폼별 수집 행 수 -->
[표 1] 플랫폼별 벤치마크 수집 현황

| 플랫폼 | 백엔드 | 구성 수 |
|---|---|---:|
| Windows PC | CPU | 160+ |
| Windows PC | GPU (CUDA, RTX 4060 Ti) | 160 |
| MacBook | CPU (Apple M1) | 160 |
| MacBook | GPU (MPS, 통합메모리) | 160 |

### 3.2 피처 스키마

메타모델 입력은 111차원 스칼라 피처이며 5개 그룹으로 구성된다.

[표 2] 피처 그룹 구성 (총 111차원)

| 그룹 | 개수 | 설명 |
|---|---:|---|
| A 공통 구조 | 33 | 파라미터·레이어·폭·FLOPs·구조 플래그 |
| B 모델 전용 | 18 | ANN/CNN/ViT/GAN 아키텍처별 피처 |
| C 입력 데이터 | 8 | 해상도·배치·데이터셋 등 |
| D 하드웨어 | 33 | 실행 시점 자동 수집(코어·메모리·대역폭 등) |
| E Op-level 분해 | 19 | 레이어별 파라미터·FLOPs 비율·메모리 트래픽 |

Op-level 그룹(E)은 "같은 FLOPs라도 연산 구성에 따라 실행 시간이 다르다"는 성질과 **메모리 읽기/쓰기 트래픽**을 피처로 담기 위한 것이다. 하드웨어 그룹(D)을 입력으로 포함함으로써 하나의 모델이 이종 플랫폼을 구분해 예측할 수 있다.

### 3.3 메타모델 학습

타깃(학습시간·추론시간·메모리)은 값의 범위가 넓어 `log1p` 변환 후 학습하고 평가 시 `expm1`로 역변환한다. 회귀 모델로 LinearRegression(베이스라인), RandomForest, GradientBoosting, XGBoost를 사용하고, 트리 계열은 GridSearchCV로 튜닝한다. 성능은 5-Fold 교차검증으로 R², R²(log), RMSE, MAE를 측정한다.

## 4. 실험 결과

### 4.1 예측 정확도

<!-- 🔴 TODO: Mac(MPS) 메타모델 학습 후 3플랫폼 통합 표로 교체.
     아래는 현행 CPU+CUDA 수치(README §5.2) — MPS 행 추가 필요. -->

[표 3] 플랫폼·타깃별 메타모델 정확도 (5-Fold CV, 최적 모델)

| 플랫폼 | 타깃 | 모델 | R² | R²(log) |
|---|---|---|---:|---:|
| CUDA GPU | 학습시간 | XGBoost | 0.930 | 0.978 |
| CUDA GPU | 추론시간 | XGBoost | 0.953 | 0.968 |
| CUDA GPU | 메모리 | XGBoost | 0.941 | 0.998 |
| CPU | 학습시간 | XGBoost | 0.881 | 0.969 |
| CPU | 추론시간 | RandomForest | 0.865 | 0.923 |
| CPU | 메모리 | GradientBoosting | 0.944 | 0.999 |
| MPS GPU | 학습시간 | 〈TBD〉 | 〈TBD〉 | 〈TBD〉 |
| MPS GPU | 추론시간 | 〈TBD〉 | 〈TBD〉 | 〈TBD〉 |
| MPS GPU | 메모리 | 〈TBD〉 | 〈TBD〉 | 〈TBD〉 |

로그 스케일 R²이 원래 스케일보다 일관되게 높은 것은 작은 값과 큰 값을 고르게 예측함을 의미한다. 장치를 피처로 통합했음에도 각 플랫폼에서 R²(log) 〈0.9x〉 이상을 유지하였다.

### 4.2 피처 중요도 — 메모리바운드 실증

GPU 학습 시간을 설명하는 상위 피처는 연산량이 아니라 **메모리 트래픽**이었다.

[표 4] 상위 피처 중요도 (GPU·학습시간·RandomForest)

| 순위 | 피처 | 중요도 |
|---:|---|---:|
| 1 | total_op_memory_write | 0.642 |
| 2 | total_op_memory_read | 0.105 |
| 3 | flops_ratio_Linear | 0.067 |
| 4 | flops | 0.029 |
| 5 | num_mult_adds | 0.025 |

메모리 쓰기/읽기 트래픽이 상위 2개를 차지하고 순수 연산량(`flops`)은 4위에 그쳤다. 이는 현대 가속기에서 실행 시간이 **메모리 바운드** 특성에 크게 좌우된다는 roofline 통설 [Roofline]과 일치하며, 이를 다양한 아키텍처에서 데이터로 확인한 결과이다.

### 4.3 실측 vs 예측

<!-- [그림 1] docs/images/scatter_train.svg, scatter_infer.svg 활용. 그림 명칭은 하단. -->
그림 1은 실측값과 예측값의 산점도이다. 대각선(y=x)에 점들이 밀집해 있어 예측이 실측을 잘 따름을 보인다. 작은 모델의 밀리초 수준 추론 시간은 GPU 상수 오버헤드로 상대 오차가 크나 절대 오차는 작았고, 벤치 분포를 벗어난 대형 모델에서는 과소 추정 경향이 관찰되었다.

[그림 1] 실측 대비 예측 산점도 (플랫폼별)

## 5. 결 론

본 논문에서는 모델 구조 피처와 하드웨어 피처를 함께 입력으로 받는 단일 메타모델로 CPU·CUDA·Apple MPS 세 이종 플랫폼의 학습·추론 시간과 메모리를 예측하였다. 6개 아키텍처 계열에 걸쳐 로그 스케일 R² 〈0.9x〉 이상을 얻었으며, 피처 중요도 분석으로 실행 시간이 FLOPs보다 **메모리 트래픽**에 지배됨을 실측으로 확인하였다. 향후 ONNX 그래프만으로 예측하는 경로와 SNN(스파이킹 신경망) 확장을 통해 프레임워크 독립적·아키텍처 확장적 예측기로 발전시킬 계획이다.

## 참고 문헌

[1] L. L. Zhang, S. Han, J. Wei, N. Zheng, T. Cao, Y. Yang, Y. Liu, "nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices," in Proc. MobiSys, 2021.

[2] "PreNeT: Leveraging Computational Features to Predict Deep Neural Network Training Time," arXiv:2412.15519, 2024.

[3] D. Justus, J. Brennan, S. Bonner, A. S. McGough, "Predicting the Computational Cost of Deep Learning Models," in Proc. IEEE Int'l Conf. on Big Data, 2018.

[4] S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Commun. ACM, vol. 52, no. 4, pp. 65–76, 2009.

[5] T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proc. KDD, 2016.

[6] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Proc. NeurIPS, 2019.

[7] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in Proc. ICLR, 2021.
