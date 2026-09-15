# RACS 2026 camera-ready 대응 설계 v2 — 지적마다 새 증거를 추가하는 계획

작성 2026-09-15 (v2). 대상: ACM RACS 2026 paper 39, 리뷰어 1 지적 12개 + 리뷰어 2.

**v1에서 고친 점.** v1은 R1-1·2·5·7·10과 R2에서 "서술 보완·톤 조정"에 기대는 부분이 있었다. v2는 항목마다 **새로 추가하는 실험·코드·표**를 먼저 정의하고, 문장 수정은 그 결과의 귀결로만 둔다. 재측정도 "GAN만"이 아니라 GPU 두 백엔드 전체로 넓혔다(비용이 감당 가능함을 확인).

## 0. 전제와 자원

- 마감 **2026-10-07**(등록 + camera-ready), 회의 09-21. 영문 제출본 확보가 1순위(저장소에 없음, 국문 v4에는 §4.4 LOFO/LOBO가 있는데 리뷰어는 없다고 함).
- 측정 비용(원시 데이터의 평균 실행시간 합 × 10회):

| 백엔드 | 전체 158~160구성 × 10회 | GAN 8구성만 | 기기 |
|---|---:|---:|---|
| Desktop CPU | 27.2 h | 0.31 h | 이 PC |
| CUDA | **2.4 h** | 0.09 h | 이 PC, RTX 4060 Ti |
| Mac CPU | 101.8 h | 0.19 h | 맥북 M4 |
| MPS | **9.1 h** | 0.11 h | 맥북 M4, 야간 1회 |

  → GPU 두 백엔드는 **전체 재측정이 가능**하다. CPU 두 백엔드는 불가하므로 그대로 둔다.
- 실행 환경: 이 PC `.venv`(Python 3.11.9, torch 2.7.1+cu118, scikit-learn 1.8.0, xgboost 3.2.0). 파이프라인: `enrich_oplevel.py` → `export_onnx_configs.py` → `eval_fill_v4i.py` → `eval_fill_v4j.py` → (신설) `eval_fill_v4k.py`. 앞의 두 산출물은 gitignored라 재생성이 필요하다.

## 1. 지적별 보충 계획 (요약)

| # | 새로 추가하는 증거 | 코드 산출물 | 논문 산출물 | 측정 |
|---|---|---|---|---|
| R1-1 | 구간·백엔드별 오차 표, 오버헤드 피처로 <10ms 오차 개선 시도, 잡음 하한, 보정 산점도 | v4k `errors_by_regime`, `overhead_features`; `make_scatter.py` 갱신 | 오차 표(잡음 열 포함) + 그림 | 없음 |
| R1-2 | 검증 프로토콜 사다리 7단계 | v4k `protocol_ladder` | 표 7을 7행으로 교체(±시드) | 없음 |
| R1-3 | 백엔드별 하드웨어 기술자 값 | v4k `backend_values` | 표 2에 열 5개 | 없음 |
| R1-4 | LOFO 복원, 사다리에 통합 | (기존 `lofo_no_family_id`) | §4.2·§4.4 | 없음 |
| R1-5 | 라벨 1개 vs 기술자 30개, LOBO 기준선, **외부 5번째 백엔드 시험** | v4k `label_vs_descriptors`, `external_backend_test`; `run_benchmark.py --subset` | 표 7 행 2개 + 외부 백엔드 표 | 외부 기기 1~2h |
| R1-6 | 통합 XGBoost permutation·gain 중요도 | v4k `unified_importance` | 표 8·9 교체 | 없음 |
| R1-7 | 실제 배치(64/1000) 기준 프록시 + arithmetic intensity, 대칭 그룹 절제, 단독 피처 기준선 | `onnx_oplevel.py` 확장, v4k `proxy_batch_scaled`, `symmetric_ablation`, `single_feature_baselines` | 표 9 재구성, §3.3 정의 갱신 | 없음 |
| R1-8 | GB·Ridge 동일 그리드 nested CV | v4k `tuned_comparison` | 표 5 갱신 | 없음 |
| R1-9 | 4자리 + 부트스트랩 CI | v4k `precision_check` | 각주 | 없음 |
| R1-10 | GPU 두 백엔드 전체 재측정(통일 워밍업, 반복별 원시값, 피크 메모리) | 러너 4개 파일 | §3.1 프로토콜, 표 3 재계산 | CUDA 2.4h + MPS 9.1h |
| R1-11 | 셀별 CV, 잡음 하한 대 예측 오차, 2σ 이내 비율 | v4k `measurement_noise`, `load_clean` std 풀링 | 오차 표 열 | R1-10과 공유 |
| R1-12 | 환경 자동 기록, 데스크톱 버전(확보) | `software_env.py`, `check_env.py --json` | §3.1 | 맥 1회 |
| R2 | 배치 스윕(CUDA), 피크 메모리(GPU), 외부 백엔드, Limitations | `--batch-sizes` 옵션 | 배치 표 1개, 메모리 결과 | 배치 스윕 0.5h |

## 2. 항목별 상세

### R1-1 헤드라인 지표와 개별 예측 정확도의 괴리

**추가 증거**
1. 오차 표: 백엔드 × {전체, ≥10ms, <10ms, GAN} × {MAPE, MdAPE, ±20%}. 같은 칸에 측정 잡음 CV 중앙값(R1-11)을 넣어 "잡음 하한 대비 예측 오차"가 한 표에서 읽히게 한다.
2. <10ms 오차 개선 시도. (a) 오버헤드 피처 추가: `num_batches_train`(MNIST 60000/64=938, CIFAR 50000/64=782), `num_batches_infer`(10000/1000=10), `ops_x_batches = num_ops × num_batches`, `log_num_ops`. (b) 소형 구간 전문 모델(학습 셀 <50ms)과 통합 모델의 <10ms MAPE 비교. (c) 목적함수 `reg:pseudohubererror` 비교. 개선되면 통합 모델에 채택하고, 아니면 "정적 피처로는 커널 실행 오버헤드가 지배하는 구간을 설명하지 못한다"를 수치로 서술한다.
3. 보정 산점도(실측 vs 예측, 로그-로그, 백엔드 색, <10ms 표시). `make_scatter.py`를 nested CV OOF 기준으로 갱신해 그림 1장 추가.

**논문**: 초록·결론은 R²(log)·MAPE·±20% 3종 병기. §4.1에 오차 표와 그림. 문장은 이 결과의 요약이다.
**판정 기준**: <10ms MAPE(현재 54.1%)가 시드 5개 평균에서 내려가는지와 무관하게 표·그림은 들어간다.

### R1-2 어떤 분할이 실제 구속 조건인가

**추가 증거** — 검증 프로토콜 사다리(모두 XGBoost nested CV, R²(log)와 MAPE(≥10ms)):
1. 무작위 5-겹 (시드 10개 평균±표준편차)
2. 구성 단위 GroupKFold (시드 10개, `shuffle=True`, sklearn 1.8 지원 확인)
3. 폭 수준 제외: 계열마다 한 폭 값(예: CNN f64, ANN h256)의 구성을 모두 홀드아웃. 이웃 폭 사이 보간이 아닌 폭 축 외삽.
4. 깊이 수준 제외: 같은 방식으로 깊이 축.
5. 최대 사분위 제외: 계열별 파라미터 수 상위 25% 구성 홀드아웃. "더 큰 후보가 너무 느린가"라는 NAS 질문 그대로.
6. 계열 제외(LOFO, 기존)
7. 백엔드 제외(LOBO, 기존)

**코드**: v4k `protocol_ladder`. 3~5단계는 `benchmark/configs/generator.py`의 폭·깊이 키(`hidden_size`, `num_filters`, `embed_dim`, `num_layers`, `num_blocks` 등)로 그룹 라벨을 만들어 GroupKFold 또는 LeaveOneGroupOut.
**논문**: 표 7을 이 사다리 7행으로 교체. §4.2는 "정확도가 어느 단계에서 꺾이는지"를 서술하고, 무작위 대 구성 단위의 0.003 차이는 시드 표준편차와 함께 한 줄로.
**판정 기준**: R²(log)가 처음 크게 떨어지는 단계가 "구속 조건"이고, 그 단계를 기준으로 적용 범위를 명시한다.

### R1-3 백엔드 구분 피처

**추가 증거**: 하드웨어 수치 30개의 백엔드별 값 표(백엔드 안에서 상수임을 함께 확인). 대표 5개(`device_type_encoded`, `cpu_cores`, `ram_total_gb`, `gpu_cores`, `tflops_fp32`)를 표 2에 열로 추가. 확인된 값: Mac CPU는 `device_type_encoded=0, gpu_cores=0, gpu_memory_gb=0`, MPS는 `2, 10, 18`. Desktop CPU와 Mac CPU는 `cpu_cores` 16 대 10, `ram_total_gb` 31.1 대 24.
**코드**: v4k `backend_values`. Mac `cpu_freq_ghz=0` 같은 미수집 필드는 "–"로 표기하고 상수 처리임을 본문에 적는다.

### R1-4 LOFO

**추가 증거**: 기존 LOFO(계열 식별자 제거판, MAPE 포함)를 사다리 6단계로 통합. 영문본에 없으면 복원하고, 서론 기여 목록에 NAS 시나리오 검증으로 명시.

### R1-5 하드웨어 피처가 "정체성" 이상의 정보를 갖는가

**추가 증거**
1. 라벨 1개 vs 기술자 30개(구성 단위 nested CV): (a) 전체 피처, (b) 기술자 30개 제거, (c) 제거 + `_backend_id` 1개, (d) 제거 + `device_type_encoded`만(두 CPU 미구분). 기대는 (c)≈(a)이고, 그 결과 자체가 표에 들어간다.
2. LOBO 기준선: 라벨 모델은 미지 백엔드에 정의되지 않으므로 "다른 세 백엔드 예측의 평균"과 "같은 기기 종류의 최근접 백엔드 예측"을 기준선으로 두고 기술자 모델의 LOBO와 비교. 기술자 모델이 기준선을 이기는 백엔드(CPU 계열)와 못 이기는 백엔드(CUDA)를 구분해 보고한다.
3. **외부 5번째 백엔드 시험**(기기가 있으면 반드시): 연구실의 다른 기기 1대(인텔/AMD 노트북 CPU, 다른 NVIDIA GPU, 또는 M1 Air를 현재 프로토콜로)에서 부분집합을 측정하고, 4백엔드로 학습한 모델로 예측. 부분집합은 계열별 균등 40~60구성 × 3회. 비용은 소형 40구성이면 수 분, 중형 포함 60구성이면 1~2h.

**코드**: `run_benchmark.py --subset stratified:60 --repeats 3` 옵션(계열별 균등 추출, 시드 고정, 결과 `results/external_<host>.json`); v4k `label_vs_descriptors`, `external_backend_test`(기술자 모델 vs 최근접 백엔드 기준선, R²(log)·MAPE·순위 상관).
**논문**: "explicit hardware features are crucial"을 1~3의 결과로 교체. 외부 백엔드 결과가 있으면 §4.4에 1행짜리 표. 이것이 R2의 "미지 하드웨어 일반화 부족"에 대한 실측 답이 된다.

### R1-6 중요도의 출처 모델

**추가 증거**: 통합 XGBoost(구성 단위 nested CV의 각 외부 폴드 best 모델)의 홀드아웃 permutation importance(n_repeats=20). 단일 피처(`total_op_memory_write`, `flops`)와 그룹 공동 셔플(MEM/FLOPs). 폴드 평균±표준편차, gain 중요도 순위 병기. 백엔드별 RF는 보조 한 줄.
**코드**: v4k `unified_importance`.
**논문**: 표 8·9의 주 결과 교체, 서론 기여문 정정.

### R1-7 메모리 프록시: 실제 배치 기준 재계산과 고유성

**추가 증거**
1. 프록시를 실제 배치로 재계산. `decompose_onnx`는 op마다 `in_elems`(활성화)·`w_params`(가중치)·`out_elems`를 이미 분리해 계산한다(`memory_read = in_elems·4 + w_params·4`, `memory_write = out_elems·4`, 배치 1). B=64(학습)·B=1000(추론)에 대해 `read_B = 4·(in_elems·B + w_params)`, `write_B = 4·out_elems·B`, `arith_intensity_B = flops·B / (read_B + write_B)`를 op별로 합산해 피처 6개를 추가한다. 학습 모델에는 B=64, 추론 모델에는 B=1000 값을 입력.
2. 대칭 그룹 절제(재학습): 전체 / −MEM / −FLOPs / −둘 다. 그룹 크기 비대칭 문제가 없는 비교.
3. 단독 피처 기준선: `mem_write_B` 단독, `flops` 단독(기존 0.558/0.468), `arith_intensity_B` 단독, `mem_write_B + backend_id`.
4. 상관 보고: 로그 공간 Spearman(mem_write, flops), 전체와 계열별.

**코드**: `benchmark/features/onnx_oplevel.py`에 `batch_size` 인자와 op별 원소 수 반환; `merged_schema.py`에 열 6개; v4k `proxy_batch_scaled`, `symmetric_ablation`, `single_feature_baselines`.
**논문**: 표 9를 "배치 기준 프록시 포함 중요도 + 대칭 절제 + 단독 기준선"으로 재구성. §3.3 프록시 정의를 배치 기준으로 갱신. 주장은 결과가 지지하는 만큼만.
**판정 기준**: 배치 기준에서도 mem_write가 flops를 앞서고 arithmetic intensity가 상위면 Roofline 해석 유지, 아니면 "일관되게 선택되는 인자"로 한정.

### R1-8 튜닝 공정성

**추가 증거**: GB에 XGBoost와 같은 그리드 `{n_estimators [200,400], max_depth [3,6], learning_rate 0.1}`로 nested CV. LR은 Ridge(alpha {0.1, 1, 10}) nested CV로 교체. 표 5 재산출.
**코드**: v4k `tuned_comparison`.

### R1-9 표 값 일치

**추가 증거**: 전체·백엔드별 R²(log)를 4자리와 부트스트랩 95% CI(1000회)로. 코드상 백엔드별 R²는 부분집합 자체 평균 기준(`r2_score(y0[bks==b], oof[bks==b])`)이고 JSON 값도 실제로 0.988/0.989로 같다. 각주 한 줄.

### R1-10 워밍업 프로토콜 통일 — GPU 전체 재측정

**추가 증거**: CUDA 158구성(2.4h, 이 PC)과 MPS 158구성(9.1h, 맥 야간)을 새 프로토콜로 전체 재측정. 새 프로토콜은 모든 계열·백엔드에서 실제 학습 배치 1개로 학습 스텝 1회 + 평가 순전파 1회 워밍업(GAN은 D+G 스텝 1회), 반복별 원시값 저장, 피크 메모리 기록(`torch.cuda.max_memory_allocated`, `torch.mps.driver_allocated_memory`).
**코드**:
- `benchmark/runner/device.py`: `warmup(device, model_fn, train_step_fn, eval_batch)`. 현재는 배치 1 no_grad 순전파 1회라 역전파 커널은 데워지지 않는다.
- `benchmark/runner/experiment.py`: `train_times`, `infer_times`, `peak_mem_mb` 저장, `discard_first` 옵션, std의 ddof 명시.
- `run_benchmark.py`: `if not is_gan:` 분기 제거, `--subset`, `--batch-sizes` 옵션.
- `benchmark/results/io.py`: 실행 헤더(시작 시각, 환경).
**병합**: `load_clean()`에 GPU 셀 override. CPU 셀은 그대로 두되 첫 실행의 oneDNN 프리미티브 생성 비용은 std로 상한 제시(10회 중 1회 δ면 평균 편향 ≤ 0.33·std).
**논문**: §3.1 프로토콜 서술을 새 프로토콜로, 표 3(플랫폼 비율) 재계산, 재측정 전후 GAN 셀 변화량 한 문장.
**대안(맥 불가 시)**: CUDA만 전체, MPS는 GAN 8구성만(0.11h).

### R1-11 측정 잡음

**추가 증거**: CPU 셀은 기존 `std_train`·`std_infer`(10회, ddof=0)로, GPU 셀은 재측정 원시값으로. 백엔드 × 타깃 CV 중앙값·p90, <10ms/≥10ms 분리, |예측−실측| ≤ 2σ인 셀 비율. 오차 표(R1-1)에 열로 통합.
예비값(원시 698행): 학습 CV 중앙값 Desktop CPU 1.2%, CUDA 2.1%, Mac CPU 0.6%, MPS 0.6%. 추론 <10ms 셀 CV 중앙값 3.9%, p90 27.9%. ≥10ms는 1.1%, 6.5%.
**코드**: `load_clean` std 풀링 수정(중복 48구성의 std가 첫 행 값으로 남는 문제) + v4k `measurement_noise`.

### R1-12 환경 기록

**추가 증거**: 데스크톱 버전 확보(Python 3.11.9, PyTorch 2.7.1+cu118, CUDA 11.8, cuDNN 9.1.0, oneDNN 3.7.1 + MKL, Windows 11 26200, 드라이버 610.62; torch 설치 2026-01-19로 측정 이전). 맥은 `check_env.py --json` 1회. 재측정 결과 파일에는 자동 기록.
**코드**: `benchmark/platform/software_env.py`, `check_env.py --json`, 결과 행 `sw_*` 필드(문자열, `STRING_FEATURES`에 추가).

### R2 고정 배치·피크 메모리·미지 하드웨어

**추가 증거**
1. 배치 스윕(CUDA, 이 PC): 계열별 균등 30구성 × 학습 배치 {32, 64, 128, 256} × 3회, 약 0.5h. `batch_size`가 변하는 피처가 된 모델의 배치 외삽 성능({32,64,128}로 학습해 256 예측) 표 1개.
2. 피크 메모리: R1-10 재측정에서 GPU 셀의 실측 피크 메모리 확보. "파라미터 메모리로 정의한 타깃" 대신 실측값 예측 결과 한 문장 또는 표 1행.
3. 미지 하드웨어: R1-5의 외부 백엔드 시험.
4. Limitations 소절: 위 실험이 다루지 못한 범위(프로토타입 규모, CPU 배치 스윕 없음)만 남긴다.

## 3. 실행 순서와 일정

- **1주차(~9/21)**: 영문본 확보. 파이프라인 재현(enrich·ONNX 재생성 → v4j 수치 일치 확인). 러너 수정 후 CUDA 전체 재측정 시작(2.4h). v4k에 사다리·라벨 대 기술자·통합 중요도·튜닝·정밀도·잡음 구현 → 월요일 1차 수치.
- **2주차(9/22~28)**: 맥 MPS 재측정(야간), 배치 스윕(0.5h), 외부 백엔드 부분집합(기기 확보 시), 배치 기준 프록시 확장·재분석, 오버헤드 피처 실험, 그림 갱신.
- **3주차(9/29~10/3)**: 영문 원고 표·그림·본문 반영, 교수님 검토.
- **10/4~7**: 양식·등록.

## 4. 월요일 결정 사항

1. 맥북 야간 MPS 전체 재측정 가능 여부(9.1h).
2. 외부 5번째 백엔드로 쓸 기기(연구실 노트북, 다른 GPU, M1 Air).
3. 배치 스윕·피크 메모리 결과를 본문에 넣을지, 분량상 한 문장으로 갈지.
4. 영문본 §4.4 포함 여부와 페이지 여유.
5. **세션 드리프트 대응(아래 §5)**: 데스크톱 CPU 전체 재측정(27h)을 할지, 부분집합 보정으로 갈지.

## 5. 2026-09-15 CUDA 재측정 결과와 세션 드리프트 (실측 후 추가)

CUDA 160구성을 v2 프로토콜로 재측정했다(`results/remeasure_cuda_v2.json`, 15:55~18:28). 원래 3월 데이터(v1)와 비교한 결과:

| 계열 | v2/v1 학습시간 중앙값 | v2/v1 추론시간 중앙값 |
|---|---:|---:|
| ANN | 1.22 | 1.36 |
| CNN | 1.00 | 0.88 |
| ResNet | 1.10 | 1.00 |
| MobileNet | 1.10 | 1.00 |
| ViT | 1.12 | 1.01 |
| GAN | 1.11 | 1.04 |

- **느려진 정도가 모델 크기에 반비례한다.** v1 학습시간 1초 미만 구성은 +20%, 1~3초 +24%, 3~10초 +10%, 10초 이상은 0%. ANN에서는 층 수가 늘수록 비가 커진다(1층 +12% → 8층 +29%). 즉 GPU 연산이 아니라 **커널 실행 1회당 지연**이 3월보다 커졌다.
- **GPU는 스로틀링 없음.** 실행 중 P2, 2670MHz, 52W/160W. 소형 모델 실행 중 SM 클럭은 405~2535MHz를 오가며(중앙값 2115) P0~P8 상태를 순환한다. 커널 사이에 클럭이 내려가는 launch-bound 특성 그대로다.
- **환경 변화 확인:** NVIDIA 드라이버가 2026-06-11에 설치됨(610.62). 3월 측정 때와 드라이버가 다르다. 전원 관리 옵션은 "균형 조정". 백그라운드 CPU 부하는 2% 미만(claude 0.5%, nexonplug 0.4%, presentmon, fvcontainer 등 게임 관련 서비스 상주).
- **CPU 쪽도 드리프트가 있다.** 소형 4구성 프로브(5회): ANN 학습 +5~10%, CNN 학습 0%. GPU보다 작다.
- **GAN 워밍업 지적(R1-10)에 대한 답은 오히려 명확해졌다.** v1에서 워밍업이 없던 GAN의 변화(+11%)가 워밍업이 있던 ResNet·MobileNet·ViT(+10~12%)와 같다. 즉 v1의 GAN 워밍업 부재는 측정값에 영향을 주지 않았고, 공통 요인은 세션 드리프트다. v2 원시값에서 첫 반복/나머지 평균 비는 중앙값 1.00, p90 1.11이다.
- **반복 변동은 v1보다 크다.** CUDA 학습 CV 중앙값 v1 2.1% → v2 5.1%.

**함의와 선택지**
- (A) 데스크톱 두 백엔드를 한 세션으로 맞춘다: CPU 전체 재측정 27h(주말 1회). 맥은 MPS 9.1h + Mac CPU는 부분집합 보정(전체 102h는 불가) 또는 4~5일 전용 실행.
- (B) GPU만 v2, CPU는 v1을 쓰고 §3.1에 세션 드리프트(프로브 결과)를 명시한다. 소형 모델의 CPU 대 GPU 비가 최대 20% 왜곡될 수 있음을 한계로 적는다.
- (C) 본문 수치는 v1을 유지하고, v2는 R1-10·11·12 답변(워밍업 무영향, 원시값 분포, 환경 기록, 드라이버 변경에 따른 드리프트)에만 쓴다.
- 어느 쪽이든 §3.1에 드라이버·전원 옵션·백그라운드 서비스를 기재하고, 재측정 전 게임 런처·오버레이(Nexon Plug, PresentMon) 종료를 프로토콜에 넣는다. 다음 재측정 전에 전원 옵션 "고성능"과 NVIDIA "최대 성능 선호" 설정에서 소형 3구성 프로브를 먼저 돌려 3월 값이 재현되는지 본다(5분).

진단 데이터: `results/diag_20260915_cuda_small.json`(ANN 2·CNN 1 × 5회), `results/diag_20260915_cpu_small.json`(ANN 3·CNN 1 × 5회), 클럭 샘플 `results/diag_20260915_gpu_clock.csv`.
