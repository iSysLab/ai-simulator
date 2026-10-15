# DNN 실행 시간 예측 프레임워크 (ai-simulator)

PyTorch DNN 모델의 **학습·추론 시간**을 하드웨어 플랫폼별로 예측하는 연구 저장소입니다.
6개 모델 계열(ANN·CNN·ResNet·MobileNet·ViT·GAN)의 구성을 네 하드웨어 플랫폼(x86 CPU, CUDA GPU, Apple ARM CPU, Apple MPS)에서 실측하고,
ONNX 그래프에서 뽑은 구조·op-level 피처와 하드웨어 정보를 입력으로 **타깃별 XGBoost 회귀기**(학습 시간용·추론 시간용)를 학습합니다. 회귀기 하나가 네 플랫폼을 함께 예측합니다.

- 논문: Junsoo Lee, Honggeun Kim, Junyoung Heo, Hong Min. *Machine Learning Model for Predicting Execution Time of Deep Neural Networks on Diverse Hardware Platform.* RACS '26, ACM, 2026. https://doi.org/10.1145/3848591.3856853
- 제출 원고 국문판: [`paper/50_draft_v4.md`](paper/50_draft_v4.md)

## 주요 결과 (RACS 2026 논문 기준)

### 데이터: 158개 구성 × 4개 플랫폼 = 632개 실측값

| 장비 | 플랫폼 | 샘플 |
|---|---|---|
| 데스크톱 (AMD Ryzen 7 7800X3D) | x86 CPU | 158 |
| 데스크톱 (NVIDIA RTX 4060 Ti) | CUDA | 158 |
| 맥북 (Apple M4) | ARM CPU | 158 |
| 맥북 (Apple M4, 24GB 통합 메모리) | MPS | 158 |

학습 시간은 학습셋 1에폭, 추론 시간은 테스트셋 전체(분류 1만 장, GAN은 생성기로 약 5만 장 생성)의 벽시계 시간이며 10회 반복 평균입니다.

### 예측 성능 (XGBoost, nested CV, R²(log))

| 설정 | 학습 시간 | 추론 시간 |
|---|---|---|
| 전체 피처, 무작위 분할 | 0.988 | 0.989 |
| 전체 피처, 구성 단위 분할 (처음 보는 구성) | **0.991** | **0.990** |
| 하드웨어 피처 제외 (구성 단위) | 0.784 | 0.672 |
| 하드웨어 피처 대신 플랫폼 라벨 (구성 단위) | 0.990 | 0.990 |
| 분석적 추정 (학습 fold에서 효율 상수 보정) | 0.78 | 0.84 |
| FLOPs 단독 (구성 단위) | 0.558 | 0.468 |
| total_params 단독 (구성 단위) | 0.171 | 0.296 |

- 오차(구성 단위): MAPE 학습 시간 16.6%, 추론 시간(10 ms 이상) 20.9%. 10 ms보다 짧은 추론은 상대 오차가 큽니다.
- 플랫폼별(무작위 분할 예측을 플랫폼별로 나눈 값) 학습/추론: x86 CPU 0.988/0.989, CUDA 0.961/0.951, ARM CPU 0.990/0.988, MPS 0.983/0.979.
- 학습 시간 예측에서는 정적 메모리 쓰기 프록시(`total_op_memory_write`)가 네 플랫폼 모두에서 피처 중요도 1위였고, FLOPs는 5~24위였습니다(RandomForest). 배치 1 순전파 그래프에서 계산한 정적 추정값이라 실제 메모리 병목을 증명하지는 않습니다.
- 같은 구성이라도 플랫폼에 따라 시간이 수십 배 차이 나고 순위가 뒤집힙니다. MobileNet은 ARM CPU가 x86 CPU보다 평균 14배 느리지만, 같은 맥의 MPS는 ARM CPU보다 약 35배 빠릅니다. ANN은 반대로 MPS가 ARM CPU보다 약 2배 느립니다.

수치 출처: [`paper/v4_measured_j.json`](paper/v4_measured_j.json)의 `t6_protocols`(위 표), `group_errors`(MAPE), `t5_backend`(플랫폼별), `t4_nested`(회귀 모델 비교). 플랫폼 라벨 행과 튜닝한 Gradient Boosting(0.988/0.989)은 [`paper/v4_measured_k_before_schema_fix.json`](paper/v4_measured_k_before_schema_fix.json)의 `label_vs_descriptors`, `tuned_comparison`.

## 논문 작성 중 바뀐 점 (2026년 7~10월)

KIISE 원고에서 시작해 RACS 2026 제출(7월), 심사 대응과 camera-ready(9~10월)까지 논문을 쓰면서 바뀐 내용입니다. 괄호 안은 커밋입니다.

### 1. 데이터: 632셀 정본

- 데스크톱 결과 파일에 섞여 있던 M4 CPU 48행을 ARM CPU로 옮겨 평균하고, 기기를 알 수 없는 ViT 12행을 뺐습니다. x86 기록이 불완전한 ViT 2개 구성은 모든 플랫폼에서 제외해 **158개 구성 × 4개 플랫폼 = 632셀**이 됐습니다. 정본 로더는 `eval_fill_v4i.py`의 `load_clean()`입니다. (37f23f4)
- 실제 실행 배치는 모든 플랫폼에서 같습니다(학습 64, 분류 추론 1000). 데스크톱 행에 비어 있던 `batch_size`를 64로 채웠습니다. (37f23f4)
- Windows CUDA 수집기가 실제 CUDA 코어 수를 기록하도록 고쳤습니다. (be38b7f)
- 데스크톱 행에는 구조 필드 24개(`input_height`, `hidden_size`, `num_filters`, `embed_dim`, `latent_dim` 등)가 비어(0) 있어, 이 열만으로 데스크톱과 맥을 구별할 수 있었습니다. `load_clean()`이 같은 구성의 맥 행에서 이 필드를 채우도록 고쳤습니다(하드웨어·측정 필드는 복사하지 않음). 고친 뒤에는 무작위 분할과 하드웨어 피처 제외 조건 등 일부 수치가 바뀝니다. 논문 표는 고치기 전 데이터 기준이고, 전후 결과는 `paper/v4_measured_k_before_schema_fix.json`과 `paper/v4_measured_k.json`에 있습니다. (f054d66)

### 2. 피처: ONNX 그래프 기준으로 통일

- 피처 스키마 v4.0: 130열(수치 120 + 문자열 10) = 구조 69 · 입력 11 · 하드웨어 36 · op-level 14. v3.0(96)과 v2.0(111)을 합쳤습니다. (25715d4)
- 합칠 때 값이 없는 v2.0 이름이 남아 피처가 0으로 학습되던 문제를 고쳐, 데이터에 값이 있는 v3.0 이름(`gpu_cores`, `cpu_cores` 등)으로 통일했습니다. (ed89aa4)
- 160개 구성을 모두 ONNX(opset 17)로 내보내고, shape inference로 연산별 FLOPs와 메모리 읽기·쓰기 바이트를 뽑는 op-level 정적 추출기를 만들었습니다. 논문의 예측 결과는 모두 이 ONNX 피처 기준입니다. (2885562)

### 3. 평가 방식

- nested 교차검증(외부 5겹 평가, 내부 4겹 GridSearchCV 튜닝)으로 튜닝과 평가를 분리했습니다. (37f23f4)
- 같은 구성의 네 플랫폼 행을 같은 fold에 넣는 **구성 단위 분할**을 추가해, 처음 보는 구성에 대한 성능을 따로 보고합니다.
- 기준선 4개를 3그룹으로 비교합니다: FLOPs 단독, total_params 단독, 분석적 추정, 하드웨어 피처 제외. 하드웨어 피처를 플랫폼 라벨로 바꾼 조건과, XGBoost와 같은 그리드로 튜닝한 Gradient Boosting도 함께 봅니다. (e62bb2d, 99da81e, f054d66)
- R²(log)와 함께 MAPE와 ±20% 이내 비율을 공개하고, 10 ms 미만 추론과 GAN은 오차를 따로 적습니다.
- 외삽 실험: 계열 하나 빼기(LOFO), 플랫폼 하나 빼기(LOBO), 기기 하나 빼기(LODO). 처음 보는 계열(ANN·GAN)과 처음 보는 플랫폼은 잘 예측하지 못합니다.

### 4. 해석과 원고 수정

- "실행 시간은 FLOPs가 아니라 메모리 트래픽이 지배한다"를 "학습 시간 예측에서 정적 메모리 쓰기 프록시가 네 플랫폼 모두에서 가장 일관된 예측 인자다"로 낮췄습니다. 배치 1 정적 추정값이고, 빼도 성능이 크게 떨어지지 않기 때문입니다.
- 하드웨어 피처를 플랫폼 라벨로 바꿔도 성능이 같아서(0.990/0.990), 지금 모델은 하드웨어 정보를 "어느 플랫폼인가"를 구별하는 데 쓴다고 적었습니다. "처음 보는 하드웨어로 일반화한다"는 주장은 넣지 않았습니다.
- MobileNet이 ARM CPU에서 느린 원인은 라이브러리별 효과를 따로 재지 않았으므로 특정 라이브러리 탓으로 단정하지 않습니다.
- 초록에 구성 단위 결과(0.991/0.990)와 MAPE(학습 16.6%, 10 ms 이상 추론 20.9%), 짧은 추론의 큰 오차를 적었고, LOFO는 학습 시간 계열별 수치(CNN·ResNet·MobileNet·ViT 0.58~0.89, ANN·GAN 음수)로 적었습니다. 한계에 학습 타깃 log1p의 소형 셀 문제(아래 6번)를 한 문장 넣었습니다.

### 5. 측정 프로토콜 v2 (camera-ready 재측정)

- 모든 플랫폼·계열에 같은 워밍업(실제 배치로 학습 1스텝 + 평가 경로 1회), 반복별 원시값, 피크 메모리, 소프트웨어 환경과 측정 직전 부하를 기록하고, 유휴 대기(`--require-idle`)와 첫 반복 제외(`--discard-first`) 옵션을 넣었습니다. (f7cc918)
- 맥북 MPS 160개 구성을 다시 재서 표 3의 방향이 재현됨을 확인했습니다. (d3df6f3, 9c1258d)
- 데스크톱 CUDA는 3월보다 느려진 드리프트 때문에 재측정값을 쓰지 않았습니다. 주된 원인은 GPU 전원 상태(낮은 부하에서 최저 클럭 고정)였고, 전원 설정을 바꾼 뒤에도 3월보다 1.17배 느려 완전히 회복되지는 않았습니다. (3bd0eb6, 5e63b98, d07af59)
- 논문 표의 632셀은 3월(v1) 측정값입니다.

### 6. 논문 이후 추가 분석 (논문 표에는 반영 안 함)

- **학습 타깃 log1p(초)의 문제**: 0.1초 아래에서는 log1p(y)≈y라 10 ms 미만 셀의 오차가 손실에 거의 잡히지 않습니다. 타깃을 log(y)로 바꾸면(모델·피처·데이터 그대로) 추론 MAPE가 58%→16%, 10 ms 미만은 222%→21%로 줄고 R²(log)는 유지됩니다. 앙상블 후보 6종보다 이 한 줄 변경이 나았습니다. → [`paper/71_log_target_findings.md`](paper/71_log_target_findings.md) (9ab9ae2)
- **외부 기기 4대**(GTX 1050, Intel i3-8100, RTX 3060, Intel i7-10700F)를 v2 프로토콜로 재서 처음 보는 하드웨어 예측을 시험했습니다. 모델은 하드웨어 사양을 속도의 연속적인 축으로 쓰지 않고 기기 구별(예: VRAM 분기)에만 써서, 사양만 바꿔서는 새 기기를 맞히지 못합니다. GTX 1050은 계열당 4개(24구성)를 재서 학습에 넣으면 기존 정확도 수준이 됐습니다. → `paper/72`~`75` (d6476a0 ~ f9b0efd)
- **GPU 3대 leave-one-GPU-out**: 트리 모델은 TFLOPS를 쓰지 않아 양끝 GPU를 외삽하지 못하고, 학습 시간은 구성별 log-TFLOPS 직선 보간이 더 정확했습니다. 같은 구성을 세 GPU에서 비교해 오버헤드·메모리·연산 세 체제를 나누고, 호스트·TFLOPS·대역폭으로 된 4-파라미터 식이 GPU 사이에 전이되는지 시험했습니다. → [`paper/76_logo_gpu_results.md`](paper/76_logo_gpu_results.md), [`paper/77_roofline_three_gpu.md`](paper/77_roofline_three_gpu.md) (1657574, 244d16e)

## 설치

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_env.py              # 환경 점검 (--json: 환경 정보를 JSON으로 출력)
```

검증한 버전: Python 3.13.5, torch 2.10.0, scikit-learn 1.8.0, xgboost 3.2.0, numpy 2.4.3, scipy 1.18.1, onnx 1.20.1. scikit-learn은 1.6 이상이 필요합니다(`GroupKFold(shuffle=...)`). 그림 스크립트(`make_fig_pipeline.py`, `make_scatter.py`)는 macOS의 AppleGothic 글꼴을 쓰고, 원고 빌드(`paper/build_pdf.py`, `paper/build_hwp_docx.py`)에는 pandoc이, PDF 출력에는 Chrome이 필요합니다(경로는 스크립트 맨 위에 있음).

## 논문 수치 재현

모든 명령은 저장소 루트에서 실행합니다. 원시 측정값(2026년 3월, v1 프로토콜)은 저장소에 있고, 중간 산출물 두 가지는 gitignore 대상이라 먼저 만들어야 합니다.

```bash
# 1. op-level 피처 보강: results/benchmark_results{,_mac}.json → *_enriched.json
python enrich_oplevel.py
# 2. 160개 구성을 ONNX(opset 17)로 내보내기 → results/onnx_configs/ (+ manifest.json)
python export_onnx_configs.py
# 3. 632셀 정리, 표 3(플랫폼 간 비율)·9(중요도)·10(절제) → paper/v4_measured_i.json
python eval_fill_v4i.py
# 4. nested CV, 표 6(모델 비교)·7(플랫폼별)·8(기준선), 오차 → paper/v4_measured_j.json
python eval_fill_v4j.py
# 5. camera-ready 보충 실험 → paper/v4_measured_k.json → paper/v4k_tables.md
python eval_fill_v4k.py          # --only precision_check,... 로 일부 절만 실행
python make_tables_v4k.py
# 6. (논문 이후) log 타깃·앙상블 비교 → paper/v4_measured_l.json → paper/v4l_tables.md
python eval_fill_v4l.py
python make_tables_v4l.py
# 7. 그림 1과 국문 원고 PDF
python make_fig_pipeline.py
python paper/build_pdf.py 50_draft_v4.md
```

다시 돌리면 논문 표와 값이 다를 수 있습니다.

- 2026-09-16에 `load_clean()`에 데스크톱 행 구조 필드 보정이 들어갔습니다("논문 작성 중 바뀐 점" 1번). 논문 표는 그 전에 만든 값으로, 커밋된 `paper/v4_measured_j.json`(7월 13일 실행)과 `paper/v4_measured_k_before_schema_fix.json`에 있습니다. 보정 후 값은 `paper/v4_measured_k.json`의 `precision_check`, `errors_by_regime`에 있습니다.
- 같은 코드라도 실행 기기에 따라 소수 넷째 자리가 달라졌습니다(예: 같은 조건의 추론 R²(log) 0.9891과 0.9899).
- `eval_fill_v4k.py`, `eval_fill_v4l.py`는 실행한 절만 JSON에 덮어써서, 한 파일 안에 실행 시점이 다른 절이 섞일 수 있습니다.

## 새로 측정하기

```bash
python check_env.py --json > paper/env_<host>.json
python run_benchmark.py --device cuda --repeats 10 --require-idle --tag remeasure_v2 \
    --output results/remeasure_cuda_v2.json
# 빠른 시험: 대표 구성 9개, 3회 반복
python run_benchmark.py --quick --output results/quick_test.json
```

- **`--output`을 꼭 지정하세요.** 기본값이 논문 원시 데이터 `results/benchmark_results.json`이고, 기존 파일에 결과를 이어 붙입니다.
- 측정 프로토콜 v2: 워밍업은 항상 켜져 있습니다. 첫 반복 제외 `--discard-first`, 유휴 대기 `--require-idle`(`--idle-cpu-max`, `--idle-gpu-max`, `--idle-wait`), 부분 집합 `--subset stratified:N|every:K|names:...`, `--batch-size`, `--tag`.
- 재측정 절차는 [`paper/61_remeasure_runbook.md`](paper/61_remeasure_runbook.md), 외부 기기 측정은 [`paper/72_external_backend_runbook.md`](paper/72_external_backend_runbook.md)를 따릅니다.

## 실행 시간 예측 (`predict_v4.py`)

```bash
# 632셀로 학습 → results/trained_models_v4/bundle_v4.pkl
python predict_v4.py --train
# 플랫폼 목록: Desktop CPU, CUDA, Mac CPU, MPS
python predict_v4.py --list-backends
python predict_v4.py --onnx model.onnx --backend MPS --dataset mnist
```

- 출력은 학습 1에폭 시간과 테스트셋 추론 시간(초)입니다.
- 기본 추정기는 log(y) 타깃 XGBoost(`--estimator log`, 논문 이후 개선판)입니다. 논문과 같은 log1p 타깃 XGBoost는 `--estimator xgb`로 학습합니다.
- 학습에는 재현 1~2단계 산출물과 `paper/v4_measured_l.json`의 하이퍼파라미터가 필요합니다. 모델 번들은 저장소에 넣지 않습니다.
- 160개 구성 manifest에 없는 ONNX는 구성 메타데이터가 0으로 들어가므로 `--config-json`으로 넘겨 주세요.

## 저장소 구조

```
ai-simulator/
├── benchmark/                      # 패키지
│   ├── models/                     # 모델 6종·레지스트리 (MNIST 4종, CIFAR-10 2종)
│   ├── configs/generator.py        # 160개 구성 (ANN 42·CNN 60·ResNet 18·
│   │                               #   MobileNet 20·ViT 12·GAN 8)
│   ├── features/
│   │   ├── merged_schema.py        # v4 스키마 130열 (수치 120)
│   │   ├── onnx_oplevel.py         # ONNX shape inference 기반 op-level 정적 피처
│   │   ├── op_profiler.py          # PyTorch 훅 기반 op-level 피처 (enrich에서 사용)
│   │   └── extractor.py, onnx_extractor.py   # 구판 96피처
│   ├── platform/                   # 하드웨어 감지, software_env.py(버전·부하)
│   ├── runner/                     # 장치·데이터 사전 적재·측정 루프(v2 프로토콜)
│   ├── predictor/ensemble.py       # log 타깃 회귀기 등 예측기 6종
│   └── results/io.py               # 증분 저장·재개
├── run_benchmark.py, check_env.py  # 측정, 환경 점검
├── enrich_oplevel.py               # 재현 1단계
├── export_onnx_configs.py          # 재현 2단계
├── train_merged.py                 # NUMERIC(수치 120열)·enrich() 정의 (v4 공용)
├── eval_fill_v4i.py ~ v4l.py       # 논문 검증 실험 (i: 632셀·표 3/9/10,
│                                   #   j: nested CV·표 6/7/8, k: camera-ready,
│                                   #   l: log 타깃)
├── make_tables_v4k.py, make_tables_v4l.py, make_fig_pipeline.py
├── predict_v4.py                   # 예측 CLI
├── eval_external_*.py, eval_logo_gpu.py   # 외부 기기 분석
├── eval_fill_v4.py, v4b ~ v4h.py   # 정리 전(698행) 실험, 일부는 v4i·v4j가 import
├── train_predictor.py, visualize_results.py   # 레거시
├── predict_from_onnx.py, export_onnx.py       # 레거시
├── paper/
│   ├── 50_draft_v4.md (+.pdf)      # RACS 2026 제출 원고 국문판
│   ├── 60~61_*.md                  # camera-ready 재측정 설계·런북
│   ├── 70~77_*.md                  # 논문 이후 분석 (log 타깃, 외부 기기, GPU 3대)
│   ├── v4_measured_*.json          # 논문·분석 수치의 실행 근거
│   └── build_pdf.py, build_hwp_docx.py   # 원고 빌드
└── results/
    ├── benchmark_results.json      # 데스크톱 원시 378행: CUDA 160, x86 CPU 218
    │                               #   (데스크톱 158 + 섞인 M4 48 + 기기 불명 ViT 12)
    ├── benchmark_results_mac.json  # 맥북 원시 320행: ARM CPU 160, MPS 160
    ├── remeasure_{cuda,mps}_v2.json    # v2 재측정 각 160행
    ├── external_*.json             # 외부 기기: GTX 1050 55행, i3-8100 42행,
    │                               #   RTX 3060 58행, i7-10700F 42행
    ├── onnx_samples/               # 구판 ONNX 샘플 14개 (opset 11)
    └── trained_models*/, figures/  # 구판 예측 모델·그림
```

## 알려진 제약

- 새로 클론한 저장소에서는 `results/*_enriched.json`과 `results/onnx_configs/`를 먼저 만들어야 합니다(재현 1~2단계). 일부 구성을 다시 만들어 같은 피처가 나오는 것을 확인했습니다(torch 2.10).
- PyTorch 훅 경로(`enrich_oplevel.py`)에서는 GAN의 op-level 피처가 0으로 남습니다. 논문 파이프라인은 ONNX에서 다시 계산하므로 영향이 없고, PyTorch 경로와 비교한 결과에만 해당합니다.
- 일부 스크립트는 main 가드가 없어 import만 해도 결과 파일을 씁니다: `eval_fill_v4b/d/g.py`, `make_fig_pipeline.py`, `make_tables_v4k.py`.
- `pyproject.toml`로는 패키지 설치가 되지 않습니다. `requirements.txt`를 쓰세요.
- 구판 `export_onnx.py`의 기본 출력 폴더는 저장소의 `results/onnx_samples/`입니다(`--output-dir`로 바꿀 수 있음).

## 피처 스키마 v4.0 (130열)

정의는 [`benchmark/features/merged_schema.py`](benchmark/features/merged_schema.py)의 `MERGED_FEATURE_COLUMNS`이고, 학습에는 문자열 10열을 뺀 수치 120열(`NUMERIC_FEATURE_COLUMNS`, `train_merged.py`의 `NUMERIC`과 같음)을 씁니다.

| 범주 | 열 수 | 대표 피처 | 출처 |
|---|---|---|---|
| 모델 구조 | 69 | `total_params`, `num_conv_layers`, `max_width` | ONNX 그래프 31 + 구성 메타데이터 38 |
| 입력 데이터 | 11 | `input_height`, `batch_size` | 벤치마크 메타데이터 |
| 하드웨어 | 36 | `cpu_cores`, `gpu_memory_gb`, `ram_total_gb` | 벤치마크 메타데이터(자동 감지) |
| op-level 분해 | 14 | `total_op_memory_write`, `flops_ratio_*` | ONNX 그래프 |

- 하드웨어 36열 중 수치 30열이 "하드웨어 피처 제외" 조건에서 빠지는 열입니다(`batch_size`는 유지).
- op-level 메모리 읽기·쓰기는 배치 1 순전파 그래프에서 계산한 바이트 수(읽기 = 입력+가중치, 쓰기 = 출력)로, 실제 하드웨어 메모리 트래픽이 아닙니다.
- `memory_bytes`(= `total_params` × 4)는 구조로 정해지는 값이라 피처와 타깃에서 모두 뺐습니다.

## 레거시: v2/v3 파이프라인

논문 이전의 PyTorch 훅 기반 96피처(v3.0) 파이프라인입니다. `run_benchmark.py`는 지금도 측정 도구로 쓰고("새로 측정하기"), 나머지는 기록용으로 남겨 두었습니다.

```bash
# 장치별 분리 모델 (--features core|full|merged)
python train_predictor.py --features full --cv 10
# 구판 그림 9종
python visualize_results.py --input-mac results/benchmark_results_mac.json \
    --output-dir results/figures_new
# 구판 ONNX 예측 (인자가 없으면 results/onnx_samples 전체)
python predict_from_onnx.py model.onnx
```

`train_predictor.py --save-models`는 `results/trained_models/`를, `visualize_results.py`는 기본값으로 `results/figures/`를 덮어씁니다.

`results/figures/`의 그림과 `results/trained_models/`의 모델은 2026년 3월의 정리 전 데이터로 만든 것이라 논문 수치와 다릅니다.

main 브랜치에는 2026년 4월 통합 때의 레거시 폴더도 함께 있습니다: `past/`(단계별 실험·데이터·모델), `benchmark/dal/`·`benchmark/support/`(`past/` 코드가 사용), `docs/`. 당시의 `scripts/` 실행 스크립트는 최상위 스크립트로 대체돼 main에서 뺐고, 4월 통합 상태 전체는 `integration` 브랜치에 있습니다.

## 팀원별 기여

### ijunsoo — 프레임워크 설계 + 논문 v4 파이프라인

- `benchmark/` 패키지 구조, 데코레이터 기반 모델 레지스트리, Feature Schema v3.0(96)·v4.0(130)
- `benchmark/platform/` 하드웨어 자동 감지, Zero-Config 실행(`check_env.py`)
- ONNX 피처 파이프라인, 632셀 데이터 정리, nested CV 검증 실험, 측정 프로토콜 v2, 외부 기기 분석, 논문(KIISE·RACS 2026)

### dal-merge (달현) — XGBoost + 하드웨어 피처

- XGBoost + GridSearchCV, log1p 타깃 변환, 하드웨어 피처 도입, 장치별 모델 분리

### khg9859 (홍근) — Transformer/GAN + ONNX + Op-Level

- ViT·GAN(G+D) 구현, ONNX export/예측 파이프라인, op-level 프로파일러, Attention FLOPs 보정

### hong (홍) — Windows/CUDA 지원 + 확장 HW 피처

- Windows CUDA 하드웨어 감지(wmic/PowerShell), 33개 HW 피처 스키마, 크로스 플랫폼 폰트
