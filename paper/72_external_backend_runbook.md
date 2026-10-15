# 외부(5번째) 백엔드 측정·평가 런북 — 다른 컴퓨터에서 Claude에게 시킬 작업

작성 2026-09-19, 브랜치 `ensemble`. 목적: 리뷰어 R1-5("하드웨어 기술자 = 백엔드 라벨")·R2("미지 하드웨어 일반화")에 대한 첫 실측 답.
632셀에서는 백엔드 4개 = 기기 4대라 하드웨어 기술자(코어 수·TFLOPS 등)가 백엔드마다 상수였고, 라벨 1개와 정보량이 같았다.
**같은 장치 종류(예: CUDA)에 다른 GPU가 하나 들어오면 이 동치가 처음 깨진다** — 4백엔드로 학습한 모델이 새 기기를 얼마나 맞히는지,
그리고 기술자 값이 "닮은 기기 실측 복사" 같은 무학습 기준선보다 나은지를 본다.

이 문서는 **다른 컴퓨터에서 Claude Code에 그대로 붙여 넣는 지시서**다. 각 단계의 명령을 순서대로 실행하고, §7의 보고 양식으로 결과를 정리한다.

---

## 0. Claude에게 주는 전제

- 저장소: `https://github.com/iSysLab/ai-simulator.git`, 브랜치 **`ensemble`**. `main`·`racs-cr`·`ijunsoo`에는 아무것도 커밋하지 않는다.
- 결과는 새 브랜치 `external-<host>`(host = 기기 별명, 예: `external-gtx1650`)에 커밋·푸시한다. `results/*.json`·`paper/*.json`·`*.log`는 gitignore라 `git add -f`가 필요하다.
- 측정 중에는 기기를 다른 용도로 쓰지 않는다. 측정 전 유휴 확인은 `--require-idle`이 한다.
- 코드는 고치지 않는다. 오류가 나면 멈추고 오류 전문과 환경(§1 출력)을 보고한다. 단, §2의 "하드웨어 기술자 확인"에서 값이 비어 있으면 그 사실만 보고하고 진행한다(코드 수정 금지).
- 모든 명령은 저장소 루트에서. Windows는 `.venv\Scripts\python`, Mac/Linux는 `.venv/bin/python`. 아래는 Linux/Mac 표기이며 Windows는 바꿔 쓴다.

## 1. 환경 준비 (10분)

```bash
git clone https://github.com/iSysLab/ai-simulator.git
cd ai-simulator
git checkout ensemble
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch torchvision numpy scikit-learn xgboost onnx joblib psutil scipy matplotlib
python check_env.py
```

- `check_env.py`가 CUDA를 보여야 한다(`디바이스: CPU, CUDA (<GPU 이름>, <VRAM>GB)`). MPS 기기면 MPS. CPU만 있어도 §3의 CPU 경로로 진행 가능.
- torch 버전은 다른 기기와 달라도 된다(환경은 §2에서 기록되고 논문에 명시한다). 단 **2.x**여야 한다.
- 데이터 `data/`는 첫 실행에서 자동 다운로드(약 180 MB). 미리 받으려면:
  ```bash
  python -c "from benchmark.runner.data import MNISTDataManager, CIFAR10DataManager; MNISTDataManager(); CIFAR10DataManager(); print('ok')"
  ```
- 논문 파이프라인 재현에 필요한 두 파일이 gitignore라 **저장소에 없다**: `results/benchmark_results_enriched.json`, `results/benchmark_results_mac_enriched.json`, 그리고 `results/onnx_configs/`(ONNX 160개 + manifest). 이 셋은 §5 분석에만 필요하고 §3 측정에는 필요 없다. 분석은 이 파일들이 있는 기기(맥북/데스크톱)에서 해도 된다 — §5 참조.

## 2. 환경·하드웨어 기술자 기록 (2분)

```bash
python check_env.py --json > paper/env_<host>.json
```

파일을 열어 `hardware.cuda`(또는 `mps`/`cpu`)에서 아래 값이 **0이 아닌지** 확인하고 보고한다:

| 키 | 의미 | 4060 Ti 참고값 |
|---|---|---|
| `accelerator_name` | GPU 이름 | NVIDIA GeForce RTX 4060 Ti |
| `gpu_core_count` | CUDA 코어 수(SM × 코어/SM) | 4352 |
| `gpu_clock_ghz` | 부스트 클럭 | 3.1 |
| `tflops_fp32` | 코어 × 클럭 × 2 | 26.98 |
| `gpu_memory_gb` | VRAM | 8.0 |
| `peak_bandwidth_gbs` | 메모리 대역폭 | 0.0 (4060 Ti도 미수집 — 비어 있어도 정상) |
| `gpu_compute_capability` | CC | 8.9 |

`gpu_core_count`·`tflops_fp32`가 0이면 `nvidia-smi`가 없거나 torch가 CUDA를 못 본 것이다. `nvidia-smi` 출력과 `python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda)"`를 보고에 첨부한다. **값이 비어도 측정은 진행한다** — 이 경우 §5 분석에서 기술자 비교는 못 하고 "라벨 없는 미지 백엔드" 예측만 본다.

## 3. 측정 (GPU 1~3시간, 저사양이면 더)

계열별 균등 60구성 × 3회 반복, 프로토콜 v2(통일 워밍업·반복별 원시값·피크 메모리·부하 기록).

```bash
python run_benchmark.py --device cuda --subset stratified:60 --repeats 3 --require-idle --tag external --output results/external_<host>.json
```

- MPS 기기면 `--device mps`, CPU만 있으면 `--device cpu`(60구성 × 3회는 CPU에서 수 시간 — `stratified:40`으로 줄여도 된다).
- 중단되면 같은 명령에 `--resume`을 붙여 이어간다.
- `--require-idle`은 CPU 20%·GPU 10% 이하가 될 때까지 최대 10분 대기한다. 계속 대기하면 백그라운드 프로그램을 끄고 다시.
- 예상 시간: 4060 Ti에서 160구성 × 10회가 2.4시간이었다. 60구성 × 3회는 그 1/9 ≈ 16분이지만 저사양 GPU는 대형 구성(CNN f128, MobileNet w1.5, ResNet)에서 몇 배 느릴 수 있다. **3시간을 넘기면** `Ctrl+C`로 멈추고 `--resume` 없이 그대로 §4로 간다(부분 결과도 쓸 수 있다).
- 시간이 남으면 CPU도(선택, 계열별 40구성 × 3회):
  ```bash
  python run_benchmark.py --device cpu --subset stratified:40 --repeats 3 --require-idle --tag external --output results/external_<host>_cpu.json
  ```

## 4. 측정 결과 확인 (1분)

```bash
python -c "import json; r=json.load(open('results/external_<host>.json')); x=r[0]; print(len(r), x['device'], x['protocol'], len(x['train_times']), x['peak_mem_mb'], x['sw_torch'], x.get('load_cpu_pct'), x.get('load_gpu_util_pct'), x['gpu_cores'], x['tflops_fp32'])"
```

확인할 것: 행 수(60 또는 중단 시점까지), `protocol == 'v2-uniform-warmup'`, `train_times` 길이 3, `peak_mem_mb > 0`, `gpu_cores`·`tflops_fp32`가 §2와 같은 값. 계열별 행 수도 보고:

```bash
python -c "import json,collections; r=json.load(open('results/external_<host>.json')); print(collections.Counter(x['model_type'] for x in r))"
```

## 5. 분석 (10분) — 632셀 파일이 있는 기기에서

`eval_external_backend.py`는 4백엔드 632셀로 log(y) 타깃 XGBoost를 학습해 외부 기기를 예측하고, 무학습 기준선 3개와 비교한다. 632셀 파일(`results/*_enriched.json`, `results/onnx_configs/`)이 없는 기기에서는 돌지 않으므로, 측정 결과 두 파일(`results/external_<host>.json`, `paper/env_<host>.json`)을 §6으로 푸시한 뒤 **맥북 또는 데스크톱에서** 실행한다.

```bash
python eval_external_backend.py --external results/external_<host>.json --env paper/env_<host>.json --name <host>
python eval_external_backend.py --external results/external_<host>.json --env paper/env_<host>.json --name <host>_norefresh --no-refresh-hw
```

두 번 돌리는 이유 — **트리 모델은 기술자 값을 보간하지 못한다.** 632셀에서 `tflops_fp32`는 CUDA 26.98·MPS 5.0·CPU 0 세 값뿐이라, 새 GPU의 값(예: 8)은 트리 분기에서 CPU 쪽이나 CUDA 쪽 어느 한쪽으로 떨어진다. 시뮬레이션(MPS를 외부로 간주)에서 기술자 갱신 시 R²(log)가 0.73 → −0.40으로 무너진 것이 이 현상이다(`paper/external_sim_MPS.json`). 따라서:
- `--no-refresh-hw`(v1 데이터의 빈 기술자 그대로) = "라벨 없는 미지 백엔드" 예측 — 사실상 LOBO.
- 갱신 모드 = 기술자 값을 실제로 쓰는 예측 — 새 GPU가 4060 Ti와 CPU 사이 어디로 분기되는지에 따라 결과가 크게 갈린다. **나쁘게 나와도 그것이 결과다**(기술자 2~3개 값으로는 전이 불가라는 정량 증거).

출력에서 볼 것(타깃마다):
- `기술자 모델`: R²(log), MAPE, MdAPE, ±20%, Spearman(순위 상관 — 절대값은 틀려도 "어느 구성이 더 느린가"는 맞히는지).
- `baseline_nearest_copy`: 같은 구성의 4060 Ti 실측을 그대로 예측값으로 쓴 것. **기술자 모델이 이걸 못 이기면 기술자는 전이 정보를 못 담는다.**
- `baseline_nearest_spec_scaled`: 4060 Ti 실측 × (26.98 / 새 GPU TFLOPS). 계산량 비례 가정. 소형 모델(launch-bound)에서는 과대 추정할 것.
- `baseline_all_backend_mean`: 4백엔드 로그 평균.
- 계열별 MAPE: 어느 계열에서 무너지는지.

## 6. 결과 회수

```bash
git checkout -b external-<host>
git add -f results/external_<host>.json paper/env_<host>.json
git add -f results/external_<host>_cpu.json          # CPU도 쟀으면
git commit -m "[add] 외부 백엔드 측정 — <GPU 이름>, 60구성×3회 v2 프로토콜"
git push -u origin external-<host>
```

분석을 다른 기기에서 했으면 그 기기에서 `paper/external_<host>.json`·`paper/external_<host>_norefresh.json`을 같은 브랜치에 추가 커밋.

## 7. 보고 양식 (Claude가 마지막에 출력할 것)

```
기기: <GPU 이름>, VRAM <GB>, CUDA 코어 <n>, TFLOPS <x>, torch <ver>, CUDA <ver>, OS <ver>
측정: <n>구성 × 3회, <시작~종료 시각>, 중단 여부, --require-idle 대기 횟수(idle_wait_s > 0 행 수)
확인: protocol v2 전부 / train_times 3개 / peak_mem 범위 / load_cpu 범위 / 계열별 행 수
4060 Ti 대비 (같은 구성 실측 비, 중앙값): 학습 <x>배, 추론 <x>배, 소형(<1 s)만 <x>배, 대형(>10 s)만 <x>배
분석 (맥/데스크톱에서 돌렸으면):
  갱신 모드     — 학습 R²/MAPE/Spearman, 추론 R²/MAPE/Spearman, 기준선 3개 각각
  no-refresh    — 같은 항목
  기술자 모델이 nearest_copy를 이긴 타깃: <없음 / 학습 / 추론 / 둘 다>
문제·이상: <있으면>
```

"4060 Ti 대비" 행은 아래로 계산한다(632셀 파일 없이도 됨 — `results/remeasure_cuda_v2.json`은 저장소에 있다):

```bash
python -c "
import json,numpy as np
ext={r['model_name']:r for r in json.load(open('results/external_<host>.json'))}
ref={r['model_name']:r for r in json.load(open('results/remeasure_cuda_v2.json'))}
common=[n for n in ext if n in ref]
for t in ['avg_train','avg_infer']:
    ratio=np.array([ext[n][t]/ref[n][t] for n in common]); yt=np.array([ref[n]['avg_train'] for n in common])
    print(t,'n',len(common),'중앙값 %.2f'%np.median(ratio),'소형(<1s) %.2f'%np.median(ratio[yt<1]) if (yt<1).any() else '','대형(>10s) %.2f'%np.median(ratio[yt>10]) if (yt>10).any() else '')
"
```

## 8. 예상 문제와 대응

| 증상 | 대응 |
|---|---|
| `torch.cuda.is_available()` False | 드라이버/CUDA 런타임 불일치. `pip install torch --index-url https://download.pytorch.org/whl/cu121`(또는 cu118) 후 재확인. 안 되면 CPU 경로로 진행하고 보고 |
| `--require-idle` 10분 대기 후 "부하 상태로 진행" 경고 | 백그라운드(브라우저·동기화·게임 런처) 종료 후 `--resume`. 경고가 반복되면 그대로 진행하고 보고에 명시 |
| VRAM 부족(OOM) | 저사양 GPU에서 CNN f128·MobileNet w1.5·ResNet 대형 구성에서 날 수 있다. 러너는 해당 구성을 건너뛰지 않으므로 중단됨 → `--resume`은 같은 구성에서 다시 죽는다. 이 경우 `--subset stratified:40`으로 다시 시작하고 OOM 난 구성 이름을 보고 |
| 첫 구성이 비정상적으로 느림 | v2 워밍업이 있어도 계열 첫 구성의 첫 반복은 20~40% 높을 수 있다(맥 관찰). 원시값(`train_times`)이 저장되므로 분석에서 처리. 조치 불필요 |
| 데이터 다운로드 실패(CIFAR 토론토 서버 느림) | `https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz`를 받아 `data/`에 두고 md5 `c58f30108f718f92721af3b95e74349a` 확인 |
| 결과 파일 저장 오류 | `results/`가 없으면 생성. 러너는 구성마다 원자적 저장(`os.replace`)이라 부분 결과가 남는다 |

## 9. 이 실험이 논문에 주는 것 (Claude는 몰라도 됨, 사람용)

- 기술자 모델이 nearest_copy를 이기면: "하드웨어 기술자가 백엔드 정체성 이상의 정보를 담는다"는 첫 증거 → R1-5 답변에 한 행, R2 "미지 하드웨어"에 실측 답.
- 못 이기면: "기술자 값이 백엔드당 하나뿐인 데이터로는 트리가 보간할 수 없다"는 정량 한계 → Limitations. 기술자 축을 채우려면 기기 수가 늘어야 한다는 후속 연구 근거.
- 어느 쪽이든 camera-ready(10/7)에는 한 문장 이상 넣기 어렵고, 후속 논문 재료다.
