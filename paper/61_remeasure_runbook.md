# GPU 재측정 런북 — 데스크톱(CUDA)과 맥북(MPS) 동시 실행

대상: RACS 2026 camera-ready 심사 대응 R1-10(워밍업 프로토콜 통일), R1-11(반복별 원시값), R1-12(환경 기록), R2(피크 메모리).
설계 문서: `paper/60_racs_camera_ready_plan.md`. 코드 변경: `benchmark/runner/experiment.py`, `benchmark/runner/device.py`, `benchmark/platform/software_env.py`, `run_benchmark.py`, `check_env.py`.

## 무엇이 달라졌나 (측정 프로토콜 v2)

| 항목 | v1 (기존 632셀) | v2 (이번 재측정) |
|---|---|---|
| 워밍업 | GPU의 비-GAN 모델만, 배치 1 no_grad 순전파 1회 | 모든 백엔드·계열, 실제 배치로 학습 스텝 1회 + 평가 순전파 1회 (GAN은 D+G 스텝) |
| 반복 | 10회, 평균·표준편차만 저장 | 10회, 반복별 원시값 `train_times`/`infer_times`도 저장 |
| 피크 메모리 | 없음 | `peak_mem_mb` (CUDA max_memory_allocated, MPS driver_allocated_memory) |
| 환경 | 미기록 | `sw_*` 필드(Python, torch, CUDA, cuDNN, oneDNN, 드라이버, OS) |
| 유휴 증빙 | 없음 | 구성마다 시작 직전 `load_cpu_pct`, `load_gpu_util_pct` 기록, `--require-idle`면 대기 |

CPU 두 백엔드는 재측정하지 않는다(27h, 102h). GPU 두 백엔드만 v2로 바꾸고 분석 단계에서 병합한다.

## 0. 공통 준비 (두 기기 모두)

1. 코드 받기. 브랜치 이름은 전달받은 것을 쓴다(예: `racs-cr`).
   ```bash
   git fetch origin
   git checkout racs-cr
   ```
2. 환경 기록. 파일 이름은 기기별로.
   ```bash
   python check_env.py --json > paper/env_desktop.json    # 데스크톱
   python check_env.py --json > paper/env_mac.json        # 맥북
   ```
3. 기기를 유휴 상태로 만든다. 게임·영상·백업·클라우드 동기화·브라우저 탭 정리. 측정 중에는 기기를 쓰지 않는다.
   - 데스크톱: 전원 옵션 고성능, 절전·화면 끄기 해제(화면은 꺼져도 되지만 절전은 안 됨).
   - 맥북: 전원 연결, 아래 명령처럼 `caffeinate -i`로 잠자기 방지. 덮개를 덮지 않는다.
4. 데이터: `data/`에 MNIST와 CIFAR-10이 있어야 한다. 없으면 첫 실행에서 자동 다운로드(약 180MB).
5. 러너는 구성 하나가 끝날 때마다 저장한다. 중단되면 같은 명령에 `--resume`을 붙여 이어간다. 이어갈 때도 기기는 유휴여야 한다.

## 1. 데스크톱 · CUDA (약 2.4시간)

```bash
.venv\Scripts\python run_benchmark.py --device cuda --repeats 10 --require-idle --tag remeasure_v2 --output results/remeasure_cuda_v2.json
```

중단 후 재개:

```bash
.venv\Scripts\python run_benchmark.py --device cuda --repeats 10 --require-idle --tag remeasure_v2 --output results/remeasure_cuda_v2.json --resume
```

## 2. 맥북 · MPS (약 9.1시간, 야간 권장)

```bash
caffeinate -i python run_benchmark.py --device mps --repeats 10 --require-idle --tag remeasure_v2 --output results/remeasure_mps_v2.json
```

중단 후 재개:

```bash
caffeinate -i python run_benchmark.py --device mps --repeats 10 --require-idle --tag remeasure_v2 --output results/remeasure_mps_v2.json --resume
```

맥에서 `--require-idle`은 CPU 사용률만 본다(MPS 사용률은 sudo 없이 읽을 수 없다).

## 3. 끝난 뒤 확인

- 행 수: CUDA 160, MPS 160 (ViT 2구성 제외는 분석 단계에서 처리).
- 각 행에 `train_times`(10개), `infer_times`(10개), `peak_mem_mb`, `protocol = v2-uniform-warmup`, `sw_torch`, `load_cpu_pct`가 있다.
- GAN 8행도 같은 프로토콜로 측정됐다(`protocol` 값 동일).
- `idle_wait_s`가 0이 아닌 행은 시작 전 대기가 있었던 구성이다. 대기 후에도 부하가 남았는지 `load_*` 값으로 본다.

빠른 확인:

```bash
python -c "import json; r=json.load(open('results/remeasure_cuda_v2.json')); print(len(r), r[0]['protocol'], len(r[0]['train_times']), r[0]['peak_mem_mb'], r[0]['sw_torch'])"
```

## 4. 결과 회수 (두 기기 모두)

`results/*.json`은 gitignore 대상이라 `-f`로 추가한다.

```bash
git add -f results/remeasure_cuda_v2.json paper/env_desktop.json     # 데스크톱
git add -f results/remeasure_mps_v2.json paper/env_mac.json          # 맥북
git commit -m "[add] GPU 재측정 v2 결과 (통일 워밍업·원시값·피크 메모리)"
git push
```

같은 브랜치에 두 기기가 각각 push하면 나중에 push하는 쪽은 `git pull --rebase` 후 push한다. 파일이 다르므로 충돌은 없다.

## 5. 선택 · 배치 스윕 (데스크톱 CUDA, 약 30분, R2 고정 배치 지적)

계열별 균등 30구성을 학습 배치 32, 128, 256으로 3회씩. 64는 본 재측정의 같은 30구성을 쓴다. 아래를 하나씩 실행한다.

```bash
.venv\Scripts\python run_benchmark.py --device cuda --subset stratified:30 --repeats 3 --batch-size 32  --require-idle --tag batch_sweep --output results/batch_sweep_cuda_b32.json
```

```bash
.venv\Scripts\python run_benchmark.py --device cuda --subset stratified:30 --repeats 3 --batch-size 128 --require-idle --tag batch_sweep --output results/batch_sweep_cuda_b128.json
```

```bash
.venv\Scripts\python run_benchmark.py --device cuda --subset stratified:30 --repeats 3 --batch-size 256 --require-idle --tag batch_sweep --output results/batch_sweep_cuda_b256.json
```

## 6. 선택 · 외부 5번째 백엔드 (다른 기기 1대, 1~2시간, R1-5·R2 미지 하드웨어)

연구실의 다른 노트북이나 PC에서 계열별 균등 60구성을 3회씩. CPU만 있어도 된다.

```bash
python run_benchmark.py --device cpu --subset stratified:60 --repeats 3 --require-idle --tag external --output results/external_<기기이름>.json
python check_env.py --json > paper/env_<기기이름>.json
```

## 7. 분석 단계에서 할 일 (데스크톱)

- `eval_fill_v4i.load_clean()`에 v2 결과로 GPU 셀을 덮어쓰는 override를 넣고 v4i → v4j → v4k 재실행.
- 재측정 전후 GAN 셀과 전체 표 수치의 변화량을 기록해 §3.1 각주로.
