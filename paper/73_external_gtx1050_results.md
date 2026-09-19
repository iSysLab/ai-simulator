# 외부 백엔드 결과 — GTX 1050 (2026-09-19)

측정: 브랜치 `external-gtx1050`(d6476a0), 런북 `72_external_backend_runbook.md` §3. 분석: `eval_external_backend.py`, 수치 `paper/external_gtx1050.json`(기술자 갱신)·`paper/external_gtx1050_norefresh.json`(라벨만).

## 1. 기기와 측정

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 2 GB, CC 6.1, CUDA 코어 640, 부스트 1.9 GHz, **2.51 TFLOPS** (4060 Ti: 4352코어, 26.98 TFLOPS, 8 GB) |
| 호스트 | Intel i3-8100 4C, 8 GB DDR4, Windows 11, Python 3.14.5, torch 2.14.0+cu126, cuDNN 9.10.2. 모니터는 내장 UHD 630에 연결 |
| 부분집합 | `stratified:60` → 58구성, **완료 55행**, OOM 3개 건너뜀(CNN_f128_l3_bn0, CNN_f128_l6_bn1, ViT_d256_l6_h8_p4). 분석은 632셀에 있는 54구성 |
| 프로토콜 | v2 통일 워밍업, 3회 반복, `PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.90`(sysmem fallback 차단), 13:24~15:09 |
| 유휴 | Windows Search 인덱서가 CPU 15~25% 상시 점유 → `--idle-cpu-max 35`로 진행, `load_cpu_pct` 0.4~33.2 기록. 대기 5행 |
| 계열 | ANN 10 · CNN 8 · ResNet 10 · MobileNet 10 · ViT 9 · GAN 8 |

**4060 Ti 대비 실측 비(같은 구성, 중앙값)**: 학습 2.94× (1 s 미만 1.84×, 10 s 이상 7.22×), 추론 3.51× (1.23×, 4.47×). TFLOPS 비 10.7×와 비교하면 소형 모델은 커널 실행 오버헤드(launch-bound)에 묶여 GPU 성능 차이가 거의 안 드러나고, 대형만 계산 비율에 가까워진다.

## 2. 4백엔드 모델의 GTX 1050 예측 (log(y) 타깃, 632셀 학습)

값은 R²(log) / MAPE / MdAPE / Spearman / 예측÷실측 중앙값.

| 예측기 | 학습시간 | 추론시간 |
|---|---|---|
| **기술자 모델 (갱신: gpu_cores 640, tflops 2.51, VRAM 2)** | **0.61 / 54 / 59 / 0.965 / 0.41** | **0.78 / 42 / 42 / 0.975 / 0.59** |
| 기술자 모델 (라벨만: v1 CUDA 기술자 + VRAM 2) | 0.77 / 73 / 76 / 0.983 / 1.56 | 0.03 / 122 / 118 / 0.966 / 2.18 |
| 기준선 a: 4060 Ti 실측 복사 | −0.11 / 70 / 70 / 0.930 / 0.30 | 0.09 / 68 / 72 / 0.943 / 0.28 |
| 기준선 b: 4060 Ti × TFLOPS 비(10.7) | 0.22 / 225 / 226 / 0.930 / 3.26 | −0.16 / 292 / 197 / 0.943 / 2.97 |
| 기준선 c: 4백엔드 로그 평균 | 0.90 / 32 / 30 / 0.979 / 0.94 | 0.83 / 49 / 32 / 0.957 / 1.19 |

계열별 MAPE(갱신 모델): 학습 ANN 58 · CNN 55 · ResNet 54 · MobileNet 68 · ViT 50 · GAN 35; 추론 ANN 41 · CNN 38 · ResNet 48 · MobileNet 41 · ViT 47 · GAN 37 — 특정 계열이 무너지지는 않는다.

## 3. 읽기

1. **기술자가 정체성 이상의 정보를 담는다는 첫 증거.** 갱신 모델이 "4060 Ti 실측 복사"를 두 타깃 모두 이긴다(R² 0.61/0.78 vs −0.11/0.09, MAPE 54/42 vs 70/68). 4060 Ti 값을 그대로 쓰면 3.3~3.6배 빠르게 예측하는데(0.30), 기술자 모델은 그 격차의 절반쯤을 메운다(0.41→, 0.59). 순위는 잘 보존된다(Spearman 0.97).
2. **그러나 절대 수준은 못 맞춘다.** 여전히 실측보다 1.7~2.4배 빠르게 예측한다. 632셀에서 `tflops_fp32`가 0·5·26.98 세 값뿐이라 트리가 2.51을 "5(MPS)에 가까운 무언가"로만 취급할 수 있기 때문. 그 결과 "4백엔드 평균"이라는 무학습 기준선이 MAPE에서 더 낫다(32/49) — GTX 1050의 속도가 우연히 네 백엔드 가운데쯤이라서이지 방법이 아니다.
3. **라벨만 주면 반대 방향으로 틀린다.** VRAM 2 GB만 실제값으로 두면 모델이 8·18·0 어느 쪽에도 없는 값을 CPU 쪽으로 분기시켜 1.6~2.2배 느리게 예측한다. 기술자 한 개만 바꾸는 것은 위험하고, 전부 함께 갱신해야 한다.
4. **TFLOPS 비례 스케일링은 최악.** 소형 모델은 launch-bound라 3배도 안 느린데 10.7배를 곱한다. Roofline 서술(§4.3)의 실측 근거: 실행시간은 계산량 비가 아니라 메모리·오버헤드 체제에 따라 갈린다.
5. MPS 시뮬레이션(0.73→−0.40)과 달리 여기서는 갱신이 도움이 됐다. 차이: GTX 1050은 4060 Ti와 **같은 장치 종류·같은 소프트웨어 스택**이라 기술자가 "CUDA 안에서 어디쯤"만 알려주면 되지만, MPS는 CUDA와 스택이 달라 기술자로 잇기 어렵다.

## 4. 논문에 주는 것

- R1-5 답변에 한 문장: "5번째 기기(GTX 1050, 같은 CUDA 스택)에서 기술자 모델은 최근접 백엔드 실측 복사보다 낫지만(R²(log) 0.61/0.78 vs −0.11/0.09) 절대 수준은 1.7~2.4배 낙관적이다 — 기술자 값이 기기당 하나뿐인 데이터로는 보간이 안 된다." 이것이 R2 "미지 하드웨어"에 대한 정직한 실측 답이다.
- camera-ready(10/7)에는 Limitations 한 문장 또는 §4.4 한 행. 표까지는 분량상 어렵다.
- 후속: 기술자 축을 채우려면 GPU 2~3대가 더 필요하다(예: 3060, 1660, T4). 그때는 `tflops`·`gpu_cores`가 연속 축이 되어 트리가 보간할 수 있다.

## 5. 재현

```bash
git fetch origin && git merge origin/external-gtx1050          # ensemble 브랜치에서
python eval_external_backend.py --external results/external_gtx1050.json --env paper/env_gtx1050.json --name gtx1050
python eval_external_backend.py --external results/external_gtx1050.json --env paper/env_gtx1050.json --name gtx1050_norefresh --no-refresh-hw
```
