# 외부 백엔드 결과 — RTX 3060 (2026-09-21)

측정: 브랜치 `external-rtx3060`(0b2af95 CUDA, d00a941 CPU), 런북 `72_external_backend_runbook.md` §3. 분석: `eval_external_backend.py`·`eval_external_fewshot.py`, 수치 `paper/external_rtx3060.json`(기술자 갱신)·`paper/external_rtx3060_norefresh.json`(라벨만)·`paper/external_rtx3060_fewshot.json`, CPU는 `paper/external_rtx3060_cpu*.json`.
73번 문서(GTX 1050) §3.1이 예측한 "VRAM 8 GB 이상인 GPU(예: 3060 12 GB)는 4060 Ti 쪽 잎으로 떨어져 복사와 같은 예측이 나올 것"을 실측으로 확인한 문서다.

> 분석은 측정 기기(RTX 3060 PC)에서 632셀 파일을 재생성해 돌렸다(`enrich_oplevel.py` → `export_onnx_configs.py`, torch 2.14.0). 두 스크립트 모두 구조적·결정적이라 맥북/데스크톱 결과와 같아야 하지만, ONNX export 버전 차이로 소수점 아래가 다를 수 있다.

## 1. 기기와 측정

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB, CC 8.6, 28 SM × 128 = CUDA 코어 3584, 최대 클럭 2.16 GHz, **15.48 TFLOPS** (4060 Ti: 4352코어, 26.98 TFLOPS, 8 GB / GTX 1050: 640코어, 2.51 TFLOPS, 2 GB) |
| 호스트 | Intel i7-10700F 8C16T 2.9 GHz, 32 GB DDR4, Windows 10 Education, Python 3.12.10, torch 2.14.0+cu126, cuDNN 9.10.2, 드라이버 616.92 |
| GPU 2장 | 같은 RTX 3060 두 장. 모니터가 붙은 GPU 0은 유휴 시에도 사용률 10~11%라 `CUDA_VISIBLE_DEVICES=1`로 GPU 1(모니터 미연결)만 노출해 측정(코드 수정 없음, `gpu_count=1`). 유휴 판정의 nvidia-smi 조회는 GPU 0을 읽으므로 `--idle-gpu-max 30` |
| 부분집합 | `stratified:60` → 58구성, **완료 58행**, OOM 0, 대기 0. 분석은 632셀에 있는 **56구성**(ViT_d256_l6_h8_p4, ViT_d256_l4_h8_p8은 632셀 정리에서 빠진 구성) |
| 프로토콜 | v2 통일 워밍업, 3회 반복, 14:04~14:35(31분), 러너 실행 1회. 학습시간 CV 중앙값 0.002 |
| 유휴 | `load_cpu_pct` 1.1~12.5, 대기 0행 |
| 계열 | ANN 10 · CNN 10 · ResNet 10 · MobileNet 10 · ViT 10 · GAN 8 |
| 기록 이상 | 3행(ResNet_2211_w16, MobileNet_w0.5_b3, ViT_d128_l4_h8_p4)은 nvidia-smi 일시 실패로 `tflops_fp32`·`gpu_clock_ghz` 0 — 분석은 `env_rtx3060.json` 기술자를 쓰므로 무관. `interconnect_type`은 유휴 링크 상태를 읽어 `pcie1`로 기록됨(4060 Ti pcie4) |

**4060 Ti 대비 실측 비(같은 구성, 중앙값)**: v2 재측정(`remeasure_cuda_v2.json`) 기준 학습 **1.10×**(1 s 미만 1.05×, 10 s 이상 1.53×), 추론 **1.16×**(1.19×, 1.25×). 모델이 학습한 632셀의 v1 CUDA 행 기준으로는 학습 1.32×(1.25×, 1.58×), 추론 1.17×. TFLOPS 비는 1.74×인데 소형은 1.05~1.25×에 머물고 대형만 1.5×대 — 1050 때와 같은 launch-bound 그림이다.

## 2. 4백엔드 모델의 RTX 3060 예측 (log(y) 타깃, 632셀 학습, 56구성 평가)

값은 R²(log) / MAPE / MdAPE / Spearman / 예측÷실측 중앙값.

| 예측기 | 학습시간 | 추론시간 |
|---|---|---|
| **기술자 모델 (갱신: gpu_cores 3584, tflops 15.48, VRAM 12)** | **0.889 / 25.0 / 26.2 / 0.984 / 0.74** | **0.970 / 23.6 / 16.2 / 0.989 / 0.86** |
| 기술자 모델 (라벨만: v1 CUDA 기술자 + VRAM 12) | 0.887 / 25.4 / 25.8 / 0.984 / 0.74 | 0.970 / 23.6 / 16.0 / 0.989 / 0.85 |
| 기준선 a: 4060 Ti 실측 복사 | 0.898 / 24.8 / 24.5 / 0.983 / 0.76 | 0.972 / 23.6 / 15.6 / 0.987 / 0.86 |
| 기준선 b: 4060 Ti × TFLOPS 비(1.74) | 0.923 / 31.0 / 31.6 / 0.983 / 1.32 | 0.732 / 61.4 / 50.4 / 0.987 / 1.49 |
| 기준선 c: 4백엔드 로그 평균 | −0.61 / 212 / 193 / 0.954 / 2.93 | −5.91 / 341 / 371 / 0.888 / 4.71 |

계열별 MAPE(갱신 모델): 학습 ANN 23 · CNN 23 · ResNet 28 · MobileNet 31 · ViT 27 · GAN 17; 추론 ANN 67 · CNN 19 · ResNet 15 · MobileNet 5 · ViT 20 · GAN 12.
계열별 예측÷실측 중앙값: 학습 ANN 0.74 · CNN 0.77 · ResNet 0.69 · MobileNet 0.68 · ViT 0.73 · GAN 0.82; 추론 ANN 0.59 · CNN 0.97 · ResNet 0.83 · MobileNet 1.02 · ViT 0.81 · GAN 0.89.

## 3. 읽기

1. **73 §3.1의 예측이 그대로 맞았다 — 기술자 모델 = 4060 Ti 복사.** 예측÷실측 0.74 vs 0.76(학습), 0.86 vs 0.86(추론), MAPE 25.0 vs 24.8 / 23.6 vs 23.6. 기술자를 실제값(코어 3584, 15.48 TFLOPS, 2.16 GHz, VRAM 12)으로 갱신한 결과와 v1 CUDA 값 그대로 둔 결과가 같다(0.74 vs 0.74). 1050에서는 `dedicated_vram_gb < 8` 분기 하나가 예측을 움직였는데, 3060은 VRAM 12 GB라 그 분기도 발화하지 않아 **16개 기술자 중 어느 것도 예측을 바꾸지 못했다.** 모델은 3060을 "4060 Ti와 같은 기기"로 본다.
2. **절대 오차가 1050보다 작은 것(MAPE 25/24 vs 54/42)은 3060이 4060 Ti에 가까워서다.** 복사만 해도 1.1~1.5배 차이니 MAPE 25%가 나온다. 사양 학습의 증거가 아니라 복사가 우연히 좋은 경우다. 학습시간은 여전히 26% 낙관(0.74), 대형(>10 s)일수록 더 낙관(계열별 ResNet·MobileNet 0.68~0.69).
3. **TFLOPS 비례 스케일링(b)은 여기서도 최악 방향으로 틀린다.** 1.74배를 곱하면 1.32~1.49배 과대. 실제 감속은 1.05~1.53배라 계산량 비의 절반도 안 된다 — Roofline 서술(§4.3)의 두 번째 실측 근거.
4. **"4백엔드 평균"은 여기서 무너진다(2.9~4.7배 과대).** 1050에서 가장 좋았던 기준선인데, 3060은 네 백엔드 중 가장 빠른 쪽 근처라 평균이 훨씬 느리다. 73 §3.3의 결론(그 기준선은 방법이 아니라 우연)을 세 번째로 확인.
5. 추론의 ANN(2~20 ms 셀)만 MAPE 67%, 예측÷실측 0.59로 따로 논다. 다른 다섯 계열은 복사가 곧 정답(MobileNet 1.02, CNN 0.97) — 3060·4060 Ti의 추론 속도가 거의 같기 때문. 소형 셀의 상대 오차는 커널 실행 오버헤드 체제라 절대값(ms 단위)으로는 작다.

### 3.1 새 기기를 학습에 포함하면 (`paper/external_rtx3060_fewshot.json`)

73 §3.2와 같은 절차: 3060의 56구성 중 k개(계열별 균등)를 632셀에 더해 학습, 나머지로 평가(log(y), 시드 5개 평균). 추론은 소형 ANN 셀 몇 개가 평균 MAPE를 끌어올려(표준편차 ±18~33) MdAPE와 ±20% 비율을 함께 본다.

| 3060 구성 학습 포함 | 학습 MAPE / MdAPE / pred÷true | 추론 MAPE / MdAPE / ±20% / pred÷true |
|---|---|---|
| 0개 (= §2의 갱신 모델) | 25.0 / 26.2 / 0.74 | 23.6 / 16.2 / 64% / 0.86 |
| **6개 (계열당 1개)** | 24.1 / 18.9 / **0.94** | 53.7±32.9 / 16.8 / 62% / **0.99** |
| 12개 | 18.7 / 14.9 / 0.92 | 44.4±34.0 / 12.7 / 73% / 0.97 |
| **24개 (계열당 4개)** | **18.5 / 14.5 / 0.97** | 35.6±18.4 / **10.7 / 71%** / 1.01 |
| (참고) 4백엔드 in-distribution | 15.9 / 12.3 | 16.4 / 12.1 |

읽기. 1050과 같은 곡선이다 — 계열당 1개면 낙관 편향이 사라지고(0.74→0.94, 0.86→0.99), 계열당 4개면 학습 MAPE 18.5%로 기존 백엔드 수준. 추론의 평균 MAPE가 오히려 오르는 것은 32개 평가 셀 중 2~3개의 소형 ANN 셀(수 ms)에서 몇 배짜리 상대 오차가 나기 때문이고, MdAPE는 16→11로 내려간다. "기기 추가 비용 = 24구성 × 3회"라는 73의 읽기가 두 번째 GPU에서도 유지된다(3060에서는 24구성 × 3회가 약 12분).

### 3.2 같은 기기의 CPU(i7-10700F) — 7번째 백엔드 (`results/external_rtx3060_cpu.json`, `paper/external_rtx3060_cpu*.json`)

42구성 × 3회 v2, 계열별 7개, 14:36~20:37(6시간 1분), 대기 0. 632셀에 있는 41구성으로 분석(ViT_d256_l4_h8_p8 제외). Desktop CPU(Ryzen 7 7800X3D) 대비 실측 비 중앙값: 학습 **2.46×**(1 s 미만 3.22×, 10 s 이상 2.32×), 추론 **2.50×**(`benchmark_results.json` v1 기준; few-shot 스크립트의 632셀 기준으로는 2.10×/2.44×). i3-8100 대비는 0.54×/0.58×.

| 예측기 | 학습 R²(log) / MAPE / pred÷true | 추론 R²(log) / MAPE / pred÷true |
|---|---|---|
| 기술자 모델 (갱신: cpu_cores 8, RAM 31.9, 2.9 GHz) | 0.824 / 51.2 / 0.48 | 0.732 / 45.9 / 0.41 |
| 기술자 모델 (라벨만) | 0.813 / 51.5 / 0.47 | 0.732 / 45.9 / 0.41 |
| 기준선 a: Desktop CPU 실측 복사 | 0.815 / 52.9 / 0.48 | 0.735 / 45.8 / 0.41 |
| 기준선 c: 4백엔드 로그 평균 | 0.574 / 65.5 / 0.28 | 0.400 / 80.4 / 0.21 |

i3와 같은 결론이 더 극단적인 형태로 나온다: 기술자 모델 = Desktop CPU 복사(0.48 vs 0.48, 0.41 vs 0.41). 이번엔 `cpu_cores`가 8 vs 8로 아예 같고 RAM도 31.9 vs 31이라 남는 구별 정보는 클럭(2.9 vs 4.2 GHz)뿐인데, 그마저 트리가 쓰지 않는다. 2.1~2.5배 차이(Zen 4 X3D vs Comet Lake, 클럭·캐시)는 기술자로는 보이지 않는다.

Few-shot(`eval_external_fewshot.py`):

| i7 구성 학습 포함 | 학습 MAPE / MdAPE / pred÷true | 추론 MAPE / MdAPE / pred÷true |
|---|---|---|
| 0개 | 51.2 / 52.3 / 0.48 | 45.9 / 58.9 / 0.41 |
| 6개 (계열당 1개) | 33.1 / 24.8 / 0.88 | 23.5 / 22.3 / 0.86 |
| 12개 | 29.1 / 18.9 / 0.98 | 18.8 / 13.8 / 0.93 |
| 24개 (계열당 4개) | **28.2 / 18.5 / 0.97** | **17.6 / 14.1 / 0.96** |

i3(20.2/23.5)보다 학습 MAPE가 조금 높은 채로 수렴하는데(28.2), 남은 평가 셀이 17개뿐이라 잡음(±10.6)이 크다. 추론은 i3와 같은 수준(17.6 vs 23.5)에 도달한다.

## 4. 논문에 주는 것

- R1-5·R2 답변에 한 문장 추가: "6번째 기기(RTX 3060, 12 GB)에서는 사양 기술자를 실제값으로 갱신해도 예측이 4060 Ti 실측 복사와 같았다(예측÷실측 0.74 vs 0.76) — VRAM 임계값 위에 있는 GPU라 GTX 1050에서 발화했던 분기조차 쓰이지 않았고, 코어 수·TFLOPS·클럭은 어느 쪽 기기에서도 예측을 움직이지 못했다."
- 1050(임계값 아래)과 3060(임계값 위) 양쪽에서 같은 기제가 확인되므로 Limitations의 "기술자 = 정체성" 문장은 두 기기 근거로 쓸 수 있다. 표는 분량상 어렵고, 73·75는 후속 논문 재료.
- 후속: 이제 CUDA 백엔드가 4060 Ti·1050·3060 세 기기라 `tflops_fp32`가 26.98·2.51·15.48 세 값의 축이 됐다. 1050과 3060을 학습에 넣고 4060 Ti(또는 세 기기 LOBO)를 맞히는 실험이 "값이 세 개면 트리가 보간을 시작하는가"에 대한 첫 답이 된다. 기기 한 대 추가 비용은 §3.1대로 24구성 × 3회.

## 5. 재현

```bash
git fetch origin && git merge origin/external-rtx3060          # ensemble 브랜치에서
python enrich_oplevel.py && python export_onnx_configs.py       # 632셀 파일이 없는 기기만
python eval_external_backend.py --external results/external_rtx3060.json --env paper/env_rtx3060.json --name rtx3060
python eval_external_backend.py --external results/external_rtx3060.json --env paper/env_rtx3060.json --name rtx3060_norefresh --no-refresh-hw
python eval_external_backend.py --external results/external_rtx3060_cpu.json --env paper/env_rtx3060.json --name rtx3060_cpu
python eval_external_backend.py --external results/external_rtx3060_cpu.json --env paper/env_rtx3060.json --name rtx3060_cpu_norefresh --no-refresh-hw
python eval_external_fewshot.py --external results/external_rtx3060.json --env paper/env_rtx3060.json --name rtx3060
python eval_external_fewshot.py --external results/external_rtx3060_cpu.json --env paper/env_rtx3060.json --name rtx3060_cpu
```
