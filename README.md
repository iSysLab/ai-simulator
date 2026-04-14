# DNN 실행 시간/공간 예측 시뮬레이션 프레임워크

PyTorch 기반 DNN 모델의 **실행 시간(학습/추론)** 및 **메모리 요구량**을 예측하는 벤치마크 프레임워크입니다.
6종의 모델 아키텍처(ANN, CNN, ResNet, MobileNet, Transformer, GAN)를 벤치마킹하고, **96개 모델 구조 + 하드웨어 피처**로부터 실행 시간을 예측하는 ML 회귀 모델을 학습합니다.

**v2.0 — Zero-Config Cross-Platform**: `python run_benchmark.py` 한 줄로 macOS/Windows/Linux 어디서든 자동으로 하드웨어를 감지하고 벤치마크를 실행합니다.

## 빠른 시작

```bash
# 1. 환경 설치
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 환경 검증
python check_env.py

# 3. 빠른 벤치마크 (대표 모델 9개, 3회 반복)
python run_benchmark.py --quick

# 4. 전체 파이프라인 (벤치마크 → 학습 → 시각화)
python run_benchmark.py --full-pipeline
```

## 실험 결과 요약

### 벤치마크 데이터 (698개 샘플, 2개 플랫폼)

#### Desktop — Windows (378개 샘플)

| 모델 | 설정 수 | 디바이스 | 파라미터 범위 | 학습시간(s) | 추론시간(s) |
|---|---|---|---|---|---|
| ANN | 84 | CPU, CUDA | 12K ~ 2.2M | 0.24 ~ 2.53 | 0.002 ~ 0.086 |
| CNN | 120 | CPU, CUDA | 2.5K ~ 8.6M | 0.72 ~ 686.5 | 0.008 ~ 41.7 |
| ResNet | 36 | CPU, CUDA | 309K ~ 17.4M | 3.54 ~ 451.0 | 0.090 ~ 25.5 |
| MobileNet | 40 | CPU, CUDA | 4.4K ~ 4.6M | 2.70 ~ 315.4 | 0.103 ~ 19.9 |
| Transformer | 34 | CPU, CUDA | 108K ~ 4.8M | 2.78 ~ 263.0 | 0.045 ~ 20.5 |
| GAN | 16 | CPU, CUDA | 1.7M ~ 7.7M | 3.05 ~ 22.1 | 0.215 ~ 1.32 |

> 측정 환경: AMD Ryzen 7 7800X3D 8C16T, 31GB RAM, NVIDIA RTX 4060 Ti (CUDA)

#### macOS — Apple Silicon (320개 샘플)

| 모델 | 설정 수 | 디바이스 | 파라미터 범위 | 학습시간(s) | 추론시간(s) |
|---|---|---|---|---|---|
| ANN | 84 | CPU, MPS | 12.7K ~ 2.2M | 0.11 ~ 1.30 | 0.002 ~ 0.088 |
| CNN | 120 | CPU, MPS | 2.5K ~ 8.6M | 0.51 ~ 673.7 | 0.028 ~ 40.3 |
| ResNet | 36 | CPU, MPS | 308K ~ 17.4M | 6.48 ~ 771.8 | 0.276 ~ 40.5 |
| MobileNet | 40 | CPU, MPS | 4.4K ~ 4.6M | 5.98 ~ 5289.5 | 0.290 ~ 71.3 |
| Transformer | 24 | CPU, MPS | 108K ~ 4.8M | 1.74 ~ 154.5 | 0.086 ~ 11.4 |
| GAN | 16 | CPU, MPS | 1.7M ~ 7.7M | 2.59 ~ 13.4 | 0.154 ~ 0.725 |

> 측정 환경: Apple M4 (4P+6E 10코어), 24GB 통합 메모리, MPS (Metal Performance Shaders)

#### 크로스 플랫폼 비교

| 비교 항목 | ANN | CNN | ResNet | MobileNet | Transformer | GAN |
|---|---|---|---|---|---|---|
| Mac CPU / Desktop CPU | 0.9x | 1.9x | 2.5x | **14.0x** | 0.8x | 0.6x |
| MPS Speedup (Mac GPU/CPU) | 0.5x | 5.1x | 7.1x | **35.0x** | 2.7x | 1.7x |

> **주요 발견**: MobileNet의 depthwise separable convolution은 Mac ARM CPU에서 14배 느림 (PyTorch ARM 빌드에 MKLDNN/oneDNN 미포함). MPS(GPU)에서는 35배 가속되어 정상 성능 발휘.

### 예측 모델 성능 (XGBoost, 5-Fold CV)

| 예측 타겟 | 디바이스 | R² | R²(log) | RMSE | MAE |
|---|---|---|---|---|---|
| 학습 시간 | CPU | 0.9465 | **0.9865** | 25.354s | 10.103s |
| 학습 시간 | CUDA | 0.9391 | **0.9724** | 1.561s | 0.663s |
| 추론 시간 | CPU | 0.9345 | **0.9772** | 1.686s | 0.642s |
| 추론 시간 | CUDA | 0.9520 | **0.9659** | 0.084s | 0.038s |
| 메모리 | CPU | 0.9445 | **0.9974** | 2.2MB | 0.4MB |
| 메모리 | CUDA | 0.9430 | **0.9979** | 2.3MB | 0.5MB |

## 시각화 결과

### 1. 파라미터 수 vs 실행 시간
![파라미터 수 vs 시간](results/figures/fig1_params_vs_time.png)

### 2. FLOPs vs 실행 시간
![FLOPs vs 시간](results/figures/fig2_flops_vs_time.png)

### 3. CPU vs GPU 디바이스별 비교
![디바이스 비교](results/figures/fig3_device_comparison.png)

### 4. GPU Speedup 비율
![Speedup](results/figures/fig4_speedup_ratio.png)

### 5. 예측 정확도 (실측 vs 예측)
![예측 정확도](results/figures/fig5_prediction_accuracy.png)

### 6. 피처 중요도 (Top 15)
![피처 중요도](results/figures/fig6_feature_importance.png)

### 7. 모델별 복잡도 히트맵
![복잡도 히트맵](results/figures/fig7_complexity_heatmap.png)

## 프로젝트 구조

```
ai-simulator/
├── benchmark/                          # 메인 패키지
│   ├── models/                         # 6종 모델 정의
│   │   ├── registry.py                 # 데코레이터 기반 모델 팩토리
│   │   ├── simple_ann.py               # SimpleANN (MNIST)
│   │   ├── simple_cnn.py               # SimpleCNN (MNIST)
│   │   ├── resnet_mnist.py             # ResNet (MNIST)
│   │   ├── mobilenet_mnist.py          # MobileNet (MNIST)
│   │   ├── transformer.py              # Vision Transformer (CIFAR-10)
│   │   └── gan.py                      # GAN Generator+Discriminator (CIFAR-10)
│   ├── features/                       # 피처 추출
│   │   ├── extractor.py                # PyTorch 모델 → 96개 피처 (Feature Schema v3.0)
│   │   ├── onnx_extractor.py           # ONNX 파일 → 96개 피처
│   │   └── op_profiler.py              # Op-level 분해/시간 측정/시뮬레이션
│   ├── platform/                       # [NEW] 크로스 플랫폼 자동 감지
│   │   ├── __init__.py                 # PlatformInfo 통합 클래스
│   │   ├── detector.py                 # OS/CPU/GPU/RAM 오케스트레이터
│   │   ├── cpu_info.py                 # CPU 상세 (P/E코어, 캐시, 주파수)
│   │   ├── gpu_info.py                 # GPU 상세 (CUDA/MPS, 텐서코어, TFLOPS)
│   │   ├── memory_info.py              # 메모리 구조 (unified vs discrete)
│   │   └── compatibility.py            # 환경 호환성 매트릭스
│   ├── runner/                         # 실험 실행
│   │   ├── device.py                   # 장치 감지/동기화/워밍업
│   │   ├── data.py                     # MNIST + CIFAR-10 데이터 로딩
│   │   └── experiment.py               # 학습/추론/GAN 시간 측정 루프
│   ├── configs/
│   │   └── generator.py                # 160개 설정 조합 자동 생성
│   └── results/
│       └── io.py                       # JSON 증분 저장/CSV 변환
├── run_benchmark.py                    # 통합 벤치마크 (--quick, --full-pipeline, --device auto)
├── train_predictor.py                  # 예측 모델 학습 (--features core/full)
├── visualize_results.py                # 시각화 (9종 그래프, 크로스 플랫폼 폰트)
├── check_env.py                        # [NEW] 환경 검증 스크립트
├── export_onnx.py                      # PyTorch → ONNX 변환
├── predict_from_onnx.py                # ONNX → 실행 시간 예측
├── pyproject.toml                      # [NEW] 패키지 설정
├── requirements.txt
└── results/
    ├── benchmark_results.json          # Desktop 벤치마크 (378개)
    ├── benchmark_results_mac.json      # Mac 벤치마크 (320개)
    ├── figures/                        # 시각화 그래프 (9개)
    └── trained_models/                 # 학습된 예측 모델 (.pkl)
```

## 사용법

### 환경 검증

```bash
python check_env.py
```

Python, PyTorch, 디바이스, 선택적 패키지, 디스크 공간, 플랫폼 감지를 자동 점검합니다.

### 벤치마크 실행

```bash
# 전체 실행 (자동 디바이스 감지, 160개 설정 x 10회 반복)
python run_benchmark.py

# 빠른 테스트 (대표 모델 9개, 3회 반복)
python run_benchmark.py --quick

# 전체 파이프라인 (벤치마크 → 예측 모델 학습 → 시각화)
python run_benchmark.py --full-pipeline

# 특정 모델/디바이스
python run_benchmark.py --model transformer --device cuda
python run_benchmark.py --model gan --device mps

# Op-level 프로파일링 포함
python run_benchmark.py --profile-ops

# 중단 후 이어서 실행
python run_benchmark.py --resume
```

### 예측 모델 학습

```bash
# 기본 실행 (1차 핵심 피처, 5-fold CV)
python train_predictor.py

# 전체 피처 사용 + 10-fold CV + 모델 저장
python train_predictor.py --features full --cv 10 --save-models
```

### 시각화

```bash
python visualize_results.py
# Mac 데이터 포함 시 크로스 플랫폼 비교 그래프도 자동 생성
python visualize_results.py --input-mac results/benchmark_results_mac.json
```

### ONNX 예측 파이프라인

```bash
python export_onnx.py                              # PyTorch → ONNX
python predict_from_onnx.py --onnx model.onnx      # ONNX → 실행 시간 예측
python predict_from_onnx.py --demo                  # 전체 샘플 예측
```

## 통합 Feature Schema v3.0 — 96개 피처

Feature_Schema.docx(v1.0) 문서를 100% 구현하여 기존 44개에서 **96개**로 확장했습니다.
`extractor.py`가 단일 소스(single source of truth)로 모든 피처를 직접 생성합니다.

### 피처 카테고리 요약

| 카테고리 | 수 | 대표 피처 | 설명 |
|---|---|---|---|
| **파라미터** | 6 | total_params, conv_params, linear_params | 레이어 타입별 파라미터 수 |
| **레이어 수** | 7 | total_layers, num_conv_layers, num_activation_layers | 연산 레이어 구성 |
| **폭** | 4 | max_width, min_width, avg_width, max_channel_width | 레이어 폭/채널 통계 |
| **연산량** | 5 | flops, model_size_mb, memory_bytes | 계산 복잡도 + 메모리 |
| **구조 플래그** | 8 | has_residual, has_depthwise, has_attention, has_skip_connection | 아키텍처 특성 (0/1) |
| **모델 분류** | 3 | model_family_encoded, model_family, model_arch | 계열 + 세부 아키텍처 |
| **ANN 전용** | 1 | hidden_size | FC 레이어 폭 |
| **CNN 전용** | 5 | num_filters, kernel_size, stride, padding, max_channels | Conv 하이퍼파라미터 |
| **Transformer 전용** | 6 | embed_dim, num_heads, ffn_dim, sequence_length, has_cls_token | 어텐션 구조 |
| **GAN 전용** | 5 | latent_dim, generator_params, discriminator_params, generator_layers, discriminator_layers | G/D 분리 |
| **입력 데이터** | 8 | batch_size, input_height, dataset_type, input_dtype, input_elements | 데이터셋 정보 |
| **HW: 디바이스 일반** | 4 | device_type_encoded, os_type, accelerator_brand, accelerator_name | 플랫폼 식별 |
| **HW: CPU** | 8 | cpu_cores_physical, cpu_perf_cores, cpu_freq_boost_ghz, cpu_cache_l2_mb | CPU 상세 |
| **HW: 메모리 구조** | 5 | memory_type, is_unified_memory, shared_memory_gb, dedicated_vram_gb | Mac/Windows 핵심 차이 |
| **HW: GPU** | 11 | gpu_core_count, gpu_tensor_core_count, tflops_fp32, fp16_support | GPU 성능 지표 |
| **HW: 인터커넥트** | 4 | interconnect_type, host_to_device_bandwidth_gbs, is_discrete_gpu | 전송 대역폭 |
| **기존 호환** | 6 | cpu_cores, cpu_freq_ghz, gpu_cores, gpu_memory_gb, ram_total_gb | v1.0 하위 호환 |
| **총합** | **96** | | |

### 피처 상세 설명

#### 1. 파라미터 관련 (6개)

| 피처 | 타입 | 단위 | 설명 | 예시 (ResNet-18) |
|---|---|---|---|---|
| `total_params` | int | 개 | 모델 전체 파라미터 수 | 11,173,962 |
| `trainable_params` | int | 개 | 학습 가능한 파라미터 수 (frozen 제외) | 11,173,962 |
| `conv_params` | int | 개 | Conv2d 레이어 파라미터 합계 | 10,950,144 |
| `linear_params` | int | 개 | Linear(FC) 레이어 파라미터 합계 | 5,130 |
| `bn_params` | int | 개 | BatchNorm 레이어 파라미터 합계 | 18,688 |
| `other_params` | int | 개 | 위 3종에 포함되지 않는 파라미터 | 200,000 |

#### 2. 레이어 수 (7개)

| 피처 | 설명 | ANN 예시 | CNN 예시 | Transformer 예시 |
|---|---|---|---|---|
| `total_layers` | Conv + Linear + BN 레이어 총합 | 3 | 12 | 26 |
| `num_hidden_layers` | Conv + Linear 레이어 수 | 3 | 8 | 20 |
| `num_conv_layers` | Conv2d 레이어 수 | 0 | 5 | 0 |
| `num_linear_layers` | Linear(FC) 레이어 수 | 3 | 3 | 20 |
| `num_bn_layers` | BatchNorm 레이어 수 | 0 | 4 | 0 |
| `num_pool_layers` | Pooling(Max/Avg/Adaptive) 레이어 수 | 0 | 2 | 0 |
| `num_activation_layers` | 활성화 함수(ReLU/GELU 등) 수 | 2 | 7 | 12 |

#### 3. 폭 (4개)

| 피처 | 설명 | 예시 |
|---|---|---|
| `max_width` | config 기반 최대 레이어 폭 | ANN hidden_size=512 → 512 |
| `min_width` | config 기반 최소 레이어 폭 | CNN [32,64,128] → 32 |
| `avg_width` | config 기반 평균 레이어 폭 | CNN [32,64,128] → 74.7 |
| `max_channel_width` | 모델 introspection 기반 최대 채널/뉴런 수 | ResNet out_channels=512 → 512 |

#### 4. 연산량 (5개)

| 피처 | 단위 | 설명 | 예시 |
|---|---|---|---|
| `flops` | 회 | Forward hook 기반 FLOPs 추정치 (batch=1) | 37,748,736 |
| `flops_per_sample` | 회 | = flops (batch=1이므로 동일) | 37,748,736 |
| `params_per_flop` | ratio | total_params / flops (파라미터 효율) | 0.296 |
| `model_size_mb` | MB | 모델 파라미터 크기 (FP32 기준) | 42.6 |
| `memory_bytes` | bytes | 파라미터 실제 메모리 점유 | 44,695,848 |

#### 5. 구조 플래그 (8개) — 모두 0 또는 1

| 피처 | =1 조건 | 해당 모델 |
|---|---|---|
| `has_residual` | Residual connection 있음 | ResNet, MobileNet, Transformer |
| `has_depthwise` | Depthwise separable conv 있음 | MobileNet |
| `has_attention` | Attention 메커니즘 있음 | Transformer |
| `has_pooling` | Pooling 레이어 있음 | CNN, ResNet, MobileNet |
| `has_batch_norm` | BatchNorm 있음 | CNN, ResNet, MobileNet, GAN |
| `has_layer_norm` | LayerNorm 있음 | Transformer |
| `has_dropout` | Dropout 있음 | (현재 모델에 미사용) |
| `has_skip_connection` | Skip connection 있음 (has_residual과 동일 대상) | ResNet, MobileNet, Transformer |

#### 6. 모델 분류 (3개)

| 피처 | 설명 | 예시 |
|---|---|---|
| `model_family_encoded` | 모델 계열 정수 인코딩 (0~5) | simple_ann→0, transformer→4 |
| `model_family` | 모델 계열 숫자 (ML 학습용, encoded와 동일) | 0~5 |
| `model_arch` | 세부 아키텍처 문자열 | ann, cnn, resnet, mobilenet, vit, dcgan |

#### 7. ANN 전용 (1개)

| 피처 | 설명 | 비해당 시 |
|---|---|---|
| `hidden_size` | FC 레이어 뉴런 수 | 0 |

#### 8. CNN 전용 (5개)

| 피처 | 설명 | 비해당 시 |
|---|---|---|
| `num_filters` | 첫 번째 Conv 필터 수 | 0 |
| `use_batchnorm` | BN 사용 여부 (0/1) | 0 |
| `kernel_size` | Conv 커널 크기 평균 | 0 |
| `stride` | Conv stride 평균 | 0 |
| `padding` | Conv padding 평균 | 0 |
| `max_channels` | 최대 채널 수 (= max_channel_width) | 0 |

#### 9. Transformer 전용 (6개)

| 피처 | 설명 | 비해당 시 |
|---|---|---|
| `embed_dim` | 임베딩 차원 | 0 |
| `num_heads` | Attention Head 수 | 0 |
| `patch_size` | ViT 패치 크기 | 0 |
| `ffn_dim` | FFN 은닉층 차원 (= embed_dim × 4) | 0 |
| `num_attention_layers` | Attention 레이어 수 | 0 |
| `sequence_length` | 시퀀스 길이 (패치 수 + CLS 토큰) | 0 |
| `has_cls_token` | CLS 토큰 사용 여부 (0/1) | 0 |

#### 10. GAN 전용 (5개)

| 피처 | 설명 | 비해당 시 |
|---|---|---|
| `latent_dim` | 노이즈 벡터 차원 | 0 |
| `generator_params` | Generator 실제 파라미터 수 | 0 |
| `discriminator_params` | Discriminator 실제 파라미터 수 | 0 |
| `generator_layers` | Generator Linear 레이어 수 | 0 |
| `discriminator_layers` | Discriminator Linear 레이어 수 | 0 |

#### 11. 입력 데이터 (8개)

| 피처 | MNIST 모델 | CIFAR-10 모델 | 설명 |
|---|---|---|---|
| `batch_size` | 64 | 64 | 배치 크기 |
| `input_height` | 28 | 32 | 입력 이미지 높이 |
| `input_width` | 28 | 32 | 입력 이미지 너비 |
| `input_channels` | 1 | 3 | 입력 채널 수 |
| `num_classes` | 10 | 10 | 출력 클래스 수 |
| `dataset_type` | mnist (0) | cifar10 (1) | 데이터셋 종류 |
| `input_dtype` | float32 | float32 | 입력 텐서 자료형 |
| `input_elements` | 784 | 3072 | 입력 원소 수 (H×W×C) |

#### 12. HW: 디바이스 일반 (4개)

| 피처 | macOS 예시 | Windows 예시 | 설명 |
|---|---|---|---|
| `device_type_encoded` | 0(CPU) / 2(MPS) | 0(CPU) / 1(CUDA) | 장치 타입 정수 인코딩 |
| `os_type` | 0 (macos) | 1 (windows) | OS 종류 정수 인코딩 |
| `accelerator_brand` | apple | nvidia | 가속기 제조사 |
| `accelerator_name` | Apple M4 | RTX 4060 Ti | 가속기 모델명 |

#### 13. HW: CPU (8개)

| 피처 | 단위 | 설명 | M4 예시 | Ryzen 예시 |
|---|---|---|---|---|
| `cpu_cores_physical` | 개 | 물리 코어 수 | 10 | 8 |
| `cpu_cores_logical` | 개 | 논리 코어 수 | 10 | 16 |
| `cpu_perf_cores` | 개 | 성능(P) 코어 수 | 4 | 8 |
| `cpu_efficiency_cores` | 개 | 효율(E) 코어 수 | 6 | 0 |
| `cpu_freq_base_ghz` | GHz | 기본 클럭 | 2.0 | 3.4 |
| `cpu_freq_boost_ghz` | GHz | 부스트 클럭 | 3.5 | 5.0 |
| `cpu_cache_l2_mb` | MB | L2 캐시 크기 | 16 | 8 |
| `cpu_cache_l3_mb` | MB | L3 캐시 크기 | 16 | 96 |

#### 14. HW: 메모리 구조 (5개) — Mac/Windows 핵심 차이

| 피처 | 단위 | 설명 | Mac (unified) | Windows (discrete) |
|---|---|---|---|---|
| `memory_type` | str | 메모리 타입 | lpddr5 | ddr5 |
| `memory_bandwidth_gbs` | GB/s | 메모리 대역폭 | 120 | 89.6 |
| `is_unified_memory` | 0/1 | 통합 메모리 여부 | 1 | 0 |
| `shared_memory_gb` | GB | CPU/GPU 공유 메모리 | 24.0 | 0.0 |
| `dedicated_vram_gb` | GB | 전용 GPU 메모리 | 0.0 | 15.6 |

#### 15. HW: GPU (11개)

| 피처 | 단위 | 설명 | MPS (M4) | CUDA (4060 Ti) |
|---|---|---|---|---|
| `gpu_count` | 개 | GPU 수 | 1 | 1 |
| `gpu_core_count` | 개 | GPU 코어 수 | 10 | 4352 |
| `gpu_tensor_core_count` | 개 | 텐서 코어 수 | 0 | 136 |
| `gpu_compute_capability` | str | CUDA Compute Capability | - | 8.9 |
| `gpu_clock_ghz` | GHz | GPU 클럭 | 1.4 | 2.31 |
| `peak_bandwidth_gbs` | GB/s | 최대 메모리 대역폭 | 120 | 288 |
| `tflops_fp32` | TFLOPS | FP32 연산 성능 | 5.0 | 22.1 |
| `tflops_fp16` | TFLOPS | FP16 연산 성능 | 10.0 | 44.2 |
| `fp16_support` | 0/1 | FP16 지원 여부 | 1 | 1 |
| `bf16_support` | 0/1 | BF16 지원 여부 | 0 | 1 |

#### 16. HW: 인터커넥트 (4개)

| 피처 | 설명 | Mac 예시 | Windows 예시 |
|---|---|---|---|
| `interconnect_type` | CPU↔GPU 연결 방식 | unified | pcie |
| `host_to_device_bandwidth_gbs` | 호스트→디바이스 대역폭 (GB/s) | 120 | 32 |
| `is_discrete_gpu` | 외장 GPU 여부 (0/1) | 0 | 1 |
| `is_integrated_gpu` | 내장 GPU 여부 (0/1) | 1 | 0 |

#### 17. 기존 호환 (6개)

| 피처 | 설명 | 매핑 원본 |
|---|---|---|
| `cpu_cores` | 논리 코어 수 (v1.0 호환) | = cpu_cores_logical |
| `cpu_freq_ghz` | 부스트 클럭 (v1.0 호환) | = cpu_freq_boost_ghz |
| `gpu_cores` | GPU 코어 수 (v1.0 호환) | = gpu_core_count |
| `gpu_memory_gb` | GPU 메모리 (v1.0 호환) | unified 추정 또는 CUDA props |
| `ram_total_gb` | 전체 RAM | psutil / sysctl |
| `device_type_encoded` | 장치 타입 인코딩 | cpu→0, cuda→1, mps→2 |

### 단계적 피처 도입 (Feature Schema 문서 7절)

```python
from benchmark.features.extractor import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS

# 1차 핵심 (50개) — 데이터 적을 때 사용 (과적합 방지)
python train_predictor.py --features core

# 전체 (96개) — 데이터 충분할 때 사용
python train_predictor.py --features full
```

### 하드웨어 자동 감지 (33개 피처)

```python
from benchmark.platform import PlatformInfo

hw = PlatformInfo.detect('mps')  # or 'cpu', 'cuda'
hw.print_summary()
```

**출력 예시 (Apple M4):**
```
  OS: macos
  가속기: apple (Apple M4)
  CPU: 10코어 (P:4, E:6), 3.5GHz boost, L2=16MB
  RAM: 24GB (unified), 120GB/s
  GPU: 10코어, 18GB, FP32 5.0 TFLOPS, FP16 10.0 TFLOPS
  인터커넥트: unified, 120GB/s
```

**감지 체인:**
1. PyTorch API (torch.cuda.get_device_properties)
2. psutil (크로스 플랫폼)
3. OS 명령 (macOS: sysctl/system_profiler, Windows: wmic→PowerShell, Linux: /proc//sys)
4. 안전한 기본값

## 팀원별 기여

### ijunsoo - 프레임워크 설계 + Zero-Config 통합

- **벤치마크 프레임워크 설계**: `benchmark/` 패키지 기반 모듈형 구조
- **모델 레지스트리**: 데코레이터 기반 팩토리 패턴
- **Feature Schema v3.0**: 44개 → 96개 피처 확장 (Feature_Schema.docx 100% 구현)
- **`benchmark/platform/` 모듈**: 33개 하드웨어 피처 크로스 플랫폼 자동 감지
- **Zero-Config 실행**: `--quick`, `--full-pipeline`, `--device auto`, `check_env.py`
- **호환성 매트릭스**: GPU 메모리 기반 설정 자동 필터링

### dal-merge (달현) - XGBoost + 하드웨어 피처

- **XGBoost + GridSearchCV**: 최적 하이퍼파라미터 자동 탐색
- **log1p 변환**: 넓은 범위(ms~수백초) 실행 시간 안정화
- **하드웨어 피처 도입**: CPU/RAM/GPU 정보 피처 추가
- **장치별 별도 모델 학습**: CPU/GPU 패턴 분리

### khg9859 (홍근) - Transformer/GAN + ONNX + Op-Level

- **Transformer/GAN 모델**: ViT, GAN(G+D) 직접 구현
- **ONNX 파이프라인**: export + predict from ONNX
- **Op-Level 프로파일러**: 개별 연산 분해/시간 측정/시뮬레이션
- **Attention FLOPs 보정**: Q@K, attn@V matmul 별도 추정

### hong (홍) - Windows/CUDA 지원 + 확장 HW 피처

- **Windows CUDA 지원**: wmic/PowerShell 기반 하드웨어 감지
- **33개 HW 피처 스키마**: fp16/bf16, P/E코어, 텐서코어, 메모리 대역폭
- **크로스 플랫폼 폰트**: Windows Malgun Gothic → NanumGothic fallback

## 핵심 인사이트

### 예측 모델
1. **XGBoost가 최적**: 모든 타겟에서 R²(log) 0.97~0.99 달성
2. **메모리는 파라미터 수로 결정**: `total_params` 중요도 0.99+
3. **추론 시간은 CPU/GPU에서 다른 피처가 중요**: CPU는 `has_residual`, GPU는 `flops`

### GPU 가속
4. **ANN은 GPU가 오히려 느림**: 모델이 작아 커널 오버헤드 지배
5. **CNN/ResNet은 GPU 12~14x 가속**
6. **MobileNet은 Mac MPS에서 35x 가속**: CPU depthwise conv 병목 해소

### 크로스 플랫폼
7. **PyTorch ARM CPU**: MKLDNN 미포함으로 depthwise conv 14배 느림
8. **BLAS 연산은 플랫폼 동등**: ANN(FC) Mac 0.9x vs Desktop
9. **메모리 구조가 핵심**: unified(Mac) vs discrete(Windows)로 데이터 전송 패턴이 달라짐
