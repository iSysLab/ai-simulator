# 통합 Feature Schema

> DNN/SNN 추론 시간 예측 · 성능 예측용 피처 통합 문서  
> v1.0

---

## 1. 통합 원칙

- 같은 의미의 피처는 이름을 통일하고 하나만 유지
- 파생 가능한 값은 원본 하나만 저장 (예: `img_size` → `input_height` / `input_width`)
- 모델 전용 피처는 공통 피처와 분리하여 별도 그룹으로 관리
- 하드웨어는 `device_type` 단일 컬럼으로 끝내지 않고 메모리 구조까지 세분화
- 범주형(categorical) 값은 raw 값과 encoded 값을 모두 보존

---

## 2. 중복·통합 대상 정리

| 기존 피처명들 | 통합 후 피처명 | 비고 |
|---|---|---|
| `num_hidden_layers` / `num_layers` / `total_layers` / `depth` | `total_layers` + `num_hidden_layers` | depth는 total_layers와 동의어, 하나만 유지 |
| `num_fc_layers` / `num_linear_layers` / `cnn_num_fc_layers` | `num_linear_layers` | PyTorch 표준 명칭(nn.Linear) 사용. 현재 코드의 cnn_num_fc_layers와 매핑 |
| `img_size` | `input_height` + `input_width` | 정사각형 가정 제거, 일반화 |
| `img_channels` / `input_channels` | `input_channels` | 통일 |
| `ram_gb` / `ram_total_gb` | `ram_total_gb` | 통일 |
| `model_type_encoded` / `model_type` | `model_family` + `model_arch` + `model_family_encoded` | 계층적으로 분리 |

---

## 3. 공통 모델 구조 피처

> 모든 모델(ANN / CNN / Transformer / GAN)에 공통 적용

### 3-1. 파라미터 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `total_params` | 전체 파라미터 수 | 핵심 피처 |
| `trainable_params` | 학습 가능한 파라미터 수 | |
| `conv_params` | 합성곱 레이어 파라미터 수 | CNN 중심 |
| `linear_params` | 선형층(FC) 파라미터 수 | |
| `bn_params` | 배치 정규화 파라미터 수 | |
| `other_params` | 기타 파라미터 수 | 위 합계 외 나머지 |

### 3-2. 레이어 수 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `total_layers` | 전체 레이어 수 | |
| `num_hidden_layers` | 은닉층 수 | |
| `num_conv_layers` | 합성곱 레이어 수 | CNN 계열 |
| `num_linear_layers` | 선형층(FC) 수 | num_fc_layers 통합 |
| `num_bn_layers` | 배치 정규화 레이어 수 | |
| `num_pool_layers` | 풀링 레이어 수 | |
| `num_activation_layers` | 활성화 함수 레이어 수 | |

### 3-3. 폭(Width) 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `max_width` | 가장 넓은 층 너비(뉴런 수) | |
| `min_width` | 가장 좁은 층 너비 | |
| `avg_width` | 평균 층 너비 | |
| `max_channel_width` | 최대 채널 폭 | CNN 중심, width와 별도 유지 |

### 3-4. 연산량 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `flops` | 총 FLOPs (부동소수점 연산 수) | 핵심 피처 |
| `flops_per_sample` | 샘플 1개당 FLOPs | |
| `params_per_flop` | 연산 대비 파라미터 비율 | 파생값 |
| `model_size_mb` | 모델 파일 크기 (MB) | 저장 크기 |
| `memory_bytes` | 추정 메모리 사용량 (bytes) | 실행 시 메모리 |

### 3-5. 구조 플래그 (0/1)

| 피처명 | 설명 | 비고 |
|---|---|---|
| `has_residual` | 잔차 연결(ResNet류) 유무 | |
| `has_depthwise` | 깊이별 합성곱 유무 | MobileNet류 |
| `has_attention` | 어텐션 메커니즘 유무 | Transformer 포함 |
| `has_pooling` | 풀링 유무 | |
| `has_batch_norm` | 배치 정규화 유무 | |
| `has_layer_norm` | 레이어 정규화 유무 | Transformer 중심 |
| `has_dropout` | 드롭아웃 유무 | |
| `has_skip_connection` | 스킵 연결 유무 | U-Net류 포함 넓은 개념 |

### 3-6. 모델 분류

| 피처명 | 설명 | 비고 |
|---|---|---|
| `model_family` | 모델 계열 (ANN/CNN/Transformer/GAN) | raw 범주형 |
| `model_arch` | 세부 아키텍처 (resnet/vit/dcgan 등) | raw 범주형 |
| `model_family_encoded` | 숫자 인코딩된 model_family | 학습용 |

---

## 4. 모델 전용 피처

> 해당 모델 계열일 때만 값을 채우고, 아닌 경우 null 또는 0으로 유지

### 4-1. ANN 전용

| 피처명 | 설명 | 비고 |
|---|---|---|
| `hidden_size` | 은닉층 크기 (히든 유닛 수) | 공통 피처만으로도 대부분 설명 가능 |

### 4-2. CNN 전용

| 피처명 | 설명 | 비고 |
|---|---|---|
| `num_filters` | 필터 수 | |
| `kernel_size` | 커널 크기 | avg / max 분리 가능 |
| `stride` | 보폭 (Stride) | |
| `padding` | 패딩 크기 | |
| `max_channels` | 최대 채널 수 | max_channel_width와 구분 |

### 4-3. Transformer 전용

| 피처명 | 설명 | 비고 |
|---|---|---|
| `embed_dim` | 임베딩 차원 | |
| `num_heads` | 멀티헤드 어텐션 헤드 수 | |
| `patch_size` | 패치 크기 | ViT에서 사용 |
| `ffn_dim` | 피드포워드 네트워크 차원 | |
| `num_attention_layers` | 어텐션 레이어 수 | |
| `sequence_length` | 시퀀스 길이 | NLP Transformer |
| `has_cls_token` | 클래스 토큰 유무 (0/1) | ViT |

### 4-4. GAN 전용

| 피처명 | 설명 | 비고 |
|---|---|---|
| `latent_dim` | 잠재 벡터 차원 | Generator 입력 |
| `generator_params` | 생성기 파라미터 수 | |
| `discriminator_params` | 판별기 파라미터 수 | |
| `generator_layers` | 생성기 레이어 수 | |
| `discriminator_layers` | 판별기 레이어 수 | |

---

## 5. 입력 데이터 피처

| 피처명 | 설명 | 비고 |
|---|---|---|
| `batch_size` | 배치 크기 | 핵심 피처 |
| `input_height` | 입력 높이 | img_size 대체 |
| `input_width` | 입력 너비 | img_size 대체 |
| `input_channels` | 입력 채널 수 | img_channels 통일 |
| `num_classes` | 클래스 수 | |
| `dataset_type` | 데이터셋 종류 (MNIST/CIFAR 등) | 범주형 |
| `input_dtype` | 입력 데이터 타입 (float32/float16 등) | |
| `input_elements` | 입력 총 원소 수 (H×W×C) | 파생값 (선택) |

---

## 6. 하드웨어 피처

> Mac(Apple Silicon, MPS)과 Windows(NVIDIA CUDA)는 메모리 구조가 근본적으로 다릅니다.  
> `device_type` 단일 컬럼만으로는 이 차이를 설명하기 부족하므로 아래 피처들을 반드시 추가합니다.

### 6-1. 디바이스 일반

| 피처명 | 설명 | 비고 |
|---|---|---|
| `device_type` | cpu / cuda / mps | 기존 피처 유지 |
| `os_type` | macos / windows / linux | 신규 추가 |
| `accelerator_brand` | apple / nvidia / amd / intel / none | 신규 추가 |
| `accelerator_name` | M1, M2, RTX 4060, RTX 4090 등 | 구체적 모델명 |

### 6-2. CPU 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `cpu_cores_physical` | 물리 코어 수 | |
| `cpu_cores_logical` | 논리 코어 수 | 하이퍼스레딩 포함 |
| `cpu_perf_cores` | 성능 코어 수 | Apple Silicon 중요 |
| `cpu_efficiency_cores` | 효율 코어 수 | Apple Silicon 중요 |
| `cpu_freq_base_ghz` | 기본 클럭 (GHz) | |
| `cpu_freq_boost_ghz` | 최대 부스트 클럭 (GHz) | |
| `cpu_cache_l2_mb` | L2 캐시 크기 (MB) | |
| `cpu_cache_l3_mb` | L3 캐시 크기 (MB) | |

### 6-3. 메모리 구조 (Mac/Windows 핵심 차이)

| 피처명 | 설명 | Mac (MPS) | Windows (CUDA) |
|---|---|---|---|
| `ram_total_gb` | 전체 RAM (GB) | 16 / 24 / 36... | 16 / 32 / 64... |
| `memory_type` | 메모리 타입 | unified | ddr4 / ddr5 |
| `memory_bandwidth_gbs` | 메모리 대역폭 (GB/s) | 높음 (공유) | 분리 (각자) |
| `is_unified_memory` | 통합 메모리 여부 (0/1) | 1 | 0 |
| `shared_memory_gb` | CPU/GPU 공유 메모리 크기 (GB) | 전체 RAM | 0 또는 소량 |
| `dedicated_vram_gb` | 전용 VRAM 크기 (GB) | 0 (없음) | 6 / 8 / 12... |

### 6-4. GPU 관련

| 피처명 | 설명 | 비고 |
|---|---|---|
| `gpu_count` | GPU 개수 | |
| `gpu_memory_gb` | GPU 메모리 (GB) | VRAM 크기 |
| `gpu_core_count` | GPU 코어 수 | |
| `gpu_tensor_core_count` | 텐서 코어 수 | NVIDIA 전용 |
| `gpu_compute_capability` | CUDA Compute Capability | NVIDIA 전용 |
| `gpu_clock_ghz` | GPU 클럭 (GHz) | |
| `peak_bandwidth_gbs` | 메모리 대역폭 (GB/s) | |
| `tflops_fp32` | FP32 최대 성능 (TFLOPS) | |
| `tflops_fp16` | FP16 최대 성능 (TFLOPS) | |
| `fp16_support` | FP16 지원 여부 (0/1) | |
| `bf16_support` | BF16 지원 여부 (0/1) | |

### 6-5. 전송/인터커넥트 (Windows/CUDA 중심)

| 피처명 | 설명 | 비고 |
|---|---|---|
| `interconnect_type` | pcie3 / pcie4 / pcie5 / unified | Mac은 unified |
| `host_to_device_bandwidth_gbs` | CPU→GPU 전송 대역폭 (GB/s) | Windows 핵심 |
| `is_discrete_gpu` | 외장 GPU 여부 (0/1) | |
| `is_integrated_gpu` | 내장 GPU 여부 (0/1) | |

---

## 7. 단계적 도입 권장 순서

> 데이터 수가 적을 경우 처음부터 모든 피처를 넣으면 과적합 위험이 있습니다.  
> **현재 프로젝트(114개 데이터) 기준: 1차 핵심 세트만 먼저 적용 권장.**  
> 92개 전체 적용 시 과적합 위험이 있으므로, 데이터 확보 후 단계적으로 확장합니다.

### 1차 핵심 세트 (시작 시 필수)

| 모델 피처 | 입력 피처 | 하드웨어 피처 |
|---|---|---|
| `total_params` | `batch_size` | `device_type` |
| `trainable_params` | `input_height` | `os_type` |
| `total_layers` | `input_width` | `cpu_cores_physical` |
| `num_conv_layers` | `input_channels` | `ram_total_gb` |
| `num_linear_layers` | `num_classes` | `is_unified_memory` |
| `flops` | | `gpu_memory_gb` |
| `model_size_mb` | | `peak_bandwidth_gbs` |
| `has_attention` | | `tflops_fp32` |
| `has_batch_norm` | | `fp16_support` |
| `model_family` | | |

### 2차 확장 세트 (성능 차이가 충분히 설명 안 될 때)

- `params_per_flop`, `memory_bytes`, `num_activation_layers`, `num_bn_layers`
- `max_width`, `avg_width`, `shared_memory_gb`, `dedicated_vram_gb`
- `interconnect_type`, `host_to_device_bandwidth_gbs`

### 3차 모델 전용 세트 (Transformer / GAN 추가 시)

- `embed_dim`, `num_heads`, `patch_size`, `ffn_dim` (Transformer)
- `latent_dim`, `generator_params`, `discriminator_params` (GAN)

---

## 8. 현재 프로젝트(33개)와의 매핑표

> 현재 `train_predictor.py`의 FEATURE_COLUMNS(33개)와 통합 스키마 간 매핑.  
> 교수님 유의사항으로 33개 유지 중이며, 향후 DNN+SNN 통합 시 이 스키마를 단계적으로 적용합니다.

| 현재 피처 (33개) | 통합 스키마 대응 | 비고 |
|---|---|---|
| `total_params` | `total_params` | 동일 |
| `log_total_params` | (파생) log1p(total_params) | 학습용 변환 |
| `model_size_mb` | `model_size_mb` | 동일 |
| `log_model_size_mb` | (파생) | 동일 |
| `num_layers` | `total_layers` | 명칭 통일 |
| `num_hidden_layers` | `num_hidden_layers` | 동일 |
| `num_conv_layers` | `num_conv_layers` | 동일 |
| `max_width` | `max_width` | 동일 |
| `log_max_width` | (파생) | 동일 |
| `min_width` | `min_width` | 동일 |
| `avg_width` | `avg_width` | 동일 |
| `base_channels` | `max_channel_width` 또는 CNN 전용 | |
| `model_type_encoded` | `model_family_encoded` | 동일 개념 |
| `cnn_has_pooling` | `has_pooling` | 동일 |
| `cnn_has_batchnorm` | `has_batch_norm` | 동일 |
| `cnn_num_fc_layers` | `num_linear_layers` | 명칭 통일 |
| `cnn_kernel_size` | `kernel_size` (CNN 전용) | 동일 |
| `device_encoded` | `device_type` | 동일 개념 |
| `cpu_cores` | `cpu_cores_logical` | |
| `cpu_freq_ghz` | `cpu_freq_boost_ghz` | |
| `cpu_cache_l2_mb` | `cpu_cache_l2_mb` | 동일 |
| `ram_total_gb` | `ram_total_gb` | 동일 |
| `gpu_memory_gb` | `gpu_memory_gb` | 동일 |
| `input_channels` | `input_channels` | 동일 |
| `input_height` | `input_height` | 동일 |
| `input_width` | `input_width` | 동일 |
| `num_classes` | `num_classes` | 동일 |
| `batch_size` | `batch_size` | 동일 |
| `dataset_encoded` | `dataset_type` (encoded) | 동일 |
| `embed_dim` | `embed_dim` | 동일 |
| `num_heads` | `num_heads` | 동일 |
| `patch_size` | `patch_size` | 동일 |
| `latent_dim` | `latent_dim` | 동일 |

---

## 요약

| 그룹 | 피처 수 |
|---|---|
| 공통 모델 구조 피처 | 33개 |
| 모델 전용 피처 (ANN/CNN/Transformer/GAN) | 18개 |
| 입력 데이터 피처 | 8개 |
| 하드웨어 피처 | 33개 |
| **총합** | **92개** |

> 한 모델 기준 실제 채워지는 피처 수: 약 **79~81개** (전용 피처는 해당 계열만 채움)
