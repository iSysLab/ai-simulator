"""통합 피처 스키마 (Merged Feature Schema) — v2.0(111) + v3.0(96) 병합, 131차원.

병합 규칙:
  - v2.0(111차원, main 브랜치)을 **전량 유지** — op-level 메모리 트래픽 피처
    (total_op_memory_write/read, flops_ratio_*, total_op_flops 등)가 논문의
    "메모리바운드" 분석 근거이므로 반드시 포함.
  - v3.0(96차원, ijunsoo)에만 있던 피처 중 **이름만 다른 중복 9개는 제외**하고,
    **새로운 20개만 추가** — 크로스플랫폼 하드웨어(gpu_tensor_core_count,
    gpu_clock_ghz 등), 레이어 카운트, GAN 층수, 파생 비율 등.

  결과 = 111 + 20 = 131.

제외한 중복 (v3 이름 → 유지한 v2 이름):
  num_filters→cnn_num_filters, kernel_size→cnn_kernel_size,
  max_channel_width→cnn_max_channels, device_type_encoded→device_encoded,
  num_attention_layers→vit_num_encoder_layers, sequence_length→seq_length,
  gpu_cores→gpu_core_count, cpu_cores→cpu_cores_physical,
  use_batchnorm→has_batch_norm

주의:
  - 문자열 피처(STRING_FEATURES)는 수치 학습에서 제외하거나 인코딩 필요.
  - op-level 피처(num_ops~std_op_flops)는 벤치 JSON에 저장돼 있지 않으므로,
    학습 전 `enrich_oplevel.py`로 모델을 재구성해 채워야 0이 아닌 값이 된다.
"""

# 131차원 통합 피처 (순서 = 학습 입력 열 순서)
MERGED_FEATURE_COLUMNS = [
    # --- [A] 공통 모델 구조 (v2.0) ---
    "total_params", "log_total_params", "trainable_params",
    "model_size_mb", "log_model_size_mb", "total_layers",
    "num_hidden_layers", "num_linear_layers", "num_conv_layers",
    "max_width", "log_max_width", "min_width", "avg_width", "base_channels",
    "model_family_encoded", "has_pooling", "has_batch_norm",
    "cnn_num_fc_layers", "cnn_kernel_size", "flops", "has_residual",
    "has_depthwise", "has_attention", "has_cls_token", "num_blocks",
    "num_mult_adds", "activation_memory_mb", "first_layer_width",
    "last_layer_width", "is_sequential", "has_skip_connection",
    "max_channels", "min_channels",
    # --- [B] 모델 전용 (v2.0) ---
    "ann_max_hidden", "ann_min_hidden", "ann_avg_hidden",
    "cnn_num_filters", "cnn_max_channels", "cnn_has_residual",
    "cnn_has_depthwise", "embed_dim", "num_heads", "patch_size", "ffn_dim",
    "vit_has_cls_token", "latent_dim", "generator_params",
    "discriminator_params", "ann_num_layers", "cnn_stem_channels",
    "vit_num_encoder_layers",
    # --- [C] 입력 데이터 (v2.0) ---
    "input_height", "input_width", "input_channels", "num_classes",
    "batch_size", "dataset_encoded", "input_pixels", "seq_length",
    # --- [D] 하드웨어 (v2.0, 33종) ---
    "device_type", "os_type", "accelerator_brand", "accelerator_name",
    "cpu_cores_physical", "cpu_cores_logical", "cpu_perf_cores",
    "cpu_efficiency_cores", "cpu_freq_base_ghz", "cpu_freq_boost_ghz",
    "cpu_cache_l2_mb", "cpu_cache_l3_mb", "ram_total_gb", "memory_type",
    "memory_bandwidth_gbs", "is_unified_memory", "shared_memory_gb",
    "dedicated_vram_gb", "gpu_count", "gpu_memory_gb", "gpu_core_count",
    "peak_bandwidth_gbs", "tflops_fp32", "tflops_fp16", "fp16_support",
    "bf16_support", "interconnect_type", "host_to_device_bandwidth_gbs",
    "is_discrete_gpu", "is_integrated_gpu", "device_encoded",
    "cpu_freq_ghz", "memory_channels",
    # --- [E] Op-level 분해 (v2.0) — 메모리바운드 분석 핵심 ---
    "conv_params", "linear_params", "bn_params", "other_params",
    "num_ops", "total_op_flops", "total_op_memory_read",
    "total_op_memory_write",
    # 주의: 'memory_bytes'(측정 메모리)는 예측 타깃이므로 피처에서 제외(누수 방지).
    "flops_ratio_Conv2d", "flops_ratio_Linear", "flops_ratio_BatchNorm2d",
    "flops_ratio_LayerNorm", "flops_ratio_MaxPool2d", "flops_ratio_ReLU",
    "flops_ratio_GELU", "max_op_flops", "avg_op_flops", "std_op_flops",
    # --- [F] v3.0에서 추가된 신규 20개 ---
    "num_bn_layers", "num_pool_layers", "num_activation_layers",
    "flops_per_sample", "params_per_flop", "has_layer_norm", "has_dropout",
    "hidden_size", "model_family", "model_arch", "stride", "padding",
    "generator_layers", "discriminator_layers", "dataset_type",
    "input_dtype", "input_elements", "gpu_tensor_core_count",
    "gpu_compute_capability", "gpu_clock_ghz",
]

# 수치 학습에서 제외할 문자열/카테고리 피처 (인코딩 버전을 대신 사용)
STRING_FEATURES = {
    "device_type", "os_type", "accelerator_brand", "accelerator_name",
    "memory_type", "interconnect_type", "model_family", "model_arch",
    "dataset_type", "input_dtype",
}

# op-level 피처 (enrich_oplevel.py가 채워야 하는 대상)
OP_LEVEL_FEATURES = [
    "num_ops", "total_op_flops", "total_op_memory_read",
    "total_op_memory_write", "flops_ratio_Conv2d", "flops_ratio_Linear",
    "flops_ratio_BatchNorm2d", "flops_ratio_LayerNorm",
    "flops_ratio_MaxPool2d", "flops_ratio_ReLU", "flops_ratio_GELU",
    "max_op_flops", "avg_op_flops", "std_op_flops",
]

NUMERIC_FEATURE_COLUMNS = [c for c in MERGED_FEATURE_COLUMNS if c not in STRING_FEATURES]

assert len(MERGED_FEATURE_COLUMNS) == 130, len(MERGED_FEATURE_COLUMNS)
