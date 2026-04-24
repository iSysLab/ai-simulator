"""ONNX 모델 추론 시간 예측 및 실측 비교

scripts/export_to_onnx.py로 내보낸 ONNX 파일을 로드하여:
  1. ONNX 그래프에서 feature 추출
  2. collect CSV 데이터로 XGBoost 즉석 학습
  3. 추론 시간 예측
  4. onnxruntime으로 실측
  5. 예측 vs 실측 비교 출력

사용법:
    python scripts/predict_from_onnx.py
"""

import os
import sys
import time

import json
import math

import joblib
import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort

os.environ.setdefault('JOBLIB_TEMP_FOLDER', 'C:/Temp/joblib')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from benchmark.support.hardware_info import get_hardware_info
from benchmark.support.device_utils import get_best_device

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR   = os.path.join(ROOT_DIR, 'data', 'onnx')
MODEL_DIR  = os.path.join(ROOT_DIR, 'predictor', 'models')
OUTPUT_CSV = os.path.join(ROOT_DIR, 'data', 'onnx_predict_results.csv')

WARMUP_RUNS  = 5
MEASURE_RUNS = 10

# hong-0311 기준 model_family_encoded 값
MODEL_FAMILY_MAP = {'ann': 0, 'cnn': 1, 'transformer': 4, 'gan': 5}

# predictor/train.py FEATURE_COLUMNS와 동일 (111개)
FEATURE_COLUMNS = [
    # 공통 모델 구조 (33개)
    'total_params', 'log_total_params', 'trainable_params',
    'model_size_mb', 'log_model_size_mb',
    'total_layers', 'num_hidden_layers', 'num_linear_layers', 'num_conv_layers',
    'max_width', 'log_max_width', 'min_width', 'avg_width',
    'base_channels', 'model_family_encoded',
    'has_pooling', 'has_batch_norm', 'cnn_num_fc_layers', 'cnn_kernel_size',
    'flops', 'has_residual', 'has_depthwise', 'has_attention', 'has_cls_token',
    'num_blocks', 'num_mult_adds', 'activation_memory_mb',
    'first_layer_width', 'last_layer_width', 'is_sequential', 'has_skip_connection',
    'max_channels', 'min_channels',
    # 모델 전용 (18개)
    'ann_max_hidden', 'ann_min_hidden', 'ann_avg_hidden',
    'cnn_num_filters', 'cnn_max_channels', 'cnn_has_residual', 'cnn_has_depthwise',
    'embed_dim', 'num_heads', 'patch_size', 'ffn_dim', 'vit_has_cls_token',
    'latent_dim', 'generator_params', 'discriminator_params',
    'ann_num_layers', 'cnn_stem_channels', 'vit_num_encoder_layers',
    # 입력 데이터 (8개)
    'input_height', 'input_width', 'input_channels', 'num_classes',
    'batch_size', 'dataset_encoded', 'input_pixels', 'seq_length',
    # 하드웨어 (33개)
    'device_type', 'os_type', 'accelerator_brand', 'accelerator_name',
    'cpu_cores_physical', 'cpu_cores_logical', 'cpu_perf_cores', 'cpu_efficiency_cores',
    'cpu_freq_base_ghz', 'cpu_freq_boost_ghz', 'cpu_cache_l2_mb', 'cpu_cache_l3_mb',
    'ram_total_gb', 'memory_type', 'memory_bandwidth_gbs', 'is_unified_memory',
    'shared_memory_gb', 'dedicated_vram_gb', 'gpu_count', 'gpu_memory_gb',
    'gpu_core_count', 'peak_bandwidth_gbs', 'tflops_fp32', 'tflops_fp16',
    'fp16_support', 'bf16_support', 'interconnect_type', 'host_to_device_bandwidth_gbs',
    'is_discrete_gpu', 'is_integrated_gpu', 'device_encoded', 'cpu_freq_ghz', 'memory_channels',
    # dal 고유 (19개)
    'conv_params', 'linear_params', 'bn_params', 'other_params',
    'num_ops', 'total_op_flops', 'total_op_memory_read', 'total_op_memory_write',
    'memory_bytes',
    'flops_ratio_Conv2d', 'flops_ratio_Linear', 'flops_ratio_BatchNorm2d',
    'flops_ratio_LayerNorm', 'flops_ratio_MaxPool2d', 'flops_ratio_ReLU', 'flops_ratio_GELU',
    'max_op_flops', 'avg_op_flops', 'std_op_flops',
]


# ── ONNX 그래프 feature 추출 ──────────────────────────────

def extract_onnx_features(onnx_path, model_type, device_str, batch_size=1):
    """ONNX 그래프를 파싱하여 feature 딕셔너리 반환

    Args:
        onnx_path: .onnx 파일 경로
        model_type: 'ann', 'cnn', 'transformer', 'gan'
        device_str: 'cpu' 또는 'cuda'
        batch_size: 추론 배치 크기

    Returns:
        dict: FEATURE_COLUMNS 키를 가진 feature 딕셔너리
    """
    # ── JSON config 로드 (export_to_onnx.py가 저장한 설정) ──
    cfg = {}
    json_path = onnx_path.replace('.onnx', '.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            cfg = json.load(f)

    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    # ── 파라미터 수 ──────────────────────────────────────
    total_params  = sum(int(np.prod(t.dims)) for t in graph.initializer)
    model_size_mb = round(total_params * 4 / (1024 ** 2), 4)

    # ── 레이어 / op 분류 ─────────────────────────────────
    type_counts = {}
    for node in graph.node:
        type_counts[node.op_type] = type_counts.get(node.op_type, 0) + 1

    conv_count   = type_counts.get('Conv', 0)
    linear_count = type_counts.get('Gemm', 0) + type_counts.get('MatMul', 0)
    bn_count     = type_counts.get('BatchNormalization', 0)
    ln_count     = type_counts.get('LayerNormalization', 0)
    relu_count   = type_counts.get('Relu', 0)
    gelu_count   = type_counts.get('Gelu', 0) + type_counts.get('FastGelu', 0)
    pool_count   = (type_counts.get('MaxPool', 0) + type_counts.get('AveragePool', 0) +
                    type_counts.get('GlobalAveragePool', 0))
    total_layers = conv_count + linear_count + bn_count
    num_ops      = len(graph.node)

    has_batch_norm = 1 if bn_count > 0 else 0
    has_pooling    = 1 if pool_count > 0 else 0
    has_attention  = 1 if model_type == 'transformer' else 0

    # ── 입력/출력 shape ───────────────────────────────────
    img_channels = cfg.get('input_channels', 0)
    input_height = cfg.get('input_height', 0)
    input_width  = cfg.get('input_width', 0)
    num_classes  = cfg.get('num_classes', 0)
    dataset_encoded = cfg.get('dataset_encoded', 0)

    # ── FLOPs 추정 ────────────────────────────────────────
    flops = total_params * 2

    # ── 파라미터 타입별 분류 ──────────────────────────────
    if total_layers > 0:
        conv_ratio   = conv_count   / total_layers
        linear_ratio = linear_count / total_layers
        bn_ratio     = bn_count     / total_layers
    else:
        conv_ratio = linear_ratio = bn_ratio = 0.0

    conv_params   = int(total_params * conv_ratio)
    linear_params = int(total_params * linear_ratio)
    bn_params     = int(total_params * bn_ratio)
    other_params  = total_params - conv_params - linear_params - bn_params

    # ── op-level 메모리/FLOPs ─────────────────────────────
    total_op_flops        = flops
    total_op_memory_read  = total_params * 4
    total_op_memory_write = total_params * 4
    memory_bytes          = total_params * 4

    flops_ratio_Conv2d      = round(conv_count   / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_Linear      = round(linear_count / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_BatchNorm2d = round(bn_count     / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_LayerNorm   = round(ln_count     / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_MaxPool2d   = round(pool_count   / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_ReLU        = round(relu_count   / num_ops, 4) if num_ops > 0 else 0.0
    flops_ratio_GELU        = round(gelu_count   / num_ops, 4) if num_ops > 0 else 0.0
    max_op_flops = flops
    avg_op_flops = round(flops / max(num_ops, 1), 0)
    std_op_flops = 0.0

    # ── 하드웨어 ──────────────────────────────────────────
    hw = get_hardware_info(device_str)

    # ── config 기반 feature ───────────────────────────────
    model_family_encoded   = MODEL_FAMILY_MAP.get(model_type, 0)

    # ANN
    hidden_size       = cfg.get('hidden_size', 0)
    num_hidden_layers = cfg.get('num_hidden_layers', 0)
    ann_max_hidden    = hidden_size if model_type == 'ann' else 0
    ann_min_hidden    = hidden_size if model_type == 'ann' else 0
    ann_avg_hidden    = float(hidden_size) if model_type == 'ann' else 0.0
    ann_num_layers    = num_hidden_layers if model_type == 'ann' else 0

    # CNN
    num_filters    = cfg.get('num_filters', 0)
    cnn_num_filters   = num_filters if model_type == 'cnn' else 0
    cnn_max_channels  = num_filters if model_type == 'cnn' else 0
    cnn_stem_channels = num_filters if model_type == 'cnn' else 0

    # Transformer
    embed_dim              = cfg.get('embed_dim', 0)
    num_heads              = cfg.get('num_heads', 0)
    patch_size             = cfg.get('patch_size', 0)
    num_transformer_layers = cfg.get('num_transformer_layers', cfg.get('num_layers', 0))
    ffn_dim                = embed_dim * 4 if model_type == 'transformer' else 0
    vit_num_encoder_layers = num_transformer_layers if model_type == 'transformer' else 0
    seq_length             = (input_height // patch_size) ** 2 if model_type == 'transformer' and patch_size > 0 else 0

    # GAN
    latent_dim = cfg.get('latent_dim', 0)
    g_hidden   = cfg.get('g_hidden_dims', [])
    generator_params     = total_params if model_type == 'gan' else 0
    discriminator_params = 0

    # 너비
    if model_type == 'ann':
        max_width = hidden_size; min_width = hidden_size; avg_width = float(hidden_size)
        first_layer_width = hidden_size; last_layer_width = hidden_size
    elif model_type == 'cnn':
        max_width = num_filters; min_width = num_filters; avg_width = float(num_filters)
        first_layer_width = num_filters; last_layer_width = num_filters
    elif model_type == 'transformer':
        max_width = embed_dim; min_width = embed_dim; avg_width = float(embed_dim)
        first_layer_width = embed_dim; last_layer_width = embed_dim
    elif model_type == 'gan' and g_hidden:
        max_width = max(g_hidden); min_width = min(g_hidden)
        avg_width = float(np.mean(g_hidden))
        first_layer_width = g_hidden[0]; last_layer_width = g_hidden[-1]
    else:
        max_width = min_width = avg_width = first_layer_width = last_layer_width = 0

    log_max_width = round(math.log1p(max_width), 6)

    feat = {
        # 공통 모델 구조 (33개)
        'total_params':          total_params,
        'log_total_params':      round(math.log1p(total_params), 6),
        'trainable_params':      total_params,
        'model_size_mb':         model_size_mb,
        'log_model_size_mb':     round(math.log1p(model_size_mb), 6),
        'total_layers':          total_layers,
        'num_hidden_layers':     num_hidden_layers,
        'num_linear_layers':     linear_count,
        'num_conv_layers':       conv_count,
        'max_width':             max_width,
        'log_max_width':         log_max_width,
        'min_width':             min_width,
        'avg_width':             round(avg_width, 4),
        'base_channels':         num_filters if model_type == 'cnn' else 0,
        'model_family_encoded':  model_family_encoded,
        'has_pooling':           has_pooling,
        'has_batch_norm':        has_batch_norm,
        'cnn_num_fc_layers':     linear_count,
        'cnn_kernel_size':       cfg.get('kernel_size', 3) if model_type == 'cnn' else 0,
        'flops':                 flops,
        'has_residual':          cfg.get('has_residual', 0),
        'has_depthwise':         0,
        'has_attention':         has_attention,
        'has_cls_token':         1 if model_type == 'transformer' else 0,
        'num_blocks':            num_transformer_layers if model_type == 'transformer' else 0,
        'num_mult_adds':         flops // 2,
        'activation_memory_mb':  0,
        'first_layer_width':     first_layer_width,
        'last_layer_width':      last_layer_width,
        'is_sequential':         1,
        'has_skip_connection':   cfg.get('has_residual', 0),
        'max_channels':          num_filters if model_type == 'cnn' else 0,
        'min_channels':          num_filters if model_type == 'cnn' else 0,
        # 모델 전용 (18개)
        'ann_max_hidden':        ann_max_hidden,
        'ann_min_hidden':        ann_min_hidden,
        'ann_avg_hidden':        ann_avg_hidden,
        'cnn_num_filters':       cnn_num_filters,
        'cnn_max_channels':      cnn_max_channels,
        'cnn_has_residual':      cfg.get('has_residual', 0) if model_type == 'cnn' else 0,
        'cnn_has_depthwise':     0,
        'embed_dim':             embed_dim,
        'num_heads':             num_heads,
        'patch_size':            patch_size,
        'ffn_dim':               ffn_dim,
        'vit_has_cls_token':     1 if model_type == 'transformer' else 0,
        'latent_dim':            latent_dim,
        'generator_params':      generator_params,
        'discriminator_params':  discriminator_params,
        'ann_num_layers':        ann_num_layers,
        'cnn_stem_channels':     cnn_stem_channels,
        'vit_num_encoder_layers': vit_num_encoder_layers,
        # 입력 데이터 (8개)
        'input_height':          input_height,
        'input_width':           input_width,
        'input_channels':        img_channels,
        'num_classes':           num_classes,
        'batch_size':            batch_size,
        'dataset_encoded':       0,
        'input_pixels':          input_height * input_width * img_channels,
        'seq_length':            seq_length,
        # 하드웨어 (33개)
        **hw,
        # dal 고유 (19개)
        'conv_params':           conv_params,
        'linear_params':         linear_params,
        'bn_params':             bn_params,
        'other_params':          other_params,
        'num_ops':               num_ops,
        'total_op_flops':        total_op_flops,
        'total_op_memory_read':  total_op_memory_read,
        'total_op_memory_write': total_op_memory_write,
        'memory_bytes':          memory_bytes,
        'flops_ratio_Conv2d':    flops_ratio_Conv2d,
        'flops_ratio_Linear':    flops_ratio_Linear,
        'flops_ratio_BatchNorm2d': flops_ratio_BatchNorm2d,
        'flops_ratio_LayerNorm': flops_ratio_LayerNorm,
        'flops_ratio_MaxPool2d': flops_ratio_MaxPool2d,
        'flops_ratio_ReLU':      flops_ratio_ReLU,
        'flops_ratio_GELU':      flops_ratio_GELU,
        'max_op_flops':          max_op_flops,
        'avg_op_flops':          avg_op_flops,
        'std_op_flops':          std_op_flops,
    }
    return feat


# ── onnxruntime 추론 시간 실측 ────────────────────────────

def measure_onnx_infer(onnx_path, dummy_np):
    """onnxruntime으로 추론 시간 측정

    Args:
        onnx_path: .onnx 파일 경로
        dummy_np: numpy 더미 입력 (float32)

    Returns:
        tuple: (평균 초, 표준편차 초)
    """
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if get_best_device() == 'cuda' else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        sess.run(None, {input_name: dummy_np})

    times = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_np})
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times))


# ── 저장된 모델 로드 ──────────────────────────────────────

def load_predictor(device_str):
    """predictor/train.py가 저장한 XGBoost 모델과 feature 컬럼 로드

    Args:
        device_str: 'cpu' 또는 'cuda'

    Returns:
        tuple: (xgb_inference 모델, feature_columns 리스트)
               모델 파일이 없으면 (None, FEATURE_COLUMNS)
    """
    device_name = 'cuda' if device_str == 'cuda' else 'cpu'
    model_path   = os.path.join(MODEL_DIR, device_name, 'xgb_inference.pkl')
    columns_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')

    if not os.path.exists(model_path):
        print(f"  [경고] 저장된 모델 없음: {model_path}")
        print("  먼저 python predictor/train.py 실행 후 재시도하세요.")
        return None, FEATURE_COLUMNS

    model   = joblib.load(model_path) # XGBoost 모델
    columns = joblib.load(columns_path) if os.path.exists(columns_path) else FEATURE_COLUMNS # feature 컬럼 목록
    print(f"  모델 로드 완료: predictor/models/{device_name}/xgb_inference.pkl")
    return model, columns


# ── ONNX 파일 열거 ────────────────────────────────────────

def find_onnx_files():
    """data/onnx/ 하위 .onnx 파일 전부 열거

    Returns:
        list: (model_type, onnx_path) 튜플 리스트
    """
    files = []
    if not os.path.isdir(ONNX_DIR):
        return files
    for model_type in os.listdir(ONNX_DIR):
        subdir = os.path.join(ONNX_DIR, model_type)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.endswith('.onnx'):
                files.append((model_type, os.path.join(subdir, fname)))
    return files


# ── 더미 입력 생성 ────────────────────────────────────────

def make_dummy(onnx_path, batch_size=1):
    """ONNX 입력 shape에 맞는 numpy 더미 텐서 생성"""
    model  = onnx.load(onnx_path)
    graph  = model.graph
    inp    = graph.input[0]
    dims   = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    dims[0] = batch_size   # batch 고정
    return np.zeros(dims, dtype=np.float32)


# ── 메인 ──────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  ONNX 추론 시간 예측 vs 실측")
    print("=" * 60)

    device_str = get_best_device()
    print(f"디바이스: {device_str}\n")

    # 저장된 예측 모델 로드
    print("[1] 예측 모델 로드")
    predictor, feat_columns = load_predictor(device_str)
    print()

    # ONNX 파일 목록
    onnx_files = find_onnx_files()
    if not onnx_files:
        print(f"ONNX 파일 없음: {ONNX_DIR}\n"
              "먼저 python scripts/export_to_onnx.py 실행 후 재시도하세요.")
        return

    print(f"[2] ONNX 파일: {len(onnx_files)}개\n")

    results = []
    for model_type, onnx_path in onnx_files:
        fname = os.path.basename(onnx_path)
        print(f"  {model_type}/{fname}")

        # feature 추출
        feat = extract_onnx_features(onnx_path, model_type, device_str)

        # 예측
        predicted_s = None
        if predictor is not None:
            x = np.array([[feat.get(c, 0) for c in feat_columns]], dtype=np.float32)
            predicted_s = float(np.expm1(predictor.predict(x)[0]))

        # 실측
        dummy_np = make_dummy(onnx_path, batch_size=1)
        try:
            actual_mean, actual_std = measure_onnx_infer(onnx_path, dummy_np)
        except Exception as e:
            print(f"    [실측 실패] {e}")
            actual_mean, actual_std = None, None

        # 출력
        pred_str   = f"{predicted_s*1000:.2f} ms" if predicted_s is not None else "N/A"
        actual_str = f"{actual_mean*1000:.2f} ms (±{actual_std*1000:.2f})" \
                     if actual_mean is not None else "N/A"
        print(f"    예측: {pred_str}  |  실측: {actual_str}")

        results.append({
            'model_type':   model_type,
            'file':         fname,
            'total_params': feat['total_params'],
            'model_size_mb': feat['model_size_mb'],
            'predicted_infer_ms': round(predicted_s * 1000, 4) if predicted_s else None,
            'actual_infer_ms':    round(actual_mean * 1000, 4) if actual_mean else None,
            'actual_infer_std_ms': round(actual_std * 1000, 4) if actual_std else None,
        })

    # 결과 저장
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {os.path.relpath(OUTPUT_CSV, ROOT_DIR)} ({len(df)}행)")


if __name__ == '__main__':
    run()
