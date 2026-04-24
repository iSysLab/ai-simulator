"""예측 모델 학습: 모델 구조 피처 → 실행 시간 예측

dal-merge: XGBoost + GridSearchCV + log 변환
khg9859:   joblib 모델 저장 + 앙상블 예측

사용법:
    python scripts/train_predictor.py
    python scripts/train_predictor.py --input results/benchmark_results.json
    python scripts/train_predictor.py --cv 10
    python scripts/train_predictor.py --save-models
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import math
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Windows: GridSearchCV n_jobs=-1 시 loky 워커 경고·불안정이 잦음 → 직렬 그리드
_GRIDSEARCH_N_JOBS = 1 if sys.platform == 'win32' else -1
_CV_PREDICT_N_JOBS = 1 if sys.platform == 'win32' else None

# XGBoost (선택적)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# joblib (모델 저장용)
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# === 통합 Feature Schema v2.0 (dal-merge 111 + ijunsoo 벤치마크) ===
# `benchmark/dal/extractor.py` 출력과 동일 순서 — collect / run_benchmark 공통
FEATURE_COLUMNS = [
    # 공통 모델 구조 (hong 33)
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
    # 모델 전용 (hong 18)
    'ann_max_hidden', 'ann_min_hidden', 'ann_avg_hidden',
    'cnn_num_filters', 'cnn_max_channels', 'cnn_has_residual', 'cnn_has_depthwise',
    'embed_dim', 'num_heads', 'patch_size', 'ffn_dim', 'vit_has_cls_token',
    'latent_dim', 'generator_params', 'discriminator_params',
    'ann_num_layers', 'cnn_stem_channels', 'vit_num_encoder_layers',
    # 입력 데이터 (8)
    'input_height', 'input_width', 'input_channels', 'num_classes',
    'batch_size', 'dataset_encoded', 'input_pixels', 'seq_length',
    # 하드웨어 (33)
    'device_type', 'os_type', 'accelerator_brand', 'accelerator_name',
    'cpu_cores_physical', 'cpu_cores_logical', 'cpu_perf_cores', 'cpu_efficiency_cores',
    'cpu_freq_base_ghz', 'cpu_freq_boost_ghz', 'cpu_cache_l2_mb', 'cpu_cache_l3_mb',
    'ram_total_gb', 'memory_type', 'memory_bandwidth_gbs', 'is_unified_memory',
    'shared_memory_gb', 'dedicated_vram_gb', 'gpu_count', 'gpu_memory_gb',
    'gpu_core_count', 'peak_bandwidth_gbs', 'tflops_fp32', 'tflops_fp16',
    'fp16_support', 'bf16_support', 'interconnect_type', 'host_to_device_bandwidth_gbs',
    'is_discrete_gpu', 'is_integrated_gpu', 'device_encoded', 'cpu_freq_ghz', 'memory_channels',
    # dal 고유 (op-level + param 분해)
    'conv_params', 'linear_params', 'bn_params', 'other_params',
    'num_ops', 'total_op_flops', 'total_op_memory_read', 'total_op_memory_write',
    'memory_bytes',
    'flops_ratio_Conv2d', 'flops_ratio_Linear', 'flops_ratio_BatchNorm2d',
    'flops_ratio_LayerNorm', 'flops_ratio_MaxPool2d', 'flops_ratio_ReLU', 'flops_ratio_GELU',
    'max_op_flops', 'avg_op_flops', 'std_op_flops',
]

# 예측 대상 (시간 + 공간 요구량)
TARGET_COLUMNS = ['avg_train', 'avg_infer', 'memory_bytes']

# 모델 계열 인코딩
MODEL_FAMILY_MAP = {
    'simple_ann': 0,
    'simple_cnn': 1,
    'resnet_mnist': 2,
    'mobilenet_mnist': 3,
    'transformer': 4,
    'gan': 5,
}

# 장치 인코딩
DEVICE_TYPE_MAP = {
    'CPU': 0,
    'GPU(CUDA)': 1,
    'GPU(MPS)': 2,
}

# 데이터셋 → 입력 정보 매핑
DATASET_INFO = {
    'simple_ann':       {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10, 'batch_size': 64},
    'simple_cnn':       {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10, 'batch_size': 64},
    'resnet_mnist':     {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10, 'batch_size': 64},
    'mobilenet_mnist':  {'input_height': 28, 'input_width': 28, 'input_channels': 1, 'num_classes': 10, 'batch_size': 64},
    'transformer':      {'input_height': 32, 'input_width': 32, 'input_channels': 3, 'num_classes': 10, 'batch_size': 64},
    'gan':              {'input_height': 32, 'input_width': 32, 'input_channels': 3, 'num_classes': 10, 'batch_size': 64},
}


def load_data(input_path):
    """벤치마크 결과 JSON 로딩"""
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"총 {len(results)}개 데이터 로딩 완료\n")
    return results


def filter_benchmark_results(
    results,
    *,
    filter_device=None,
    min_avg_accuracy=None,
    dedupe=False,
    dedupe_keep='last',
):
    """메타모델 학습용 품질 게이트 (원본 JSON은 변경하지 않음).

    Args:
        results: 벤치마크 JSON 리스트
        filter_device: 이 라벨과 일치하는 `device`만 유지 (예: 'CPU', 'GPU(CUDA)')
        min_avg_accuracy: 분류 모델만 적용. `avg_accuracy`가 이 값 미만이면 제외.
            `gan`은 정확도가 없으므로 항상 유지.
        dedupe: True이면 (model_name, device)당 1행만 유지
        dedupe_keep: 'last' | 'first' — 중복 시 마지막/첫 번째 행 유지

    Returns:
        필터된 리스트
    """
    out = list(results)

    n0 = len(out)

    if filter_device:
        out = [r for r in out if r.get('device') == filter_device]
        print(f"  [필터] device == {filter_device!r}: {n0} → {len(out)}개")

    if min_avg_accuracy is not None:
        thr = float(min_avg_accuracy)
        before = len(out)
        kept = []
        for r in out:
            if r.get('model_type') == 'gan':
                kept.append(r)
                continue
            acc = float(r.get('avg_accuracy', 0) or 0)
            if acc >= thr:
                kept.append(r)
        out = kept
        print(f"  [필터] avg_accuracy >= {thr}% (GAN 제외): {before} → {len(out)}개")

    if dedupe:
        before = len(out)
        if dedupe_keep == 'last':
            by_key = {}
            for r in out:
                key = (r.get('model_name'), r.get('device'))
                by_key[key] = r
            out = [by_key[k] for k in sorted(by_key.keys())]
        else:
            by_key = {}
            for r in out:
                key = (r.get('model_name'), r.get('device'))
                if key not in by_key:
                    by_key[key] = r
            out = [by_key[k] for k in sorted(by_key.keys())]
        print(f"  [필터] (model_name, device) 중복 제거 ({dedupe_keep}): "
              f"{before} → {len(out)}개")

    print(f"  품질 게이트 후: {len(out)}개 (원본 대비 {n0} → {len(out)})\n")
    return out


def enrich_result(r):
    """벤치마크 JSON 1건을 111차원 스키마에 맞게 보정 (구버전·부분 필드 호환)."""
    model_type = r.get('model_type', '')
    config = r.get('config', {})
    enriched = dict(r)
    tp = float(r.get('total_params', 0) or 0)
    ms = float(r.get('model_size_mb', 0) or 0)

    enriched.setdefault('log_total_params', round(math.log1p(tp), 6))
    enriched.setdefault('log_model_size_mb', round(math.log1p(ms), 6))
    enriched.setdefault('num_hidden_layers',
                        r.get('num_hidden_layers',
                              r.get('num_conv_layers', 0) + r.get('num_linear_layers', 0)))
    mw = r.get('max_width', r.get('max_channel_width', 0))
    enriched.setdefault('max_width', mw)
    enriched.setdefault('log_max_width', round(math.log1p(float(mw or 0)), 6))
    enriched.setdefault('min_width', r.get('min_width', 0))
    enriched.setdefault('avg_width', r.get('avg_width', mw or 0))
    enriched.setdefault('base_channels', r.get('base_channels', r.get('num_filters', 0)))

    enriched.setdefault('model_family_encoded', MODEL_FAMILY_MAP.get(model_type, -1))

    enriched.setdefault('has_pooling', 1 if r.get('num_pool_layers', 0) > 0 else r.get('has_pooling', 0))
    enriched.setdefault('has_batch_norm', 1 if r.get('num_bn_layers', 0) > 0 else r.get('has_batch_norm', 0))
    enriched.setdefault('cnn_num_fc_layers', r.get('cnn_num_fc_layers', 1))
    enriched.setdefault('cnn_kernel_size', r.get('cnn_kernel_size', 3))
    enriched.setdefault('num_mult_adds', r.get('num_mult_adds', r.get('flops', 0) // 2))
    enriched.setdefault('activation_memory_mb', r.get('activation_memory_mb', 0))
    enriched.setdefault('first_layer_width', r.get('first_layer_width', mw or 0))
    enriched.setdefault('last_layer_width', r.get('last_layer_width', mw or 0))
    enriched.setdefault('is_sequential', r.get('is_sequential', 1))
    enriched.setdefault('has_skip_connection', r.get('has_skip_connection', r.get('has_residual', 0)))
    enriched.setdefault('max_channels', r.get('max_channels', r.get('max_channel_width', 0)))
    enriched.setdefault('min_channels', r.get('min_channels', 0))

    enriched.setdefault('ann_max_hidden', config.get('hidden_size', 0) if model_type == 'simple_ann' else 0)
    enriched.setdefault('ann_min_hidden', enriched['ann_max_hidden'])
    enriched.setdefault('ann_avg_hidden', float(enriched['ann_max_hidden']))
    enriched.setdefault('ann_num_layers', config.get('num_layers', 0) if model_type == 'simple_ann' else 0)

    nf = config.get('num_filters', 0)
    enriched.setdefault('cnn_num_filters', nf if 'cnn' in model_type or model_type in ('simple_cnn', 'resnet_mnist', 'mobilenet_mnist') else 0)
    enriched.setdefault('cnn_max_channels', r.get('cnn_max_channels', enriched['cnn_num_filters']))
    enriched.setdefault('cnn_has_residual', r.get('cnn_has_residual', 1 if model_type == 'resnet_mnist' else 0))
    enriched.setdefault('cnn_has_depthwise', r.get('cnn_has_depthwise', 1 if model_type == 'mobilenet_mnist' else 0))
    enriched.setdefault('cnn_stem_channels', r.get('cnn_stem_channels', nf))

    ed = config.get('embed_dim', 0)
    enriched.setdefault('embed_dim', ed)
    enriched.setdefault('num_heads', config.get('num_heads', 0))
    enriched.setdefault('patch_size', config.get('patch_size', 0))
    enriched.setdefault('ffn_dim', r.get('ffn_dim', ed * 4 if model_type == 'transformer' else 0))
    enriched.setdefault('vit_has_cls_token', r.get('vit_has_cls_token', 1 if model_type == 'transformer' else 0))
    enriched.setdefault('vit_num_encoder_layers', r.get('vit_num_encoder_layers', config.get('num_layers', 0)))

    enriched.setdefault('latent_dim', config.get('latent_dim', 0))
    enriched.setdefault('generator_params', r.get('generator_params', 0))
    enriched.setdefault('discriminator_params', r.get('discriminator_params', 0))

    ds_info = DATASET_INFO.get(model_type, {})
    enriched.setdefault('input_height', r.get('input_height', ds_info.get('input_height', 28)))
    enriched.setdefault('input_width', r.get('input_width', ds_info.get('input_width', 28)))
    enriched.setdefault('input_channels', r.get('input_channels', ds_info.get('input_channels', 1)))
    enriched.setdefault('num_classes', r.get('num_classes', ds_info.get('num_classes', 10)))
    enriched.setdefault('batch_size', r.get('batch_size', ds_info.get('batch_size', 64)))
    de = 1 if model_type in ('transformer', 'gan') else 0
    enriched.setdefault('dataset_encoded', r.get('dataset_encoded', de))
    ih, iw, ic = enriched['input_height'], enriched['input_width'], enriched['input_channels']
    enriched.setdefault('input_pixels', r.get('input_pixels', ih * iw * ic))
    ps = enriched.get('patch_size', 0)
    enriched.setdefault('seq_length', r.get('seq_length', (ih // ps) ** 2 if model_type == 'transformer' and ps else 0))

    # 하드웨어: 구 JSON에는 일부만 있음 → 나머지 0
    dev = r.get('device', 'CPU')
    enriched.setdefault('device_type', r.get('device_type', 1 if dev != 'CPU' else 0))
    enriched.setdefault('device_encoded', r.get('device_encoded', 1 if dev != 'CPU' else 0))
    enriched.setdefault('cpu_freq_ghz', r.get('cpu_freq_ghz', 0))
    enriched.setdefault('cpu_cores_physical', r.get('cpu_cores_physical', r.get('cpu_cores', 0)))
    enriched.setdefault('cpu_cores_logical', r.get('cpu_cores_logical', r.get('cpu_cores', 0)))

    for col in FEATURE_COLUMNS:
        enriched.setdefault(col, 0.0)

    return enriched


def prepare_features(results):
    """결과에서 피처 행렬(X)과 타겟 벡터(Y) 추출

    Schema v1.0에 맞춰 config/하드웨어/입력 피처를 주입한 후 추출.
    """
    X = []
    y_train = []
    y_infer = []
    y_memory = []
    devices = []
    names = []

    for r in results:
        enriched = enrich_result(r)
        row = []
        for col in FEATURE_COLUMNS:
            val = enriched.get(col)
            if val is None:
                val = 0
            row.append(float(val))

        X.append(row)
        y_train.append(r['avg_train'])
        y_infer.append(r['avg_infer'])
        y_memory.append(r.get('memory_bytes', 0))
        devices.append(r['device'])
        names.append(r['model_name'])

    return (np.array(X), np.array(y_train), np.array(y_infer),
            np.array(y_memory), devices, names)


def evaluate(y_true_log, y_pred_log, model_name, target_name):
    """log 공간 예측값을 원래 단위로 변환 후 성능 평가 (dal-merge 방식)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2_log = r2_score(y_true_log, y_pred_log)

    print(f"  {model_name:25s} | "
          f"R²: {r2:.4f} | R²(log): {r2_log:.4f} | "
          f"RMSE: {rmse:.5f} | MAE: {mae:.5f}")

    return {
        'model': model_name,
        'target': target_name,
        'R2': round(r2, 4),
        'R2_log': round(r2_log, 4),
        'RMSE': round(rmse, 5),
        'MAE': round(mae, 5),
    }


def train_and_evaluate(X, y_log, target_name, cv_folds=5):
    """여러 회귀 모델 학습 + GridSearchCV + K-Fold 교차 검증

    dal-merge: XGBoost + GridSearchCV
    khg9859:   log 변환 + 교차 검증 예측값 수집

    Returns:
        tuple: (결과 목록, 최적 모델 dict)
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    best_models = {}

    # 1. RandomForest + GridSearchCV
    print(f"\n  RandomForest + GridSearchCV 학습 중...")
    # GridSearchCV가 병렬이면 베이스 추정기는 n_jobs=1 (중첩 병렬 방지, Windows loky 경고 완화)
    rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }
    rf_gs = GridSearchCV(rf_base, rf_grid, cv=kf, scoring='r2',
                         n_jobs=_GRIDSEARCH_N_JOBS, verbose=0)
    rf_gs.fit(X, y_log)
    print(f"    최적 파라미터: {rf_gs.best_params_}")
    print(f"    CV R²(log): {rf_gs.best_score_:.4f}")

    y_pred = cross_val_predict(
        rf_gs.best_estimator_, X, y_log, cv=kf, n_jobs=_CV_PREDICT_N_JOBS)
    r = evaluate(y_log, y_pred, 'RandomForest', target_name)
    results.append(r)
    best_models['rf'] = rf_gs.best_estimator_

    # 2. GradientBoosting
    print(f"\n  GradientBoosting 학습 중...")
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X, y_log)
    y_pred = cross_val_predict(gb, X, y_log, cv=kf, n_jobs=_CV_PREDICT_N_JOBS)
    r = evaluate(y_log, y_pred, 'GradientBoosting', target_name)
    results.append(r)
    best_models['gb'] = gb

    # 3. XGBoost + GridSearchCV (선택적, dal-merge에서 병합)
    if HAS_XGBOOST:
        print(f"\n  XGBoost + GridSearchCV 학습 중...")
        xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1)
        xgb_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5],
        }
        xgb_gs = GridSearchCV(xgb_base, xgb_grid, cv=kf, scoring='r2',
                               n_jobs=_GRIDSEARCH_N_JOBS, verbose=0)
        xgb_gs.fit(X, y_log)
        print(f"    최적 파라미터: {xgb_gs.best_params_}")
        print(f"    CV R²(log): {xgb_gs.best_score_:.4f}")

        y_pred = cross_val_predict(
            xgb_gs.best_estimator_, X, y_log, cv=kf, n_jobs=_CV_PREDICT_N_JOBS)
        r = evaluate(y_log, y_pred, 'XGBoost', target_name)
        results.append(r)
        best_models['xgb'] = xgb_gs.best_estimator_

    return results, best_models


def feature_importance(model, feature_names, target_name, top_n=10):
    """피처 중요도 분석 (RandomForest/XGBoost)"""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  피처 중요도 (상위 {top_n}개) -{target_name}:")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1:2d}. {feature_names[idx]:<30s}: "
              f"{importances[idx]:.4f}")


def save_models(models_dict, target_name, output_dir):
    """학습된 모델을 joblib으로 저장 (khg9859에서 병합)"""
    if not HAS_JOBLIB:
        print("  joblib 미설치 -모델 저장 건너뜀")
        return

    os.makedirs(output_dir, exist_ok=True)

    for key, model in models_dict.items():
        fname = f"{key}_{target_name}.pkl"
        path = os.path.join(output_dir, fname)
        joblib.dump(model, path)
        print(f"  저장: {path}")

    # 피처 컬럼 순서도 저장
    fc_path = os.path.join(output_dir, 'feature_columns.pkl')
    if not os.path.exists(fc_path):
        joblib.dump(FEATURE_COLUMNS, fc_path)
        print(f"  저장: {fc_path}")


def main():
    parser = argparse.ArgumentParser(description='실행 시간 예측 모델 학습')
    parser.add_argument('--input', type=str,
                        default='results/benchmark_results.json',
                        help='벤치마크 결과 JSON 경로')
    parser.add_argument('--cv', type=int, default=5,
                        help='교차 검증 폴드 수 (기본: 5)')
    parser.add_argument('--save-models', action='store_true',
                        help='학습된 모델을 joblib으로 저장')
    parser.add_argument('--model-dir', type=str,
                        default='results/trained_models',
                        help='모델 저장 디렉토리')
    parser.add_argument('--filter-device', type=str, default=None,
                        help="장치 라벨로만 학습 (예: CPU, GPU(CUDA), GPU(MPS))")
    parser.add_argument('--min-avg-accuracy', type=float, default=None,
                        help='분류 모델만: avg_accuracy(%%) 미만 행 제외 (GAN은 제외하지 않음)')
    parser.add_argument('--dedupe', action='store_true',
                        help='(model_name, device) 중복 시 1행만 유지')
    parser.add_argument('--dedupe-keep', type=str, default='last',
                        choices=['first', 'last'],
                        help='--dedupe 시 동일 키에서 유지할 행 (기본: last)')
    args = parser.parse_args()

    print("=" * 60)
    print("  DNN 실행 시간 예측 모델 학습")
    models_used = "RF + GB"
    if HAS_XGBOOST:
        models_used += " + XGBoost"
    print(f"  모델: {models_used}")
    print(f"  평가: {args.cv}-Fold CV + GridSearchCV")
    print(f"  타겟 변환: log1p (dal-merge/khg9859 병합)")
    print("=" * 60)

    # 데이터 로딩
    results = load_data(args.input)

    if (args.filter_device is not None or args.min_avg_accuracy is not None
            or args.dedupe):
        print("학습용 데이터 품질 게이트 적용 중...")
        results = filter_benchmark_results(
            results,
            filter_device=args.filter_device,
            min_avg_accuracy=args.min_avg_accuracy,
            dedupe=args.dedupe,
            dedupe_keep=args.dedupe_keep,
        )
        if not results:
            print("오류: 필터 후 남은 샘플이 없습니다. 조건을 완화하세요.",
                  file=sys.stderr)
            sys.exit(1)

    # 피처 준비
    X, y_train, y_infer, y_memory, devices, names = prepare_features(results)
    print(f"피처 행렬: {X.shape[0]}개 샘플 × {X.shape[1]}개 피처")
    print(f"학습 시간 범위: {y_train.min():.5f}s ~ {y_train.max():.5f}s")
    print(f"추론 시간 범위: {y_infer.min():.5f}s ~ {y_infer.max():.5f}s")
    print(f"메모리 범위: {y_memory.min():.0f}B ~ {y_memory.max():.0f}B\n")

    # 장치별 분리 (dal-merge 방식: CPU와 GPU 관계가 다르므로)
    unique_devices = sorted(set(devices))
    all_results = []

    for dev in unique_devices:
        mask = np.array([d == dev for d in devices])
        X_dev = X[mask]
        y_train_dev = y_train[mask]
        y_infer_dev = y_infer[mask]
        y_memory_dev = y_memory[mask]

        if len(X_dev) < args.cv:
            print(f"[{dev}] 데이터 부족 ({len(X_dev)}개) -건너뜀\n")
            continue

        print(f"{'='*60}")
        print(f"[{dev}] 데이터: {len(X_dev)}개")
        print(f"{'='*60}")

        # log 변환 (dal-merge + khg9859 핵심 기법)
        y_train_log = np.log1p(y_train_dev)
        y_infer_log = np.log1p(y_infer_dev)

        # 학습 시간 예측
        print(f"\n--- 학습 시간 예측 ---")
        train_results, train_models = train_and_evaluate(
            X_dev, y_train_log, f'학습시간 [{dev}]', cv_folds=args.cv)
        all_results.extend(train_results)

        # 피처 중요도 (최적 모델)
        for key in ['rf', 'xgb']:
            if key in train_models:
                feature_importance(
                    train_models[key], FEATURE_COLUMNS,
                    f'학습시간 [{dev}] -{key.upper()}')

        # 모델 저장
        if args.save_models:
            save_models(train_models, 'training', args.model_dir)

        # 추론 시간 예측
        print(f"\n--- 추론 시간 예측 ---")
        infer_results, infer_models = train_and_evaluate(
            X_dev, y_infer_log, f'추론시간 [{dev}]', cv_folds=args.cv)
        all_results.extend(infer_results)

        for key in ['rf', 'xgb']:
            if key in infer_models:
                feature_importance(
                    infer_models[key], FEATURE_COLUMNS,
                    f'추론시간 [{dev}] -{key.upper()}')

        if args.save_models:
            save_models(infer_models, 'inference', args.model_dir)

        # 메모리/공간 요구량 예측
        if y_memory_dev.max() > 0:
            print(f"\n--- 메모리 요구량 예측 ---")
            y_memory_log = np.log1p(y_memory_dev)
            mem_results, mem_models = train_and_evaluate(
                X_dev, y_memory_log, f'메모리 [{dev}]', cv_folds=args.cv)
            all_results.extend(mem_results)

            for key in ['rf', 'xgb']:
                if key in mem_models:
                    feature_importance(
                        mem_models[key], FEATURE_COLUMNS,
                        f'메모리 [{dev}] -{key.upper()}')

            if args.save_models:
                save_models(mem_models, 'memory', args.model_dir)

        print()

    # 최종 성능 비교 테이블
    print(f"\n{'='*60}")
    print("  [최종 성능 비교]")
    print(f"{'='*60}")
    print(f"  {'모델':<25s} | {'타겟':<20s} | {'R²':>6s} | {'R²(log)':>8s} | "
          f"{'RMSE':>10s} | {'MAE':>10s}")
    print(f"  {'-'*25} | {'-'*20} | {'-'*6} | {'-'*8} | {'-'*10} | {'-'*10}")
    for r in all_results:
        print(f"  {r['model']:<25s} | {r['target']:<20s} | "
              f"{r['R2']:>6.4f} | {r['R2_log']:>8.4f} | "
              f"{r['RMSE']:>10.5f} | {r['MAE']:>10.5f}")

    print("\n예측 모델 학습 완료!")


if __name__ == '__main__':
    main()
