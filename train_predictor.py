"""예측 모델 학습: 모델 구조 피처 → 실행 시간 예측

dal-merge: XGBoost + GridSearchCV + log 변환
khg9859:   joblib 모델 저장 + 앙상블 예측

사용법:
    python train_predictor.py                                    # 기본 실행
    python train_predictor.py --input results/benchmark_results.json
    python train_predictor.py --cv 10                            # 10-fold CV
    python train_predictor.py --save-models                      # 학습된 모델 저장
"""
import os
import argparse
import json
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


# === 통합 Feature Schema (v1.0) ===
# extractor.py 출력 + config에서 주입 + 하드웨어 외부 주입

# 1차 핵심 세트 + 2차 확장 (공통 모델 구조)
FEATURE_COLUMNS = [
    # 3-1. 파라미터 관련
    'total_params', 'trainable_params', 'conv_params', 'linear_params',
    'bn_params', 'other_params',
    # 3-2. 레이어 수
    'total_layers', 'num_hidden_layers', 'num_conv_layers', 'num_linear_layers',
    'num_bn_layers', 'num_pool_layers', 'num_activation_layers',
    # 3-3. 폭(Width)
    'max_width', 'min_width', 'avg_width', 'max_channel_width',
    # 3-4. 연산량
    'flops', 'flops_per_sample', 'params_per_flop',
    'model_size_mb', 'memory_bytes',
    # 3-5. 구조 플래그
    'has_residual', 'has_depthwise', 'has_attention',
    'has_pooling', 'has_batch_norm', 'has_layer_norm', 'has_dropout',
    # 3-6. 모델 분류
    'model_family_encoded',
    # 4. 모델 전용 피처
    # ANN
    'hidden_size',
    # CNN
    'num_filters', 'use_batchnorm',
    # Transformer
    'embed_dim', 'num_heads', 'patch_size',
    # GAN
    'latent_dim', 'generator_params', 'discriminator_params',
    # 5. 입력 데이터 피처
    'batch_size', 'input_height', 'input_width', 'input_channels', 'num_classes',
    # 6. 하드웨어 피처 (1차 핵심)
    'device_type_encoded',
    'cpu_cores', 'cpu_freq_ghz', 'ram_total_gb',
    'gpu_cores', 'gpu_memory_gb',
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


def enrich_result(r):
    """벤치마크 결과 1건의 피처 보정 (하위 호환용)

    extractor.py가 44개 피처를 모두 생성하므로, 여기서는
    구버전 JSON 데이터에 누락된 필드만 fallback으로 채움.
    """
    model_type = r['model_type']
    config = r.get('config', {})
    enriched = dict(r)

    # extractor v2 이전 데이터 호환: 누락 필드 보정
    enriched.setdefault('num_hidden_layers',
                        r.get('num_conv_layers', 0) + r.get('num_linear_layers', 0))

    enriched.setdefault('max_width', r.get('max_channel_width', 0))
    enriched.setdefault('min_width', 0)
    enriched.setdefault('avg_width', 0)

    flops = r.get('flops', 0)
    enriched.setdefault('flops_per_sample', flops)
    enriched.setdefault('params_per_flop',
                        r.get('total_params', 0) / flops if flops > 0 else 0)

    enriched.setdefault('has_pooling', 1 if r.get('num_pool_layers', 0) > 0 else 0)
    enriched.setdefault('has_batch_norm', 1 if r.get('num_bn_layers', 0) > 0 else 0)
    enriched.setdefault('has_layer_norm', 1 if model_type == 'transformer' else 0)
    enriched.setdefault('has_dropout', 0)

    enriched.setdefault('model_family_encoded', MODEL_FAMILY_MAP.get(model_type, -1))

    # 모델 전용 피처 (구버전 호환)
    enriched.setdefault('hidden_size', config.get('hidden_size', 0))
    enriched.setdefault('num_filters', config.get('num_filters', 0))
    enriched.setdefault('use_batchnorm', 1 if config.get('use_batchnorm', False) else 0)
    enriched.setdefault('embed_dim', config.get('embed_dim', 0))
    enriched.setdefault('num_heads', config.get('num_heads', 0))
    enriched.setdefault('patch_size', config.get('patch_size', 0))
    enriched.setdefault('latent_dim', config.get('latent_dim', 0))
    enriched.setdefault('generator_params', 0)
    enriched.setdefault('discriminator_params', 0)

    # 입력 데이터 피처
    ds_info = DATASET_INFO.get(model_type, {})
    for k, v in ds_info.items():
        enriched.setdefault(k, v)

    # 하드웨어 피처
    enriched.setdefault('device_type_encoded',
                        DEVICE_TYPE_MAP.get(r.get('device', 'CPU'), 0))
    enriched.setdefault('gpu_cores', 0)

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

    # 1. LinearRegression (baseline)
    print(f"\n  LinearRegression 학습 중...")
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    lr.fit(X, y_log)
    y_pred = cross_val_predict(lr, X, y_log, cv=kf)
    r = evaluate(y_log, y_pred, 'LinearRegression', target_name)
    results.append(r)
    best_models['lr'] = lr

    # 2. RandomForest + GridSearchCV
    print(f"\n  RandomForest + GridSearchCV 학습 중...")
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }
    rf_gs = GridSearchCV(rf_base, rf_grid, cv=kf, scoring='r2',
                         n_jobs=-1, verbose=0)
    rf_gs.fit(X, y_log)
    print(f"    최적 파라미터: {rf_gs.best_params_}")
    print(f"    CV R²(log): {rf_gs.best_score_:.4f}")

    y_pred = cross_val_predict(rf_gs.best_estimator_, X, y_log, cv=kf)
    r = evaluate(y_log, y_pred, 'RandomForest', target_name)
    results.append(r)
    best_models['rf'] = rf_gs.best_estimator_

    # 3. GradientBoosting
    print(f"\n  GradientBoosting 학습 중...")
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X, y_log)
    y_pred = cross_val_predict(gb, X, y_log, cv=kf)
    r = evaluate(y_log, y_pred, 'GradientBoosting', target_name)
    results.append(r)
    best_models['gb'] = gb

    # 4. XGBoost + GridSearchCV (선택적, dal-merge에서 병합)
    if HAS_XGBOOST:
        print(f"\n  XGBoost + GridSearchCV 학습 중...")
        xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0)
        xgb_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5],
        }
        xgb_gs = GridSearchCV(xgb_base, xgb_grid, cv=kf, scoring='r2',
                               n_jobs=-1, verbose=0)
        xgb_gs.fit(X, y_log)
        print(f"    최적 파라미터: {xgb_gs.best_params_}")
        print(f"    CV R²(log): {xgb_gs.best_score_:.4f}")

        y_pred = cross_val_predict(xgb_gs.best_estimator_, X, y_log, cv=kf)
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
    args = parser.parse_args()

    print("=" * 60)
    print("  DNN 실행 시간 예측 모델 학습")
    models_used = "LR + RF + GB"
    if HAS_XGBOOST:
        models_used += " + XGBoost"
    print(f"  모델: {models_used}")
    print(f"  평가: {args.cv}-Fold CV + GridSearchCV")
    print(f"  타겟 변환: log1p (dal-merge/khg9859 병합)")
    print("=" * 60)

    # 데이터 로딩
    results = load_data(args.input)

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
