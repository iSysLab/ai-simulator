"""ANN + CNN + Transformer + GAN 실행 시간 예측 모델 학습

수집된 ann_results.csv / cnn_results.csv / transformer_results.csv / gan_results.csv를 읽어서
LinearRegression / Random Forest / XGBoost로
학습 시간 및 추론 시간 예측 모델을 학습하고 K-Fold 교차검증으로 성능 평가

사용법:
    python predictor/train.py
"""

import os
import sys
import numpy as np
import pandas as pd

# joblib 임시 폴더를 한글 없는 경로로 설정 (UnicodeEncodeError 방지)
os.environ.setdefault('JOBLIB_TEMP_FOLDER', 'C:/Temp/joblib')
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

ROOT_DIR                  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR                 = os.path.join(ROOT_DIR, 'predictor', 'models')
ANN_CSV_PATH              = os.path.join(ROOT_DIR, 'data', 'ann_results.csv')
CNN_CSV_PATH              = os.path.join(ROOT_DIR, 'data', 'cnn_results.csv')
TRANSFORMER_CSV_PATH      = os.path.join(ROOT_DIR, 'data', 'transformer_results.csv')
GAN_CSV_PATH              = os.path.join(ROOT_DIR, 'data', 'gan_results.csv')
RESULT_CSV_PATH           = os.path.join(ROOT_DIR, 'data', 'predictor_results.csv')
IMPORTANCE_CSV_PATH       = os.path.join(ROOT_DIR, 'data', 'feature_importance_results.csv')

# ── Feature / Target 설정 ─────────────────────────────────
FEATURE_COLUMNS = [
    # ── 공통 모델 구조 (hong 33개) ────────────────────────
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
    # ── 모델 전용 (hong 18개) ─────────────────────────────
    'ann_max_hidden', 'ann_min_hidden', 'ann_avg_hidden',
    'cnn_num_filters', 'cnn_max_channels', 'cnn_has_residual', 'cnn_has_depthwise',
    'embed_dim', 'num_heads', 'patch_size', 'ffn_dim', 'vit_has_cls_token',
    'latent_dim', 'generator_params', 'discriminator_params',
    'ann_num_layers', 'cnn_stem_channels', 'vit_num_encoder_layers',
    # ── 입력 데이터 (hong 8개) ────────────────────────────
    'input_height', 'input_width', 'input_channels', 'num_classes',
    'batch_size', 'dataset_encoded', 'input_pixels', 'seq_length',
    # ── 하드웨어 (hong 33개) ──────────────────────────────
    'device_type', 'os_type', 'accelerator_brand', 'accelerator_name',
    'cpu_cores_physical', 'cpu_cores_logical', 'cpu_perf_cores', 'cpu_efficiency_cores',
    'cpu_freq_base_ghz', 'cpu_freq_boost_ghz', 'cpu_cache_l2_mb', 'cpu_cache_l3_mb',
    'ram_total_gb', 'memory_type', 'memory_bandwidth_gbs', 'is_unified_memory',
    'shared_memory_gb', 'dedicated_vram_gb', 'gpu_count', 'gpu_memory_gb',
    'gpu_core_count', 'peak_bandwidth_gbs', 'tflops_fp32', 'tflops_fp16',
    'fp16_support', 'bf16_support', 'interconnect_type', 'host_to_device_bandwidth_gbs',
    'is_discrete_gpu', 'is_integrated_gpu', 'device_encoded', 'cpu_freq_ghz', 'memory_channels',
    # ── dal 고유 (op-level 15개 + param types 4개) ────────
    'conv_params', 'linear_params', 'bn_params', 'other_params',
    'num_ops', 'total_op_flops', 'total_op_memory_read', 'total_op_memory_write',
    'memory_bytes',
    'flops_ratio_Conv2d', 'flops_ratio_Linear', 'flops_ratio_BatchNorm2d',
    'flops_ratio_LayerNorm', 'flops_ratio_MaxPool2d', 'flops_ratio_ReLU', 'flops_ratio_GELU',
    'max_op_flops', 'avg_op_flops', 'std_op_flops',
]

TARGET_TRAIN = 'training_time_mean_sec'
TARGET_INFER = 'inference_time_mean_ms'


# ── 데이터 로드 ───────────────────────────────────────────

def load_data():
    """ANN + CNN + Transformer + GAN CSV 로드 및 전처리

    - 존재하는 CSV만 읽어서 합침 (일부 없어도 동작)
    - device 문자열 → 숫자 인코딩 (cpu=0, cuda=1)
    - NaN → 0 대체
    """
    dfs = []
    sources = [
        (ANN_CSV_PATH,         'ANN'),
        (CNN_CSV_PATH,         'CNN'),
        (TRANSFORMER_CSV_PATH, 'Transformer'),
        (GAN_CSV_PATH,         'GAN'),
    ]
    for path, name in sources:
        if os.path.exists(path):
            d = pd.read_csv(path)
            print(f"  {name} 데이터: {len(d)}행")
            dfs.append(d)
        else:
            print(f"  {name} 데이터: 없음 (건너뜀)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"합계: {len(df)}행\n")

    df['device'] = df['device'].map({'cpu': 0, 'cuda': 1})
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    print(f"  CPU 데이터: {(df['device'] == 0).sum()}개")
    print(f"  CUDA 데이터: {(df['device'] == 1).sum()}개")
    print(f"  ANN 데이터: {(df['model_family_encoded'] == 0).sum()}개")
    print(f"  CNN 데이터: {(df['model_family_encoded'] == 1).sum()}개")
    print(f"  Transformer 데이터: {(df['model_family_encoded'] == 4).sum()}개")
    print(f"  GAN 데이터: {(df['model_family_encoded'] == 5).sum()}개")
    return df


# ── 성능 평가 ─────────────────────────────────────────────

def evaluate(y_true_log, y_pred_log, model_name, target_name):
    """log 공간 예측값을 원래 단위로 변환 후 성능 평가

    log1p 역변환(expm1)으로 원래 단위(초) 복원 후
    R², RMSE, MAE 계산

    평가 지표 설명
    - R²      : 모델이 데이터를 얼마나 잘 설명하는지 나타내는 지표
    - RMSE    : 예측값과 실제값 차이의 제곱 평균의 루트 (큰 오차에 민감)
    - MAE     : 예측값과 실제값 차이의 절대값 평균
    - R²_log  : log 공간에서의 모델 설명력
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    r2     = r2_score(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2_log = r2_score(y_true_log, y_pred_log)

    print(f"  [{model_name}] {target_name}")
    print(f"    R² (원래 단위): {r2:.4f} | R² (log): {r2_log:.4f} | "
          f"RMSE: {rmse:.5f} | MAE: {mae:.5f}")

    return {
        'model':      model_name,
        'target':     target_name,
        'R2':         round(r2, 4),
        'R2_log':     round(r2_log, 4),
        'RMSE':       round(rmse, 5),
        'MAE':        round(mae, 5),
        'best_params': None,
        'cv_r2_log':  None,
    }


# ── feature 중요도 출력 ───────────────────────────────────

def print_feature_importance(model, target_name):
    """RandomForest / XGBoost feature 중요도 상위 10개 출력 및 반환

    Args:
        model: 학습이 끝난 RF 또는 XGBoost 모델 객체
               (model.feature_importances_ 속성으로 중요도 접근)
        target_name: 출력용 타겟 이름
    """
    # 학습된 트리 구조에서 각 feature의 기여도를 배열로 추출 (합계=1)
    importances = model.feature_importances_
    # 중요도 높은 순으로 feature 인덱스 정렬
    indices = np.argsort(importances)[::-1]
    print(f"\n  [feature 중요도 - {target_name}] 상위 10개")
    rows = []
    for i in range(len(FEATURE_COLUMNS)):
        idx = indices[i]
        if i < 10:
            print(f"    {i+1:2d}. {FEATURE_COLUMNS[idx]:<25s}: {importances[idx]:.4f}")
        rows.append({
            'target':    target_name,
            'rank':      i + 1,
            'feature':   FEATURE_COLUMNS[idx],
            'importance': round(float(importances[idx]), 6),
        })
    return rows


# ── 학습 및 교차검증 ─────────────────────────────────────

def train_and_evaluate(X, y_log, model_type, target_name):
    """GridSearchCV로 하이퍼파라미터 튜닝 + K-Fold(5) 교차검증

    Args:
        X: feature 행렬
        y_log: log1p 변환된 타겟 벡터
        model_type: 'lr' (LinearRegression), 'rf' (Random Forest), 'xgb' (XGBoost)
        target_name: 출력용 타겟 이름

    Returns:
        tuple: (최적 모델, 교차검증 예측값)
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    name_map = {'lr': 'LinearRegression', 'rf': 'RandomForest', 'xgb': 'XGBoost'}
    name = name_map[model_type]
    print(f"\n  {name} × {target_name} 학습 중...")

    if model_type == 'lr':
        # LinearRegression은 파라미터 없으므로 GridSearch 없이 직접 교차검증
        model = LinearRegression()
        model.fit(X, y_log)
        y_pred_log = cross_val_predict(model, X, y_log, cv=kf)
        return model, y_pred_log, None, None

    if model_type == 'rf':
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators':     [100, 200],
            'max_depth':        [None, 5, 10],
            'min_samples_split':[2, 5],
        }
    else:
        base = xgb.XGBRegressor(random_state=42, verbosity=0)
        param_grid = {
            'n_estimators':  [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth':     [3, 5],
        }

    # param_grid에 정의된 여러 하이퍼파라미터 조합을 시험하고 가장 좋은 조합 반환
    gs = GridSearchCV(base, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X, y_log)

    print(f"    최적 파라미터: {gs.best_params_}")
    print(f"    K-Fold CV R² (log): {gs.best_score_:.4f}")

    # 최적 모델로 교차검증 예측값 수집
    y_pred_log = cross_val_predict(gs.best_estimator_, X, y_log, cv=kf)

    return gs.best_estimator_, y_pred_log, gs.best_params_, round(gs.best_score_, 4)


# ── 메인 ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANN + CNN 실행 시간 예측 모델 학습")
    print("  LinearRegression + RandomForest + XGBoost")
    print("  K-Fold(5) + GridSearchCV")
    print("=" * 60)

    df = load_data()
    print(f"학습 시간 범위: {df[TARGET_TRAIN].min():.4f}s ~ {df[TARGET_TRAIN].max():.4f}s")
    print(f"추론 시간 범위: {df[TARGET_INFER].min():.4f}s ~ {df[TARGET_INFER].max():.4f}s")
    print(f"Feature 수: {len(FEATURE_COLUMNS)}개")

    # ── 장치별 분리 학습 ──────────────────────────────────
    device_map = {0: 'CPU', 1: 'CUDA'}
    all_results = []
    all_importances = []

    for device_code, device_name in device_map.items():
        subset = df[df['device'] == device_code]
        if len(subset) < 5:
            print(f"\n[{device_name}] 데이터 부족 ({len(subset)}개) — 건너뜀")
            continue

        print(f"\n{'='*60}")
        print(f"  [{device_name}] 데이터: {len(subset)}개")
        print(f"{'='*60}")

        X = subset[FEATURE_COLUMNS].values
        y_train_log = np.log1p(subset[TARGET_TRAIN].values)
        y_infer_log = np.log1p(subset[TARGET_INFER].values)

        # 학습 시간 예측
        print(f"\n  [학습 시간 예측]")
        lr_tr,  lr_tr_pred,  _,         _          = train_and_evaluate(X, y_train_log, 'lr',  '학습시간')
        rf_tr,  rf_tr_pred,  rf_tr_p,   rf_tr_cv   = train_and_evaluate(X, y_train_log, 'rf',  '학습시간')
        xgb_tr, xgb_tr_pred, xgb_tr_p,  xgb_tr_cv  = train_and_evaluate(X, y_train_log, 'xgb', '학습시간')
        print()
        r = evaluate(y_train_log, lr_tr_pred,  'LinearRegression', f'학습시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_train_log, rf_tr_pred,  'RandomForest',     f'학습시간 [{device_name}]')
        r['best_params'] = str(rf_tr_p);  r['cv_r2_log'] = rf_tr_cv
        all_results.append(r)
        r = evaluate(y_train_log, xgb_tr_pred, 'XGBoost',          f'학습시간 [{device_name}]')
        r['best_params'] = str(xgb_tr_p); r['cv_r2_log'] = xgb_tr_cv
        all_results.append(r)
        for row in print_feature_importance(rf_tr,  f'학습시간 [{device_name}] - RandomForest'):
            row['device'] = device_name; all_importances.append(row)
        for row in print_feature_importance(xgb_tr, f'학습시간 [{device_name}] - XGBoost'):
            row['device'] = device_name; all_importances.append(row)

        # 추론 시간 예측
        print(f"\n  [추론 시간 예측]")
        lr_inf,  lr_inf_pred,  _,          _           = train_and_evaluate(X, y_infer_log, 'lr',  '추론시간')
        rf_inf,  rf_inf_pred,  rf_inf_p,   rf_inf_cv   = train_and_evaluate(X, y_infer_log, 'rf',  '추론시간')
        xgb_inf, xgb_inf_pred, xgb_inf_p,  xgb_inf_cv  = train_and_evaluate(X, y_infer_log, 'xgb', '추론시간')
        print()
        r = evaluate(y_infer_log, lr_inf_pred,  'LinearRegression', f'추론시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_infer_log, rf_inf_pred,  'RandomForest',     f'추론시간 [{device_name}]')
        r['best_params'] = str(rf_inf_p);  r['cv_r2_log'] = rf_inf_cv
        all_results.append(r)
        r = evaluate(y_infer_log, xgb_inf_pred, 'XGBoost',          f'추론시간 [{device_name}]')
        r['best_params'] = str(xgb_inf_p); r['cv_r2_log'] = xgb_inf_cv
        all_results.append(r)
        for row in print_feature_importance(rf_inf,  f'추론시간 [{device_name}] - RandomForest'):
            row['device'] = device_name; all_importances.append(row)
        for row in print_feature_importance(xgb_inf, f'추론시간 [{device_name}] - XGBoost'):
            row['device'] = device_name; all_importances.append(row)

        # ── 모델 저장 ─────────────────────────────────────
        save_dir = os.path.join(MODEL_DIR, device_name.lower())
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(xgb_tr,  os.path.join(save_dir, 'xgb_training.pkl'))
        joblib.dump(xgb_inf, os.path.join(save_dir, 'xgb_inference.pkl'))
        joblib.dump(rf_tr,   os.path.join(save_dir, 'rf_training.pkl'))
        joblib.dump(rf_inf,  os.path.join(save_dir, 'rf_inference.pkl'))
        print(f"\n  모델 저장 완료: predictor/models/{device_name.lower()}/")

    # feature 컬럼 순서 저장 (predict_from_onnx.py에서 사용)
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, 'feature_columns.pkl'))

    # ── 최종 결과 출력 ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  [최종 성능 비교]")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results)
    print(results_df[['model', 'target', 'R2', 'R2_log', 'RMSE', 'MAE']].to_string(index=False))
    print()

    # ── 결과 CSV 저장 ─────────────────────────────────────
    os.makedirs(os.path.dirname(RESULT_CSV_PATH), exist_ok=True)
    results_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"성능 결과 저장: {RESULT_CSV_PATH} ({len(results_df)}행)")

    imp_df = pd.DataFrame(all_importances)
    imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"중요도 결과 저장: {IMPORTANCE_CSV_PATH} ({len(imp_df)}행)")


if __name__ == '__main__':
    main()