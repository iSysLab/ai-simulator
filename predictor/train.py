"""ANN + CNN 실행 시간 예측 모델 학습

수집된 ann_results.csv / cnn_results.csv를 읽어서
LinearRegression / Random Forest / XGBoost로
학습 시간 및 추론 시간 예측 모델을 학습하고 K-Fold 교차검증으로 성능 평가

사용법:
    python predictor/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

ROOT_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_CSV_PATH     = os.path.join(ROOT_DIR, 'data', 'ann_results.csv')
CNN_CSV_PATH     = os.path.join(ROOT_DIR, 'data', 'cnn_results.csv')
RESULT_CSV_PATH  = os.path.join(ROOT_DIR, 'data', 'predictor_results.csv')

# ── Feature / Target 설정 ─────────────────────────────────
FEATURE_COLUMNS = [
    # 모델 구조 (공통)
    'total_params', 'trainable_params', 'linear_params',
    'flops', 'model_size_mb', 'num_layers', 'model_type',
    # ANN 전용
    'hidden_size', 'num_hidden_layers',
    # CNN 전용 (ANN은 0)
    'conv_params', 'num_conv_layers', 'num_filters', 'has_batchnorm',
    'has_pooling', 'kernel_size', 'num_fc_layers',
    # 하드웨어
    'device', 'cpu_cores', 'cpu_freq_ghz', 'cpu_cache_l2_mb',
    'ram_total_gb', 'gpu_memory_gb',
    # 입력 데이터
    'batch_size', 'input_channels', 'input_height', 'input_width', 'num_classes',
]

TARGET_TRAIN = 'train_time_mean'
TARGET_INFER = 'infer_time_mean'


# ── 데이터 로드 ───────────────────────────────────────────

def load_data():
    """ANN + CNN CSV 로드 및 전처리

    - 존재하는 CSV만 읽어서 합침 (ANN만 있어도 동작)
    - device 문자열 → 숫자 인코딩 (cpu=0, cuda=1, mps=2)
    - NaN → 0 대체
    """
    dfs = []
    for path, name in [(ANN_CSV_PATH, 'ANN'), (CNN_CSV_PATH, 'CNN')]:
        if os.path.exists(path):
            d = pd.read_csv(path)
            print(f"  {name} 데이터: {len(d)}행")
            dfs.append(d)
        else:
            print(f"  {name} 데이터: 없음 (건너뜀)")

    df = pd.concat(dfs, ignore_index=True)
    print(f"합계: {len(df)}행\n")

    df['device'] = df['device'].map({'cpu': 0, 'cuda': 1, 'mps': 2})
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    print(f"  CPU 데이터: {(df['device'] == 0).sum()}개")
    print(f"  CUDA 데이터: {(df['device'] == 1).sum()}개")
    print(f"  ANN 데이터: {(df['model_type'] == 0).sum()}개")
    print(f"  CNN 데이터: {(df['model_type'] == 1).sum()}개")
    return df


# ── 성능 평가 ─────────────────────────────────────────────

def evaluate(y_true_log, y_pred_log, model_name, target_name):
    """log 공간 예측값을 원래 단위로 변환 후 성능 평가

    log1p 역변환(expm1)으로 원래 단위(초) 복원 후
    R², RMSE, MAE 계산
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
        'model':  model_name,
        'target': target_name,
        'R2':     round(r2, 4),
        'R2_log': round(r2_log, 4),
        'RMSE':   round(rmse, 5),
        'MAE':    round(mae, 5),
    }


# ── feature 중요도 출력 ───────────────────────────────────

def print_feature_importance(model, target_name):
    """RandomForest / XGBoost feature 중요도 상위 10개 출력"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\n  [feature 중요도 - {target_name}] 상위 10개")
    for i in range(min(10, len(FEATURE_COLUMNS))):
        idx = indices[i]
        print(f"    {i+1:2d}. {FEATURE_COLUMNS[idx]:<25s}: {importances[idx]:.4f}")


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
        return model, y_pred_log

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

    gs = GridSearchCV(base, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X, y_log)

    print(f"    최적 파라미터: {gs.best_params_}")
    print(f"    K-Fold CV R² (log): {gs.best_score_:.4f}")

    # 최적 모델로 교차검증 예측값 수집
    y_pred_log = cross_val_predict(gs.best_estimator_, X, y_log, cv=kf)

    return gs.best_estimator_, y_pred_log


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
    device_map = {0: 'CPU', 1: 'CUDA', 2: 'MPS'}
    all_results = []

    for device_code, device_name in device_map.items():
        subset = df[df['device'] == device_code]
        if len(subset) < 5:
            print(f"\n[{device_name}] 데이터 부족 ({len(subset)}개) — 건너뜀")
            continue

        print(f"\n{'='*60}")
        print(f"  [{device_name}] 데이터: {len(subset)}개")
        print(f"{'='*60}")

        X = subset[FEATURE_COLUMNS].values
        # 타겟에 log1p 변환 적용
        # 값 범위가 넓을 때(예: 0.001s ~ 100s) 그대로 학습하면 큰 값에 편향됨
        # log1p 변환으로 범위를 균일하게 만들어 예측 성능 향상
        y_train_log = np.log1p(subset[TARGET_TRAIN].values)
        y_infer_log = np.log1p(subset[TARGET_INFER].values)

        # 학습 시간 예측
        print(f"\n  [학습 시간 예측]")
        lr_tr,  lr_tr_pred  = train_and_evaluate(X, y_train_log, 'lr',  '학습시간')
        rf_tr,  rf_tr_pred  = train_and_evaluate(X, y_train_log, 'rf',  '학습시간')
        xgb_tr, xgb_tr_pred = train_and_evaluate(X, y_train_log, 'xgb', '학습시간')
        print()
        r = evaluate(y_train_log, lr_tr_pred,  'LinearRegression', f'학습시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_train_log, rf_tr_pred,  'RandomForest',     f'학습시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_train_log, xgb_tr_pred, 'XGBoost',          f'학습시간 [{device_name}]')
        all_results.append(r)
        print_feature_importance(rf_tr,  f'학습시간 [{device_name}] - RandomForest')
        print_feature_importance(xgb_tr, f'학습시간 [{device_name}] - XGBoost')

        # 추론 시간 예측
        print(f"\n  [추론 시간 예측]")
        lr_inf,  lr_inf_pred  = train_and_evaluate(X, y_infer_log, 'lr',  '추론시간')
        rf_inf,  rf_inf_pred  = train_and_evaluate(X, y_infer_log, 'rf',  '추론시간')
        xgb_inf, xgb_inf_pred = train_and_evaluate(X, y_infer_log, 'xgb', '추론시간')
        print()
        r = evaluate(y_infer_log, lr_inf_pred,  'LinearRegression', f'추론시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_infer_log, rf_inf_pred,  'RandomForest',     f'추론시간 [{device_name}]')
        all_results.append(r)
        r = evaluate(y_infer_log, xgb_inf_pred, 'XGBoost',          f'추론시간 [{device_name}]')
        all_results.append(r)
        print_feature_importance(rf_inf,  f'추론시간 [{device_name}] - RandomForest')
        print_feature_importance(xgb_inf, f'추론시간 [{device_name}] - XGBoost')

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
    print(f"결과 저장 완료: {RESULT_CSV_PATH} ({len(results_df)}행)")


if __name__ == '__main__':
    main()
