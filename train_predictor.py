"""예측 모델 학습: 모델 구조 피처 → 실행 시간 예측

사용법:
    python train_predictor.py                                    # 기본 실행
    python train_predictor.py --input results/benchmark_results.json
    python train_predictor.py --cv 10                            # 10-fold CV
"""
import argparse
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# 입력 피처 목록
FEATURE_COLUMNS = [
    'total_params', 'trainable_params', 'conv_params', 'linear_params',
    'bn_params', 'other_params',
    'num_conv_layers', 'num_linear_layers', 'num_bn_layers',
    'num_pool_layers', 'num_activation_layers', 'total_layers',
    'flops', 'memory_bytes', 'depth', 'max_channel_width',
    'has_residual', 'has_depthwise', 'has_attention',
    'model_type_simple_ann', 'model_type_simple_cnn',
    'model_type_resnet_mnist', 'model_type_mobilenet_mnist',
]

# 예측 대상
TARGET_COLUMNS = ['avg_train', 'avg_infer']


def load_data(input_path):
    """벤치마크 결과 JSON 로딩"""
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"총 {len(results)}개 데이터 로딩 완료\n")
    return results


def prepare_features(results):
    """결과에서 피처 행렬(X)과 타겟 벡터(Y) 추출"""
    X = []
    y_train = []
    y_infer = []
    devices = []
    names = []

    for r in results:
        row = []
        valid = True
        for col in FEATURE_COLUMNS:
            val = r.get(col)
            if val is None:
                valid = False
                break
            row.append(float(val))

        if not valid:
            continue

        X.append(row)
        y_train.append(r['avg_train'])
        y_infer.append(r['avg_infer'])
        devices.append(r['device'])
        names.append(r['model_name'])

    return (np.array(X), np.array(y_train), np.array(y_infer),
            devices, names)


def train_and_evaluate(X, y, target_name, cv_folds=5):
    """여러 회귀 모델 학습 및 교차 검증 평가

    Returns:
        dict: 모델별 R², MAE 점수
    """
    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        mae_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

        results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
        }

        print(f"  {name:25s} | "
              f"R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f}) | "
              f"MAE: {-mae_scores.mean():.4f}s (±{mae_scores.std():.4f})")

    return results


def feature_importance(X, y, feature_names):
    """RandomForest로 피처 중요도 분석"""
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n  피처 중요도 (상위 10개):")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1:2d}. {feature_names[idx]:<30s}: "
              f"{importances[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='실행 시간 예측 모델 학습')
    parser.add_argument('--input', type=str,
                        default='results/benchmark_results.json',
                        help='벤치마크 결과 JSON 경로')
    parser.add_argument('--cv', type=int, default=5,
                        help='교차 검증 폴드 수 (기본: 5)')
    args = parser.parse_args()

    # 데이터 로딩
    results = load_data(args.input)

    # 피처 준비
    X, y_train, y_infer, devices, names = prepare_features(results)
    print(f"피처 행렬: {X.shape[0]}개 샘플 × {X.shape[1]}개 피처\n")

    # 장치별 분리
    unique_devices = sorted(set(devices))

    for dev in unique_devices:
        mask = np.array([d == dev for d in devices])
        X_dev = X[mask]
        y_train_dev = y_train[mask]
        y_infer_dev = y_infer[mask]

        if len(X_dev) < args.cv:
            print(f"[{dev}] 데이터 부족 ({len(X_dev)}개) — 건너뜀\n")
            continue

        print(f"{'='*60}")
        print(f"[{dev}] 데이터: {len(X_dev)}개")
        print(f"{'='*60}")

        # 학습 시간 예측
        print(f"\n--- 학습 시간 예측 ---")
        train_results = train_and_evaluate(
            X_dev, y_train_dev, '학습시간', cv_folds=args.cv)
        feature_importance(X_dev, y_train_dev, FEATURE_COLUMNS)

        # 추론 시간 예측
        print(f"\n--- 추론 시간 예측 ---")
        infer_results = train_and_evaluate(
            X_dev, y_infer_dev, '추론시간', cv_folds=args.cv)
        feature_importance(X_dev, y_infer_dev, FEATURE_COLUMNS)

        print()

    print("\n예측 모델 학습 완료!")


if __name__ == '__main__':
    main()
