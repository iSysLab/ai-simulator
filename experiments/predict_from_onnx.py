# ============================================================
# Stage 5: ONNX 모델 → 실행 시간 예측 메인 CLI 스크립트
# ============================================================
#
# 역할:
#   ONNX 파일을 입력받아 Stage 3에서 학습된 XGBoost/RandomForest 모델로
#   추론 시간(ms)과 학습 시간(sec/epoch)을 예측합니다.
#
# 사용법:
#   # 기본 (ANN/CNN: 자동 판별)
#   python predict_from_onnx.py data/stage5/onnx_samples/ann_l2_w256.onnx --device cpu
#
#   # Transformer (embed_dim, num_heads, patch_size 직접 지정)
#   python predict_from_onnx.py data/stage5/onnx_samples/vit_dim128_heads4_l4.onnx \
#       --device cpu --embed-dim 128 --num-heads 4 --patch-size 4
#
#   # GAN (latent_dim 직접 지정)
#   python predict_from_onnx.py data/stage5/onnx_samples/gan_gen_z100_h256_512.onnx \
#       --device cpu --latent-dim 100
#
# 왜 Transformer/GAN은 직접 지정해야 하나?
#   ONNX 파일에는 "embed_dim이 128이다" 같은 정보가 명시적으로 없습니다.
#   Conv, Gemm 등의 연산 수로 추정하지만, Transformer는 일반 FC와 구별이 어렵습니다.
#   따라서 사용자가 직접 입력해야 정확한 예측이 가능합니다.
#
# 하드웨어: MacBook Air M1
# 작성자: 김홍근
# ============================================================

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

# sklearn 내부 경고 억제 (feature names 불일치 관련 무해한 경고)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 프로젝트 루트 경로 설정 (dnn/ 폴더)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.onnx_feature_extractor import extract_features_from_onnx

# 학습된 예측 모델 저장 경로
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained')


def load_prediction_models():
    """
    train_predictor.py에서 저장한 joblib 모델들을 불러옵니다.

    joblib.load()는 joblib.dump()로 저장한 파일을 읽어서
    학습 완료된 XGBoost, RandomForest 객체를 복원합니다.

    Returns:
        tuple: (xgb_inf, xgb_tr, rf_inf, rf_tr, feature_columns)
               - xgb_inf    : 추론 시간 예측 XGBoost 모델
               - xgb_tr     : 학습 시간 예측 XGBoost 모델
               - rf_inf     : 추론 시간 예측 RandomForest 모델
               - rf_tr      : 학습 시간 예측 RandomForest 모델
               - feature_columns: Feature 이름 목록 (순서 중요!)
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib이 설치되지 않았습니다. pip install joblib")

    required_files = [
        'xgb_inference.pkl',
        'xgb_training.pkl',
        'rf_inference.pkl',
        'rf_training.pkl',
        'feature_columns.pkl',
    ]

    # 모델 파일 존재 여부 확인
    for fname in required_files:
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {fpath}\n"
                "먼저 train_predictor.py를 실행하여 모델을 저장하세요."
            )

    print("  예측 모델 로드 중...")
    xgb_inf        = joblib.load(os.path.join(MODEL_DIR, 'xgb_inference.pkl'))
    xgb_tr         = joblib.load(os.path.join(MODEL_DIR, 'xgb_training.pkl'))
    rf_inf         = joblib.load(os.path.join(MODEL_DIR, 'rf_inference.pkl'))
    rf_tr          = joblib.load(os.path.join(MODEL_DIR, 'rf_training.pkl'))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    print(f"  → XGBoost 모델 2개, RandomForest 모델 2개 로드 완료")
    print(f"  → Feature 수: {len(feature_columns)}개")

    return xgb_inf, xgb_tr, rf_inf, rf_tr, feature_columns


def features_to_dataframe(features: dict, feature_columns: list) -> pd.DataFrame:
    """
    Feature 딕셔너리를 예측 모델이 사용할 수 있는 DataFrame으로 변환합니다.

    중요:
      feature_columns에 정의된 순서대로 Feature를 배열해야 합니다.
      순서가 다르면 예측 결과가 완전히 틀릴 수 있습니다.

    누락된 Feature는 0으로 채웁니다 (Transformer/GAN 전용 Feature 등).

    Args:
        features       (dict): extract_features_from_onnx()의 반환값
        feature_columns (list): Feature 이름 순서 목록

    Returns:
        pd.DataFrame: 1행, n열 DataFrame
    """
    row = {}
    for col in feature_columns:
        row[col] = features.get(col, 0)  # 없는 Feature는 0으로 처리
    return pd.DataFrame([row])[feature_columns]


def predict(onnx_path: str, device: str, embed_dim: int, num_heads: int,
            patch_size: int, latent_dim: int, batch_size: int):
    """
    ONNX 모델 파일로부터 실행 시간을 예측합니다.

    전체 흐름:
        ONNX 파일 → Feature 추출 → DataFrame 변환 → 예측 모델 실행 → 결과 출력

    로그 역변환(expm1):
        train_predictor.py에서 target을 log1p(y)로 변환하여 학습했으므로,
        예측값도 expm1(y_pred)로 변환해야 원래 단위(ms, sec)로 복원됩니다.
        expm1(x) = exp(x) - 1 (log1p의 역함수)
    """
    print("\n" + "=" * 60)
    print(f"  ONNX 모델 실행 시간 예측")
    print(f"  파일: {os.path.basename(onnx_path)}")
    print(f"  디바이스: {device}")
    print("=" * 60)

    # ── Step 1: ONNX에서 Feature 추출 ─────────────────────
    print("\n[1단계] ONNX 파싱 및 Feature 추출")
    features = extract_features_from_onnx(
        onnx_path=onnx_path,
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        latent_dim=latent_dim,
        batch_size=batch_size,
    )

    # ── Step 2: 예측 모델 로드 ─────────────────────────────
    print("\n[2단계] 학습된 예측 모델 로드")
    xgb_inf, xgb_tr, rf_inf, rf_tr, feature_columns = load_prediction_models()

    # ── Step 3: Feature → DataFrame ─────────────────────────
    print("\n[3단계] Feature 벡터 생성")
    X = features_to_dataframe(features, feature_columns)
    print(f"  Feature 벡터 완성: {X.shape[1]}개 Feature")

    # ── Step 4: 예측 (log 공간에서 예측 후 역변환) ──────────
    # 모델은 log1p(y)를 예측하도록 학습됨
    # 따라서 예측값에 expm1() 적용하여 원래 단위로 복원
    print("\n[4단계] 실행 시간 예측")

    xgb_inf_pred_log = xgb_inf.predict(X)[0]
    xgb_tr_pred_log  = xgb_tr.predict(X)[0]
    rf_inf_pred_log  = rf_inf.predict(X)[0]
    rf_tr_pred_log   = rf_tr.predict(X)[0]

    xgb_inf_pred_ms  = float(np.expm1(xgb_inf_pred_log))
    xgb_tr_pred_sec  = float(np.expm1(xgb_tr_pred_log))
    rf_inf_pred_ms   = float(np.expm1(rf_inf_pred_log))
    rf_tr_pred_sec   = float(np.expm1(rf_tr_pred_log))

    # ── Step 5: 결과 출력 ────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [예측 결과]")
    print(f"  파일: {os.path.basename(onnx_path)}")
    print(f"  디바이스: {device}")
    print(f"  배치 크기: {batch_size}")
    print("-" * 60)
    print(f"  추론 시간 예측 (XGBoost)     : {xgb_inf_pred_ms:10.2f} ms/batch")
    print(f"  추론 시간 예측 (RandomForest): {rf_inf_pred_ms:10.2f} ms/batch")
    print(f"  학습 시간 예측 (XGBoost)     : {xgb_tr_pred_sec:10.2f} sec/epoch")
    print(f"  학습 시간 예측 (RandomForest): {rf_tr_pred_sec:10.2f} sec/epoch")
    print("-" * 60)

    # XGBoost와 RandomForest 앙상블 평균 (두 모델 평균)
    avg_inf_ms  = (xgb_inf_pred_ms + rf_inf_pred_ms) / 2
    avg_tr_sec  = (xgb_tr_pred_sec + rf_tr_pred_sec) / 2
    print(f"  추론 시간 앙상블 평균         : {avg_inf_ms:10.2f} ms/batch")
    print(f"  학습 시간 앙상블 평균         : {avg_tr_sec:10.2f} sec/epoch")
    print("=" * 60)
    print()

    return {
        'xgb_inference_ms':  xgb_inf_pred_ms,
        'xgb_training_sec':  xgb_tr_pred_sec,
        'rf_inference_ms':   rf_inf_pred_ms,
        'rf_training_sec':   rf_tr_pred_sec,
        'avg_inference_ms':  avg_inf_ms,
        'avg_training_sec':  avg_tr_sec,
    }


def parse_args():
    """
    CLI 인자 파서 정의

    argparse: 터미널에서 스크립트 실행 시 인자를 편리하게 받는 파이썬 표준 라이브러리
    """
    parser = argparse.ArgumentParser(
        description=(
            'ONNX 모델 파일에서 추론/학습 시간을 예측합니다.\n\n'
            '예시:\n'
            '  python predict_from_onnx.py model.onnx --device cpu\n'
            '  python predict_from_onnx.py vit.onnx --device mps --embed-dim 128 --num-heads 4 --patch-size 4\n'
            '  python predict_from_onnx.py gan.onnx --device cpu --latent-dim 100'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 필수 인자
    parser.add_argument(
        'onnx_path',
        type=str,
        help='예측할 ONNX 파일 경로 (예: data/stage5/onnx_samples/ann_l2_w256.onnx)'
    )

    # 선택 인자 (기본값 있음)
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'],
        help='예측 디바이스 (cpu, mps[Mac], cuda[Windows/Linux], 기본값: cpu)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='배치 크기 (기본값: 64)'
    )

    # Transformer 전용 인자
    parser.add_argument(
        '--embed-dim', type=int, default=0,
        help='[Transformer 전용] 임베딩 차원 (예: 64, 128, 256)'
    )
    parser.add_argument(
        '--num-heads', type=int, default=0,
        help='[Transformer 전용] Attention 헤드 수 (예: 4, 8)'
    )
    parser.add_argument(
        '--patch-size', type=int, default=0,
        help='[Transformer 전용] 패치 크기 (예: 4, 8)'
    )

    # GAN 전용 인자
    parser.add_argument(
        '--latent-dim', type=int, default=0,
        help='[GAN 전용] 노이즈 벡터 차원 (예: 64, 100, 128)'
    )

    return parser.parse_args()


def demo_mode():
    """
    ONNX 파일이 없을 때 사용하는 데모 모드.
    onnx_samples 폴더에 있는 파일들을 순서대로 예측합니다.
    """
    onnx_dir = os.path.join(BASE_DIR, 'data', 'stage5', 'onnx_samples')
    if not os.path.exists(onnx_dir):
        print(f"ONNX 샘플 폴더가 없습니다: {onnx_dir}")
        print("먼저 export_to_onnx.py를 실행하여 ONNX 파일을 생성하세요.")
        return

    onnx_files = sorted([f for f in os.listdir(onnx_dir) if f.endswith('.onnx')])
    if not onnx_files:
        print("ONNX 파일이 없습니다. export_to_onnx.py를 먼저 실행하세요.")
        return

    print(f"\n[데모 모드] {len(onnx_files)}개 ONNX 파일 예측")

    # 각 파일별 Transformer/GAN 특성 매핑 (파일명으로 판별)
    demo_configs = {
        'vit_dim64_heads4_l3':      {'embed_dim': 64,  'num_heads': 4, 'patch_size': 4},
        'vit_dim128_heads4_l4':     {'embed_dim': 128, 'num_heads': 4, 'patch_size': 4},
        'gan_gen_z100_h256_512':    {'latent_dim': 100},
        'gan_gen_z64_h128_256_512': {'latent_dim': 64},
    }

    for fname in onnx_files:
        onnx_path = os.path.join(onnx_dir, fname)
        stem = fname.replace('.onnx', '')
        cfg = demo_configs.get(stem, {})

        predict(
            onnx_path=onnx_path,
            device='cpu',
            embed_dim=cfg.get('embed_dim', 0),
            num_heads=cfg.get('num_heads', 0),
            patch_size=cfg.get('patch_size', 0),
            latent_dim=cfg.get('latent_dim', 0),
            batch_size=64,
        )


if __name__ == '__main__':
    # 인자가 없으면 데모 모드 (모든 ONNX 샘플 파일 순서대로 예측)
    if len(sys.argv) == 1:
        print("인자가 없어 데모 모드로 실행합니다.")
        print("(사용법은 --help 참조)\n")
        demo_mode()
    else:
        args = parse_args()

        # ONNX 파일 존재 여부 확인
        if not os.path.exists(args.onnx_path):
            print(f"오류: ONNX 파일을 찾을 수 없습니다: {args.onnx_path}")
            sys.exit(1)

        predict(
            onnx_path=args.onnx_path,
            device=args.device,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            patch_size=args.patch_size,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
        )
