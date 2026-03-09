"""ONNX 모델 → 실행 시간 예측 CLI 스크립트

학습된 예측 모델(joblib)을 로드하고, ONNX 파일에서 피처를 추출하여
추론/학습 시간을 예측.

사용법:
    # ANN/CNN (자동 판별)
    python predict_from_onnx.py results/onnx_samples/ann_h256_l2.onnx --device cpu

    # Transformer (embed_dim, num_heads, patch_size 직접 지정)
    python predict_from_onnx.py results/onnx_samples/vit_d128_l4_h4.onnx \\
        --device cpu --embed-dim 128 --num-heads 4 --patch-size 4

    # GAN (latent_dim 직접 지정)
    python predict_from_onnx.py results/onnx_samples/gan_z128_G256_512_1024.onnx \\
        --device cpu --latent-dim 128

    # 데모 모드 (모든 ONNX 샘플)
    python predict_from_onnx.py
"""
import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from benchmark.features.onnx_extractor import extract_features_from_onnx


def load_prediction_models(model_dir):
    """joblib으로 저장된 예측 모델 로드"""
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib 필요: pip install joblib")

    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('.pkl') and fname != 'feature_columns.pkl':
            key = fname.replace('.pkl', '')
            models[key] = joblib.load(os.path.join(model_dir, fname))

    fc_path = os.path.join(model_dir, 'feature_columns.pkl')
    if os.path.exists(fc_path):
        feature_columns = joblib.load(fc_path)
    else:
        raise FileNotFoundError(f"feature_columns.pkl 없음: {fc_path}")

    print(f"  모델 {len(models)}개 로드, 피처 {len(feature_columns)}개")
    return models, feature_columns


def features_to_array(features, feature_columns):
    """피처 딕셔너리를 numpy 배열로 변환 (순서 맞춤)"""
    row = [features.get(col, 0) for col in feature_columns]
    return np.array([row])


def predict(onnx_path, device, model_dir, embed_dim=0, num_heads=0,
            patch_size=0, latent_dim=0, batch_size=64):
    """ONNX → 피처 추출 → 실행 시간 예측"""
    print(f"\n{'='*60}")
    print(f"  ONNX 실행 시간 예측: {os.path.basename(onnx_path)}")
    print(f"  디바이스: {device}")
    print(f"{'='*60}")

    # 피처 추출
    features = extract_features_from_onnx(
        onnx_path, device=device,
        embed_dim=embed_dim, num_heads=num_heads,
        patch_size=patch_size, latent_dim=latent_dim,
        batch_size=batch_size)

    # 예측 모델 로드
    models, feature_columns = load_prediction_models(model_dir)
    X = features_to_array(features, feature_columns)

    # 예측 (log 공간 → expm1 역변환)
    print(f"\n  {'모델':<30s} | {'예측값':>12s}")
    print(f"  {'-'*30} | {'-'*12}")

    for name, model in sorted(models.items()):
        y_pred_log = model.predict(X)[0]
        y_pred = float(np.expm1(y_pred_log))
        unit = 's/epoch' if 'training' in name else 's'
        print(f"  {name:<30s} | {y_pred:>10.4f} {unit}")

    print()


def demo_mode(model_dir):
    """results/onnx_samples 폴더의 모든 ONNX 파일 예측"""
    onnx_dir = 'results/onnx_samples'
    if not os.path.exists(onnx_dir):
        print(f"ONNX 샘플 없음: {onnx_dir}")
        print("먼저 export_onnx.py를 실행하세요.")
        return

    onnx_files = sorted(f for f in os.listdir(onnx_dir) if f.endswith('.onnx'))
    print(f"\n[데모 모드] {len(onnx_files)}개 ONNX 파일 예측\n")

    # 파일명으로 Transformer/GAN 파라미터 추정
    for fname in onnx_files:
        path = os.path.join(onnx_dir, fname)
        stem = fname.replace('.onnx', '')

        kwargs = {}
        if 'vit_' in stem:
            parts = stem.split('_')
            for p in parts:
                if p.startswith('d'):
                    kwargs['embed_dim'] = int(p[1:])
                elif p.startswith('h'):
                    kwargs['num_heads'] = int(p[1:])
            kwargs['patch_size'] = 4
        elif 'gan_' in stem:
            parts = stem.split('_')
            for p in parts:
                if p.startswith('z'):
                    kwargs['latent_dim'] = int(p[1:])

        predict(path, 'cpu', model_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='ONNX 모델 실행 시간 예측')
    parser.add_argument('onnx_path', nargs='?', default=None,
                        help='ONNX 파일 경로 (없으면 데모 모드)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--model-dir', type=str,
                        default='results/trained_models')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embed-dim', type=int, default=0)
    parser.add_argument('--num-heads', type=int, default=0)
    parser.add_argument('--patch-size', type=int, default=0)
    parser.add_argument('--latent-dim', type=int, default=0)
    args = parser.parse_args()

    if args.onnx_path is None:
        demo_mode(args.model_dir)
    else:
        if not os.path.exists(args.onnx_path):
            print(f"파일 없음: {args.onnx_path}")
            sys.exit(1)
        predict(
            args.onnx_path, args.device, args.model_dir,
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            patch_size=args.patch_size, latent_dim=args.latent_dim,
            batch_size=args.batch_size)


if __name__ == '__main__':
    main()
