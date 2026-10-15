"""PyTorch 모델 → ONNX 파일 변환 스크립트

벤치마크에 사용된 모든 모델 타입의 대표 구성을 ONNX로 저장.
khg9859 브랜치에서 포팅 + 모듈형 구조에 맞게 수정.

사용법:
    python export_onnx.py
    python export_onnx.py --output-dir results/onnx_samples
"""
import os
import argparse
import logging
import warnings
import torch

logging.getLogger("torch.onnx").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from benchmark.models import simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist
from benchmark.models import transformer, gan
from benchmark.models.registry import create_model


def export_model(model, dummy_input, save_path, input_names=None,
                 output_names=None, opset_version=11):
    """PyTorch 모델을 ONNX로 저장"""
    model.eval()
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    print(f"  {os.path.basename(save_path)}", end='', flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model, dummy_input, save_path,
                dynamo=False,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
            )
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        params = sum(p.numel() for p in model.parameters())
        print(f" → {size_mb:.2f} MB, {params:,} params")
        return True
    except Exception as e:
        print(f" → 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='모델 ONNX 변환')
    parser.add_argument('--output-dir', type=str,
                        default='results/onnx_samples',
                        help='ONNX 파일 저장 디렉토리')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  PyTorch → ONNX 변환")
    print(f"  저장 위치: {args.output_dir}")
    print("=" * 60)

    # ANN 대표 모델
    print("\n[ANN]")
    for h, l in [(128, 1), (256, 2), (512, 3)]:
        m = create_model('simple_ann', hidden_size=h, num_layers=l)
        dummy = torch.randn(1, 1, 28, 28)
        path = os.path.join(args.output_dir, f"ann_h{h}_l{l}.onnx")
        export_model(m, dummy, path)

    # CNN 대표 모델
    print("\n[CNN]")
    for f, l in [(32, 2), (64, 3), (128, 4)]:
        m = create_model('simple_cnn', num_filters=f, num_conv_layers=l)
        dummy = torch.randn(1, 1, 28, 28)
        path = os.path.join(args.output_dir, f"cnn_f{f}_l{l}.onnx")
        export_model(m, dummy, path)

    # ResNet 대표 모델
    print("\n[ResNet]")
    for bw in [16, 32, 64]:
        m = create_model('resnet_mnist', layers=[2, 2, 2, 2], base_width=bw)
        dummy = torch.randn(1, 1, 28, 28)
        path = os.path.join(args.output_dir, f"resnet_2222_w{bw}.onnx")
        export_model(m, dummy, path)

    # Transformer 대표 모델
    print("\n[Transformer]")
    for ed, nl, nh in [(64, 2, 4), (128, 4, 4), (256, 4, 8)]:
        m = create_model('transformer', img_size=32, patch_size=4,
                         in_channels=3, embed_dim=ed, num_layers=nl,
                         num_heads=nh)
        dummy = torch.randn(1, 3, 32, 32)
        path = os.path.join(args.output_dir, f"vit_d{ed}_l{nl}_h{nh}.onnx")
        export_model(m, dummy, path)

    # GAN Generator 대표 모델
    print("\n[GAN Generator]")
    for ld, hdims in [(64, [128, 256]), (128, [256, 512, 1024])]:
        m = create_model('gan', latent_dim=ld, img_size=32, img_channels=3,
                         g_hidden_dims=hdims)
        # GAN forward = Generator forward
        dummy = torch.randn(1, ld)
        h_tag = '_'.join(str(d) for d in hdims)
        path = os.path.join(args.output_dir, f"gan_z{ld}_G{h_tag}.onnx")
        export_model(m, dummy, path)

    # ONNX 검증
    print("\n[검증]")
    try:
        import onnx
        for fname in sorted(os.listdir(args.output_dir)):
            if not fname.endswith('.onnx'):
                continue
            fpath = os.path.join(args.output_dir, fname)
            try:
                model = onnx.load(fpath)
                onnx.checker.check_model(model)
                print(f"  OK: {fname}")
            except Exception as e:
                print(f"  FAIL: {fname}: {e}")
    except ImportError:
        print("  onnx 미설치 — 검증 건너뜀")

    print(f"\n변환 완료! 저장 위치: {args.output_dir}")


if __name__ == '__main__':
    main()
