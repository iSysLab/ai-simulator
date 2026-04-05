"""ONNX 내보내기 실험

4가지 모델 타입(ANN, CNN, Transformer, GAN)을 ONNX 포맷으로 내보냅니다.

출력 경로: data/onnx/<model_type>/<model_name>.onnx

사용법:
    python experiments/export_to_onnx.py
"""

import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ann import SimpleANN
from models.cnn import SimpleCNN
from models.transformer import SimpleViT
from models.gan import SimpleGAN

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(ROOT_DIR, 'data', 'onnx')

OPSET = 17


# ── 내보낼 모델 정의 ──────────────────────────────────────

ANN_CONFIGS = [
    {'hidden_size': 64,  'num_hidden_layers': 2},
    {'hidden_size': 128, 'num_hidden_layers': 2},
    {'hidden_size': 256, 'num_hidden_layers': 4},
]

CNN_CONFIGS = [
    {'num_filters': 16, 'num_conv_layers': 2, 'use_batchnorm': False},
    {'num_filters': 32, 'num_conv_layers': 3, 'use_batchnorm': False},
    {'num_filters': 32, 'num_conv_layers': 3, 'use_batchnorm': True},
]

TRANSFORMER_CONFIGS = [
    {'embed_dim': 64,  'num_layers': 2, 'num_heads': 4, 'patch_size': 4},
    {'embed_dim': 128, 'num_layers': 4, 'num_heads': 4, 'patch_size': 4},
    {'embed_dim': 256, 'num_layers': 4, 'num_heads': 8, 'patch_size': 4},
]

GAN_CONFIGS = [
    {'latent_dim': 64,  'g_hidden_dims': [128, 256]},
    {'latent_dim': 128, 'g_hidden_dims': [256, 512, 1024]},
    {'latent_dim': 256, 'g_hidden_dims': [256, 512, 1024]},
]


# ── 내보내기 함수 ──────────────────────────────────────────

def export(model, dummy_input, save_path, config=None):
    """모델을 ONNX로 내보냅니다.

    Args:
        model: PyTorch 모델 (eval 모드로 전환 후 내보냄)
        dummy_input: 모델 입력 더미 텐서
        save_path: 저장 경로 (.onnx)
        config: 모델 설정 dict (함께 JSON으로 저장)
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=OPSET,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # config JSON 저장 (predict_from_onnx.py에서 feature 추출에 사용)
    if config is not None:
        json_path = save_path.replace('.onnx', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"  저장: {os.path.relpath(save_path, ROOT_DIR)}  ({size_kb:.1f} KB)")


# ── 메인 ──────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  ONNX 내보내기")
    print(f"  opset={OPSET}  /  출력: data/onnx/")
    print("=" * 60)

    exported = 0

    # ── ANN ────────────────────────────────────────────────
    print("\n[ANN]  입력: (1, 784)  데이터: MNIST")
    ann_dir = os.path.join(OUTPUT_DIR, 'ann') #  저장 폴더 경로 설정
    for cfg in ANN_CONFIGS:
        model = SimpleANN(
            hidden_size=cfg['hidden_size'],
            num_hidden_layers=cfg['num_hidden_layers'],
        )
        dummy = torch.zeros(1, 1, 28, 28)
        name  = f"ann_h{cfg['hidden_size']}_l{cfg['num_hidden_layers']}.onnx"
        export(model, dummy, os.path.join(ann_dir, name), config={**cfg, 'model_type': 'ann', 'dataset_encoded': 0, 'input_channels': 1, 'input_height': 28, 'input_width': 28, 'num_classes': 10})
        exported += 1

    # ── CNN ────────────────────────────────────────────────
    print("\n[CNN]  입력: (1, 1, 28, 28)  데이터: MNIST")
    cnn_dir = os.path.join(OUTPUT_DIR, 'cnn')
    for cfg in CNN_CONFIGS:
        model = SimpleCNN(
            num_filters=cfg['num_filters'],
            num_conv_layers=cfg['num_conv_layers'],
            use_batchnorm=cfg['use_batchnorm'],
        )
        dummy = torch.zeros(1, 1, 28, 28)
        bn_tag = 'bn' if cfg['use_batchnorm'] else 'nobn'
        name   = f"cnn_f{cfg['num_filters']}_l{cfg['num_conv_layers']}_{bn_tag}.onnx"
        export(model, dummy, os.path.join(cnn_dir, name), config={**cfg, 'model_type': 'cnn', 'dataset_encoded': 0, 'input_channels': 1, 'input_height': 28, 'input_width': 28, 'num_classes': 10, 'has_batchnorm': int(cfg['use_batchnorm'])})
        exported += 1

    # ── Transformer ────────────────────────────────────────
    print("\n[Transformer]  입력: (1, 3, 32, 32)  데이터: CIFAR-10")
    vit_dir = os.path.join(OUTPUT_DIR, 'transformer')
    for cfg in TRANSFORMER_CONFIGS:
        model = SimpleViT(
            img_size=32,
            patch_size=cfg['patch_size'],
            in_channels=3,
            num_classes=10,
            embed_dim=cfg['embed_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
        )
        dummy = torch.zeros(1, 3, 32, 32)
        name  = (f"vit_e{cfg['embed_dim']}_l{cfg['num_layers']}"
                 f"_h{cfg['num_heads']}_p{cfg['patch_size']}.onnx")
        export(model, dummy, os.path.join(vit_dir, name), config={**cfg, 'model_type': 'transformer', 'num_transformer_layers': cfg['num_layers'], 'dataset_encoded': 1, 'input_channels': 3, 'input_height': 32, 'input_width': 32, 'num_classes': 10})
        exported += 1

    # ── GAN (Generator only) ───────────────────────────────
    print("\n[GAN Generator]  입력: (1, latent_dim)  데이터: MNIST")
    gan_dir = os.path.join(OUTPUT_DIR, 'gan')
    for cfg in GAN_CONFIGS:
        gan   = SimpleGAN(
            latent_dim=cfg['latent_dim'],
            g_hidden_dims=cfg['g_hidden_dims'],
        )
        model = gan.generator          # Generator만 내보내기
        dummy = torch.zeros(1, cfg['latent_dim'])
        depth = len(cfg['g_hidden_dims'])
        name  = f"gan_gen_z{cfg['latent_dim']}_d{depth}.onnx"
        export(model, dummy, os.path.join(gan_dir, name), config={**cfg, 'model_type': 'gan', 'dataset_encoded': 0, 'input_channels': 1, 'input_height': 28, 'input_width': 28, 'num_classes': 0})
        exported += 1

    print(f"\n완료: 총 {exported}개 ONNX 파일 저장  →  {os.path.relpath(OUTPUT_DIR, ROOT_DIR)}/")


if __name__ == '__main__':
    run()
