# ============================================================
# Stage 5: PyTorch 모델 → ONNX 파일 변환 스크립트
# ============================================================
#
# 역할:
#   Stage 1~4에서 만든 PyTorch 모델들(ANN, CNN, Transformer, GAN)을
#   ONNX 표준 파일 형식으로 저장합니다.
#   저장된 .onnx 파일은 predict_from_onnx.py에서 테스트용으로 사용됩니다.
#
# ONNX 변환 원리:
#   PyTorch에서는 torch.onnx.export()를 사용합니다.
#   이 함수는 "트레이싱(Tracing)" 방식으로 동작합니다:
#     1. 더미 입력(dummy_input)을 모델에 통과시킴
#     2. 이 과정에서 실행된 연산들을 그래프로 기록
#     3. 이 그래프를 ONNX 형식으로 저장
#
# 주의사항:
#   - 동적 제어 흐름(if/for 등이 입력값에 따라 달라지는 경우)은
#     트레이싱으로 잡히지 않을 수 있음
#   - 더미 입력은 실제 추론 때와 동일한 shape이어야 함
#
# 하드웨어: MacBook Air M1 (CPU 전용, ONNX 변환은 CPU에서 진행)
# 작성자: 김홍근
# ============================================================

import os
import sys
import logging
import warnings
import torch

# PyTorch ONNX exporter 내부 로거 경고 억제
# (torch.export, onnxscript 등에서 나오는 INFO/WARNING 로그)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)

# 프로젝트 루트 경로 설정 (dnn/ 폴더)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.ann_models import SimpleANN
from models.cnn_models import SimpleCNN
from models.transformer_models import SimpleViT
from models.gan_models import Generator

# ──────────────────────────────────────────────────────────
# 저장 경로 설정
# ──────────────────────────────────────────────────────────
STAGE5_ONNX_DIR = os.path.join(BASE_DIR, 'data', 'stage5', 'onnx_samples')
os.makedirs(STAGE5_ONNX_DIR, exist_ok=True)


def export_model_to_onnx(model, dummy_input, save_path, input_names, output_names, opset_version=11):
    """
    PyTorch 모델을 ONNX 파일로 저장하는 공통 함수

    Args:
        model        (nn.Module): 저장할 PyTorch 모델 (eval 모드로 전환 후 사용)
        dummy_input  (Tensor)   : 모델에 넣을 더미 입력 (shape만 맞으면 됨)
        save_path    (str)      : 저장할 .onnx 파일 경로
        input_names  (list[str]): ONNX 그래프 입력 노드 이름
        output_names (list[str]): ONNX 그래프 출력 노드 이름
        opset_version (int)     : ONNX opset 버전 (PyTorch 2.x 기준 18 권장)
    """
    # 평가 모드로 전환 (Dropout, BatchNorm 등의 동작이 달라지므로 반드시 필요)
    model.eval()

    print(f"  저장 중: {os.path.basename(save_path)}", end='', flush=True)
    try:
        # PyTorch 2.10+에서 dynamo=False로 레거시 TorchScript exporter를 명시적으로 사용
        # dynamo=True (기본값): triton, inductor 등 무거운 모듈 로드 → 매우 느림
        # dynamo=False          : 기존 TorchScript 기반 → 빠르고 안정적
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                dynamo=False,              # 레거시 TorchScript exporter 사용 (핵심!)
                export_params=True,        # 학습된 가중치도 함께 저장
                opset_version=opset_version,
                do_constant_folding=True,  # 상수 연산 미리 계산 (모델 최적화)
                input_names=input_names,
                output_names=output_names,
            )

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f" → 완료 ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f" → 실패: {e}")
        return False


def export_ann_models():
    """
    SimpleANN 모델들을 ONNX로 변환

    MNIST 입력 기준: 배치 × 1채널 × 28×28 = 배치 × 784 (flatten 후)
    """
    print("\n[ANN 모델 ONNX 변환]")
    # 대표적인 ANN 구조 3가지 (레이어 수, 너비 다양화)
    ann_configs = [
        {'hidden_sizes': [128],           'name': 'ann_l1_w128'},
        {'hidden_sizes': [256, 128],      'name': 'ann_l2_w256'},
        {'hidden_sizes': [512, 256, 128], 'name': 'ann_l3_w512'},
    ]

    for cfg in ann_configs:
        model = SimpleANN(input_size=784, hidden_sizes=cfg['hidden_sizes'], num_classes=10)
        # ANN 더미 입력: (1, 1, 28, 28) → 모델 내부에서 28×28=784로 flatten
        dummy_input = torch.randn(1, 1, 28, 28)
        save_path = os.path.join(STAGE5_ONNX_DIR, f"{cfg['name']}.onnx")
        export_model_to_onnx(
            model, dummy_input, save_path,
            input_names=['mnist_image'],
            output_names=['class_logits']
        )


def export_cnn_models():
    """
    SimpleCNN 모델들을 ONNX로 변환

    CIFAR-10 입력 기준: 배치 × 3채널 × 32×32
    """
    print("\n[CNN 모델 ONNX 변환]")
    cnn_configs = [
        {'num_conv_layers': 2, 'base_channels': 32,  'name': 'cnn_l2_c32'},
        {'num_conv_layers': 3, 'base_channels': 64,  'name': 'cnn_l3_c64'},
        {'num_conv_layers': 4, 'base_channels': 128, 'name': 'cnn_l4_c128'},
    ]

    for cfg in cnn_configs:
        model = SimpleCNN(
            num_classes=10,
            num_conv_layers=cfg['num_conv_layers'],
            base_channels=cfg['base_channels']
        )
        # CIFAR-10 더미 입력: (1, 3, 32, 32)
        dummy_input = torch.randn(1, 3, 32, 32)
        save_path = os.path.join(STAGE5_ONNX_DIR, f"{cfg['name']}.onnx")
        export_model_to_onnx(
            model, dummy_input, save_path,
            input_names=['cifar10_image'],
            output_names=['class_logits']
        )


def export_transformer_models():
    """
    SimpleViT(Vision Transformer) 모델들을 ONNX로 변환

    CIFAR-10 입력 기준: 배치 × 3채널 × 32×32
    img_size=32, patch_size=4 → 8×8=64개의 패치
    """
    print("\n[Transformer 모델 ONNX 변환]")
    vit_configs = [
        {
            'embed_dim': 64, 'num_heads': 4, 'num_layers': 3,
            'patch_size': 4, 'name': 'vit_dim64_heads4_l3'
        },
        {
            'embed_dim': 128, 'num_heads': 4, 'num_layers': 4,
            'patch_size': 4, 'name': 'vit_dim128_heads4_l4'
        },
    ]

    for cfg in vit_configs:
        model = SimpleViT(
            img_size=32,
            patch_size=cfg['patch_size'],
            in_channels=3,
            num_classes=10,
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            mlp_ratio=4.0,
            dropout=0.0  # ONNX 변환 시 dropout 0으로 고정 (eval에서도 동일)
        )
        # CIFAR-10 더미 입력
        dummy_input = torch.randn(1, 3, 32, 32)
        save_path = os.path.join(STAGE5_ONNX_DIR, f"{cfg['name']}.onnx")
        export_model_to_onnx(
            model, dummy_input, save_path,
            input_names=['cifar10_image'],
            output_names=['class_logits']
        )


def export_gan_generator():
    """
    GAN의 Generator 모델을 ONNX로 변환

    GAN Generator 입력: 랜덤 노이즈 벡터 (latent_dim 차원)
    출력: MNIST 이미지 (1×28×28 → flatten → 784)
    """
    print("\n[GAN Generator ONNX 변환]")
    gan_configs = [
        {'latent_dim': 100, 'hidden_dims': [256, 512],       'name': 'gan_gen_z100_h256_512'},
        {'latent_dim': 64,  'hidden_dims': [128, 256, 512],  'name': 'gan_gen_z64_h128_256_512'},
    ]

    for cfg in gan_configs:
        # Generator는 img_size=28, img_channels=1로 MNIST 이미지(28×28 흑백)를 생성
        # output_dim = img_size × img_size × img_channels = 28 × 28 × 1 = 784
        model = Generator(
            latent_dim=cfg['latent_dim'],
            hidden_dims=cfg['hidden_dims'],
            img_size=28,
            img_channels=1,
        )
        # GAN 더미 입력: (1, latent_dim) — 랜덤 노이즈 벡터
        dummy_input = torch.randn(1, cfg['latent_dim'])
        save_path = os.path.join(STAGE5_ONNX_DIR, f"{cfg['name']}.onnx")
        export_model_to_onnx(
            model, dummy_input, save_path,
            input_names=['noise_vector'],
            output_names=['generated_image']
        )


def verify_onnx_files():
    """
    저장된 ONNX 파일들이 유효한지 검증

    onnx.checker.check_model()은 모델 구조가 ONNX 스펙에 맞는지 확인합니다.
    이 검증을 통과하면 대부분의 ONNX 런타임에서 정상 실행 가능합니다.
    """
    print("\n[ONNX 파일 유효성 검증]")
    try:
        import onnx
    except ImportError:
        print("  onnx 미설치 → 건너뜀 (pip install onnx)")
        return

    onnx_files = [f for f in os.listdir(STAGE5_ONNX_DIR) if f.endswith('.onnx')]
    onnx_files.sort()

    for fname in onnx_files:
        fpath = os.path.join(STAGE5_ONNX_DIR, fname)
        try:
            model = onnx.load(fpath)
            onnx.checker.check_model(model)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  ✓ {fname:45s} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")


def main():
    print("=" * 60)
    print("  Stage 5: PyTorch 모델 → ONNX 변환")
    print(f"  저장 위치: {STAGE5_ONNX_DIR}")
    print("=" * 60)

    # 각 모델 타입별로 ONNX 변환 실행
    export_ann_models()
    export_cnn_models()
    export_transformer_models()
    export_gan_generator()

    # 저장된 파일 목록 및 유효성 확인
    verify_onnx_files()

    print("\n" + "=" * 60)
    print("  변환 완료!")
    print(f"  저장 폴더: {STAGE5_ONNX_DIR}")
    print("  다음 단계: predict_from_onnx.py로 실행 시간 예측")
    print("=" * 60)


if __name__ == '__main__':
    main()
