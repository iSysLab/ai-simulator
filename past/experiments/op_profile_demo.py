# ============================================================
# Op-Level 프로파일러 데모 스크립트
# ============================================================
#
# 역할: utils/op_profiler.py를 ANN, CNN 샘플 모델에 적용하여
#       각 연산(op)별 시간 측정 결과를 출력합니다.
#
# 실행: python experiments/op_profile_demo.py [--device cpu|mps]
#
# 출처: ijunsoo 브랜치 op_profiler 포팅 후 데모용으로 작성
# 작성자: 김홍근
# ============================================================

import os
import sys
import argparse

# 프로젝트 루트(ai-simulator)를 path에 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "past"))

import torch

# 모델 로드
from models.ann_models import SimpleANN
from models.cnn_models import SimpleCNN

# Op-Level 프로파일러 (ijunsoo 브랜치에서 포팅)
from benchmark.support.op_profiler import (
    decompose_model,
    measure_op_times,
    print_op_profile,
)


def run_ann_demo(device='cpu', batch_size=64):
    """ANN 모델 Op-Level 프로파일 데모"""
    print("\n" + "=" * 70)
    print("  [1] ANN 모델 (SimpleANN [256, 256]) Op-Level 프로파일")
    print("=" * 70)

    # MNIST 입력 shape: (배치, 채널, 높이, 너비)
    input_shape = (batch_size, 1, 28, 28)

    model = SimpleANN(hidden_sizes=[256, 256])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델: SimpleANN [256, 256], 파라미터: {total_params:,}개")
    print(f"  입력 shape: {input_shape}, 디바이스: {device}")

    # 1) Op 분해만 (시간 측정 없이)
    ops = decompose_model(model, input_shape)
    print(f"\n  [Op 분해] 총 {len(ops)}개 연산 감지")

    # 2) 각 Op 시간 측정
    ops = measure_op_times(
        model, input_shape, device=device,
        warmup=3, repeats=10
    )

    # 3) 결과 출력
    print_op_profile(ops, top_n=10)


def run_cnn_demo(device='cpu', batch_size=64):
    """CNN 모델 Op-Level 프로파일 데모"""
    print("\n" + "=" * 70)
    print("  [2] CNN 모델 (SimpleCNN L3 C32) Op-Level 프로파일")
    print("=" * 70)

    # CIFAR-10 입력 shape
    input_shape = (batch_size, 3, 32, 32)

    model = SimpleCNN(num_conv_layers=3, base_channels=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델: SimpleCNN L3 C32, 파라미터: {total_params:,}개")
    print(f"  입력 shape: {input_shape}, 디바이스: {device}")

    ops = decompose_model(model, input_shape)
    print(f"\n  [Op 분해] 총 {len(ops)}개 연산 감지")

    ops = measure_op_times(
        model, input_shape, device=device,
        warmup=3, repeats=10
    )

    print_op_profile(ops, top_n=12)


def main():
    parser = argparse.ArgumentParser(description='Op-Level 프로파일러 데모')
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'mps', 'cuda'],
        help='측정 디바이스 (cpu, mps, cuda)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='배치 크기 (기본 64)'
    )
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['all', 'ann', 'cnn'],
        help='프로파일할 모델 (all=ANN+CNN, ann, cnn)'
    )
    args = parser.parse_args()

    # MPS 사용 가능 여부 확인
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("  [경고] MPS를 사용할 수 없습니다. CPU로 대체합니다.")
        args.device = 'cpu'

    print("\n  Op-Level 프로파일러 데모 (ijunsoo 브랜치 포팅)")
    print("  교수님 요구사항 '방법 2': 모델을 연산 단위로 분해 → 각 op 시간 측정 → 합산")

    if args.model in ('all', 'ann'):
        run_ann_demo(device=args.device, batch_size=args.batch_size)
    if args.model in ('all', 'cnn'):
        run_cnn_demo(device=args.device, batch_size=args.batch_size)

    print("\n  [완료] Op-Level 프로파일 데모 종료\n")


if __name__ == '__main__':
    main()
