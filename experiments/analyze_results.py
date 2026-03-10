import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def analyze_results():
    """
    실험 결과 분석 및 시각화
    """
    # 데이터 로드
    df = pd.read_csv('../data/ann_mnist_results.csv')

    print("=" * 80)
    print("ANN MNIST 실험 결과 분석")
    print("=" * 80)

    # 기본 통계
    print("\n1. 기본 통계")
    print("-" * 80)
    print(df.groupby('device')[['inference_time_mean_ms', 'training_time_mean_sec']].describe())

    # CPU vs MPS 비교
    print("\n2. CPU vs MPS 비교")
    print("-" * 80)
    cpu_data = df[df['device'] == 'cpu']
    mps_data = df[df['device'] == 'mps']

    print(f"평균 추론 시간 - CPU: {cpu_data['inference_time_mean_ms'].mean():.2f} ms")
    print(f"평균 추론 시간 - MPS: {mps_data['inference_time_mean_ms'].mean():.2f} ms")
    print(f"평균 학습 시간 - CPU: {cpu_data['training_time_mean_sec'].mean():.2f} 초")
    print(f"평균 학습 시간 - MPS: {mps_data['training_time_mean_sec'].mean():.2f} 초")

    # 파라미터 수와 시간의 상관관계
    print("\n3. 상관관계 분석")
    print("-" * 80)
    for device in ['cpu', 'mps']:
        device_data = df[df['device'] == device]
        corr_inf = device_data['total_params'].corr(device_data['inference_time_mean_ms'])
        corr_train = device_data['total_params'].corr(device_data['training_time_mean_sec'])
        print(f"{device.upper()} - 파라미터 수 vs 추론 시간: {corr_inf:.4f}")
        print(f"{device.upper()} - 파라미터 수 vs 학습 시간: {corr_train:.4f}")

    # 시각화
    create_visualizations(df)


def create_visualizations(df):
    """
    결과 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 파라미터 수 vs 추론 시간
    ax1 = axes[0, 0]
    for device in ['cpu', 'mps']:
        device_data = df[df['device'] == device]
        ax1.scatter(device_data['total_params'], device_data['inference_time_mean_ms'],
                    label=device.upper(), alpha=0.6, s=100)
    ax1.set_xlabel('총 파라미터 수', fontsize=12)
    ax1.set_ylabel('추론 시간 (ms)', fontsize=12)
    ax1.set_title('파라미터 수 vs 추론 시간', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 파라미터 수 vs 학습 시간
    ax2 = axes[0, 1]
    for device in ['cpu', 'mps']:
        device_data = df[df['device'] == device]
        ax2.scatter(device_data['total_params'], device_data['training_time_mean_sec'],
                    label=device.upper(), alpha=0.6, s=100)
    ax2.set_xlabel('총 파라미터 수', fontsize=12)
    ax2.set_ylabel('학습 시간 (초)', fontsize=12)
    ax2.set_title('파라미터 수 vs 학습 시간', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 레이어 수별 추론 시간 비교
    ax3 = axes[1, 0]
    cpu_data = df[df['device'] == 'cpu']
    mps_data = df[df['device'] == 'mps']

    x = np.arange(len(cpu_data))
    width = 0.35

    ax3.bar(x - width / 2, cpu_data['inference_time_mean_ms'], width, label='CPU', alpha=0.8)
    ax3.bar(x + width / 2, mps_data['inference_time_mean_ms'], width, label='MPS', alpha=0.8)
    ax3.set_xlabel('모델 인덱스', fontsize=12)
    ax3.set_ylabel('추론 시간 (ms)', fontsize=12)
    ax3.set_title('모델별 추론 시간 (CPU vs MPS)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 레이어 수별 학습 시간 비교
    ax4 = axes[1, 1]
    ax4.bar(x - width / 2, cpu_data['training_time_mean_sec'], width, label='CPU', alpha=0.8)
    ax4.bar(x + width / 2, mps_data['training_time_mean_sec'], width, label='MPS', alpha=0.8)
    ax4.set_xlabel('모델 인덱스', fontsize=12)
    ax4.set_ylabel('학습 시간 (초)', fontsize=12)
    ax4.set_title('모델별 학습 시간 (CPU vs MPS)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/ann_results_visualization.png', dpi=300, bbox_inches='tight')
    print("\n4. 시각화")
    print("-" * 80)
    print("그래프가 ../data/ann_results_visualization.png 에 저장되었습니다")

    plt.show()


if __name__ == "__main__":
    analyze_results()