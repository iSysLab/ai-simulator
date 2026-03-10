import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def analyze_cnn_results():
    """
    CNN 실험 결과 분석 및 시각화
    """
    # 데이터 로드
    df = pd.read_csv('../data/cnn_cifar10_results.csv')

    print("=" * 80)
    print("CNN CIFAR-10 실험 결과 분석")
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

    # 모델별 CPU/MPS 속도 비교
    print("\n3. 모델별 MPS 가속 비율")
    print("-" * 80)
    for model_name in df['model_name'].unique():
        cpu_inf = df[(df['model_name'] == model_name) & (df['device'] == 'cpu')]['inference_time_mean_ms'].values[0]
        mps_inf = df[(df['model_name'] == model_name) & (df['device'] == 'mps')]['inference_time_mean_ms'].values[0]
        speedup = cpu_inf / mps_inf
        print(f"{model_name:20s}: {speedup:.2f}x 빠름 (CPU {cpu_inf:.2f}ms → MPS {mps_inf:.2f}ms)")

    # 파라미터 수와 시간의 상관관계
    print("\n4. 상관관계 분석")
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
    fig = plt.figure(figsize=(18, 12))

    # 2x3 레이아웃
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. 파라미터 수 vs 추론 시간 (로그 스케일)
    ax1 = fig.add_subplot(gs[0, 0])
    for device in ['cpu', 'mps']:
        device_data = df[df['device'] == device]
        ax1.scatter(device_data['total_params'], device_data['inference_time_mean_ms'],
                    label=device.upper(), alpha=0.6, s=100)
    ax1.set_xlabel('총 파라미터 수', fontsize=12)
    ax1.set_ylabel('추론 시간 (ms, 로그)', fontsize=12)
    ax1.set_title('파라미터 수 vs 추론 시간', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 파라미터 수 vs 학습 시간 (로그 스케일)
    ax2 = fig.add_subplot(gs[0, 1])
    for device in ['cpu', 'mps']:
        device_data = df[df['device'] == device]
        ax2.scatter(device_data['total_params'], device_data['training_time_mean_sec'],
                    label=device.upper(), alpha=0.6, s=100)
    ax2.set_xlabel('총 파라미터 수', fontsize=12)
    ax2.set_ylabel('학습 시간 (초, 로그)', fontsize=12)
    ax2.set_title('파라미터 수 vs 학습 시간', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MPS 가속 비율 막대 그래프
    ax3 = fig.add_subplot(gs[0, 2])
    speedup_data = []
    model_names = []
    for model_name in df['model_name'].unique():
        cpu_inf = df[(df['model_name'] == model_name) & (df['device'] == 'cpu')]['inference_time_mean_ms'].values[0]
        mps_inf = df[(df['model_name'] == model_name) & (df['device'] == 'mps')]['inference_time_mean_ms'].values[0]
        speedup = cpu_inf / mps_inf
        speedup_data.append(speedup)
        # 모델 이름 축약
        short_name = model_name.replace('SimpleCNN_', '')
        model_names.append(short_name)

    colors = ['skyblue' if 'L' in name else 'coral' for name in model_names]
    bars = ax3.barh(model_names, speedup_data, color=colors, alpha=0.7)
    ax3.set_xlabel('MPS 가속 비율 (배)', fontsize=12)
    ax3.set_title('모델별 MPS 가속 효과', fontsize=14, fontweight='bold')
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='동일 속도')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. 모델별 추론 시간 비교 (CPU vs MPS)
    ax4 = fig.add_subplot(gs[1, 0])
    cpu_data = df[df['device'] == 'cpu'].sort_values('model_idx')
    mps_data = df[df['device'] == 'mps'].sort_values('model_idx')

    x = np.arange(len(cpu_data))
    width = 0.35

    ax4.bar(x - width / 2, cpu_data['inference_time_mean_ms'], width,
            label='CPU', alpha=0.8, color='steelblue')
    ax4.bar(x + width / 2, mps_data['inference_time_mean_ms'], width,
            label='MPS', alpha=0.8, color='coral')
    ax4.set_xlabel('모델 인덱스', fontsize=12)
    ax4.set_ylabel('추론 시간 (ms, 로그)', fontsize=12)
    ax4.set_title('모델별 추론 시간 (CPU vs MPS)', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 모델별 학습 시간 비교 (CPU vs MPS)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x - width / 2, cpu_data['training_time_mean_sec'], width,
            label='CPU', alpha=0.8, color='steelblue')
    ax5.bar(x + width / 2, mps_data['training_time_mean_sec'], width,
            label='MPS', alpha=0.8, color='coral')
    ax5.set_xlabel('모델 인덱스', fontsize=12)
    ax5.set_ylabel('학습 시간 (초, 로그)', fontsize=12)
    ax5.set_title('모델별 학습 시간 (CPU vs MPS)', fontsize=14, fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. SimpleCNN 레이어별 성능
    ax6 = fig.add_subplot(gs[1, 2])
    simple_cnn = df[df['model_type'] == 'SimpleCNN']
    for device in ['cpu', 'mps']:
        device_data = simple_cnn[simple_cnn['device'] == device]
        # 레이어 수별로 그룹화하여 평균
        layer_groups = device_data.groupby('num_conv_layers')['inference_time_mean_ms'].mean()
        ax6.plot(layer_groups.index, layer_groups.values,
                 marker='o', label=device.upper(), linewidth=2, markersize=8)
    ax6.set_xlabel('Convolution 레이어 수', fontsize=12)
    ax6.set_ylabel('평균 추론 시간 (ms)', fontsize=12)
    ax6.set_title('SimpleCNN 레이어 수별 성능', fontsize=14, fontweight='bold')
    ax6.set_xticks([2, 3, 4])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/cnn_results_visualization.png', dpi=300, bbox_inches='tight')
    print("\n5. 시각화")
    print("-" * 80)
    print("그래프가 ../data/cnn_results_visualization.png 에 저장되었습니다")

    plt.show()


if __name__ == "__main__":
    analyze_cnn_results()