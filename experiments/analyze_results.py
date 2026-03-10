import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
import platform
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except Exception:
        pass


def analyze_results():
    """
    Stage 1 ANN 실험 결과 분석 및 시각화
    """
    csv_path = os.path.join(BASE_DIR, 'data', 'stage1', 'ann_mnist_results.csv')
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("ANN MNIST 실험 결과 분석")
    print("=" * 80)

    # 기본 통계
    print("\n1. 기본 통계")
    print("-" * 80)
    print(df.groupby('device')[['inference_time_mean_ms', 'training_time_mean_sec']].describe())

    # CPU vs GPU(MPS/CUDA) 비교
    print("\n2. CPU vs GPU 비교")
    print("-" * 80)
    cpu_data = df[df['device'] == 'cpu']
    gpu_data = df[df['device'].isin(['mps', 'cuda'])]

    print(f"평균 추론 시간 - CPU: {cpu_data['inference_time_mean_ms'].mean():.2f} ms")
    print(f"평균 학습 시간 - CPU: {cpu_data['training_time_mean_sec'].mean():.2f} 초")
    if len(gpu_data) > 0:
        gpu_dev = gpu_data['device'].iloc[0].upper()
        print(f"평균 추론 시간 - GPU({gpu_dev}): {gpu_data['inference_time_mean_ms'].mean():.2f} ms")
        print(f"평균 학습 시간 - GPU({gpu_dev}): {gpu_data['training_time_mean_sec'].mean():.2f} 초")

    # 파라미터 수와 시간의 상관관계
    print("\n3. 상관관계 분석")
    print("-" * 80)
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        if len(device_data) < 2:
            continue
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
    for device in df['device'].unique():
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
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        ax2.scatter(device_data['total_params'], device_data['training_time_mean_sec'],
                    label=device.upper(), alpha=0.6, s=100)
    ax2.set_xlabel('총 파라미터 수', fontsize=12)
    ax2.set_ylabel('학습 시간 (초)', fontsize=12)
    ax2.set_title('파라미터 수 vs 학습 시간', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 모델별 추론 시간 비교 (CPU vs GPU) — 공통 config만 사용
    ax3 = axes[1, 0]
    cpu_df = df[df['device'] == 'cpu'][['config', 'inference_time_mean_ms', 'training_time_mean_sec']].copy() if 'config' in df.columns else df[df['device'] == 'cpu'][['model_idx', 'inference_time_mean_ms', 'training_time_mean_sec']].copy()
    gpu_df = df[df['device'].isin(['mps', 'cuda'])][['config', 'inference_time_mean_ms', 'training_time_mean_sec']].copy() if 'config' in df.columns else df[df['device'].isin(['mps', 'cuda'])][['model_idx', 'inference_time_mean_ms', 'training_time_mean_sec']].copy()
    if len(gpu_df) > 0:
        gpu_df = gpu_df.groupby('config' if 'config' in df.columns else 'model_idx').first().reset_index()
    merge_key = 'config' if 'config' in df.columns else 'model_idx'
    merged = pd.merge(cpu_df, gpu_df, on=merge_key, suffixes=('_cpu', '_gpu'), how='inner') if len(gpu_df) > 0 else pd.DataFrame()
    n = len(merged)
    if n == 0:
        ax3.text(0.5, 0.5, 'CPU/GPU 공통 데이터 없음', ha='center', va='center', transform=ax3.transAxes)
    else:
        x = np.arange(n)
        width = 0.35
        ax3.bar(x - width / 2, merged['inference_time_mean_ms_cpu'].values, width, label='CPU', alpha=0.8)
        ax3.bar(x + width / 2, merged['inference_time_mean_ms_gpu'].values, width, label='GPU', alpha=0.8)
    ax3.set_xlabel('모델 인덱스', fontsize=12)
    ax3.set_ylabel('추론 시간 (ms)', fontsize=12)
    ax3.set_title('모델별 추론 시간 (CPU vs GPU)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 모델별 학습 시간 비교 (CPU vs MPS)
    ax4 = axes[1, 1]
    if n > 0:
        ax4.bar(x - width / 2, merged['training_time_mean_sec_cpu'].values, width, label='CPU', alpha=0.8)
        ax4.bar(x + width / 2, merged['training_time_mean_sec_gpu'].values, width, label='GPU', alpha=0.8)
    else:
        ax4.text(0.5, 0.5, 'CPU/GPU 공통 데이터 없음', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel('모델 인덱스', fontsize=12)
    ax4.set_ylabel('학습 시간 (초)', fontsize=12)
    ax4.set_title('모델별 학습 시간 (CPU vs GPU)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    out_dir = os.path.join(BASE_DIR, 'data', 'stage1')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'ann_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("\n4. 시각화")
    print("-" * 80)
    print(f"그래프가 {save_path} 에 저장되었습니다")


if __name__ == "__main__":
    analyze_results()