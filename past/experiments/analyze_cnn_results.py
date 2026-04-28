import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 한글 폰트 설정
import platform
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except Exception:
        pass


def analyze_cnn_results():
    """
    Stage 2 CNN 실험 결과 분석 및 시각화
    """
    csv_path = os.path.join(BASE_DIR, 'data', 'stage2', 'cnn_cifar10_results.csv')
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("CNN CIFAR-10 실험 결과 분석")
    print("=" * 80)

    # 기본 통계
    print("\n1. 기본 통계")
    print("-" * 80)
    print(df.groupby('device')[['inference_time_mean_ms', 'training_time_mean_sec']].describe())

    # CPU vs GPU 비교
    devices = sorted(df['device'].unique())
    print("\n2. 디바이스별 비교")
    print("-" * 80)
    for dev in devices:
        dev_data = df[df['device'] == dev]
        print(f"평균 추론 시간 - {dev.upper()}: {dev_data['inference_time_mean_ms'].mean():.2f} ms")
        print(f"평균 학습 시간 - {dev.upper()}: {dev_data['training_time_mean_sec'].mean():.2f} 초")

    # 모델별 CPU vs GPU 속도 비교
    gpu_devices = [d for d in devices if d in ('mps', 'cuda')]
    print("\n3. 모델별 GPU 가속 비율")
    print("-" * 80)
    cpu_data = df[df['device'] == 'cpu']
    for model_name in df['model_name'].unique():
        cpu_row = df[(df['model_name'] == model_name) & (df['device'] == 'cpu')]
        if cpu_row.empty:
            continue
        cpu_inf = cpu_row['inference_time_mean_ms'].values[0]
        for gpu in gpu_devices:
            gpu_row = df[(df['model_name'] == model_name) & (df['device'] == gpu)]
            if not gpu_row.empty:
                gpu_inf = gpu_row['inference_time_mean_ms'].values[0]
                speedup = cpu_inf / gpu_inf
                print(f"{model_name:20s} ({gpu.upper()}): {speedup:.2f}x 빠름 (CPU {cpu_inf:.2f}ms → {gpu.upper()} {gpu_inf:.2f}ms)")

    # 파라미터 수와 시간의 상관관계
    print("\n4. 상관관계 분석")
    print("-" * 80)
    for device in devices:
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
    결과 시각화 (CPU, MPS, CUDA 등 df에 있는 모든 디바이스 지원)
    """
    devices = sorted(df['device'].unique())
    device_colors = {'cpu': 'steelblue', 'mps': 'coral', 'cuda': 'seagreen'}

    fig = plt.figure(figsize=(18, 12))

    # 2x3 레이아웃
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. 파라미터 수 vs 추론 시간 (로그 스케일)
    ax1 = fig.add_subplot(gs[0, 0])
    for device in devices:
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
    for device in devices:
        device_data = df[df['device'] == device]
        ax2.scatter(device_data['total_params'], device_data['training_time_mean_sec'],
                    label=device.upper(), alpha=0.6, s=100)
    ax2.set_xlabel('총 파라미터 수', fontsize=12)
    ax2.set_ylabel('학습 시간 (초, 로그)', fontsize=12)
    ax2.set_title('파라미터 수 vs 학습 시간', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. GPU 가속 비율 막대 그래프 (CPU vs MPS, CPU vs CUDA)
    ax3 = fig.add_subplot(gs[0, 2])
    gpu_devices = [d for d in devices if d in ('mps', 'cuda')]
    cpu_df = df[df['device'] == 'cpu']
    if gpu_devices and not cpu_df.empty:
        model_names = []
        speedup_by_gpu = {g: [] for g in gpu_devices}
        for model_name in df['model_name'].unique():
            cpu_row = df[(df['model_name'] == model_name) & (df['device'] == 'cpu')]
            if cpu_row.empty:
                continue
            cpu_inf = cpu_row['inference_time_mean_ms'].values[0]
            short_name = model_name.replace('SimpleCNN_', '')
            model_names.append(short_name)
            for gpu in gpu_devices:
                gpu_row = df[(df['model_name'] == model_name) & (df['device'] == gpu)]
                if not gpu_row.empty:
                    speedup_by_gpu[gpu].append(cpu_inf / gpu_row['inference_time_mean_ms'].values[0])
                else:
                    speedup_by_gpu[gpu].append(np.nan)

        y_pos = np.arange(len(model_names))
        bar_height = 0.8 / len(gpu_devices) if len(gpu_devices) > 1 else 0.6
        for i, gpu in enumerate(gpu_devices):
            offset = (i - len(gpu_devices) / 2 + 0.5) * bar_height
            ax3.barh(y_pos + offset, speedup_by_gpu[gpu], bar_height * 0.9, label=gpu.upper(),
                     alpha=0.8, color=device_colors.get(gpu, 'gray'))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(model_names)
        ax3.set_xlabel('가속 비율 (배)', fontsize=12)
        ax3.set_title('모델별 GPU 가속 효과', fontsize=14, fontweight='bold')
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='동일 속도')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'GPU 데이터 없음', ha='center', va='center', transform=ax3.transAxes)

    # 4. 모델별 추론 시간 비교 (모든 디바이스)
    ax4 = fig.add_subplot(gs[1, 0])
    base_device = devices[0]
    base_data = df[df['device'] == base_device].sort_values('model_idx')
    n_models = len(base_data)
    model_indices = base_data['model_idx'].values
    x = np.arange(n_models)
    width = 0.8 / len(devices) if devices else 0.35
    for i, device in enumerate(devices):
        dev_df = df[df['device'] == device]
        vals = [dev_df[dev_df['model_idx'] == mid]['inference_time_mean_ms'].values[0]
                if len(dev_df[dev_df['model_idx'] == mid]) > 0 else np.nan
                for mid in model_indices]
        offset = (i - len(devices) / 2 + 0.5) * width
        ax4.bar(x + offset, vals, width, label=device.upper(), alpha=0.8,
                color=device_colors.get(device, 'gray'))
    ax4.set_xlabel('모델 인덱스', fontsize=12)
    ax4.set_ylabel('추론 시간 (ms, 로그)', fontsize=12)
    ax4.set_title('모델별 추론 시간', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 모델별 학습 시간 비교 (모든 디바이스)
    ax5 = fig.add_subplot(gs[1, 1])
    for i, device in enumerate(devices):
        dev_df = df[df['device'] == device]
        vals = [dev_df[dev_df['model_idx'] == mid]['training_time_mean_sec'].values[0]
                if len(dev_df[dev_df['model_idx'] == mid]) > 0 else np.nan
                for mid in model_indices]
        offset = (i - len(devices) / 2 + 0.5) * width
        ax5.bar(x + offset, vals, width, label=device.upper(), alpha=0.8,
                color=device_colors.get(device, 'gray'))
    ax5.set_xlabel('모델 인덱스', fontsize=12)
    ax5.set_ylabel('학습 시간 (초, 로그)', fontsize=12)
    ax5.set_title('모델별 학습 시간', fontsize=14, fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. SimpleCNN 레이어별 성능
    ax6 = fig.add_subplot(gs[1, 2])
    simple_cnn = df[df['model_type'] == 'SimpleCNN']
    for device in devices:
        device_data = simple_cnn[simple_cnn['device'] == device]
        if device_data.empty:
            continue
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
    out_dir = os.path.join(BASE_DIR, 'data', 'stage2')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'cnn_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("\n5. 시각화")
    print("-" * 80)
    print(f"그래프가 {save_path} 에 저장되었습니다")


if __name__ == "__main__":
    analyze_cnn_results()