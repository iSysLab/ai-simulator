# ============================================================
# Stage 4 Transformer (ViT) 실험 결과 분석 및 시각화
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import platform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except Exception:
        pass


def analyze_transformer_results():
    """Stage 4 Transformer 실험 결과 분석 및 시각화"""
    csv_path = os.path.join(BASE_DIR, 'data', 'stage4', 'transformer_results.csv')
    df = pd.read_csv(csv_path)

    # device 통일 (cuda -> mps로 취급하여 GPU로 그룹화)
    df = df.copy()
    df['device_group'] = df['device'].replace({'cuda': 'GPU', 'mps': 'GPU', 'cpu': 'CPU'})

    print("=" * 80)
    print("Transformer (ViT) 실험 결과 분석")
    print("=" * 80)
    print(f"\n총 {len(df)}개 데이터 (모델 구성 x 디바이스)")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. 파라미터 수 vs 추론 시간
    ax1 = fig.add_subplot(gs[0, 0])
    for dev in ['cpu', 'mps', 'cuda']:
        sub = df[df['device'] == dev]
        if len(sub) > 0:
            ax1.scatter(sub['total_params'], sub['inference_time_mean_ms'],
                        label=dev.upper(), alpha=0.6, s=80)
    ax1.set_xlabel('총 파라미터 수')
    ax1.set_ylabel('추론 시간 (ms)')
    ax1.set_title('파라미터 수 vs 추론 시간')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. 파라미터 수 vs 학습 시간
    ax2 = fig.add_subplot(gs[0, 1])
    for dev in ['cpu', 'mps', 'cuda']:
        sub = df[df['device'] == dev]
        if len(sub) > 0:
            ax2.scatter(sub['total_params'], sub['training_time_mean_sec'],
                        label=dev.upper(), alpha=0.6, s=80)
    ax2.set_xlabel('총 파라미터 수')
    ax2.set_ylabel('학습 시간 (초)')
    ax2.set_title('파라미터 수 vs 학습 시간')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. embed_dim별 평균 추론 시간
    ax3 = fig.add_subplot(gs[0, 2])
    for device in df['device'].unique():
        sub = df[df['device'] == device].groupby('embed_dim')['inference_time_mean_ms'].mean()
        ax3.plot(sub.index, sub.values, marker='o', label=device.upper(), linewidth=2)
    ax3.set_xlabel('embed_dim')
    ax3.set_ylabel('평균 추론 시간 (ms)')
    ax3.set_title('Embedding 차원별 추론 시간')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. num_layers별 추론 시간 (CPU vs GPU)
    ax4 = fig.add_subplot(gs[1, 0])
    cpu_sub = df[df['device'] == 'cpu'].groupby('num_layers')['inference_time_mean_ms'].mean()
    gpu_sub = df[df['device_group'] == 'GPU'].groupby('num_layers')['inference_time_mean_ms'].mean()
    x = np.arange(max(len(cpu_sub), len(gpu_sub)))
    w = 0.35
    ax4.bar(x[:len(cpu_sub)] - w/2, cpu_sub.values, w, label='CPU', alpha=0.8)
    ax4.bar(x[:len(gpu_sub)] + w/2, gpu_sub.values, w, label='GPU', alpha=0.8)
    ax4.set_xlabel('Encoder 레이어 수')
    ax4.set_ylabel('평균 추론 시간 (ms)')
    ax4.set_title('레이어 수별 추론 시간')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. config_str별 추론 시간 (상위 12개)
    ax5 = fig.add_subplot(gs[1, 1])
    cfg_avg = df.groupby('config_str')['inference_time_mean_ms'].mean().sort_values(ascending=True).tail(12)
    ax5.barh(range(len(cfg_avg)), cfg_avg.values, alpha=0.8)
    ax5.set_yticks(range(len(cfg_avg)))
    ax5.set_yticklabels([c[:20] + '..' if len(str(c)) > 20 else c for c in cfg_avg.index], fontsize=8)
    ax5.set_xlabel('추론 시간 (ms)')
    ax5.set_title('구성별 추론 시간 (상위 12개)')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. CPU vs GPU 속도비
    ax6 = fig.add_subplot(gs[1, 2])
    configs = df['config_str'].unique()
    speedups = []
    labels = []
    for cfg in configs[:10]:
        cpu_t = df[(df['config_str'] == cfg) & (df['device'] == 'cpu')]['inference_time_mean_ms'].values
        gpu_t = df[(df['config_str'] == cfg) & (df['device_group'] == 'GPU')]['inference_time_mean_ms'].values
        if len(cpu_t) > 0 and len(gpu_t) > 0 and gpu_t[0] > 0:
            speedups.append(cpu_t[0] / gpu_t[0])
            labels.append(cfg[:15] + '..' if len(cfg) > 15 else cfg)
    if speedups:
        ax6.barh(range(len(speedups)), speedups, color='steelblue', alpha=0.7)
        ax6.set_yticks(range(len(labels)))
        ax6.set_yticklabels(labels, fontsize=8)
        ax6.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('CPU/GPU 속도비')
    ax6.set_title('GPU 가속 효과')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Stage 4: Transformer (ViT) 실험 결과 시각화', fontsize=14, fontweight='bold', y=1.02)
    out_dir = os.path.join(BASE_DIR, 'data', 'stage4')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'transformer_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n시각화 저장: {save_path}")


if __name__ == "__main__":
    analyze_transformer_results()
