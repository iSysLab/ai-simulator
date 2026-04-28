# ============================================================
# 전체 Stage 합본 시각화
# Stage 1~4 데이터 + Stage 3 예측 성능 요약을 한 화면에
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import platform
import warnings
import logging
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# U+2212 glyph 경고 억제
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.mathtext').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message=".*glyph.*")

# 마이너스 기호: ASCII 하이픈 사용
matplotlib.rcParams['axes.unicode_minus'] = False
# 로그 스케일 틱 라벨: DejaVu Sans 사용 (U+2212 지원)
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# 한글 폰트 설정
def _setup_korean_font():
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        available = {f.name for f in fm.fontManager.ttflist}
        for name in ['Malgun Gothic', 'MalgunGothic', 'Gulim', 'Dotum', 'Batang']:
            if name in available:
                plt.rcParams['font.family'] = name
                return
        # 폰트명 부분 일치 검색
        for f in fm.fontManager.ttflist:
            if 'Malgun' in f.name or 'Gulim' in f.name:
                plt.rcParams['font.family'] = f.name
                return

_setup_korean_font()


def load_stage_data():
    """각 Stage CSV 로드 (없으면 빈 DataFrame)"""
    data = {}
    paths = {
        'stage1': ('data', 'stage1', 'ann_mnist_results.csv'),
        'stage2': ('data', 'stage2', 'cnn_cifar10_results.csv'),
        'stage4_transformer': ('data', 'stage4', 'transformer_results.csv'),
        'stage4_gan': ('data', 'stage4', 'gan_results.csv'),
        'stage3': ('data', 'stage3', 'v3', 'stage3_v3_prediction_results.csv'),
    }
    for key, path_tuple in paths.items():
        p = os.path.join(BASE_DIR, *path_tuple)
        if os.path.exists(p):
            data[key] = pd.read_csv(p)
        else:
            data[key] = pd.DataFrame()
    return data


def create_summary_visualization():
    """전체 Stage 합본 시각화"""
    data = load_stage_data()

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # ---- 1. Stage 1 ANN: 모델 타입별 데이터 수 & 평균 추론 시간 ----
    ax1 = fig.add_subplot(gs[0, 0])
    if not data['stage1'].empty:
        df1 = data['stage1']
        cpu_avg = df1[df1['device'] == 'cpu']['inference_time_mean_ms'].mean()
        gpu_avg = df1[df1['device'].isin(['mps', 'cuda'])]['inference_time_mean_ms'].mean()
        ax1.bar(['ANN (CPU)', 'ANN (GPU)'], [cpu_avg, gpu_avg], color=['steelblue', 'coral'], alpha=0.8)
        ax1.set_ylabel('평균 추론 시간 (ms)')
        ax1.set_title(f'Stage 1: ANN ({len(df1)}개 데이터)')
    else:
        ax1.text(0.5, 0.5, 'Stage 1 데이터 없음', ha='center', va='center')
    ax1.grid(True, alpha=0.3, axis='y')

    # ---- 2. Stage 2 CNN: 모델별 추론 시간 ----
    ax2 = fig.add_subplot(gs[0, 1])
    if not data['stage2'].empty:
        df2 = data['stage2']
        cpu_df = df2[df2['device'] == 'cpu'].sort_values('model_idx')
        gpu_df = df2[df2['device'].isin(['mps', 'cuda'])].sort_values('model_idx')
        x = np.arange(min(len(cpu_df), len(gpu_df)))
        w = 0.35
        ax2.bar(x - w/2, cpu_df['inference_time_mean_ms'].values[:len(x)], w, label='CPU', alpha=0.8)
        ax2.bar(x + w/2, gpu_df['inference_time_mean_ms'].values[:len(x)], w, label='GPU', alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_ylabel('추론 시간 (ms)')
        ax2.set_title(f'Stage 2: CNN ({len(df2)}개 데이터)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Stage 2 데이터 없음', ha='center', va='center')
    ax2.grid(True, alpha=0.3, axis='y')

    # ---- 3. Stage 4 Transformer: 파라미터 vs 추론 시간 ----
    ax3 = fig.add_subplot(gs[0, 2])
    if not data['stage4_transformer'].empty:
        df4t = data['stage4_transformer']
        ax3.scatter(df4t['total_params'], df4t['inference_time_mean_ms'],
                    c=df4t['device'].map({'cpu': 0, 'mps': 1, 'cuda': 1}), cmap='coolwarm', alpha=0.6)
        ax3.set_xlabel('파라미터 수')
        ax3.set_ylabel('추론 시간 (ms)')
        ax3.set_yscale('log')
        ax3.set_title(f'Stage 4: Transformer ({len(df4t)}개)')
    else:
        ax3.text(0.5, 0.5, 'Transformer 데이터 없음', ha='center', va='center')
    ax3.grid(True, alpha=0.3)

    # ---- 4. Stage 4 GAN: 파라미터 vs 추론 시간 ----
    ax4 = fig.add_subplot(gs[1, 0])
    if not data['stage4_gan'].empty:
        df4g = data['stage4_gan']
        ax4.scatter(df4g['total_params'], df4g['inference_time_mean_ms'],
                    c=df4g['device'].map({'cpu': 0, 'mps': 1, 'cuda': 1}), cmap='viridis', alpha=0.6)
        ax4.set_xlabel('파라미터 수')
        ax4.set_ylabel('추론 시간 (ms)')
        ax4.set_yscale('log')
        ax4.set_title(f'Stage 4: GAN ({len(df4g)}개)')
    else:
        ax4.text(0.5, 0.5, 'GAN 데이터 없음', ha='center', va='center')
    ax4.grid(True, alpha=0.3)

    # ---- 5. 전체 데이터 분포 (모델 타입별) ----
    ax5 = fig.add_subplot(gs[1, 1])
    counts = []
    labels = []
    if not data['stage1'].empty:
        counts.append(len(data['stage1']) // 2)  # cpu/mps 쌍
        labels.append('ANN')
    if not data['stage2'].empty:
        counts.append(len(data['stage2']) // 2)
        labels.append('CNN')
    if not data['stage4_transformer'].empty:
        counts.append(len(data['stage4_transformer']) // 2)
        labels.append('Transformer')
    if not data['stage4_gan'].empty:
        counts.append(len(data['stage4_gan']) // 2)
        labels.append('GAN')
    if counts:
        ax5.bar(labels, counts, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
        ax5.set_ylabel('모델 구성 수')
        ax5.set_title('모델 타입별 데이터 수')
    else:
        ax5.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
    ax5.grid(True, alpha=0.3, axis='y')

    # ---- 6. Stage 3 예측 성능 요약 (R² 등) ----
    ax6 = fig.add_subplot(gs[1, 2])
    perf_text = []
    perf_text.append("Stage 3 예측 모델 성능")
    perf_text.append("(92 Feature, XGBoost/RF)")
    perf_text.append("")
    perf_text.append("추론 R²_log: 0.93 (XGB)")
    perf_text.append("학습 R²_log: 0.88 (XGB)")
    perf_text.append("")
    perf_text.append("데이터: 114개")
    perf_text.append("(ANN 52 + CNN 22 + ViT 24 + GAN 16)")
    ax6.text(0.1, 0.9, '\n'.join(perf_text), transform=ax6.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Stage 3: 예측 모델 요약')

    # ---- 7. 전체 파이프라인 요약 (텍스트) ----
    ax7 = fig.add_subplot(gs[2, :])
    pipeline_text = """
    [전체 파이프라인 요약]
    Stage 1 (ANN) → Stage 2 (CNN) → Stage 4 (Transformer, GAN) : 실행 시간 데이터 수집
    Stage 3 : 92개 Feature 기반 XGBoost/RandomForest 예측 모델 학습
    Stage 5 : ONNX 파일 입력 → Feature 추출 → 실행 시간 예측
    """
    ax7.text(0.5, 0.5, pipeline_text, transform=ax7.transAxes,
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax7.axis('off')

    plt.suptitle('DNN 실행 시간 예측 시스템 — 전체 Stage 합본 시각화', fontsize=16, fontweight='bold', y=1.02)

    out_dir = os.path.join(BASE_DIR, 'reports', 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'all_stages_summary.png')
    # U+2212 glyph 경고가 stderr로 직접 출력되는 경우 억제
    with open(os.devnull, 'w') as devnull:
        _stderr = sys.stderr
        sys.stderr = devnull
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        finally:
            sys.stderr = _stderr
    plt.close()
    print(f"합본 시각화 저장: {save_path}")
    return save_path


def run_all_visualizations():
    """모든 Stage 시각화 + 합본 실행"""
    import subprocess
    scripts = [
        ('Stage 1 ANN', 'analyze_results.py'),
        ('Stage 2 CNN', 'analyze_cnn_results.py'),
        ('Stage 4 Transformer', 'analyze_transformer_results.py'),
        ('Stage 4 GAN', 'analyze_gan_results.py'),
    ]
    exp_dir = os.path.join(BASE_DIR, 'experiments')
    for name, script in scripts:
        path = os.path.join(exp_dir, script)
        if os.path.exists(path):
            print(f"\n--- {name} 시각화 실행 ---")
            subprocess.run(['python', path], cwd=BASE_DIR, check=False)
    print("\n--- 합본 시각화 ---")
    create_summary_visualization()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        run_all_visualizations()
    else:
        create_summary_visualization()
