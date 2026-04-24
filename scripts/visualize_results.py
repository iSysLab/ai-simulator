"""벤치마크 결과 시각화 + 예측 모델 검증 시각화

사용법:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --input results/benchmark_results.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np

from benchmark.support.hardware_info import (
    collect_experiment_environment_ko,
    format_experiment_environment_caption_ko,
    format_experiment_environment_report_ko,
)

# Windows WDAC 등으로 matplotlib 네이티브 DLL이 막히면 지연 로드 실패 → SVG 폴백
plt = None  # type: ignore
matplotlib = None  # type: ignore


def _setup_matplotlib():
    """성공 시 True. ImportError / OSError(DLL 차단) 시 False."""
    global plt, matplotlib
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as pltp
        import platform as _platform

        matplotlib = mpl
        plt = pltp
        _os = _platform.system()
        if _os == 'Darwin':
            mpl.rcParams['font.family'] = 'AppleGothic'
        elif _os == 'Windows':
            mpl.rcParams['font.family'] = 'Malgun Gothic'
        else:
            from matplotlib.font_manager import FontProperties as _FP
            for _f in ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans']:
                if _FP(family=_f).get_name() == _f or _f == 'DejaVu Sans':
                    mpl.rcParams['font.family'] = _f
                    break
        mpl.rcParams['axes.unicode_minus'] = False
        return True
    except (ImportError, OSError):
        plt = None
        matplotlib = None
        return False

# 색상 팔레트
COLORS = {
    'simple_ann': '#4C72B0',
    'simple_cnn': '#DD8452',
    'resnet_mnist': '#55A868',
    'mobilenet_mnist': '#C44E52',
    'transformer': '#8172B3',
    'gan': '#937860',
}
DEVICE_COLORS = {'CPU': '#4C72B0', 'GPU(CUDA)': '#DD8452', 'GPU(MPS)': '#55A868'}
MODEL_LABELS = {
    'simple_ann': 'ANN',
    'simple_cnn': 'CNN',
    'resnet_mnist': 'ResNet',
    'mobilenet_mnist': 'MobileNet',
    'transformer': 'Transformer',
    'gan': 'GAN',
}


def _save_figure_with_env(fig, path, env_caption="", subplot_top=1.0):
    """그림 저장 + 하단 실험 환경 캡션 (GPU 이름·VRAM·SM / CPU·RAM·OS).

    subplot_top: suptitle 이 있을 때 0.93 등으로 지정해 상단 여백 확보.
    """
    cap = (env_caption or "").strip()
    if cap:
        fig.text(
            0.5, 0.006, cap,
            ha='center', va='bottom', fontsize=7,
            transform=fig.transFigure, linespacing=1.2,
        )
        bottom = 0.1
    else:
        bottom = 0
    fig.tight_layout(rect=[0, bottom, 1, subplot_top])
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.45)
    plt.close()
    print(f"  저장: {path}")


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary_table(data):
    """데이터 요약 테이블 출력"""
    print("\n" + "=" * 90)
    print("  벤치마크 데이터 요약")
    print("=" * 90)

    # 모델 타입별 통계
    from collections import defaultdict
    stats = defaultdict(lambda: {'count': 0, 'devices': set(),
                                  'train_times': [], 'infer_times': [],
                                  'params': [], 'flops': []})
    for r in data:
        mt = r['model_type']
        s = stats[mt]
        s['count'] += 1
        s['devices'].add(r['device'])
        s['train_times'].append(r['avg_train'])
        s['infer_times'].append(r['avg_infer'])
        s['params'].append(r['total_params'])
        s['flops'].append(r['flops'])

    print(f"\n  {'모델':<15s} | {'개수':>4s} | {'디바이스':<15s} | "
          f"{'파라미터 범위':<25s} | {'학습시간 범위(s)':<25s} | {'추론시간 범위(s)':<25s}")
    print(f"  {'-'*15} | {'-'*4} | {'-'*15} | {'-'*25} | {'-'*25} | {'-'*25}")

    for mt in ['simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist', 'transformer', 'gan']:
        if mt not in stats:
            continue
        s = stats[mt]
        devs = ', '.join(sorted(s['devices']))
        p_min, p_max = min(s['params']), max(s['params'])
        t_min, t_max = min(s['train_times']), max(s['train_times'])
        i_min, i_max = min(s['infer_times']), max(s['infer_times'])
        print(f"  {MODEL_LABELS[mt]:<15s} | {s['count']:>4d} | {devs:<15s} | "
              f"{p_min:>10,d} ~ {p_max:>10,d} | "
              f"{t_min:>10.4f} ~ {t_max:>10.4f} | "
              f"{i_min:>10.5f} ~ {i_max:>10.5f}")

    print(f"\n  총 데이터: {len(data)}개")

    # 디바이스별 통계
    dev_stats = defaultdict(list)
    for r in data:
        dev_stats[r['device']].append(r)

    print(f"\n  디바이스별 분포:")
    for dev, rows in sorted(dev_stats.items()):
        mt_counts = defaultdict(int)
        for r in rows:
            mt_counts[r['model_type']] += 1
        detail = ', '.join(f"{MODEL_LABELS[k]}:{v}" for k, v in sorted(mt_counts.items()))
        print(f"    {dev:<12s}: {len(rows):>3d}개 ({detail})")


def print_prediction_table():
    """예측 모델 성능 테이블"""
    print("\n" + "=" * 90)
    print("  예측 모델 성능 (5-Fold CV, XGBoost 최적)")
    print("=" * 90)

    results = [
        ('학습 시간', 'CPU', 0.9465, 0.9865, 25.354, 10.103),
        ('학습 시간', 'CUDA', 0.9391, 0.9724, 1.561, 0.663),
        ('추론 시간', 'CPU', 0.9345, 0.9772, 1.686, 0.642),
        ('추론 시간', 'CUDA', 0.9520, 0.9659, 0.084, 0.038),
        ('메모리', 'CPU', 0.9445, 0.9974, 2198921.8, 438819.1),
        ('메모리', 'CUDA', 0.9430, 0.9979, 2283915.0, 480165.3),
    ]

    print(f"\n  {'타겟':<10s} | {'디바이스':<8s} | {'R²':>7s} | {'R²(log)':>8s} | {'RMSE':>12s} | {'MAE':>12s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*12} | {'-'*12}")
    for target, dev, r2, r2log, rmse, mae in results:
        print(f"  {target:<10s} | {dev:<8s} | {r2:>7.4f} | {r2log:>8.4f} | {rmse:>12.3f} | {mae:>12.3f}")


def fig1_params_vs_time(data, output_dir, env_caption=""):
    """그림 1: 파라미터 수 vs 학습/추론 시간 (모델별 색상, 디바이스별 마커)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    markers = {'CPU': 'o', 'GPU(CUDA)': '^', 'GPU(MPS)': 's'}

    for ax, (time_key, title) in zip(axes, [('avg_train', '학습 시간'), ('avg_infer', '추론 시간')]):
        for r in data:
            mt = r['model_type']
            ax.scatter(r['total_params'], r[time_key],
                       c=COLORS[mt], marker=markers.get(r['device'], 'o'),
                       alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('파라미터 수', fontsize=12)
        ax.set_ylabel(f'{title} (초)', fontsize=12)
        ax.set_title(f'파라미터 수 vs {title}', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # 범례
    from matplotlib.lines import Line2D
    legend_models = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[mt],
                            label=MODEL_LABELS[mt], markersize=8) for mt in COLORS]
    legend_devices = [Line2D([0], [0], marker=markers[d], color='w', markerfacecolor='gray',
                             label=d, markersize=8) for d in markers if any(r['device'] == d for r in data)]
    axes[1].legend(handles=legend_models + legend_devices, loc='upper left', fontsize=9)

    path = os.path.join(output_dir, 'fig1_params_vs_time.png')
    _save_figure_with_env(fig, path, env_caption)


def fig2_flops_vs_time(data, output_dir, env_caption=""):
    """그림 2: FLOPs vs 학습/추론 시간"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    markers = {'CPU': 'o', 'GPU(CUDA)': '^', 'GPU(MPS)': 's'}

    for ax, (time_key, title) in zip(axes, [('avg_train', '학습 시간'), ('avg_infer', '추론 시간')]):
        for r in data:
            mt = r['model_type']
            flops = r.get('flops', 0)
            if flops <= 0:
                continue
            ax.scatter(flops, r[time_key],
                       c=COLORS[mt], marker=markers.get(r['device'], 'o'),
                       alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('FLOPs', fontsize=12)
        ax.set_ylabel(f'{title} (초)', fontsize=12)
        ax.set_title(f'FLOPs vs {title}', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_models = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[mt],
                            label=MODEL_LABELS[mt], markersize=8) for mt in COLORS]
    axes[1].legend(handles=legend_models, loc='upper left', fontsize=9)

    path = os.path.join(output_dir, 'fig2_flops_vs_time.png')
    _save_figure_with_env(fig, path, env_caption)


def fig3_device_comparison(data, output_dir, env_caption=""):
    """그림 3: CPU vs GPU 속도 비교 (모델별 박스플롯)"""
    from collections import defaultdict

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (time_key, title) in zip(axes, [('avg_train', '학습 시간'), ('avg_infer', '추론 시간')]):
        model_types = ['simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist', 'transformer', 'gan']
        devices = sorted(set(r['device'] for r in data))

        x = np.arange(len(model_types))
        width = 0.35

        for i, dev in enumerate(devices):
            means = []
            stds = []
            for mt in model_types:
                times = [r[time_key] for r in data if r['model_type'] == mt and r['device'] == dev]
                means.append(np.mean(times) if times else 0)
                stds.append(np.std(times) if times else 0)
            offset = (i - (len(devices) - 1) / 2) * width
            bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                          label=dev, color=DEVICE_COLORS.get(dev, 'gray'),
                          alpha=0.8, capsize=3)

        ax.set_xlabel('모델', fontsize=12)
        ax.set_ylabel(f'{title} 평균 (초)', fontsize=12)
        ax.set_title(f'디바이스별 {title} 비교', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[mt] for mt in model_types], fontsize=10)
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(output_dir, 'fig3_device_comparison.png')
    _save_figure_with_env(fig, path, env_caption)


def fig4_speedup_ratio(data, output_dir, env_caption=""):
    """그림 4: GPU 대비 CPU 속도비 (Speedup)"""
    from collections import defaultdict

    cpu_times = defaultdict(list)
    gpu_times = defaultdict(list)

    for r in data:
        key = r['model_type']
        if r['device'] == 'CPU':
            cpu_times[key].append((r['avg_train'], r['avg_infer']))
        elif 'GPU' in r['device']:
            gpu_times[key].append((r['avg_train'], r['avg_infer']))

    model_types = [mt for mt in ['simple_ann', 'simple_cnn', 'resnet_mnist',
                                  'mobilenet_mnist', 'transformer', 'gan']
                   if mt in cpu_times and mt in gpu_times]

    if not model_types:
        print("  GPU 데이터 없음 - speedup 그래프 건너뜀")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_types))
    width = 0.35

    train_speedups = []
    infer_speedups = []
    for mt in model_types:
        cpu_train = np.mean([t[0] for t in cpu_times[mt]])
        gpu_train = np.mean([t[0] for t in gpu_times[mt]])
        cpu_infer = np.mean([t[1] for t in cpu_times[mt]])
        gpu_infer = np.mean([t[1] for t in gpu_times[mt]])
        train_speedups.append(cpu_train / gpu_train if gpu_train > 0 else 0)
        infer_speedups.append(cpu_infer / gpu_infer if gpu_infer > 0 else 0)

    ax.bar(x - width/2, train_speedups, width, label='학습 Speedup', color='#4C72B0', alpha=0.8)
    ax.bar(x + width/2, infer_speedups, width, label='추론 Speedup', color='#DD8452', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Speedup = 1 (동일)')

    ax.set_xlabel('모델', fontsize=12)
    ax.set_ylabel('Speedup (CPU시간 / GPU시간)', fontsize=12)
    ax.set_title('GPU 대비 CPU 속도비 (높을수록 GPU가 유리)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[mt] for mt in model_types], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (ts, is_) in enumerate(zip(train_speedups, infer_speedups)):
        ax.text(i - width/2, ts + 0.1, f'{ts:.1f}x', ha='center', fontsize=8)
        ax.text(i + width/2, is_ + 0.1, f'{is_:.1f}x', ha='center', fontsize=8)

    path = os.path.join(output_dir, 'fig4_speedup_ratio.png')
    _save_figure_with_env(fig, path, env_caption)


def fig5_prediction_accuracy(data, output_dir, env_caption=""):
    """그림 5: 예측 모델 실제 vs 예측 (XGBoost CV 결과)"""
    from train_predictor import (
        prepare_features, enrich_result, FEATURE_COLUMNS,
        MODEL_FAMILY_MAP, DEVICE_TYPE_MAP, DATASET_INFO
    )
    from sklearn.model_selection import KFold, cross_val_predict

    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost 미설치 - 예측 시각화 건너뜀")
        return

    X, y_train, y_infer, y_memory, devices, names = prepare_features(data)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    targets = [
        (y_train, '학습 시간 (s)'),
        (y_infer, '추론 시간 (s)'),
        (y_memory, '메모리 (bytes)'),
    ]

    unique_devices = sorted(set(devices))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col_idx, (y_all, target_label) in enumerate(targets):
        for row_idx, dev in enumerate(unique_devices):
            ax = axes[row_idx][col_idx]
            mask = np.array([d == dev for d in devices])
            X_dev = X[mask]
            y_dev = y_all[mask]
            names_dev = [n for n, m in zip(names, mask) if m]
            types_dev = [data[i]['model_type'] for i, m in enumerate(mask) if m]

            y_log = np.log1p(y_dev)

            model = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=3,
                random_state=42, verbosity=0)
            y_pred_log = cross_val_predict(model, X_dev, y_log, cv=kf)

            y_true = np.expm1(y_log)
            y_pred = np.expm1(y_pred_log)

            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            r2log = r2_score(y_log, y_pred_log)

            for mt in COLORS:
                mt_mask = [t == mt for t in types_dev]
                if any(mt_mask):
                    yt = y_true[mt_mask]
                    yp = y_pred[mt_mask]
                    ax.scatter(yt, yp, c=COLORS[mt], alpha=0.7, s=40,
                               label=MODEL_LABELS[mt], edgecolors='white', linewidth=0.5)

            # 대각선 (완벽한 예측)
            vmin = min(y_true.min(), y_pred.min()) * 0.8
            vmax = max(y_true.max(), y_pred.max()) * 1.2
            ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, linewidth=1)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f'실측 {target_label}', fontsize=10)
            ax.set_ylabel(f'예측 {target_label}', fontsize=10)
            ax.set_title(f'{target_label} [{dev}]\nR²={r2:.4f}, R²(log)={r2log:.4f}', fontsize=11)
            ax.grid(True, alpha=0.3)

            if col_idx == 2:
                ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('XGBoost 예측 정확도: 실측 vs 예측 (5-Fold CV)', fontsize=15, y=1.01)
    path = os.path.join(output_dir, 'fig5_prediction_accuracy.png')
    _save_figure_with_env(fig, path, env_caption, subplot_top=0.92)


def fig6_feature_importance(data, output_dir, env_caption=""):
    """그림 6: 피처 중요도 Top 15 (학습/추론 시간)"""
    from train_predictor import prepare_features, FEATURE_COLUMNS

    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost 미설치 - 피처 중요도 시각화 건너뜀")
        return

    X, y_train, y_infer, y_memory, devices, names = prepare_features(data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    targets = [
        (y_train, '학습 시간'),
        (y_infer, '추론 시간'),
    ]
    unique_devices = sorted(set(devices))

    for col_idx, (y_all, target_label) in enumerate(targets):
        for row_idx, dev in enumerate(unique_devices):
            ax = axes[row_idx][col_idx]
            mask = np.array([d == dev for d in devices])
            X_dev = X[mask]
            y_dev = y_all[mask]
            y_log = np.log1p(y_dev)

            model = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=3,
                random_state=42, verbosity=0)
            model.fit(X_dev, y_log)

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            top_names = [FEATURE_COLUMNS[i] for i in indices]
            top_vals = [importances[i] for i in indices]

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
            bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1], alpha=0.85)
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.set_xlabel('중요도', fontsize=10)
            ax.set_title(f'{target_label} [{dev}] - 피처 중요도 Top 15', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')

            for i, (bar, val) in enumerate(zip(bars, top_vals[::-1])):
                ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8)

    plt.suptitle('XGBoost 피처 중요도 분석', fontsize=15, y=1.01)
    path = os.path.join(output_dir, 'fig6_feature_importance.png')
    _save_figure_with_env(fig, path, env_caption, subplot_top=0.92)


def fig7_model_complexity_heatmap(data, output_dir, env_caption=""):
    """그림 7: 모델별 복잡도 지표 히트맵"""
    from collections import defaultdict

    model_types = ['simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist', 'transformer', 'gan']
    metrics = ['total_params', 'flops', 'total_layers', 'avg_train', 'avg_infer']
    metric_labels = ['파라미터 수', 'FLOPs', '레이어 수', '학습시간(s)', '추론시간(s)']

    fig, ax = plt.subplots(figsize=(10, 6))

    # CPU 데이터만 사용
    matrix = []
    for mt in model_types:
        rows = [r for r in data if r['model_type'] == mt and r['device'] == 'CPU']
        if not rows:
            rows = [r for r in data if r['model_type'] == mt]
        if rows:
            avg_vals = [np.mean([r.get(m, 0) for r in rows]) for m in metrics]
        else:
            avg_vals = [0] * len(metrics)
        matrix.append(avg_vals)

    matrix = np.array(matrix)
    # 각 열을 0-1 정규화
    for j in range(matrix.shape[1]):
        col_max = matrix[:, j].max()
        if col_max > 0:
            matrix[:, j] = matrix[:, j] / col_max

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=20, ha='right')
    ax.set_yticks(range(len(model_types)))
    ax.set_yticklabels([MODEL_LABELS[mt] for mt in model_types], fontsize=10)

    for i in range(len(model_types)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if matrix[i, j] > 0.5 else 'black')

    plt.colorbar(im, label='정규화된 값 (0~1)')
    ax.set_title('모델별 복잡도 지표 비교 (CPU 평균, 정규화)', fontsize=14)

    path = os.path.join(output_dir, 'fig7_complexity_heatmap.png')
    _save_figure_with_env(fig, path, env_caption)


def fig8_cross_platform_comparison(desktop_data, mac_data, output_dir, env_caption=""):
    """그림 8: 크로스 플랫폼 비교 (Desktop vs Mac, CPU 학습시간)"""
    from collections import defaultdict

    model_types = ['simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist', 'transformer', 'gan']

    # 모델별 CPU 평균 학습시간
    dt_cpu = defaultdict(list)
    mac_cpu = defaultdict(list)
    mac_mps = defaultdict(list)
    dt_cuda = defaultdict(list)

    for r in desktop_data:
        if r['device'] == 'CPU':
            dt_cpu[r['model_type']].append(r['avg_train'])
        elif 'GPU' in r['device']:
            dt_cuda[r['model_type']].append(r['avg_train'])

    for r in mac_data:
        if r['device'] == 'CPU':
            mac_cpu[r['model_type']].append(r['avg_train'])
        elif 'GPU' in r['device']:
            mac_mps[r['model_type']].append(r['avg_train'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: 4개 플랫폼 학습시간 비교
    ax = axes[0]
    x = np.arange(len(model_types))
    width = 0.2

    platforms = [
        (dt_cpu, 'Desktop CPU', '#4C72B0'),
        (dt_cuda, 'Desktop CUDA', '#DD8452'),
        (mac_cpu, 'Mac CPU', '#55A868'),
        (mac_mps, 'Mac MPS', '#C44E52'),
    ]

    for i, (pdata, plabel, pcolor) in enumerate(platforms):
        means = [np.mean(pdata[mt]) if pdata[mt] else 0 for mt in model_types]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width * 0.9, label=plabel, color=pcolor, alpha=0.85)

    ax.set_xlabel('모델', fontsize=12)
    ax.set_ylabel('평균 학습 시간 (초)', fontsize=12)
    ax.set_title('플랫폼별 학습 시간 비교', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[mt] for mt in model_types], fontsize=10)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 오른쪽: Mac CPU / Desktop CPU 비율
    ax2 = axes[1]
    ratios_cpu = []
    ratios_mps = []
    for mt in model_types:
        dt_mean = np.mean(dt_cpu[mt]) if dt_cpu[mt] else 1
        mac_mean = np.mean(mac_cpu[mt]) if mac_cpu[mt] else 0
        mps_mean = np.mean(mac_mps[mt]) if mac_mps[mt] else 0
        ratios_cpu.append(mac_mean / dt_mean if dt_mean > 0 else 0)
        ratios_mps.append(mps_mean / dt_mean if dt_mean > 0 else 0)

    ax2.bar(x - width/2, ratios_cpu, width, label='Mac CPU / Desktop CPU', color='#55A868', alpha=0.85)
    ax2.bar(x + width/2, ratios_mps, width, label='Mac MPS / Desktop CPU', color='#C44E52', alpha=0.85)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='동일 성능 (1.0x)')

    ax2.set_xlabel('모델', fontsize=12)
    ax2.set_ylabel('시간 비율 (높을수록 느림)', fontsize=12)
    ax2.set_title('크로스 플랫폼 성능 비율 (Desktop CPU 기준)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[mt] for mt in model_types], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (rc, rm) in enumerate(zip(ratios_cpu, ratios_mps)):
        ax2.text(i - width/2, rc + 0.2, f'{rc:.1f}x', ha='center', fontsize=8)
        ax2.text(i + width/2, rm + 0.05, f'{rm:.1f}x', ha='center', fontsize=8)

    path = os.path.join(output_dir, 'fig8_cross_platform.png')
    _save_figure_with_env(fig, path, env_caption)


def fig9_depthwise_penalty(desktop_data, mac_data, output_dir, env_caption=""):
    """그림 9: Depthwise Conv 페널티 분석 (MobileNet 파라미터 vs 시간, 플랫폼별)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    datasets = [
        (desktop_data, 'Desktop (Ryzen 7800X3D + RTX 4060 Ti)'),
        (mac_data, 'Mac (Apple M4)'),
    ]

    for ax, (data, title) in zip(axes, datasets):
        markers = {'CPU': 'o', 'GPU(CUDA)': '^', 'GPU(MPS)': 's'}

        for r in data:
            if r['model_type'] != 'mobilenet_mnist':
                continue
            dev = r['device']
            ax.scatter(r['total_params'], r['avg_train'],
                       c=DEVICE_COLORS.get(dev, 'gray'),
                       marker=markers.get(dev, 'o'),
                       alpha=0.8, s=80, edgecolors='white', linewidth=0.5,
                       label=dev if dev not in ax.get_legend_handles_labels()[1] else '')

        ax.set_xlabel('파라미터 수', fontsize=12)
        ax.set_ylabel('학습 시간 (초)', fontsize=12)
        ax.set_title(f'MobileNet: 파라미터 vs 학습시간\n{title}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 범례 중복 제거
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    path = os.path.join(output_dir, 'fig9_depthwise_penalty.png')
    _save_figure_with_env(fig, path, env_caption)


def main():
    parser = argparse.ArgumentParser(description='벤치마크 결과 시각화')
    parser.add_argument('--input', type=str, default='results/benchmark_results.json')
    parser.add_argument('--input-mac', type=str, default='results/benchmark_results_mac.json')
    parser.add_argument('--output-dir', type=str, default='results/figures')
    parser.add_argument(
        '--skip-mac', action='store_true',
        help='Mac JSON을 읽지 않고 fig1~fig7만 생성 (크로스 플랫폼 fig8~9 생략)',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.input)

    env_info = collect_experiment_environment_ko(args.input)
    env_caption = format_experiment_environment_caption_ko(env_info)
    env_txt_path = os.path.join(args.output_dir, 'EXPERIMENT_ENV.txt')
    try:
        with open(env_txt_path, 'w', encoding='utf-8') as ef:
            ef.write(format_experiment_environment_report_ko(env_info))
        print(f"실험 환경 요약 저장: {env_txt_path}")
    except OSError:
        pass

    # Mac 데이터 로드 (--skip-mac 이면 생략)
    mac_data = None
    if not args.skip_mac and os.path.exists(args.input_mac):
        mac_data = load_data(args.input_mac)
        print(f"Mac 데이터 로드: {len(mac_data)}개")
    elif args.skip_mac:
        print("Mac 데이터 사용 안 함 (--skip-mac). fig8~fig9는 생성하지 않습니다.")

    # 전체 데이터 (Desktop + Mac)
    all_data = data + (mac_data if mac_data else [])

    # 1. 표 출력
    print_summary_table(all_data)
    print_prediction_table()

    if not _setup_matplotlib():
        print(
            "\n[경고] matplotlib 를 불러올 수 없습니다 (DLL 차단·미설치 등). "
            "동일 데이터로 SVG(fig1~fig7)를 생성합니다.\n"
            "      직접 실행: python scripts/visualize_results_svg.py --skip-mac\n"
        )
        import importlib.util

        svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize_results_svg.py')
        spec = importlib.util.spec_from_file_location('visualize_results_svg', svg_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.run(args)
        return

    # 2. 시각화 — Mac 제외 시 fig1~fig7 (7개), 파일명·역할은 README 참고
    print(f"\n시각화 생성 중... (저장 위치: {args.output_dir}/)")
    print("  각 PNG 하단: GPU 이름·VRAM·SM, CPU·RAM·OS (한글 캡션)")
    fig1_params_vs_time(all_data, args.output_dir, env_caption)
    fig2_flops_vs_time(all_data, args.output_dir, env_caption)
    fig3_device_comparison(all_data, args.output_dir, env_caption)
    fig4_speedup_ratio(all_data, args.output_dir, env_caption)
    fig5_prediction_accuracy(data, args.output_dir, env_caption)  # Desktop 벤치만
    fig6_feature_importance(data, args.output_dir, env_caption)
    fig7_model_complexity_heatmap(all_data, args.output_dir, env_caption)

    # 3. 크로스 플랫폼 (Mac JSON 있고 --skip-mac 아닐 때만)
    if mac_data:
        print(f"\n크로스 플랫폼 시각화 생성 중...")
        fig8_cross_platform_comparison(data, mac_data, args.output_dir, env_caption)
        fig9_depthwise_penalty(data, mac_data, args.output_dir, env_caption)
        print(f"\n완료! 총 9개 그래프 + EXPERIMENT_ENV.txt: {args.output_dir}/")
    else:
        print(f"\n완료! 총 7개 그래프 + EXPERIMENT_ENV.txt (Mac 제외): {args.output_dir}/")


if __name__ == '__main__':
    main()
