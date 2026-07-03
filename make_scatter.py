"""그림 1: 통합 단일 메타모델의 실측 vs 예측 산점도 (2패널, 4백엔드).

- 좌: 학습시간(XGBoost), 우: 추론시간(GradientBoosting) — 각 타깃의 최적 모델
- 5-겹 CV 예측(cross_val_predict), 색은 4개 백엔드 (Desktop CPU/CUDA/Mac CPU/MPS)
- 출력: paper/figures/fig1_scatter.png (300dpi, HWP용) + .svg (미리보기용)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

import train_merged as tm

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BACKENDS = ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]
COLOR = {"Desktop CPU": "#2ca02c", "CUDA": "#d62728",
         "Mac CPU": "#ff7f0e", "MPS": "#1f77b4"}
MARKER = {"Desktop CPU": "o", "CUDA": "s", "Mac CPU": "^", "MPS": "D"}


def main():
    desk = json.load(open("results/benchmark_results_enriched.json"))
    mac = json.load(open("results/benchmark_results_mac_enriched.json"))
    data = desk + mac
    backends = []
    for i, r in enumerate(data):
        if r["device"] == "CPU":
            backends.append("Desktop CPU" if i < len(desk) else "Mac CPU")
        else:
            backends.append({"GPU(CUDA)": "CUDA", "GPU(MPS)": "MPS"}[r["device"]])
    backends = np.array(backends)

    kf = KFold(5, shuffle=True, random_state=42)
    panels = [
        ("avg_train", "학습시간",
         XGBRegressor(n_estimators=400, max_depth=6, random_state=42,
                      n_jobs=1, verbosity=0)),
        ("avg_infer", "추론시간", GradientBoostingRegressor(random_state=42)),
    ]

    # 2단 조판의 단 폭(약 8cm)에 맞도록 세로 스택, 지면 절약형 높이
    fig, axes = plt.subplots(2, 1, figsize=(4.7, 5.0))
    for ax, (target, tlabel, est) in zip(axes, panels):
        X, y = tm.build_xy(data, target)
        ylog = np.log1p(y)
        pred_log = cross_val_predict(est, X, ylog, cv=kf, n_jobs=1)
        pred = np.expm1(pred_log)
        r2l = r2_score(ylog, pred_log)

        lo = min(y[y > 0].min(), pred[pred > 0].min()) * 0.5
        hi = max(y.max(), pred.max()) * 2
        ax.plot([lo, hi], [lo, hi], ls="--", c="#999", lw=1, zorder=1)
        for b in BACKENDS:
            m = backends == b
            ax.scatter(y[m], pred[m], s=14, c=COLOR[b], marker=MARKER[b],
                       alpha=0.55, linewidths=0, label=b, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        # AppleGothic에 U+2212 글리프가 없어 mathtext 지수 대신 일반 숫자 표기
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(plain)
        ax.yaxis.set_major_formatter(plain)
        ax.set_xlabel(f"실측 {tlabel} (s)")
        ax.set_ylabel(f"예측 {tlabel} (s)")
        ax.set_title(f"{tlabel}  R²(log)={r2l:.3f}", fontsize=11)
        ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
        print(f"{tlabel}: R2(log)={r2l:.4f}  n={len(y)}")

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9,
                   handletextpad=0.3, borderpad=0.4)
    fig.tight_layout()

    os.makedirs("paper/figures", exist_ok=True)
    for ext in ["png", "svg"]:
        out = f"paper/figures/fig1_scatter.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
