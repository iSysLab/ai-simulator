"""스마트미디어저널 원고 그림 3: 실측 시간 대 예측 시간 산점도(학습·추론, 바깥 폴드 예측).

색은 장치 종류(CPU 파랑·GPU 주황, 색각 이상 검사 통과 2색), 마커는 기기(데스크톱 ●·맥북 ▲)로 구분해
흑백 인쇄에서도 네 플랫폼을 읽을 수 있게 한다.
사용: python paper/smj/make_fig_smj_scatter.py --method platform_w [--seed 0]
입력: paper/smj/results/preds_avg_train.npz, preds_avg_infer.npz → paper/smj/figures/fig3_smj_scatter.png
"""
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
# 로그 눈금은 10^-2 대신 0.01처럼 소수로 쓴다(AppleGothic에 수식용 빼기 기호 U+2212가 없음)
PLAIN = mticker.FuncFormatter(lambda v, _: f"{v:g}")

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures", "fig3_smj_scatter.png")
CPU, GPU = "#2a78d6", "#eb6834"
STYLE = {"Desktop CPU": ("x86 CPU", CPU, "o"), "CUDA": ("CUDA GPU", GPU, "o"),
         "Mac CPU": ("ARM CPU", CPU, "^"), "MPS": ("MPS", GPU, "^")}
PANELS = [("avg_train", "(a) 학습 시간"), ("avg_infer", "(b) 추론 시간")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="platform_w")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # 원고에 단 폭(79 mm)으로 넣으므로 그림 크기를 그 폭에 맞추고, 글씨는 7 pt 안팎으로 둔다(축소되지 않음)
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 1.72), dpi=300)
    for ax, (tgt, title) in zip(axes, PANELS):
        d = np.load(os.path.join(RES, f"preds_{tgt}.npz"))
        y, p, plat = d["y"], d[args.method][args.seed], d["plat"]
        lo, hi = 10 ** np.floor(np.log10(min(y.min(), p.min()))), 10 ** np.ceil(np.log10(max(y.max(), p.max())))
        line = np.array([lo, hi])
        ax.plot(line, line, color="#0b0b0b", lw=0.6, zorder=1)
        for f in (0.8, 1.2):
            ax.plot(line, line * f, color="#8a8984", lw=0.5, ls="--", zorder=1)
        for key, (label, color, marker) in STYLE.items():
            m = plat == key
            ax.scatter(y[m], p[m], s=6, c=color, marker=marker, linewidths=0.2, edgecolors="white",
                       alpha=0.85, label=label, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            # 5자리 범위라 눈금 글씨는 두 자리(100배)마다, 10배 눈금은 짧은 눈금으로만 둔다
            axis.set_major_locator(mticker.LogLocator(base=100))
            axis.set_minor_locator(mticker.LogLocator(base=10))
            axis.set_major_formatter(PLAIN)
            axis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.set_xlabel("실측 (초)", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5, length=2.5, width=0.4, pad=1)
        ax.tick_params(which="minor", length=1.5, width=0.3)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color("#52514e")
    axes[0].set_ylabel("예측 (초)", fontsize=7, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               handletextpad=0.2, columnspacing=0.8, markerscale=1.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.09, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
