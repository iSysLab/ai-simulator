"""실측 vs 예측 산점도(log-log)를 순수 Python SVG로 생성 (matplotlib 불필요).

통합 단일 메타모델의 5-겹 CV 예측을 백엔드별 색으로 그려 그림 1로 사용.
"""
import json
import math

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

import train_merged as tm

BACKEND_COLOR = {
    "GPU(CUDA)": "#d62728",
    "GPU(MPS)": "#1f77b4",
    "CPU": "#2ca02c",
}
BACKEND_LABEL = {"GPU(CUDA)": "Desktop CUDA", "GPU(MPS)": "Mac MPS", "CPU": "CPU"}

W, H, PAD = 460, 440, 60


def main():
    data = []
    for f in ["results/benchmark_results_enriched.json",
              "results/benchmark_results_mac_enriched.json"]:
        data += json.load(open(f))

    X, y = tm.build_xy(data, "avg_train")
    ylog = np.log1p(y)
    kf = KFold(5, shuffle=True, random_state=42)
    est = XGBRegressor(n_estimators=400, max_depth=6, random_state=42,
                       n_jobs=1, verbosity=0)
    pred = np.expm1(cross_val_predict(est, X, ylog, cv=kf, n_jobs=1))
    devices = [r["device"] for r in data]

    # log10 좌표 범위 (0.01s ~ 1000s)
    lo, hi = math.log10(0.05), math.log10(1000)

    def sx(v):
        return PAD + (math.log10(max(v, 0.01)) - lo) / (hi - lo) * (W - 2 * PAD)

    def sy(v):
        return H - PAD - (math.log10(max(v, 0.01)) - lo) / (hi - lo) * (H - 2 * PAD)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>']

    # 축·격자
    for e in range(-1, 4):
        v = 10 ** e
        x, yy = sx(v), sy(v)
        parts.append(f'<line x1="{x:.1f}" y1="{PAD}" x2="{x:.1f}" y2="{H-PAD}" stroke="#eee"/>')
        parts.append(f'<line x1="{PAD}" y1="{yy:.1f}" x2="{W-PAD}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-PAD+16:.1f}" text-anchor="middle" fill="#555">{v:g}s</text>')
        parts.append(f'<text x="{PAD-8:.1f}" y="{yy+4:.1f}" text-anchor="end" fill="#555">{v:g}</text>')

    # y=x 기준선
    parts.append(f'<line x1="{sx(10**lo):.1f}" y1="{sy(10**lo):.1f}" '
                 f'x2="{sx(10**hi):.1f}" y2="{sy(10**hi):.1f}" '
                 f'stroke="#999" stroke-dasharray="5,4"/>')

    # 점
    for d, yt, yp in zip(devices, y, pred):
        c = BACKEND_COLOR.get(d, "#888")
        parts.append(f'<circle cx="{sx(yt):.1f}" cy="{sy(yp):.1f}" r="2.6" '
                     f'fill="{c}" fill-opacity="0.55"/>')

    # 라벨·범례
    parts.append(f'<text x="{W/2:.0f}" y="{H-18}" text-anchor="middle">실측 학습시간 (s, log)</text>')
    parts.append(f'<text x="18" y="{H/2:.0f}" text-anchor="middle" '
                 f'transform="rotate(-90 18 {H/2:.0f})">예측 학습시간 (s, log)</text>')
    parts.append(f'<text x="{W/2:.0f}" y="24" text-anchor="middle" font-weight="bold">'
                 f'통합 단일 메타모델: 실측 vs 예측 (R²log=0.985)</text>')
    ly = PAD + 6
    for d in ["GPU(CUDA)", "GPU(MPS)", "CPU"]:
        parts.append(f'<circle cx="{W-PAD-88}" cy="{ly-4}" r="4" fill="{BACKEND_COLOR[d]}"/>')
        parts.append(f'<text x="{W-PAD-78}" y="{ly}" fill="#333">{BACKEND_LABEL[d]}</text>')
        ly += 18
    parts.append("</svg>")

    out = "paper/figures/fig1_scatter_train.svg"
    import os
    os.makedirs("paper/figures", exist_ok=True)
    open(out, "w").write("\n".join(parts))
    print(f"저장: {out}  (점 {len(y)}개)")


if __name__ == "__main__":
    main()
