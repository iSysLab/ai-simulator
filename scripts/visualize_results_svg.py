"""matplotlib DLL 차단(Windows WDAC 등) 시 벤치마크 fig1~fig7 을 SVG로 생성.

`python scripts/visualize_results.py` 가 matplotlib 로드에 실패하면 자동 호출된다.
직접 실행: `python scripts/visualize_results_svg.py --skip-mac`

출력: results/figures/fig1_*.svg … fig7_*.svg + EXPERIMENT_ENV.txt
(fig8~9 크로스 플랫폼은 SVG 미구현 — matplotlib 필요)
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.support.hardware_info import (
    collect_experiment_environment_ko,
    format_experiment_environment_caption_ko,
    format_experiment_environment_report_ko,
)

COLORS = {
    "simple_ann": "#4C72B0",
    "simple_cnn": "#DD8452",
    "resnet_mnist": "#55A868",
    "mobilenet_mnist": "#C44E52",
    "transformer": "#8172B3",
    "gan": "#937860",
}
DEVICE_COLORS = {"CPU": "#4C72B0", "GPU(CUDA)": "#DD8452", "GPU(MPS)": "#55A868"}
MODEL_LABELS = {
    "simple_ann": "ANN",
    "simple_cnn": "CNN",
    "resnet_mnist": "ResNet",
    "mobilenet_mnist": "MobileNet",
    "transformer": "Transformer",
    "gan": "GAN",
}


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _svg_open(w: int, h: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n',
        '<rect width="100%" height="100%" fill="#fafafa"/>\n',
    ]


def _svg_close() -> str:
    return "</svg>\n"


def _env_block(W: int, y0: int, env_caption: str) -> list[str]:
    lines: list[str] = []
    cap = (env_caption or "").strip()
    if not cap:
        return lines
    parts = cap.split("\n")
    lines.append(
        f'<text x="{W / 2:.1f}" y="{y0}" text-anchor="middle" font-size="10" '
        f'fill="#555" font-family="Malgun Gothic, sans-serif">[실험 환경]</text>\n'
    )
    for i, ln in enumerate(parts):
        lines.append(
            f'<text x="{W / 2:.1f}" y="{y0 + 14 + i * 13:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#333" font-family="Malgun Gothic, sans-serif">'
            f"{escape(ln)}</text>\n"
        )
    return lines


def _log_axes(
    xs: list[float],
    ys: list[float],
    ml: float,
    mt: float,
    pw: float,
    ph: float,
):
    pos = [float(v) for v in xs + ys if math.isfinite(v) and v > 0]
    if not pos:
        lo, hi = 1e-9, 1e-6
    else:
        lo = min(pos) * 0.85
        hi = max(pos) * 1.15
    lo = max(lo, 1e-15)
    if hi <= lo:
        hi = lo * 10
    ll = math.log10(lo)
    lh = math.log10(hi)
    span = lh - ll if lh > ll else 1e-9

    def tx(v: float) -> float:
        vv = max(float(v), lo)
        return ml + (math.log10(vv) - ll) / span * pw

    def ty(v: float) -> float:
        vv = max(float(v), lo)
        return mt + ph - (math.log10(vv) - ll) / span * ph

    return tx, ty, lo, hi


def _marker(cx: float, cy: float, dev: str, color: str, r: float = 5.0) -> str:
    if dev == "CPU":
        return (
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{color}" '
            f'fill-opacity="0.85" stroke="#fff" stroke-width="0.5"/>\n'
        )
    if "CUDA" in dev:
        s = r * 1.2
        pts = f"{cx:.2f},{cy - s:.2f} {cx - s:.2f},{cy + s * 0.6:.2f} {cx + s:.2f},{cy + s * 0.6:.2f}"
        return f'<polygon points="{pts}" fill="{color}" fill-opacity="0.85" stroke="#fff" stroke-width="0.5"/>\n'
    s = r * 0.9
    return (
        f'<rect x="{cx - s:.2f}" y="{cy - s:.2f}" width="{2 * s:.2f}" height="{2 * s:.2f}" '
        f'fill="{color}" fill-opacity="0.85" stroke="#fff" stroke-width="0.5"/>\n'
    )


def _panel_frame(ml: float, mt: float, pw: float, ph: float) -> str:
    return (
        f'<rect x="{ml:.1f}" y="{mt:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
        'fill="none" stroke="#333" stroke-width="1"/>\n'
    )


def fig1_svg(data, out_dir: str, env_caption: str) -> None:
    W, H = 1520, 640
    pw, ph = 660, 480
    ml0, ml1 = 70, 70 + pw + 40
    mt = 56
    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="32" text-anchor="middle" font-size="16" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림1: 파라미터 수 vs 학습·추론 시간</text>\n"
    )

    for col, (ykey, sub) in enumerate(
        [
            ("avg_train", "학습 시간 (s)"),
            ("avg_infer", "추론 시간 (s)"),
        ]
    ):
        ml = ml0 if col == 0 else ml1
        pts = []
        for r in data:
            x = float(r["total_params"])
            y = float(r[ykey])
            if x <= 0 or y <= 0:
                continue
            pts.append((x, y, COLORS[r["model_type"]], r["device"]))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if not xs:
            continue
        tx, ty, _, _ = _log_axes(xs, ys, ml, mt, pw, ph)
        parts.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt - 8:.1f}" text-anchor="middle" font-size="12" '
            f'font-family="Malgun Gothic, sans-serif">{escape(sub)}</text>\n'
        )
        for x, y, c, d in pts:
            parts.append(_marker(tx(x), ty(y), d, c))
        parts.append(_panel_frame(ml, mt, pw, ph))
        parts.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt + ph + 28:.1f}" text-anchor="middle" font-size="11" '
            f'font-family="Malgun Gothic, sans-serif">파라미터 수 (log)</text>\n'
        )

    parts.append(
        f'<text transform="rotate(-90 24 {(mt + ph / 2):.1f})" x="24" y="{mt + ph / 2:.1f}" '
        f'text-anchor="middle" font-size="11" font-family="Malgun Gothic, sans-serif">'
        "시간 (log)</text>\n"
    )
    parts.extend(_env_block(W, H - 52, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig1_params_vs_time.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def fig2_svg(data, out_dir: str, env_caption: str) -> None:
    W, H = 1520, 640
    pw, ph = 660, 480
    ml0, ml1 = 70, 70 + pw + 40
    mt = 56
    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="32" text-anchor="middle" font-size="16" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림2: FLOPs vs 학습·추론 시간</text>\n"
    )
    for col, ykey in enumerate(["avg_train", "avg_infer"]):
        ml = ml0 if col == 0 else ml1
        sub = "학습 시간 (s)" if col == 0 else "추론 시간 (s)"
        pts = []
        for r in data:
            x = float(r.get("flops") or 0)
            if x <= 0:
                continue
            y = float(r[ykey])
            if y <= 0:
                continue
            pts.append((x, y, COLORS[r["model_type"]], r["device"]))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if not xs:
            parts.append(
                f'<text x="{ml + pw / 2:.1f}" y="{mt + ph / 2:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#888" font-family="Malgun Gothic, sans-serif">'
                "FLOPs&gt;0 데이터 없음</text>\n"
            )
            continue
        tx, ty, _, _ = _log_axes(xs, ys, ml, mt, pw, ph)
        parts.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt - 8:.1f}" text-anchor="middle" font-size="12" '
            f'font-family="Malgun Gothic, sans-serif">{escape(sub)}</text>\n'
        )
        for x, y, c, d in pts:
            parts.append(_marker(tx(x), ty(y), d, c))
        parts.append(_panel_frame(ml, mt, pw, ph))
        parts.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt + ph + 28:.1f}" text-anchor="middle" font-size="11" '
            f'font-family="Malgun Gothic, sans-serif">FLOPs (log)</text>\n'
        )
    parts.append(
        f'<text transform="rotate(-90 24 {(mt + ph / 2):.1f})" x="24" y="{mt + ph / 2:.1f}" '
        f'text-anchor="middle" font-size="11" font-family="Malgun Gothic, sans-serif">'
        "시간 (log)</text>\n"
    )
    parts.extend(_env_block(W, H - 52, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig2_flops_vs_time.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def fig3_svg(data, out_dir: str, env_caption: str) -> None:
    model_types = [
        "simple_ann",
        "simple_cnn",
        "resnet_mnist",
        "mobilenet_mnist",
        "transformer",
        "gan",
    ]
    devices = sorted({r["device"] for r in data})
    W, H = 1500, 620
    mt, mb = 80, 120
    ml, mr = 100, 80
    pw = W - ml - mr
    ph = H - mt - mb
    n = len(model_types)
    n_dev = len(devices)
    group_w = pw / max(n, 1)
    bar_w = group_w / max(n_dev + 1, 1) * 0.85

    def means_for(time_key: str) -> np.ndarray:
        M = np.zeros((len(devices), len(model_types)))
        for j, mt_ in enumerate(model_types):
            for i, dev in enumerate(devices):
                t = [r[time_key] for r in data if r["model_type"] == mt_ and r["device"] == dev]
                M[i, j] = float(np.mean(t)) if t else 0.0
        return M

    parts = _svg_open(W, H * 2 + 100)
    y_off = 0
    for ti, (time_key, title) in enumerate([("avg_train", "학습"), ("avg_infer", "추론")]):
        M = means_for(time_key)
        vmax = float(M.max()) or 1.0
        vmin = max(float(M[M > 0].min()) if np.any(M > 0) else 0.001, 1e-4)
        ll = math.log10(vmin)
        lh = math.log10(vmax * 1.1)
        span = lh - ll if lh > ll else 1e-9

        def ty_log(v: float) -> float:
            if v <= 0:
                v = vmin
            return y_off + mt + ph - (math.log10(v) - ll) / span * ph

        parts.append(
            f'<text x="{W / 2:.1f}" y="{y_off + 36:.1f}" text-anchor="middle" font-size="15" '
            f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
            f"그림3: 디바이스별 {title} 시간 비교 (log)</text>\n"
        )
        base_y = y_off + mt + ph
        for j in range(n):
            gx = ml + j * group_w + group_w * 0.1
            for i, dev in enumerate(devices):
                v = M[i, j]
                if v <= 0:
                    continue
                bw = bar_w
                x0 = gx + i * (group_w * 0.85 / max(n_dev, 1))
                y0 = ty_log(v)
                h = base_y - y0
                col = DEVICE_COLORS.get(dev, "#888")
                parts.append(
                    f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" '
                    f'fill="{col}" fill-opacity="0.85"/>\n'
                )
        parts.append(
            f'<line x1="{ml:.1f}" y1="{base_y:.1f}" x2="{ml + pw:.1f}" y2="{base_y:.1f}" '
            'stroke="#333" stroke-width="1"/>\n'
        )
        for j, mt_ in enumerate(model_types):
            cx = ml + j * group_w + group_w / 2
            parts.append(
                f'<text x="{cx:.1f}" y="{y_off + mt + ph + 22:.1f}" text-anchor="middle" '
                f'font-size="10" font-family="Malgun Gothic, sans-serif">'
                f"{escape(MODEL_LABELS[mt_])}</text>\n"
            )
        lx = ml + pw + 8
        ly = y_off + mt + 20
        for i, dev in enumerate(devices):
            parts.append(
                f'<rect x="{lx:.1f}" y="{ly + i * 18:.1f}" width="12" height="12" '
                f'fill="{DEVICE_COLORS.get(dev, "#888")}"/>\n'
                f'<text x="{lx + 18:.1f}" y="{ly + 11 + i * 18:.1f}" font-size="10" '
                f'font-family="Malgun Gothic, sans-serif">{escape(dev)}</text>\n'
            )
        y_off += H

    parts.extend(_env_block(W, y_off + 30, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig3_device_comparison.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def fig4_svg(data, out_dir: str, env_caption: str) -> None:
    cpu_t = defaultdict(list)
    gpu_t = defaultdict(list)
    for r in data:
        k = r["model_type"]
        if r["device"] == "CPU":
            cpu_t[k].append((r["avg_train"], r["avg_infer"]))
        elif "GPU" in r["device"]:
            gpu_t[k].append((r["avg_train"], r["avg_infer"]))
    model_types = [
        mt
        for mt in [
            "simple_ann",
            "simple_cnn",
            "resnet_mnist",
            "mobilenet_mnist",
            "transformer",
            "gan",
        ]
        if mt in cpu_t and mt in gpu_t
    ]
    if not model_types:
        print("  GPU 데이터 없음 - fig4 SVG 건너뜀")
        return
    train_s = []
    infer_s = []
    for mt in model_types:
        c_tr = np.mean([t[0] for t in cpu_t[mt]])
        g_tr = np.mean([t[0] for t in gpu_t[mt]])
        c_i = np.mean([t[1] for t in cpu_t[mt]])
        g_i = np.mean([t[1] for t in gpu_t[mt]])
        train_s.append(c_tr / g_tr if g_tr > 0 else 0)
        infer_s.append(c_i / g_i if g_i > 0 else 0)

    W, H = 1000, 520
    ml, mtop, pw, ph = 100, 70, 780, 360
    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="38" text-anchor="middle" font-size="15" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림4: GPU 대비 CPU Speedup (CPU시간/GPU시간)</text>\n"
    )
    n = len(model_types)
    bw = pw / max(n * 2 + 1, 1) * 0.35
    ymax = max(max(train_s), max(infer_s), 1.1) * 1.15
    base_y = mtop + ph
    for i, mkey in enumerate(model_types):
        x0 = ml + i * (pw / max(n, 1)) + pw / max(n, 1) * 0.15
        h1 = train_s[i] / ymax * ph
        h2 = infer_s[i] / ymax * ph
        parts.append(
            f'<rect x="{x0:.1f}" y="{base_y - h1:.1f}" width="{bw:.1f}" height="{h1:.1f}" '
            f'fill="#4C72B0" fill-opacity="0.85"/>\n'
            f'<rect x="{x0 + bw + 4:.1f}" y="{base_y - h2:.1f}" width="{bw:.1f}" height="{h2:.1f}" '
            f'fill="#DD8452" fill-opacity="0.85"/>\n'
        )
        parts.append(
            f'<text x="{x0 + bw:.1f}" y="{base_y + 18:.1f}" text-anchor="middle" font-size="9" '
            f'font-family="Malgun Gothic, sans-serif">{escape(MODEL_LABELS[mkey])}</text>\n'
        )
    parts.append(
        f'<line x1="{ml:.1f}" y1="{base_y:.1f}" x2="{ml + pw:.1f}" y2="{base_y:.1f}" stroke="#333"/>\n'
    )
    parts.append(
        f'<text x="{ml + 20:.1f}" y="{base_y + 50:.1f}" font-size="10" font-family="Malgun Gothic, sans-serif">'
        "■ 학습 Speedup  ■ 추론 Speedup</text>\n"
    )
    parts.extend(_env_block(W, H - 48, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig4_speedup_ratio.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def fig5_svg(data, out_dir: str, env_caption: str) -> None:
    try:
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold, cross_val_predict

        import xgboost as xgb
    except ImportError as e:
        print(f"  fig5 SVG 건너뜀 (의존성): {e}")
        return

    from scripts.train_predictor import prepare_features

    X, y_train, y_infer, y_memory, devices, names = prepare_features(data)
    targets = [
        (y_train, "학습 시간 (s)"),
        (y_infer, "추론 시간 (s)"),
        (y_memory, "메모리 (bytes)"),
    ]
    unique_devices = sorted(set(devices))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    W, H = 1180, 820
    cols = 3
    rows = len(unique_devices)
    cell_w = 360
    cell_h = 240
    ox, oy = 50, 50

    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="30" text-anchor="middle" font-size="14" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림5: XGBoost 실측 vs 예측 (5-Fold CV)</text>\n"
    )

    for row_idx, dev in enumerate(unique_devices):
        mask = np.array([d == dev for d in devices])
        types_dev = [data[i]["model_type"] for i, m in enumerate(mask) if m]
        for col_idx, (y_all, target_label) in enumerate(targets):
            X_dev = X[mask]
            y_dev = y_all[mask]
            y_log = np.log1p(y_dev)
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbosity=0,
            )
            y_pred_log = cross_val_predict(model, X_dev, y_log, cv=kf)
            y_true = np.expm1(y_log)
            y_pred = np.expm1(y_pred_log)
            r2 = r2_score(y_true, y_pred)

            ml = ox + col_idx * cell_w
            mt = oy + row_idx * cell_h
            pw, ph = cell_w - 40, cell_h - 55
            pts = []
            for ti, mt_ in enumerate(COLORS.keys()):
                idx = [i for i, t in enumerate(types_dev) if t == mt_]
                if not idx:
                    continue
                for i in idx:
                    yt = float(y_true[i])
                    yp = float(y_pred[i])
                    if yt <= 0 or yp <= 0:
                        continue
                    pts.append((yt, yp, COLORS[mt_], dev))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if not xs:
                continue
            tx, ty, vmin, vmax = _log_axes(xs, ys, ml, mt + 20, pw, ph)
            parts.append(
                f'<text x="{ml + pw / 2:.1f}" y="{mt:.1f}" text-anchor="middle" font-size="10" '
                f'font-family="Malgun Gothic, sans-serif">'
                f"{escape(target_label)} [{escape(dev)}] R²={r2:.3f}</text>\n"
            )
            x1, y1 = tx(vmin), ty(vmin)
            x2, y2 = tx(vmax), ty(vmax)
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                'stroke="#c44e52" stroke-width="1" stroke-dasharray="4,3" opacity="0.8"/>\n'
            )
            for x, y, c, d in pts:
                parts.append(_marker(tx(x), ty(y), d, c, 3.8))
            parts.append(_panel_frame(ml, mt + 20, pw, ph))

    parts.extend(_env_block(W, H - 55, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig5_prediction_accuracy.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def fig6_svg(data, out_dir: str, env_caption: str) -> None:
    try:
        import xgboost as xgb
    except ImportError as e:
        print(f"  fig6 SVG 건너뜀: {e}")
        return

    from scripts.train_predictor import FEATURE_COLUMNS, prepare_features

    X, y_train, y_infer, _, devices, _ = prepare_features(data)
    targets = [(y_train, "학습 시간"), (y_infer, "추론 시간")]
    unique_devices = sorted(set(devices))

    W, H = 1200, 900
    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="28" text-anchor="middle" font-size="14" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림6: XGBoost 피처 중요도 (Top 15)</text>\n"
    )
    cell_w, cell_h = 560, 380
    ox, oy = 40, 50

    vir = ["#440154", "#31688e", "#35b779", "#fde725"]

    for col_idx, (y_all, target_label) in enumerate(targets):
        for row_idx, dev in enumerate(unique_devices):
            mask = np.array([d == dev for d in devices])
            X_dev = X[mask]
            y_dev = y_all[mask]
            y_log = np.log1p(y_dev)
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_dev, y_log)
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            names = [FEATURE_COLUMNS[i] for i in idx]
            vals = [float(imp[i]) for i in idx]
            ml = ox + col_idx * (cell_w + 30)
            mt = oy + row_idx * (cell_h + 40)
            pw = cell_w - 180
            ph = cell_h - 60
            vmax = max(vals) if vals else 1.0
            n = len(names)
            bh = (ph - 4 * (n - 1)) / n if n else 1
            parts.append(
                f'<text x="{ml + pw / 2 + 60:.1f}" y="{mt - 6:.1f}" text-anchor="middle" font-size="11" '
                f'font-family="Malgun Gothic, sans-serif">'
                f"{escape(target_label)} [{escape(dev)}]</text>\n"
            )
            for i, (name, v) in enumerate(zip(reversed(names), reversed(vals))):
                y = mt + i * (bh + 4)
                bw = (v / vmax) * pw if vmax > 0 else 0
                ci = i % len(vir)
                parts.append(
                    f'<text x="{ml + 55:.1f}" y="{y + bh / 2 + 4:.1f}" text-anchor="end" font-size="8" '
                    f'font-family="Consolas, monospace">{escape(name[:28])}</text>\n'
                    f'<rect x="{ml + 60:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                    f'fill="{vir[ci]}" fill-opacity="0.9"/>\n'
                )

    parts.extend(_env_block(W, H - 50, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig6_feature_importance.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def _ylorrd(t: float) -> str:
    t = max(0.0, min(1.0, t))
    stops = ["#ffffcc", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]
    x = t * (len(stops) - 1)
    i = int(x)
    return stops[min(i, len(stops) - 1)]


def fig7_svg(data, out_dir: str, env_caption: str) -> None:
    model_types = [
        "simple_ann",
        "simple_cnn",
        "resnet_mnist",
        "mobilenet_mnist",
        "transformer",
        "gan",
    ]
    metrics = ["total_params", "flops", "total_layers", "avg_train", "avg_infer"]
    metric_labels = ["파라미터", "FLOPs", "레이어", "학습(s)", "추론(s)"]
    matrix = []
    for mt in model_types:
        rows = [r for r in data if r["model_type"] == mt and r["device"] == "CPU"]
        if not rows:
            rows = [r for r in data if r["model_type"] == mt]
        if rows:
            matrix.append([float(np.mean([r.get(m, 0) for r in rows])) for m in metrics])
        else:
            matrix.append([0.0] * len(metrics))
    M = np.array(matrix)
    for j in range(M.shape[1]):
        cmax = M[:, j].max()
        if cmax > 0:
            M[:, j] = M[:, j] / cmax

    W, H = 920, 520
    ml, mtop = 160, 70
    cw = 120
    ch = 50
    parts = _svg_open(W, H)
    parts.append(
        f'<text x="{W / 2:.1f}" y="36" text-anchor="middle" font-size="14" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "그림7: 모델 복잡도 (CPU 평균, 열 정규화)</text>\n"
    )
    for j, lab in enumerate(metric_labels):
        parts.append(
            f'<text x="{ml + j * cw + cw / 2:.1f}" y="{mtop - 8:.1f}" text-anchor="middle" '
            f'font-size="10" font-family="Malgun Gothic, sans-serif">{escape(lab)}</text>\n'
        )
    for i, mkey in enumerate(model_types):
        parts.append(
            f'<text x="{ml - 10:.1f}" y="{mtop + i * ch + ch / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" font-family="Malgun Gothic, sans-serif">'
            f"{escape(MODEL_LABELS[mkey])}</text>\n"
        )
        for j in range(len(metrics)):
            v = float(M[i, j])
            col = _ylorrd(v)
            x0 = ml + j * cw
            y0 = mtop + i * ch
            tc = "#fff" if v > 0.5 else "#222"
            parts.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cw - 4:.1f}" height="{ch - 4:.1f}" '
                f'fill="{col}" stroke="#fff" stroke-width="1"/>\n'
                f'<text x="{x0 + cw / 2 - 2:.1f}" y="{y0 + ch / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{tc}">{v:.2f}</text>\n'
            )
    parts.extend(_env_block(W, H - 48, env_caption))
    parts.append(_svg_close())
    path = os.path.join(out_dir, "fig7_complexity_heatmap.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"  저장: {path}")


def run(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.input)

    env_info = collect_experiment_environment_ko(args.input)
    env_caption = format_experiment_environment_caption_ko(env_info)
    env_txt_path = os.path.join(args.output_dir, "EXPERIMENT_ENV.txt")
    try:
        with open(env_txt_path, "w", encoding="utf-8") as ef:
            ef.write(format_experiment_environment_report_ko(env_info))
        print(f"실험 환경 요약 저장: {env_txt_path}")
    except OSError:
        pass

    mac_data = None
    if not args.skip_mac and os.path.exists(args.input_mac):
        mac_data = load_data(args.input_mac)
        print(f"Mac 데이터 로드: {len(mac_data)}개")
    elif args.skip_mac:
        print("Mac 데이터 사용 안 함 (--skip-mac).")

    all_data = data + (mac_data if mac_data else [])

    print(f"\nSVG 시각화 (matplotlib 불필요) → {args.output_dir}/")
    fig1_svg(all_data, args.output_dir, env_caption)
    fig2_svg(all_data, args.output_dir, env_caption)
    fig3_svg(all_data, args.output_dir, env_caption)
    fig4_svg(all_data, args.output_dir, env_caption)
    fig5_svg(data, args.output_dir, env_caption)
    fig6_svg(data, args.output_dir, env_caption)
    fig7_svg(all_data, args.output_dir, env_caption)

    if mac_data:
        print(
            "\n[안내] fig8~fig9 크로스 플랫폼 SVG는 미구현입니다. "
            "matplotlib 정책 해제 후 visualize_results.py 를 사용하세요."
        )
    print(f"\n완료! SVG fig1~fig7 + EXPERIMENT_ENV.txt: {args.output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="벤치마크 SVG 시각화 (matplotlib 없음)")
    parser.add_argument("--input", type=str, default="results/benchmark_results.json")
    parser.add_argument("--input-mac", type=str, default="results/benchmark_results_mac.json")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--skip-mac", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
