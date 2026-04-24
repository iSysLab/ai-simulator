"""보고용 그림 생성 (실측 vs 예측 산점도 + 피처 중요도).

`scripts/visualize_results.py` 의 fig5 스타일에 맞춤:
  - GPU / CPU 를 한 축에 섞지 않고 패널로 분리 (축 범위는 패널 안 데이터만 사용)
  - 점 색: 모델 계열 (ANN / CNN / ResNet / Transformer / GAN)
  - 한글 제목·축 라벨

matplotlib DLL 차단 환경 대비: 표준 라이브러리만으로 SVG 생성.

사용:
    python scripts/plot_report_figures.py

생성물: docs/images/*.svg
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import joblib
except ImportError:
    joblib = None

from benchmark.support.hardware_info import (
    collect_experiment_environment_ko,
    format_experiment_environment_caption_ko,
)
from scripts.train_predictor import FEATURE_COLUMNS

IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# visualize_results.py 와 동일 팔레트 (results/figures 스타일 통일)
COLORS = {
    "simple_ann": "#4C72B0",
    "simple_cnn": "#DD8452",
    "resnet_mnist": "#55A868",
    "mobilenet_mnist": "#C44E52",
    "transformer": "#8172B3",
    "gan": "#937860",
}
MODEL_LABELS = {
    "simple_ann": "ANN",
    "simple_cnn": "CNN",
    "resnet_mnist": "ResNet",
    "mobilenet_mnist": "MobileNet",
    "transformer": "ViT",
    "gan": "GAN",
}

def _onnx_stem_to_family(onnx_file: str) -> str:
    s = (onnx_file or "").strip().lower().replace(".onnx", "")
    if s.startswith("ann_"):
        return "simple_ann"
    if s.startswith("resnet_"):
        return "resnet_mnist"
    if s.startswith("vit_"):
        return "transformer"
    if s.startswith("gan_"):
        return "gan"
    if s.startswith("cnn_"):
        # export_onnx: MobileNet 변형은 파일명 cnn_f128_l4 등
        if "f128" in s or "mobilenet" in s:
            return "mobilenet_mnist"
        return "simple_cnn"
    return "simple_cnn"


def _load_comp_csv() -> list[dict]:
    path = ROOT / "results" / "report_onnx_vs_benchmark.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    try:
        if x is None or x == "" or x == "None":
            return None
        return float(x)
    except Exception:
        return None


def _svg_header(w: int, h: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n'
        '<rect width="100%" height="100%" fill="#fafafa"/>\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _log_projector(vmin: float, vmax: float, ml: float, mt: float, pw: float, ph: float):
    log_lo = math.log10(vmin)
    log_hi = math.log10(vmax)
    span = log_hi - log_lo if log_hi > log_lo else 1e-9

    def tx(v: float) -> float:
        return ml + (math.log10(v) - log_lo) / span * pw

    def ty(v: float) -> float:
        return mt + ph - (math.log10(v) - log_lo) / span * ph

    return tx, ty, log_lo, log_hi, span


def _fmt_tick(v: float) -> str:
    if v <= 0:
        return "0"
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    if v >= 0.01:
        return f"{v:.2f}"
    return f"{v:.2g}"


def _panel_scatter(
    parts: list[str],
    rows_dev: list[dict],
    bench_key: str,
    pred_key: str,
    ml: float,
    mt: float,
    pw: float,
    ph: float,
    panel_title: str,
) -> None:
    pts: list[tuple[float, float, str]] = []
    for r in rows_dev:
        x = _to_float(r.get(bench_key))
        y = _to_float(r.get(pred_key))
        if x is None or y is None or x <= 0 or y <= 0:
            continue
        fam = _onnx_stem_to_family(r.get("onnx_file", ""))
        pts.append((x, y, fam))

    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{mt - 8:.1f}" text-anchor="middle" '
        f'font-size="13" font-weight="600" font-family="Malgun Gothic, sans-serif">'
        f"{escape(panel_title)}</text>\n"
    )

    if not pts:
        parts.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt + ph / 2:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#666" font-family="Malgun Gothic, sans-serif">'
            "데이터 없음</text>\n"
        )
        parts.append(
            f'<rect x="{ml:.1f}" y="{mt:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
            'fill="none" stroke="#ccc" stroke-width="1"/>\n'
        )
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    vmin = min(min(xs), min(ys)) * 0.82
    vmax = max(max(xs), max(ys)) * 1.18
    vmin = max(vmin, 1e-6)
    if vmax <= vmin:
        vmax = vmin * 10

    tx, ty, log_lo, log_hi, _span = _log_projector(vmin, vmax, ml, mt, pw, ph)

    # 그리드 (10의 거듭제곱)
    k0 = math.floor(log_lo)
    k1 = math.ceil(log_hi)
    for k in range(k0, k1 + 1):
        xv = 10.0**k
        if vmin <= xv <= vmax:
            xpx = tx(xv)
            parts.append(
                f'<line x1="{xpx:.2f}" y1="{mt}" x2="{xpx:.2f}" y2="{mt + ph}" '
                'stroke="#e8e8e8" stroke-width="1"/>\n'
            )
            ypx = ty(xv)
            parts.append(
                f'<line x1="{ml}" y1="{ypx:.2f}" x2="{ml + pw}" y2="{ypx:.2f}" '
                'stroke="#e8e8e8" stroke-width="1"/>\n'
            )

    x1, y1 = tx(vmin), ty(vmin)
    x2, y2 = tx(vmax), ty(vmax)
    parts.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        'stroke="#c44e52" stroke-width="1.5" stroke-dasharray="5,4" opacity="0.75"/>\n'
    )

    rdot = 5.0
    for x, y, fam in pts:
        col = COLORS.get(fam, "#333333")
        cx, cy = tx(x), ty(y)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rdot}" fill="{col}" '
            'fill-opacity="0.85" stroke="#fff" stroke-width="0.6"/>\n'
        )

    parts.append(
        f'<rect x="{ml:.1f}" y="{mt:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
        'fill="none" stroke="#333" stroke-width="1"/>\n'
    )

    # x 눈금 라벨 (하단)
    for k in range(k0, k1 + 1):
        xv = 10.0**k
        if vmin <= xv <= vmax:
            xpx = tx(xv)
            parts.append(
                f'<text x="{xpx:.1f}" y="{mt + ph + 16:.1f}" text-anchor="middle" '
                f'font-size="9" fill="#444" font-family="Malgun Gothic, sans-serif">'
                f"{escape(_fmt_tick(xv))}</text>\n"
            )


def _scatter_dual_svg(
    rows: list[dict],
    target: str,
    main_title: str,
    env_lines: list[str] | None = None,
) -> str | None:
    bench_key = "bench_avg_train" if target == "train" else "bench_avg_infer"
    pred_key = "pred_rf_train" if target == "train" else "pred_rf_infer"

    if not rows:
        return None

    env_lines = env_lines or []
    W = 1100
    H = 620 + (34 if env_lines else 0)
    mt_main = 36
    # 좌 CPU, 우 GPU — fig5 행 순서와 동일
    pw, ph = 460, 420
    ml_cpu = 72
    ml_gpu = 72 + pw + 56
    mt_plot = 58

    parts: list[str] = [_svg_header(W, H)]
    parts.append(
        f'<text x="{W / 2:.1f}" y="{mt_main}" text-anchor="middle" font-size="15" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">{escape(main_title)}</text>\n'
    )

    for dev, ml in (("CPU", ml_cpu), ("GPU(CUDA)", ml_gpu)):
        sub = [r for r in rows if r.get("device") == dev]
        label = "CPU" if dev == "CPU" else "GPU (CUDA)"
        _panel_scatter(parts, sub, bench_key, pred_key, ml, mt_plot, pw, ph, label)

    # 공통 y축 라벨
    parts.append(
        f'<text transform="rotate(-90 22 {(mt_plot + ph / 2):.1f})" x="22" y="{mt_plot + ph / 2:.1f}" '
        f'text-anchor="middle" font-size="12" font-family="Malgun Gothic, sans-serif">'
        f'{"RF 예측 학습 시간 (s)" if target == "train" else "RF 예측 추론 시간 (s)"}</text>\n'
    )

    # 공통 x축 라벨 (패널 아래)
    y_xlab = mt_plot + ph + 36
    cx_mid = ml_cpu + pw / 2
    cx_mid2 = ml_gpu + pw / 2
    xlab = "실측 학습 시간 (s)" if target == "train" else "실측 추론 시간 (s)"
    parts.append(
        f'<text x="{cx_mid:.1f}" y="{y_xlab}" text-anchor="middle" font-size="11" '
        f'font-family="Malgun Gothic, sans-serif">{escape(xlab)}</text>\n'
        f'<text x="{cx_mid2:.1f}" y="{y_xlab}" text-anchor="middle" font-size="11" '
        f'font-family="Malgun Gothic, sans-serif">{escape(xlab)}</text>\n'
    )

    # 범례 (모델 계열) — fig1~5 와 동일 의미
    leg_y = y_xlab + 28
    leg_x = W / 2 - 220
    order = [
        "simple_ann",
        "simple_cnn",
        "resnet_mnist",
        "mobilenet_mnist",
        "transformer",
        "gan",
    ]
    parts.append(
        f'<text x="{W / 2:.1f}" y="{leg_y - 6:.1f}" text-anchor="middle" font-size="10" '
        f'fill="#555" font-family="Malgun Gothic, sans-serif">'
        "점 색 = ONNX 샘플 계열 (벤치마크 시각화와 동일 팔레트)</text>\n"
    )
    lx = leg_x
    for i, fam in enumerate(order):
        col = COLORS[fam]
        lab = MODEL_LABELS[fam]
        xi = lx + (i % 6) * 150
        yi = leg_y + (i // 6) * 18
        parts.append(
            f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="4.5" fill="{col}"/>\n'
            f'<text x="{xi + 10:.1f}" y="{yi + 4:.1f}" font-size="10" '
            f'font-family="Malgun Gothic, sans-serif">{escape(lab)}</text>\n'
        )

    if env_lines:
        y_env = leg_y + 22
        parts.append(
            f'<text x="{W / 2:.1f}" y="{y_env - 2:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#666" font-family="Malgun Gothic, sans-serif">'
            "[실험 환경]</text>\n"
        )
        for i, ln in enumerate(env_lines):
            parts.append(
                f'<text x="{W / 2:.1f}" y="{y_env + 12 + i * 14:.1f}" text-anchor="middle" '
                f'font-size="9" fill="#333" font-family="Malgun Gothic, sans-serif">'
                f"{escape(ln)}</text>\n"
            )

    parts.append(_svg_footer())
    return "".join(parts)


def _viridis_like(i: int, n: int) -> str:
    """간단 그라데이션 (matplotlib viridis 느낌)."""
    if n <= 1:
        return "#440154"
    t = i / max(n - 1, 1)
    # 보라 → 청록 → 노랑
    if t < 0.25:
        return "#440154"
    if t < 0.5:
        return "#31688e"
    if t < 0.75:
        return "#35b779"
    return "#fde725"


def _feature_importance_svg(env_lines: list[str] | None = None) -> str | None:
    if joblib is None:
        return None
    model_path = ROOT / "results" / "trained_models_gpu" / "rf_training.pkl"
    if not model_path.exists():
        return None
    rf = joblib.load(model_path)
    if not hasattr(rf, "feature_importances_"):
        return None
    im = list(rf.feature_importances_)
    indexed = sorted(enumerate(im), key=lambda x: x[1], reverse=True)[:10]
    if not indexed:
        return None

    top_i, top_v = indexed[0]
    top_name = FEATURE_COLUMNS[top_i]
    rest = indexed[1:]  # 2~10위

    env_lines = env_lines or []
    W = 760
    mt = 72
    ph = 300  # 막대 영역 고정 (H 늘리면 막대만 길어지지 않게)
    extra = (len(env_lines) * 13 + 22) if env_lines else 0
    mb = 48 + extra
    H = mt + ph + mb
    ml, mr = 240, 100
    pw = W - ml - mr

    parts: list[str] = [_svg_header(W, H)]
    parts.append(
        f'<text x="{W / 2:.1f}" y="32" text-anchor="middle" font-size="15" '
        f'font-weight="600" font-family="Malgun Gothic, sans-serif">'
        "RF 피처 중요도 (GPU · 학습 시간)</text>\n"
    )
    parts.append(
        f'<text x="{W / 2:.1f}" y="52" text-anchor="middle" font-size="11" fill="#333" '
        f'font-family="Malgun Gothic, sans-serif">'
        f"1위: {escape(top_name)} — {top_v:.4f}  ·  아래: 2~10위 (막대만 확대)</text>\n"
    )

    if not rest:
        parts.append(_svg_footer())
        return "".join(parts)

    names = [FEATURE_COLUMNS[i] for i in reversed([i for i, _ in rest])]
    vals = [float(im[i]) for i in reversed([i for i, _ in rest])]
    vmax = max(vals) if vals else 1.0

    n = len(names)
    gap = 5
    bh = (ph - gap * (n - 1)) / n if n else 0

    for i, (name, v) in enumerate(zip(names, vals)):
        y = mt + i * (bh + gap)
        bw = (v / vmax) * pw if vmax > 0 else 0
        color = _viridis_like(i, n)
        parts.append(
            f'<text x="{ml - 8:.1f}" y="{y + bh / 2 + 4:.1f}" text-anchor="end" font-size="10" '
            f'font-family="Consolas, monospace">{escape(name)}</text>\n'
            f'<rect x="{ml:.1f}" y="{y:.1f}" width="{bw:.2f}" height="{bh:.1f}" '
            f'fill="{color}" fill-opacity="0.9"/>\n'
            f'<text x="{ml + bw + 6:.1f}" y="{y + bh / 2 + 4:.1f}" font-size="9" '
            f'font-family="Malgun Gothic, sans-serif">{v:.4f}</text>\n'
        )

    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{mt + ph + 22:.1f}" text-anchor="middle" font-size="11" '
        f'font-family="Malgun Gothic, sans-serif">중요도 (상대 비교)</text>\n'
    )
    if env_lines:
        y0 = mt + ph + 40
        parts.append(
            f'<text x="{W / 2:.1f}" y="{y0:.1f}" text-anchor="middle" font-size="9" fill="#666" '
            f'font-family="Malgun Gothic, sans-serif">[실험 환경]</text>\n'
        )
        for i, ln in enumerate(env_lines):
            parts.append(
                f'<text x="{W / 2:.1f}" y="{y0 + 12 + i * 13:.1f}" text-anchor="middle" '
                f'font-size="9" fill="#333" font-family="Malgun Gothic, sans-serif">'
                f"{escape(ln)}</text>\n"
            )
    parts.append(_svg_footer())
    return "".join(parts)


def main():
    rows = _load_comp_csv()
    outputs: list[Path] = []

    bench_json = ROOT / "results" / "benchmark_results.json"
    cap = format_experiment_environment_caption_ko(collect_experiment_environment_ko(bench_json))
    env_lines = [ln.strip() for ln in cap.split("\n") if ln.strip()]

    if rows:
        s_train = _scatter_dual_svg(
            rows,
            "train",
            "ONNX 검증: 실측 vs RF 예측 — 학습 시간 (장치별)",
            env_lines,
        )
        if s_train:
            p = IMG_DIR / "scatter_train.svg"
            p.write_text(s_train, encoding="utf-8")
            outputs.append(p)

        s_infer = _scatter_dual_svg(
            rows,
            "infer",
            "ONNX 검증: 실측 vs RF 예측 — 추론 시간 (장치별)",
            env_lines,
        )
        if s_infer:
            p = IMG_DIR / "scatter_infer.svg"
            p.write_text(s_infer, encoding="utf-8")
            outputs.append(p)
    else:
        print(
            "경고: results/report_onnx_vs_benchmark.csv 가 없어 산점도는 건너뜁니다. "
            "먼저 `python scripts/generate_research_report.py`를 실행하세요."
        )

    fi = _feature_importance_svg(env_lines)
    if fi:
        p = IMG_DIR / "feature_importance_gpu_train.svg"
        p.write_text(fi, encoding="utf-8")
        outputs.append(p)
    else:
        print(
            "경고: results/trained_models_gpu/rf_training.pkl 이 없어 중요도 그림은 건너뜁니다."
        )

    if outputs:
        print("저장 완료 (SVG, matplotlib 불필요):")
        for p in outputs:
            print(f"  {p.relative_to(ROOT)}")
        html = ROOT / "docs" / "preview_figures.html"
        if html.exists():
            print(f"  (브라우저 미리보기) {html.relative_to(ROOT)}")
    print(
        "참고: 산점도는 visualize_results.py fig5 처럼 장치별 패널·모델 계열 색을 사용합니다."
    )
    print(
        "참고: IDE 마크다운 미리보기에서 SVG가 안 보이면 docs/preview_figures.html 을 브라우저로 여세요."
    )


if __name__ == "__main__":
    main()
