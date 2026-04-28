"""보고/논문용 표 생성: 메타모델 CV 지표, ONNX vs 벤치마크, 피처 중요도 요약

사용 (프로젝트 루트에서):
    python scripts/generate_research_report.py
    python scripts/generate_research_report.py --input results/benchmark_results.json
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.features.onnx_extractor import extract_features_from_onnx
from scripts.predict_from_onnx import features_to_array, load_prediction_models
from scripts.train_predictor import (
    FEATURE_COLUMNS,
    filter_benchmark_results,
    load_data,
    prepare_features,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "results" / "benchmark_results.json"
OUT_MD = ROOT / "docs" / "RESEARCH_REPORT.md"
OUT_DIR = ROOT / "results"


# export_onnx.py와 동일한 샘플 목록 + ViT/GAN 힌트
ONNX_JOBS: list[tuple[str, dict]] = [
    ("results/onnx_samples/ann_h128_l1.onnx", {}),
    ("results/onnx_samples/ann_h256_l2.onnx", {}),
    ("results/onnx_samples/ann_h512_l3.onnx", {}),
    ("results/onnx_samples/cnn_f32_l2.onnx", {}),
    ("results/onnx_samples/cnn_f64_l3.onnx", {}),
    ("results/onnx_samples/cnn_f128_l4.onnx", {}),
    ("results/onnx_samples/resnet_2222_w16.onnx", {}),
    ("results/onnx_samples/resnet_2222_w32.onnx", {}),
    ("results/onnx_samples/resnet_2222_w64.onnx", {}),
    ("results/onnx_samples/vit_d64_l2_h4.onnx", {"embed_dim": 64, "num_heads": 4, "patch_size": 4}),
    ("results/onnx_samples/vit_d128_l4_h4.onnx", {"embed_dim": 128, "num_heads": 4, "patch_size": 4}),
    ("results/onnx_samples/vit_d256_l4_h8.onnx", {"embed_dim": 256, "num_heads": 8, "patch_size": 4}),
    ("results/onnx_samples/gan_z64_G128_256.onnx", {"latent_dim": 64}),
    ("results/onnx_samples/gan_z128_G256_512_1024.onnx", {"latent_dim": 128}),
]


def onnx_stem_to_benchmark_name(stem: str) -> str:
    """ONNX 파일 stem → benchmark_results.json 의 model_name."""
    stem = stem.replace(".onnx", "")
    if stem.startswith("ann_"):
        return "ANN_" + stem[4:]
    if stem.startswith("cnn_"):
        return "CNN_" + stem[4:] + "_bn0"
    if stem.startswith("resnet_"):
        return "ResNet_" + stem[7:]
    if stem.startswith("vit_"):
        return "ViT_" + stem[4:] + "_p4"
    if stem.startswith("gan_"):
        return "GAN_" + stem[4:]
    return stem


def find_benchmark_row(rows: list, model_name: str, device: str) -> dict | None:
    matches = [r for r in rows if r.get("model_name") == model_name and r.get("device") == device]
    return matches[-1] if matches else None


def collect_meta_metrics(results: list, device_label: str, cv: int = 5) -> list[dict]:
    """한 장치 필터 후 train/infer/memory CV 지표 (RF, GB, XGB)."""
    out: list[dict] = []
    with contextlib.redirect_stdout(io.StringIO()):
        filtered = filter_benchmark_results(
            list(results),
            filter_device=device_label,
            dedupe=True,
            dedupe_keep="last",
        )
        X, y_train, y_infer, y_memory, devices, _names = prepare_features(filtered)
        mask = np.array([d == device_label for d in devices])
        Xd = X[mask]
        yt = np.log1p(y_train[mask])
        yi = np.log1p(y_infer[mask])
        ym = np.log1p(y_memory[mask])

        targets = [
            ("학습시간", yt),
            ("추론시간", yi),
        ]
        if ym.max() > 0:
            targets.append(("메모리", ym))

        for target_short, y_log in targets:
            metrics, models = train_and_evaluate(
                Xd, y_log, f"{target_short} [{device_label}]", cv_folds=cv
            )
            for m in metrics:
                out.append(
                    {
                        "device": device_label,
                        "target": target_short,
                        "model": m["model"],
                        "R2": m["R2"],
                        "R2_log": m["R2_log"],
                        "RMSE": m["RMSE"],
                        "MAE": m["MAE"],
                    }
                )
    return out


def predict_onnx_file(
    onnx_rel: str,
    device_py: str,
    model_dir: Path,
    extra: dict,
) -> dict[str, float]:
    path = ROOT / onnx_rel
    if not path.is_file():
        return {}
    feats = extract_features_from_onnx(
        str(path),
        device=device_py,
        embed_dim=extra.get("embed_dim", 0),
        num_heads=extra.get("num_heads", 0),
        patch_size=extra.get("patch_size", 0),
        latent_dim=extra.get("latent_dim", 0),
        batch_size=64,
    )
    models, cols = load_prediction_models(str(model_dir))
    X = features_to_array(feats, cols)
    preds: dict[str, float] = {}
    for name, model in sorted(models.items()):
        y_log = model.predict(X)[0]
        preds[name] = float(np.expm1(y_log))
    return preds


def top_importances_text(rf_model, top_n: int = 8) -> list[tuple[str, float]]:
    im = rf_model.feature_importances_
    idx = np.argsort(im)[::-1][:top_n]
    return [(FEATURE_COLUMNS[i], float(im[i])) for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_JSON))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--skip-meta", action="store_true", help="메타모델 재학습 생략(시간 절약)")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_data(args.input)
    rows_all = list(raw)

    lines: list[str] = []
    lines.append("# 연구 보고용 요약 표 (자동 생성)\n")
    lines.append(
        "본 문서는 `python scripts/generate_research_report.py`로 생성되었습니다. "
        "벤치마크 JSON·학습된 joblib·ONNX 샘플 경로가 있어야 합니다.\n"
    )

    # --- 1) 메타모델 CV 표 ---
    lines.append("## 1. 메타모델 교차검증 성능 (RF / GB / XGB)\n")
    lines.append(
        "동일 조건: `--dedupe`, 장치별 필터, `log1p` 타깃, 5-Fold CV `cross_val_predict` 기준 RMSE·MAE는 **원래 단위**입니다. "
        "메모리 행은 타깃이 `memory_bytes` 스케일입니다.\n"
    )

    meta_rows: list[dict] = []
    if not args.skip_meta:
        for dev in ("GPU(CUDA)", "CPU"):
            print(f"[메타모델 CV 계산 중] {dev} …", flush=True)
            meta_rows.extend(collect_meta_metrics(rows_all, dev, cv=args.cv))
    else:
        lines.append("*(이 실행에서는 `--skip-meta`로 메타모델 표를 생략했습니다.)*\n")

    if meta_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        meta_csv = OUT_DIR / "report_meta_model_cv.csv"
        with meta_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["device", "target", "model", "R2", "R2_log", "RMSE", "MAE"],
            )
            w.writeheader()
            for r in meta_rows:
                w.writerow(r)
        lines.append(f"- Excel용 CSV: `{meta_csv.relative_to(ROOT)}`\n")

        lines.append("| 장치 | 타깃 | 모델 | R² | R²(log) | RMSE | MAE |")
        lines.append("|------|------|------|-----|---------|------|-----|")
        for r in meta_rows:
            lines.append(
                f"| {r['device']} | {r['target']} | {r['model']} | {r['R2']} | "
                f"{r['R2_log']} | {r['RMSE']} | {r['MAE']} |"
            )
        lines.append("")

    # --- 2) ONNX vs 벤치 (training / inference / memory bytes) ---
    lines.append("## 2. ONNX 예측 vs 벤치마크 (대표 샘플)\n")
    lines.append(
        "`*_memory` 예측값은 **바이트(byte) 추정**입니다 (스크립트가 `s`로 표시하는 것은 UI 버그). "
        "벤치의 `memory_bytes`와 비교하세요.\n"
    )

    gpu_dir = ROOT / "results" / "trained_models_gpu"
    cpu_dir = ROOT / "results" / "trained_models_cpu"
    comp_rows: list[dict] = []

    for onnx_rel, extra in ONNX_JOBS:
        stem = Path(onnx_rel).stem
        bname = onnx_stem_to_benchmark_name(stem)

        for device_label, py_dev, mdir in (
            ("GPU(CUDA)", "cuda", gpu_dir),
            ("CPU", "cpu", cpu_dir),
        ):
            if not mdir.is_dir():
                continue
            bench = find_benchmark_row(rows_all, bname, device_label)
            with contextlib.redirect_stdout(io.StringIO()):
                preds = predict_onnx_file(onnx_rel, py_dev, mdir, extra)
            if not preds:
                continue
            row: dict = {
                "onnx_file": stem,
                "benchmark_model_name": bname,
                "device": device_label,
                "bench_avg_train": bench["avg_train"] if bench else None,
                "bench_avg_infer": bench["avg_infer"] if bench else None,
                "bench_memory_bytes": bench.get("memory_bytes") if bench else None,
                "pred_rf_train": preds.get("rf_training"),
                "pred_rf_infer": preds.get("rf_inference"),
                "pred_rf_mem": preds.get("rf_memory"),
                "pred_xgb_train": preds.get("xgb_training"),
                "pred_xgb_infer": preds.get("xgb_inference"),
                "pred_xgb_mem": preds.get("xgb_memory"),
            }
            if bench:
                row["err_rf_train_pct"] = round(
                    100.0 * (preds["rf_training"] - bench["avg_train"]) / max(bench["avg_train"], 1e-9),
                    2,
                )
                row["err_rf_infer_pct"] = round(
                    100.0 * (preds["rf_inference"] - bench["avg_infer"]) / max(bench["avg_infer"], 1e-9),
                    2,
                )
            comp_rows.append(row)

    if not comp_rows:
        lines.append(
            "*(학습된 모델 폴더 `results/trained_models_gpu` / `trained_models_cpu` 또는 "
            "ONNX 샘플이 없어 비교 표를 채우지 못했습니다.)*\n"
        )

    if comp_rows:
        comp_csv = OUT_DIR / "report_onnx_vs_benchmark.csv"
        with comp_csv.open("w", newline="", encoding="utf-8-sig") as f:
            keys = list(comp_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(comp_rows)
        lines.append(f"- Excel용 CSV: `{comp_csv.relative_to(ROOT)}`\n")

        # 검증 하이라이트: ANN_h128_l1, resnet_2222_w32 GPU
        lines.append("### 검증 예시 (동일 model_name·장치)\n")
        highlights = [
            ("ann_h128_l1", "GPU(CUDA)"),
            ("resnet_2222_w32", "GPU(CUDA)"),
        ]
        for stem_h, dev_h in highlights:
            sub = [r for r in comp_rows if r["onnx_file"] == stem_h and r["device"] == dev_h]
            if not sub:
                continue
            r = sub[0]
            lines.append(f"- **{r['benchmark_model_name']}** ({dev_h}): ")
            if r.get("bench_avg_train") is not None:
                lines.append(
                    f"벤치 학습 {r['bench_avg_train']:.4f}s vs RF예측 {r['pred_rf_train']:.4f}s "
                    f"({r.get('err_rf_train_pct', 'N/A')}%), "
                    f"벤치 추론 {r['bench_avg_infer']:.4f}s vs RF예측 {r['pred_rf_infer']:.4f}s "
                    f"({r.get('err_rf_infer_pct', 'N/A')}%). "
                )
            lines.append("\n")

        lines.append("| ONNX | 장치 | 벤치 train(s) | RF train | 벤치 infer(s) | RF infer | 벤치 mem(B) | RF mem(B) |")
        lines.append("|------|------|---------------|----------|---------------|----------|-------------|-----------|")
        for r in comp_rows:
            bt = f"{r['bench_avg_train']:.4f}" if r.get("bench_avg_train") is not None else "-"
            bi = f"{r['bench_avg_infer']:.4f}" if r.get("bench_avg_infer") is not None else "-"
            bm = str(int(r["bench_memory_bytes"])) if r.get("bench_memory_bytes") is not None else "-"
            pt = f"{r['pred_rf_train']:.4f}" if r.get("pred_rf_train") is not None else "-"
            pi = f"{r['pred_rf_infer']:.4f}" if r.get("pred_rf_infer") is not None else "-"
            pm = f"{r['pred_rf_mem']:.0f}" if r.get("pred_rf_mem") is not None else "-"
            lines.append(
                f"| {r['onnx_file']} | {r['device']} | {bt} | {pt} | {bi} | {pi} | {bm} | {pm} |"
            )
        lines.append("")

    # --- 3) 피처 중요도 (GPU, 학습시간, RF) ---
    lines.append("## 3. 피처 중요도 (GPU, 학습 시간, RandomForest)\n")
    with contextlib.redirect_stdout(io.StringIO()):
        filtered = filter_benchmark_results(
            list(rows_all),
            filter_device="GPU(CUDA)",
            dedupe=True,
            dedupe_keep="last",
        )
        X, y_train, _yi, _ym, devices, _ = prepare_features(filtered)
    mask = np.array([d == "GPU(CUDA)" for d in devices])
    Xd = X[mask]
    yt = np.log1p(y_train[mask])
    with contextlib.redirect_stdout(io.StringIO()):
        _metrics, models = train_and_evaluate(
            Xd, yt, "학습시간 [GPU] report", cv_folds=min(5, len(Xd))
        )
    rf = models.get("rf")
    if rf is not None:
        tops = top_importances_text(rf, top_n=8)
        lines.append("상위 피처와 해석 초안:\n")
        for i, (name, val) in enumerate(tops, 1):
            lines.append(f"{i}. `{name}` — 중요도 {val:.4f}")
        lines.append("")
        lines.append(
            "**한 단락 설명(초안):** GPU 벤치마크에서 학습 시간을 설명할 때, "
            "RandomForest 기준으로는 op·메모리 트래픽 관련 지표(`total_op_memory_read/write`), "
            "연산 비율(`flops_ratio_*`), 그리고 `flops`·`model_family_encoded` 등이 상위에 나타난다. "
            "즉 **연산량뿐 아니라 메모리 접근 패턴과 모델 계열**이 실행 시간 예측에 함께 기여하는 형태로 볼 수 있다.\n"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"작성 완료: {OUT_MD}")
    if meta_rows:
        print(f"작성 완료: {OUT_DIR / 'report_meta_model_cv.csv'}")
    if comp_rows:
        print(f"작성 완료: {OUT_DIR / 'report_onnx_vs_benchmark.csv'}")


if __name__ == "__main__":
    main()
