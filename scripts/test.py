#!/usr/bin/env python3
"""원클릭 ONNX 예측 + 결과 JSON/CSV 저장 CLI

사용법:
  python scripts/test.py model.onnx
  python scripts/test.py model.onnx --device cuda
  python scripts/test.py model.onnx --embed-dim 128 --num-heads 4 --patch-size 4
  python scripts/test.py model.onnx --latent-dim 128
  python scripts/test.py --all
  python scripts/test.py --benchmark --model simple_ann
  python scripts/test.py --benchmark --repeats 3 --device cpu

결과:
  - results/test_results_<timestamp>.json  : 예측, 피처, 하드웨어 스냅샷
  - results/test_results_<timestamp>.csv   : 동일 내용 (평탄화, 엑셀용)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.features.onnx_extractor import extract_features_from_onnx  # noqa: E402
from scripts.predict_from_onnx import features_to_array, load_prediction_models  # noqa: E402
from benchmark.support.hardware_info import get_hardware_info  # noqa: E402


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_onnx_prediction(
    onnx_path: str,
    device: str,
    model_dir: str,
    *,
    embed_dim: int = 0,
    num_heads: int = 0,
    patch_size: int = 0,
    latent_dim: int = 0,
    batch_size: int = 64,
) -> dict:
    """ONNX → 피처 → joblib 메타모델 예측. 저장용 dict 반환."""
    features = extract_features_from_onnx(
        onnx_path,
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        latent_dim=latent_dim,
        batch_size=batch_size,
    )
    models, feature_columns = load_prediction_models(model_dir)
    X = features_to_array(features, feature_columns)

    predictions: dict[str, float] = {}
    for name, model in sorted(models.items()):
        y_pred_log = model.predict(X)[0]
        predictions[name] = float(np.expm1(y_pred_log))

    aligned_features = {
        str(col): float(features.get(col, 0) or 0) for col in feature_columns
    }

    payload = {
        "meta": {
            "onnx_path": os.path.abspath(onnx_path),
            "onnx_basename": os.path.basename(onnx_path),
            "device": device,
            "model_dir": os.path.abspath(model_dir),
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "patch_size": patch_size,
            "latent_dim": latent_dim,
            "timestamp": _timestamp(),
        },
        "predictions": predictions,
        "features": aligned_features,
        "feature_columns": [str(c) for c in feature_columns],
        "hardware": {k: _jsonable(v) for k, v in get_hardware_info(device).items()},
    }
    return payload


def _jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _print_summary(payload: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  ONNX: {payload['meta']['onnx_basename']}")
    print(f"  디바이스: {payload['meta']['device']}")
    print(f"{'=' * 60}")
    print(f"\n  {'모델(메타)':<32s} | {'예측값':>14s}")
    print(f"  {'-' * 32} | {'-' * 14}")
    for name, val in sorted(payload["predictions"].items()):
        unit = "s/epoch" if "training" in name.lower() or "train" in name.lower() else "s"
        print(f"  {name:<32s} | {val:>12.4f} {unit}")
    print()


def save_results(payload: dict, output_dir: str, prefix: str = "test_results") -> tuple[str, str]:
    """JSON + CSV 저장. (json_path, csv_path) 반환."""
    os.makedirs(output_dir, exist_ok=True)
    ts = payload["meta"].get("timestamp") or _timestamp()
    base = f"{prefix}_{ts}"
    json_path = os.path.join(output_dir, f"{base}.json")
    csv_path = os.path.join(output_dir, f"{base}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    flat: dict[str, object] = {}
    for k, v in payload["meta"].items():
        flat[f"meta_{k}"] = v
    for k, v in payload["predictions"].items():
        flat[f"pred_{k}"] = v
    for k, v in payload["features"].items():
        flat[f"feat_{k}"] = v
    for k, v in payload["hardware"].items():
        flat[f"hw_{k}"] = v

    fieldnames = list(flat.keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(flat)

    print(f"결과 저장 완료: {json_path}")
    print(f"결과 저장 완료: {csv_path}")
    return json_path, csv_path


def _kwargs_from_stem(stem: str) -> dict:
    kwargs: dict = {}
    if "vit_" in stem:
        parts = stem.split("_")
        for p in parts:
            if p.startswith("d") and p[1:].isdigit():
                kwargs["embed_dim"] = int(p[1:])
            elif p.startswith("h") and p[1:].isdigit():
                kwargs["num_heads"] = int(p[1:])
        kwargs.setdefault("patch_size", 4)
    elif "gan_" in stem:
        for p in stem.split("_"):
            if p.startswith("z") and p[1:].isdigit():
                kwargs["latent_dim"] = int(p[1:])
    return kwargs


def demo_all_onnx(model_dir: str, output_dir: str, device: str, batch_size: int) -> None:
    onnx_dir = ROOT / "results" / "onnx_samples"
    if not onnx_dir.is_dir():
        print(f"ONNX 샘플 없음: {onnx_dir}")
        print("먼저 scripts/export_onnx.py를 실행하세요.")
        sys.exit(1)

    files = sorted(f for f in os.listdir(onnx_dir) if f.endswith(".onnx"))
    print(f"\n[--all] {len(files)}개 ONNX 예측\n")
    for fname in files:
        path = str(onnx_dir / fname)
        stem = fname.replace(".onnx", "")
        kw = _kwargs_from_stem(stem)
        payload = run_onnx_prediction(
            path,
            device,
            model_dir,
            batch_size=batch_size,
            **{**dict(embed_dim=0, num_heads=0, patch_size=0, latent_dim=0), **kw},
        )
        _print_summary(payload)
        save_results(payload, output_dir, prefix="test_results")


def run_benchmark_subprocess(
    output_dir: str,
    *,
    model: str | None,
    device: str | None,
    repeats: int,
    profile_ops: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"benchmark_results_{_timestamp()}.json")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_benchmark.py"), "--output", out_json, "--repeats", str(repeats)]
    if model:
        cmd.extend(["--model", model])
    if device:
        cmd.extend(["--device", device])
    if profile_ops:
        cmd.append("--profile-ops")
    print("실행:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"벤치마크 실패 (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"벤치마크 결과 저장: {out_json}")


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX 원클릭 예측 + JSON/CSV 저장")
    p.add_argument("onnx_path", nargs="?", default=None, help="ONNX 파일 경로")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--model-dir", type=str, default="results/trained_models")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=0)
    p.add_argument("--num-heads", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=0)
    p.add_argument("--all", action="store_true", help="results/onnx_samples/ 내 모든 ONNX 예측")
    p.add_argument("--benchmark", action="store_true", help="scripts/run_benchmark.py 실행 후 결과 JSON 저장")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "simple_ann",
            "simple_cnn",
            "resnet_mnist",
            "mobilenet_mnist",
            "transformer",
            "gan",
        ],
        help="--benchmark 시 단일 모델만 실행",
    )
    p.add_argument("--repeats", type=int, default=10, help="--benchmark 반복 횟수")
    p.add_argument("--profile-ops", action="store_true", help="--benchmark 시 op 프로파일")
    args = p.parse_args()

    if args.benchmark:
        run_benchmark_subprocess(
            args.output_dir,
            model=args.model,
            device=args.device,
            repeats=args.repeats,
            profile_ops=args.profile_ops,
        )
        return

    if args.all:
        demo_all_onnx(args.model_dir, args.output_dir, args.device, args.batch_size)
        return

    if not args.onnx_path:
        p.print_help()
        print("\n오류: ONNX 경로를 주거나 --all / --benchmark 를 사용하세요.", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.onnx_path):
        print(f"파일 없음: {args.onnx_path}", file=sys.stderr)
        sys.exit(1)

    payload = run_onnx_prediction(
        args.onnx_path,
        args.device,
        args.model_dir,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )
    _print_summary(payload)
    save_results(payload, args.output_dir, prefix="test_results")


if __name__ == "__main__":
    main()
