"""병합 스키마(131차원) 전용 학습 스크립트 — torch 비의존, 단일 스레드.

train_predictor.py는 extractor.py를 통해 torch를 import(이 환경에서 매우 느림)하고
GridSearchCV n_jobs=-1(loky) 병렬이 hang되는 문제가 있다. 본 스크립트는:
  - merged_schema만 import(순수 리스트, torch 없음)
  - 모든 추정기 n_jobs=1
  - 핵심 파생 피처(log 변환·플래그)는 인라인 계산
로 이종 플랫폼 데이터의 디바이스별 메타모델을 안정적으로 학습·평가한다.

사용:
  python train_merged.py --input results/benchmark_results_enriched.json
  python train_merged.py --input results/benchmark_results_mac_enriched.json --out results/report_merged_mac.csv
"""
import argparse
import csv
import json
import math

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from benchmark.features.merged_schema import (
    MERGED_FEATURE_COLUMNS, STRING_FEATURES,
)

NUMERIC = [c for c in MERGED_FEATURE_COLUMNS if c not in STRING_FEATURES]
TARGETS = [("avg_train", "학습시간"), ("avg_infer", "추론시간"),
           ("memory_bytes", "메모리")]


def enrich(r):
    """핵심 파생 피처 인라인 보강 (train_predictor.enrich_result의 축약)."""
    e = dict(r)
    tp = e.get("total_params", 0) or 0
    ms = e.get("model_size_mb", 0) or 0
    mw = e.get("max_width", 0) or e.get("max_channel_width", 0) or 0
    fl = e.get("flops", 0) or 0
    e.setdefault("log_total_params", math.log1p(tp))
    e.setdefault("log_model_size_mb", math.log1p(ms))
    e.setdefault("log_max_width", math.log1p(mw))
    e.setdefault("num_hidden_layers",
                 (e.get("num_conv_layers", 0) or 0) + (e.get("num_linear_layers", 0) or 0))
    e.setdefault("max_width", mw)
    e.setdefault("has_pooling", 1 if (e.get("num_pool_layers", 0) or 0) > 0 else 0)
    e.setdefault("has_batch_norm", 1 if (e.get("num_bn_layers", 0) or 0) > 0 else 0)
    e.setdefault("num_mult_adds", fl // 2)
    e.setdefault("flops_per_sample", fl)
    e.setdefault("params_per_flop", tp / fl if fl > 0 else 0)
    ih, iw, ic = e.get("input_height", 0) or 0, e.get("input_width", 0) or 0, e.get("input_channels", 0) or 0
    e.setdefault("input_pixels", ih * iw * ic)
    e.setdefault("input_elements", ih * iw * ic)
    e.setdefault("model_family_encoded", {
        "simple_ann": 0, "simple_cnn": 1, "resnet_mnist": 2,
        "mobilenet_mnist": 3, "transformer": 4, "gan": 5,
    }.get(e.get("model_type", ""), -1))
    return e


def build_xy(rows, target):
    X, y = [], []
    for r in rows:
        e = enrich(r)
        X.append([float(e.get(c, 0) or 0) for c in NUMERIC])
        y.append(float(r.get(target, 0) or 0))
    return np.array(X), np.array(y)


def make_models():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return {
        "LinearRegression": (LinearRegression(), None),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=1),
                         {"n_estimators": [100, 200], "max_depth": [None, 12]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), None),
        "XGBoost": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
                    {"n_estimators": [200, 400], "max_depth": [3, 6],
                     "learning_rate": [0.1]}),
    }, kf


def evaluate(name, est, grid, X, ylog, kf):
    if grid:
        gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1)
        gs.fit(X, ylog)
        est = gs.best_estimator_
    pred_log = cross_val_predict(est, X, ylog, cv=kf, n_jobs=1)
    pred = np.expm1(pred_log)
    y = np.expm1(ylog)
    return {
        "r2": r2_score(y, pred),
        "r2log": r2_score(ylog, pred_log),
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "est": est,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.input))
    devices = sorted(set(r["device"] for r in data))
    print(f"입력 {args.input} | {len(data)}행 | 디바이스 {devices}")
    print(f"수치 피처 {len(NUMERIC)}개 (병합 스키마 {len(MERGED_FEATURE_COLUMNS)} 중 문자열 {len(STRING_FEATURES)} 제외)\n")

    report_rows = []
    importance_dump = {}

    for dev in devices:
        rows = [r for r in data if r["device"] == dev]
        print(f"===== {dev}  ({len(rows)}행) =====")
        for target, tlabel in TARGETS:
            X, y = build_xy(rows, target)
            ylog = np.log1p(y)
            models, kf = make_models()
            best = None
            for name, (est, grid) in models.items():
                try:
                    res = evaluate(name, est, grid, X, ylog, kf)
                except Exception as e:  # noqa: BLE001
                    print(f"   [{tlabel}] {name} 실패: {e}")
                    continue
                report_rows.append([dev, tlabel, name,
                                    round(res["r2"], 4), round(res["r2log"], 4),
                                    round(res["rmse"], 4), round(res["mae"], 4)])
                if best is None or res["r2log"] > best[1]["r2log"]:
                    best = (name, res)
            if best:
                bname, bres = best
                print(f"   [{tlabel}] 최적={bname}  R²={bres['r2']:.4f}  R²(log)={bres['r2log']:.4f}")
                # 피처 중요도 (RF 기준)
                if target == "avg_train" and hasattr(bres["est"], "feature_importances_"):
                    imp = sorted(zip(NUMERIC, bres["est"].feature_importances_),
                                 key=lambda t: -t[1])[:12]
                    importance_dump[dev] = imp
        print()

    # 피처 중요도 출력 (학습시간 기준)
    print("===== 피처 중요도 (학습시간, 최적 모델) =====")
    for dev, imp in importance_dump.items():
        print(f"[{dev}]")
        for f, v in imp:
            print(f"   {v:.4f}  {f}")
        print()

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["device", "target", "model", "R2", "R2_log", "RMSE", "MAE"])
            w.writerows(report_rows)
        print(f"CSV 저장: {args.out}")


if __name__ == "__main__":
    main()
