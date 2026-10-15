"""ONNX 기반 피처 재구성 실험 — "피처를 ONNX 그래프에서 뽑아도 되는가".

각 벤치마크 행의 구조·op-level 피처를 PyTorch 훅 값 대신 ONNX 그래프
(shape inference)에서 계산한 값으로 교체하고, 동일 프로토콜(XGBoost,
5-겹 CV)로 재학습해 R²(log)을 비교한다. 하드웨어 피처·타깃(실측 시간)은
행 그대로 유지 — ONNX가 대체하는 것은 '모델 구조 → 피처' 경로다.

산출: paper/v4_measured_e.json
사용: python eval_fill_v4e.py   (export_onnx_configs.py 선행 필요)
"""
import json
import os

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from benchmark.features.onnx_oplevel import (
    get_op_level_features_onnx, get_structural_features_onnx,
)
from eval_fill_v4 import load698, models
from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(BASE, "results/onnx_configs")

# ONNX 유래 값으로 교체하는 키 (구조 + op-level)
REPLACE_DERIVED = [  # enrich()가 재파생하도록 삭제
    "log_total_params", "log_model_size_mb", "log_max_width",
    "num_mult_adds", "flops_per_sample", "params_per_flop",
]


def config_key(mt, cfg):
    return mt + "::" + json.dumps(cfg, sort_keys=True)


def onnx_feats_for(manifest, key, cache):
    if key not in cache:
        entry = manifest.get(key)
        if entry is None:
            cache[key] = None
        else:
            path = os.path.join(ONNX_DIR, entry["file"])
            f = dict(get_structural_features_onnx(path))
            f.update(get_op_level_features_onnx(path))
            f["flops"] = f["total_op_flops"]
            cache[key] = f
    return cache[key]


def to_onnx_row(r, of):
    e = {k: v for k, v in r.items() if k not in REPLACE_DERIVED}
    e.update(of)
    return e


def xgb_cv(rows, label):
    defs, kf = models()
    est0, grid = defs["XGBoost"]
    out = {}
    for tgt in ["avg_train", "avg_infer"]:
        X = np.array([[float(enrich(r).get(c, 0) or 0) for c in NUMERIC] for r in rows])
        y = np.log1p(np.array([float(r.get(tgt, 0) or 0) for r in rows]))
        gs = GridSearchCV(est0, grid, cv=kf, scoring="r2", n_jobs=1)
        gs.fit(X, y)
        pl = cross_val_predict(gs.best_estimator_, X, y, cv=kf, n_jobs=1)
        out[tgt] = round(r2_score(y, pl), 3)
        print(f"[{label}] {tgt} R2log={out[tgt]:.3f}", flush=True)
    return out


def main():
    manifest = json.load(open(os.path.join(ONNX_DIR, "manifest.json")))
    rows = load698()
    cache = {}

    onnx_rows, skipped = [], 0
    agree = {k: [] for k in ["flops", "total_op_memory_read",
                             "total_op_memory_write", "total_params", "num_ops"]}
    seen_cfg = set()
    for r in rows:
        key = config_key(r["model_type"], r.get("config", {}) or {})
        of = onnx_feats_for(manifest, key, cache)
        if of is None:
            skipped += 1
            continue
        onnx_rows.append(to_onnx_row(r, of))
        if key not in seen_cfg:  # 피처 일치도는 unique config당 1번
            seen_cfg.add(key)
            for k in agree:
                pt = float(r.get(k, 0) or 0)
                ox = float(of.get(k, 0) or 0)
                if pt > 0:
                    agree[k].append(abs(ox - pt) / pt)

    print(f"행 {len(rows)} | ONNX 매칭 {len(onnx_rows)} | 제외 {skipped} "
          f"(export 실패 config)", flush=True)

    result = {
        "n_rows": len(onnx_rows), "n_skipped": skipped,
        "feature_agreement_relerr": {
            k: {"median": round(float(np.median(v)), 4),
                "p90": round(float(np.percentile(v, 90)), 4), "n": len(v)}
            for k, v in agree.items() if v
        },
        "onnx_features": xgb_cv(onnx_rows, "ONNX 피처"),
        "pytorch_features_same_rows": xgb_cv(
            [r for r in rows if config_key(r["model_type"], r.get("config", {}) or {}) in manifest],
            "PyTorch 피처(동일 행)"),
    }
    path = os.path.join(BASE, "paper/v4_measured_e.json")
    json.dump(result, open(path, "w"), ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    main()
