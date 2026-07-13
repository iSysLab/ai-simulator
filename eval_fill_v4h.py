"""구판 ONNX 파일 14개에 대한 외부 파일 예측 검증 — §4.5 보강.

main 브랜치에서 받아온 results/onnx_samples/*.onnx(구판 파이프라인이 opset 11로
export한 파일)를 새 추출기(onnx_oplevel)로 읽어 피처를 만들고, 해당 14개 구성을
학습에서 통째로 제외한 통합 XGBoost로 네 백엔드의 실측 시간을 예측한다.
비교 기준: 구판 경로의 오차는 results/report_onnx_vs_benchmark.csv에 기록돼 있다.

산출: paper/v4_measured_h.json
사용: python eval_fill_v4h.py
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from benchmark.features.onnx_oplevel import (
    get_op_level_features_onnx, get_structural_features_onnx,
)
from eval_fill_v4e import config_key, onnx_feats_for, to_onnx_row, ONNX_DIR
from eval_fill_v4 import load698
from eval_fill_v4f import backend_of
from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE, "results/onnx_samples")


def load_mapping():
    """구판 리포트에서 onnx_file → benchmark_model_name 매핑."""
    m = {}
    with open(os.path.join(BASE, "results/report_onnx_vs_benchmark.csv"),
              encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m[row["onnx_file"]] = row["benchmark_model_name"]
    return m


def sample_feats(onnx_file):
    path = os.path.join(SAMPLE_DIR, onnx_file + ".onnx")
    f = dict(get_structural_features_onnx(path))
    f.update(get_op_level_features_onnx(path))
    f["flops"] = f["total_op_flops"]
    return f


def xy(subset, target):
    X = np.array([[float(e.get(c, 0) or 0) for c in NUMERIC] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def main():
    mapping = load_mapping()
    held_names = set(mapping.values())
    rows = load698()

    # 학습셋: 신판 ONNX 피처(160구성), 14개 매칭 구성은 통째로 제외
    manifest = json.load(open(os.path.join(ONNX_DIR, "manifest.json")))
    cache = {}
    train_rows = []
    test_raw = []
    for r in rows:
        if r["model_name"] in held_names:
            test_raw.append(r)
            continue
        of = onnx_feats_for(manifest, config_key(r["model_type"], r.get("config", {}) or {}), cache)
        train_rows.append(enrich(to_onnx_row(r, of)))

    # 테스트셋: 피처는 '구판 파일'에서 추출 (구성 힌트·하드웨어는 행에서)
    name2feats = {mapping[s]: sample_feats(s) for s in mapping}
    test_rows = [enrich(to_onnx_row(r, name2feats[r["model_name"]])) for r in test_raw]
    print(f"학습 {len(train_rows)}행 | 테스트 {len(test_rows)}행 "
          f"(구성 {len(held_names)}개 × 백엔드)", flush=True)

    out = {"n_train": len(train_rows), "n_test": len(test_rows),
           "per_target": {}, "cases": []}
    preds = {}
    for tgt in ["avg_train", "avg_infer"]:
        Xtr, ytr = xy(train_rows, tgt)
        Xte, yte = xy(test_rows, tgt)
        est = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=1, verbosity=0).fit(Xtr, ytr)
        pl = est.predict(Xte)
        y, p = np.expm1(yte), np.expm1(pl)
        pe = np.abs(y - p) / np.clip(np.abs(y), 1e-9, None)
        m10 = y >= 0.01
        out["per_target"][tgt] = {
            "r2log": round(r2_score(yte, pl), 3),
            "mape": round(float(np.mean(pe) * 100), 1),
            "mape_ge10ms": round(float(np.mean(pe[m10]) * 100), 1),
            "w20_ge10ms": round(float(np.mean(pe[m10] < 0.2) * 100), 1),
        }
        preds[tgt] = p
        print(tgt, out["per_target"][tgt], flush=True)

    # 대표 사례 (구판 리포트와 같은 구성끼리 비교 가능하게)
    for i, r in enumerate(test_raw):
        out["cases"].append({
            "model_name": r["model_name"], "backend": backend_of(r),
            "train_meas": round(float(r["avg_train"]), 3),
            "train_pred": round(float(preds["avg_train"][i]), 3),
            "infer_meas": round(float(r["avg_infer"]), 4),
            "infer_pred": round(float(preds["avg_infer"][i]), 4),
        })

    path = os.path.join(BASE, "paper/v4_measured_h.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    main()
