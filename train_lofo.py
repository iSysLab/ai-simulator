"""Leave-one-family-out 실험 — 통합(698샘플, 4백엔드) 메타모델.

각 모델 계열을 통째로 제외하고 학습한 뒤 해당 계열을 예측한다.
피처·타깃·전처리는 train_merged.py와 동일 (log1p, 131차원 병합 스키마).
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = ["simple_ann", "simple_cnn", "resnet_mnist",
            "mobilenet_mnist", "transformer", "gan"]
TARGETS = [("avg_train", "학습시간"), ("avg_infer", "추론시간")]

data = (json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
        + json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json"))))
rows = [enrich(r) for r in data]
print(f"총 {len(rows)}행")


def xy(subset, target):
    X = np.array([[float(e.get(c, 0) or 0) for c in NUMERIC] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def make_models():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                                random_state=42, n_jobs=1, verbosity=0),
    }


for target, tlabel in TARGETS:
    print(f"\n===== {tlabel} ({target}) — leave-one-family-out =====")
    pooled = {name: {"y": [], "p": []} for name in make_models()}
    print(f"{'제외 계열':<16} {'n_test':>6}", end="")
    names = list(make_models())
    for name in names:
        print(f" {name+' R2log':>22}", end="")
    print()
    for fam in FAMILIES:
        tr = [e for e in rows if e["model_type"] != fam]
        te = [e for e in rows if e["model_type"] == fam]
        Xtr, ytr = xy(tr, target)
        Xte, yte = xy(te, target)
        print(f"{fam:<16} {len(te):>6}", end="")
        for name, est in make_models().items():
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
            r2l = r2_score(yte, pred)
            pooled[name]["y"].append(yte)
            pooled[name]["p"].append(pred)
            print(f" {r2l:>22.3f}", end="")
        print()
    print(f"{'전체(pooled)':<16} {len(rows):>6}", end="")
    for name in names:
        yy = np.concatenate(pooled[name]["y"])
        pp = np.concatenate(pooled[name]["p"])
        print(f" {r2_score(yy, pp):>22.3f}", end="")
    print()
