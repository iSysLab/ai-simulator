"""검증 프로토콜·기준선 실험 — 통합(698샘플) 메타모델.

심사 보강용 두 가지 실험을 수행한다.
  1) 검증 프로토콜: random 5-fold / GroupKFold(구성 단위 분할) /
     leave-one-backend-out / leave-one-device-out
  2) 피처 기준선: 전체 130피처 / 하드웨어 제외 / FLOPs 단독 / 파라미터 수 단독
     (random 5-fold와 GroupKFold 모두)

사용: python train_protocols.py
"""
import json
import os
import re

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from xgboost import XGBRegressor

from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))

HW_PAT = re.compile(r"cpu_|gpu_|memory_bandwidth|unified|shared_memory"
                    r"|memory_channels|device_type")
HW_COLS = [c for c in NUMERIC if HW_PAT.search(c)]

FEATURE_SETS = {
    "전체(130)": NUMERIC,
    "하드웨어 제외": [c for c in NUMERIC if c not in HW_COLS],
    "FLOPs 단독": ["flops"],
    "Params 단독": ["total_params"],
}

desk = json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
mac = json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json")))
rows = [enrich(r) for r in desk + mac]
for i, e in enumerate(rows):
    e["_device"] = "desktop" if i < len(desk) else "mac"
    e["_backend"] = (("Desktop CPU" if e["_device"] == "desktop" else "Mac CPU")
                     if e["device"] == "CPU"
                     else {"GPU(CUDA)": "CUDA", "GPU(MPS)": "MPS"}[e["device"]])
groups = np.array([f'{e["model_type"]}/{e["model_name"]}' for e in rows])


def xy(subset, cols, target):
    X = np.array([[float(e.get(c, 0) or 0) for c in cols] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def make_est():
    # make_scatter.py·본문 §4.1과 동일 설정 (수치 일관성)
    return XGBRegressor(n_estimators=400, max_depth=6,
                        random_state=42, n_jobs=1, verbosity=0)


def holdout_r2(train_rows, test_rows, cols, target):
    Xtr, ytr = xy(train_rows, cols, target)
    Xte, yte = xy(test_rows, cols, target)
    est = make_est().fit(Xtr, ytr)
    return r2_score(yte, est.predict(Xte))


print(f"하드웨어 피처 {len(HW_COLS)}개 제외 대상: {HW_COLS}\n")

# ---- 1) 검증 프로토콜 (전체 피처) ----
print("===== 검증 프로토콜별 R²(log) — 통합 모델, 전체 피처, XGBoost =====")
print(f"{'프로토콜':<28} {'학습시간':>8} {'추론시간':>8}")
res = {}
for target in ["avg_train", "avg_infer"]:
    X, y = xy(rows, NUMERIC, target)
    r_rand = r2_score(y, cross_val_predict(
        make_est(), X, y, cv=KFold(5, shuffle=True, random_state=42), n_jobs=1))
    r_group = r2_score(y, cross_val_predict(
        make_est(), X, y, cv=GroupKFold(5), groups=groups, n_jobs=1))
    res[target] = {"random 5-fold": r_rand, "GroupKFold(구성)": r_group}
for name in ["random 5-fold", "GroupKFold(구성)"]:
    print(f"{name:<28} {res['avg_train'][name]:>8.3f} {res['avg_infer'][name]:>8.3f}")

for b in ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]:
    tr = [e for e in rows if e["_backend"] != b]
    te = [e for e in rows if e["_backend"] == b]
    rt = holdout_r2(tr, te, NUMERIC, "avg_train")
    ri = holdout_r2(tr, te, NUMERIC, "avg_infer")
    print(f"{'LOBO: '+b:<28} {rt:>8.3f} {ri:>8.3f}")

for dv, label in [("mac", "LODO: desktop→mac"), ("desktop", "LODO: mac→desktop")]:
    tr = [e for e in rows if e["_device"] != dv]
    te = [e for e in rows if e["_device"] == dv]
    rt = holdout_r2(tr, te, NUMERIC, "avg_train")
    ri = holdout_r2(tr, te, NUMERIC, "avg_infer")
    print(f"{label:<28} {rt:>8.3f} {ri:>8.3f}")

# ---- 2) 피처 기준선 ----
print("\n===== 피처 기준선 R²(log) — 통합 모델, XGBoost =====")
print(f"{'피처 집합':<16} {'CV':<14} {'학습시간':>8} {'추론시간':>8}")
for fname, cols in FEATURE_SETS.items():
    for cvname, cv, g in [("random 5-fold", KFold(5, shuffle=True, random_state=42), None),
                          ("GroupKFold", GroupKFold(5), groups)]:
        vals = []
        for target in ["avg_train", "avg_infer"]:
            X, y = xy(rows, cols, target)
            pred = cross_val_predict(make_est(), X, y, cv=cv, groups=g, n_jobs=1)
            vals.append(r2_score(y, pred))
        print(f"{fname:<16} {cvname:<14} {vals[0]:>8.3f} {vals[1]:>8.3f}")
