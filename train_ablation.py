"""메모리 트래픽 피처 절제(ablation) 실험 — §4.3 메모리바운드 주장의 인과적 보강.

RandomForest 중요도는 상관된 피처에 몰릴 수 있으므로, 두 가지 추가 증거를 계산한다.
  1) 절제: 메모리 트래픽 피처를 제거하고 5-겹 CV R²(log)의 하락폭을 측정
     - A: total_op_memory_read/write 제거
     - B: A + activation_memory_mb 제거 (쓰기 트래픽의 대리 피처까지 제거)
  2) permutation importance(홀드아웃 25%): total_op_memory_write와 flops를 각각
     무작위 셔플했을 때 R²(log) 하락폭 비교

사용: python train_ablation.py
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))

ABLATE_A = ["total_op_memory_read", "total_op_memory_write"]
ABLATE_B = ABLATE_A + ["activation_memory_mb"]

data = (json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
        + json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json"))))
rows = [enrich(r) for r in data]

# 백엔드 라벨: 기기(파일 출처)로 CPU를 구분
n_desk = 378
for i, e in enumerate(rows):
    dev = e["device"]
    if dev == "CPU":
        e["_backend"] = "Desktop CPU" if i < n_desk else "Mac CPU"
    else:
        e["_backend"] = {"GPU(CUDA)": "CUDA", "GPU(MPS)": "MPS"}[dev]

BACKENDS = ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]


def xy(subset, cols, target="avg_train"):
    X = np.array([[float(e.get(c, 0) or 0) for c in cols] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def cv_r2log(subset, cols):
    X, y = xy(subset, cols)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
    pred = cross_val_predict(est, X, y, cv=kf, n_jobs=1)
    return r2_score(y, pred)


idx_w = NUMERIC.index("total_op_memory_write")
idx_f = NUMERIC.index("flops")
print(f"{'백엔드':<12} {'전체피처':>9} {'절제A':>9} {'절제B':>9} {'perm(write)':>12} {'perm(flops)':>12}")
for b in BACKENDS:
    sub = [e for e in rows if e["_backend"] == b]
    full = cv_r2log(sub, NUMERIC)
    abl_a = cv_r2log(sub, [c for c in NUMERIC if c not in ABLATE_A])
    abl_b = cv_r2log(sub, [c for c in NUMERIC if c not in ABLATE_B])
    # permutation importance — 홀드아웃 25%에서 셔플 (in-sample 과대평가 방지)
    X, y = xy(sub, NUMERIC)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1).fit(Xtr, ytr)
    pi = permutation_importance(est, Xte, yte, n_repeats=20, random_state=42, n_jobs=1)
    print(f"{b:<12} {full:>9.4f} {abl_a:>9.4f} {abl_b:>9.4f} "
          f"{pi.importances_mean[idx_w]:>12.3f} {pi.importances_mean[idx_f]:>12.3f}")
