"""v4 보강판 표 수치: ① 4개 회귀 모델 비교 ② 통합 XGBoost의 백엔드별 R²(log).
   ai-simulator/ 루트에서 실행, 결과는 paper/v4_measured_d.json에 저장."""
import json, os
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score

from eval_fill_v4 import load698, build_xy, models

BASE = os.path.dirname(os.path.abspath(__file__))


def backend_of(r):
    if r["device"] == "CPU":
        return "Desktop CPU" if r["_src"] == "win" else "Mac CPU"
    return {"GPU(CUDA)": "CUDA", "GPU(MPS)": "MPS"}[r["device"]]


rows = load698()
backends = np.array([backend_of(r) for r in rows])
out = {"model_comparison": {}, "backend_breakdown": {}}

for tgt in ["avg_train", "avg_infer"]:
    X, y = build_xy(rows, tgt)
    ylog = np.log1p(y)
    defs, kf = models()
    comp = {}
    for name, (est, grid) in defs.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1)
            gs.fit(X, ylog)
            est = gs.best_estimator_
        pl = cross_val_predict(est, X, ylog, cv=kf, n_jobs=1)
        comp[name] = round(r2_score(ylog, pl), 3)
        if name == "XGBoost":
            bb = {}
            for b in ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]:
                m = backends == b
                bb[b] = {"n": int(m.sum()),
                         "r2log": round(r2_score(ylog[m], pl[m]), 3)}
            out["backend_breakdown"][tgt] = bb
        print(f"[{tgt}] {name:18s} R2log={comp[name]:.3f}", flush=True)
    out["model_comparison"][tgt] = comp

path = os.path.join(BASE, "paper/v4_measured_d.json")
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
print("저장:", path)
