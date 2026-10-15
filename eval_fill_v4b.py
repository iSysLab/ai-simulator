"""추론 MAPE의 소값 왜곡 확인: 전체 vs ≥10ms 구간 분리 보고 (best=XGBoost 고정)."""
import json, os, numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from train_merged import enrich, NUMERIC
from eval_fill_v4 import load698, build_xy

BASE = os.path.dirname(os.path.abspath(__file__))
rows = load698()
kf = KFold(5, shuffle=True, random_state=42)
res = {}
for tgt, label in [("avg_train", "학습시간"), ("avg_infer", "추론시간")]:
    X, y = build_xy(rows, tgt)
    ylog = np.log1p(y)
    est = XGBRegressor(random_state=42, n_jobs=1, verbosity=0, n_estimators=400, max_depth=6, learning_rate=0.1)
    pred = np.expm1(cross_val_predict(est, X, ylog, cv=kf, n_jobs=1))
    pe = np.abs(y - pred) / np.clip(np.abs(y), 1e-9, None)
    def stat(mask):
        m = pe[mask]
        return dict(n=int(mask.sum()), mape=float(m.mean()*100),
                    w10=float((m<0.1).mean()*100), w20=float((m<0.2).mean()*100))
    allm = stat(np.ones_like(y, bool))
    big = stat(y >= 0.01)  # ≥10ms
    res[tgt] = {"all": allm, "ge10ms": big,
                "frac_lt10ms": float((y < 0.01).mean()*100)}
    print(f"[{label}] 전체 MAPE={allm['mape']:.1f}% ±20%={allm['w20']:.1f}% | "
          f"≥10ms MAPE={big['mape']:.1f}% ±20%={big['w20']:.1f}% (n={big['n']}) | "
          f"<10ms 비율={res[tgt]['frac_lt10ms']:.1f}%")
json.dump(res, open(os.path.join(BASE, "paper/v4_measured_b.json"), "w"), indent=2, ensure_ascii=False)
print("saved paper/v4_measured_b.json")
