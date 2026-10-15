"""v4 초안의 [측정] 셀 채우기: 통합 모델 MAPE/±20% + 분석적 baseline.
   ai-simulator/ 루트에서 실행 (benchmark 패키지 import 필요)."""
import json, math, os
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from train_merged import enrich, NUMERIC  # 동일 피처·전처리 재사용

BASE = os.path.dirname(os.path.abspath(__file__))

def load698():
    a = json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
    b = json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json")))
    for r in a: r["_src"] = "win"   # CPU = x86 Ryzen
    for r in b: r["_src"] = "mac"   # CPU = ARM M4
    return a + b

def build_xy(rows, target):
    X, y = [], []
    for r in rows:
        e = enrich(r)
        X.append([float(e.get(c, 0) or 0) for c in NUMERIC])
        y.append(float(r.get(target, 0) or 0))
    return np.array(X), np.array(y)

def models():
    kf = KFold(5, shuffle=True, random_state=42)
    return {
        "LinearRegression": (LinearRegression(), None),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=1),
                         {"n_estimators": [100, 200], "max_depth": [None, 12]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), None),
        "XGBoost": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
                    {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}),
    }, kf

def metrics(y, pred, ylog, pred_log):
    pe = np.abs(y - pred) / np.clip(np.abs(y), 1e-9, None)
    return {"r2log": r2_score(ylog, pred_log),
            "mape": float(np.mean(pe) * 100),
            "within10": float(np.mean(pe < 0.10) * 100),
            "within20": float(np.mean(pe < 0.20) * 100)}

def unified(rows, target):
    X, y = build_xy(rows, target)
    ylog = np.log1p(y)
    mm, kf = models()
    best = None
    for name, (est, grid) in mm.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1); gs.fit(X, ylog)
            est = gs.best_estimator_
        pred_log = cross_val_predict(est, X, ylog, cv=kf, n_jobs=1)
        m = metrics(y, np.expm1(pred_log), ylog, pred_log)
        m["model"] = name
        if best is None or m["r2log"] > best["r2log"]:
            best = m
    return best

# ---- 분석적 baseline (Paleo식) : 데이터시트 peak 스펙 사용 ----
# (FP32 TFLOPS, 메모리 대역폭 GB/s) — 데이터시트 값(측정 아님)
SPECS = {
    ("win", "CPU"):        (1.28,  83.0),   # Ryzen 7 7800X3D: 8c×5.0GHz×32flop/cyc; DDR5-5200 듀얼
    ("win", "GPU(CUDA)"):  (22.06, 288.0),  # RTX 4060 Ti
    ("mac", "CPU"):        (1.50, 120.0),   # Apple M4 CPU (NEON), 통합메모리 대역폭
    ("mac", "GPU(MPS)"):   (3.50, 120.0),   # Apple M4 GPU 10-core, 통합메모리
}
def backend_key(r):
    return (r["_src"], r["device"])

def analytic(rows, target):
    # 워크로드 총 연산/바이트 대략치: 순전파 flops·op-memory를 대표값으로,
    # 학습은 fwd+bwd≈3배. 절대 스케일은 백엔드별 PPP가 흡수하므로 상대 구성만 반영.
    mult = 3.0 if target == "avg_train" else 1.0
    est_by_bk, y_by_bk = {}, {}
    for r in rows:
        bk = backend_key(r)
        if bk not in SPECS: continue
        tflops, bw = SPECS[bk]
        fl = float(r.get("flops", 0) or 0) * mult
        byt = (float(r.get("total_op_memory_read", 0) or 0) +
               float(r.get("total_op_memory_write", 0) or 0)) * mult
        t_hat = fl / (tflops * 1e12) + byt / (bw * 1e9)   # 초
        y = float(r.get(target, 0) or 0)
        est_by_bk.setdefault(bk, []).append(t_hat)
        y_by_bk.setdefault(bk, []).append(y)
    # 백엔드별 PPP = median(y / t_hat), 로그 공간 평가
    yl, pl = [], []
    for bk in est_by_bk:
        e = np.array(est_by_bk[bk]); yv = np.array(y_by_bk[bk])
        ratio = yv / np.clip(e, 1e-12, None)
        ppp = np.median(ratio[np.isfinite(ratio) & (ratio > 0)])
        pred = e * ppp
        yl.append(np.log1p(yv)); pl.append(np.log1p(np.clip(pred, 0, None)))
    yl = np.concatenate(yl); pl = np.concatenate(pl)
    return r2_score(yl, pl)

if __name__ == "__main__":
    rows = load698()
    print(f"rows={len(rows)}  features={len(NUMERIC)}")
    out = {}
    for tgt, label in [("avg_train", "학습시간"), ("avg_infer", "추론시간")]:
        b = unified(rows, tgt)
        a = analytic(rows, tgt)
        out[tgt] = {**b, "analytic_r2log": a}
        print(f"[{label}] best={b['model']}  R2log={b['r2log']:.4f}  "
              f"MAPE={b['mape']:.1f}%  ±10%={b['within10']:.1f}%  ±20%={b['within20']:.1f}%  "
              f"| analytic R2log={a:.3f}")
    json.dump(out, open(os.path.join(BASE, "paper/v4_measured.json"), "w"),
              indent=2, ensure_ascii=False)
    print("saved paper/v4_measured.json")
