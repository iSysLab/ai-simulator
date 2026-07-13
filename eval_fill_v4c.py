"""통합 모델의 4개 회귀 모델별 성능 비교 (논문 §4에 넣을 표)."""
import os, numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from eval_fill_v4 import load698, build_xy

rows = load698()
kf = KFold(5, shuffle=True, random_state=42)
defs = {
    "LinearRegression": (LinearRegression(), None),
    "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=1),
                     {"n_estimators": [100, 200], "max_depth": [None, 12]}),
    "GradientBoosting": (GradientBoostingRegressor(random_state=42), None),
    "XGBoost": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
                {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}),
}
for tgt, label in [("avg_train", "학습시간"), ("avg_infer", "추론시간")]:
    X, y = build_xy(rows, tgt); ylog = np.log1p(y)
    print(f"\n[{label}]")
    for name, (est, grid) in defs.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1); gs.fit(X, ylog)
            est = gs.best_estimator_
        pl = cross_val_predict(est, X, ylog, cv=kf, n_jobs=1)
        r2 = r2_score(y, np.expm1(pl)); r2log = r2_score(ylog, pl)
        print(f"  {name:18s} R2={r2:7.3f}  R2log={r2log:.4f}")
