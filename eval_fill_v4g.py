"""표5·LOBO·LODO 재계산 — 표5는 표3과 동일한 GridSearchCV 절차로 통일.
산출: paper/v4_measured_g.json"""
import json
import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict)
from xgboost import XGBRegressor

from eval_fill_v4f import (load_onnx_rows, backend_of, xy, BACKENDS,
                           HW_COLS, TARGETS)
from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))

GRID = {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}


def est():
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                        random_state=42, n_jobs=1, verbosity=0)


rows = [enrich(r) for r in load_onnx_rows()]
for e in rows:
    e["_backend"] = backend_of(e)
groups = np.array([f'{e["model_type"]}/{e["model_name"]}' for e in rows])

out = {"t5": {}, "lobo": {}, "lodo": {}}
for label, cols, kind in [("full_random", NUMERIC, "kfold"),
                          ("full_group", NUMERIC, "group"),
                          ("no_hw_group", [c for c in NUMERIC if c not in HW_COLS], "group"),
                          ("flops_group", ["flops"], "group"),
                          ("params_group", ["total_params"], "group")]:
    vals = {}
    for tgt in TARGETS:
        X, y = xy(rows, cols, tgt)
        cv = KFold(5, shuffle=True, random_state=42) if kind == "kfold" else GroupKFold(5)
        g = None if kind == "kfold" else groups
        base = XGBRegressor(random_state=42, n_jobs=1, verbosity=0)
        gs = GridSearchCV(base, GRID, cv=cv, scoring="r2", n_jobs=1)
        gs.fit(X, y, groups=g) if g is not None else gs.fit(X, y)
        pl = cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=g, n_jobs=1)
        vals[tgt] = round(r2_score(y, pl), 3)
    out["t5"][label] = vals
    print(label, vals, flush=True)


def holdout(tr, te, tgt):
    Xtr, ytr = xy(tr, NUMERIC, tgt)
    Xte, yte = xy(te, NUMERIC, tgt)
    return round(r2_score(yte, est().fit(Xtr, ytr).predict(Xte)), 3)


for b in BACKENDS:
    out["lobo"][b] = {t: holdout([e for e in rows if e["_backend"] != b],
                                 [e for e in rows if e["_backend"] == b], t)
                      for t in TARGETS}
    print("LOBO", b, out["lobo"][b], flush=True)
for dv, lb in [("mac", "desktop→mac"), ("desktop", "mac→desktop")]:
    te_is = lambda e: (e["_src"] == "mac") == (dv == "mac")  # noqa: E731
    out["lodo"][lb] = {t: holdout([e for e in rows if not te_is(e)],
                                  [e for e in rows if te_is(e)], t)
                       for t in TARGETS}
    print("LODO", lb, out["lodo"][lb], flush=True)

path = os.path.join(BASE, "paper/v4_measured_g.json")
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
print("저장:", path)
