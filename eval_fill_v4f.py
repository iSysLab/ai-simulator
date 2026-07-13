"""논문 전체 표를 ONNX 피처 기준으로 재계산 — 표3~8 통일판.

eval_fill_v4e의 ONNX 행 재구성을 공용으로 쓰고, 기존 실험 스크립트
(eval_fill_v4/v4b/v4d, train_protocols, train_ablation, train_lofo)의
절차를 동일하게 재현한다. 산출: paper/v4_measured_f.json

사용: python eval_fill_v4f.py   (export_onnx_configs.py 선행 필요)
"""
import json
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict, train_test_split)
from xgboost import XGBRegressor

from eval_fill_v4 import load698, models, SPECS, backend_key
from eval_fill_v4e import config_key, onnx_feats_for, to_onnx_row, ONNX_DIR
from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
BACKENDS = ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]
FAMILIES = ["simple_ann", "simple_cnn", "resnet_mnist",
            "mobilenet_mnist", "transformer", "gan"]
TARGETS = ["avg_train", "avg_infer"]

HW_PAT = re.compile(r"cpu_|gpu_|memory_bandwidth|unified|shared_memory"
                    r"|memory_channels|device_type")
HW_EXTRA = ["ram_total_gb", "dedicated_vram_gb", "peak_bandwidth_gbs",
            "tflops_fp32", "tflops_fp16", "host_to_device_bandwidth_gbs",
            "batch_size"]
HW_COLS = [c for c in NUMERIC if HW_PAT.search(c)] + \
          [c for c in HW_EXTRA if c in NUMERIC]
ABLATE_A = ["total_op_memory_read", "total_op_memory_write"]
ABLATE_B = ABLATE_A + ["activation_memory_mb"]


def load_onnx_rows():
    manifest = json.load(open(os.path.join(ONNX_DIR, "manifest.json")))
    cache = {}
    out = []
    for r in load698():
        of = onnx_feats_for(manifest, config_key(r["model_type"], r.get("config", {}) or {}), cache)
        assert of is not None, r["model_name"]
        out.append(to_onnx_row(r, of))
    return out


def backend_of(e):
    if e["device"] == "CPU":
        return "Desktop CPU" if e["_src"] == "win" else "Mac CPU"
    return {"GPU(CUDA)": "CUDA", "GPU(MPS)": "MPS"}[e["device"]]


def xy(subset, cols, target):
    X = np.array([[float(e.get(c, 0) or 0) for c in cols] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def xgb_fixed():
    return XGBRegressor(n_estimators=400, max_depth=6,
                        random_state=42, n_jobs=1, verbosity=0)


def main():
    rows = [enrich(r) for r in load_onnx_rows()]
    for e in rows:
        e["_backend"] = backend_of(e)
    groups = np.array([f'{e["model_type"]}/{e["model_name"]}' for e in rows])
    out = {}
    log = lambda *a: print(*a, flush=True)

    # ---- 표3: 4개 회귀 모델 비교 + 표4: 백엔드별 분해 + MAPE ----
    comp, bb, mape = {}, {}, {}
    for tgt in TARGETS:
        X, y0 = xy(rows, NUMERIC, tgt)
        y = np.expm1(y0)  # 원 단위
        defs, kf = models()
        comp[tgt] = {}
        for name, (est, grid) in defs.items():
            if grid:
                gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1)
                gs.fit(X, y0)
                est = gs.best_estimator_
            pl = cross_val_predict(est, X, y0, cv=kf, n_jobs=1)
            comp[tgt][name] = round(r2_score(y0, pl), 3)
            log(f"[표3 {tgt}] {name} {comp[tgt][name]}")
            if name == "XGBoost":
                bks = np.array([e["_backend"] for e in rows])
                bb[tgt] = {b: round(r2_score(y0[bks == b], pl[bks == b]), 3)
                           for b in BACKENDS}
                pred = np.expm1(pl)
                pe = np.abs(y - pred) / np.clip(np.abs(y), 1e-9, None)
                m = y >= (0.01 if tgt == "avg_infer" else 0)
                mape[tgt] = {"mape_all": round(float(np.mean(pe) * 100), 1),
                             "w20_all": round(float(np.mean(pe < 0.2) * 100), 1),
                             "mape_ge10ms": round(float(np.mean(pe[m]) * 100), 1),
                             "w20_ge10ms": round(float(np.mean(pe[m] < 0.2) * 100), 1)}
    out["t3_model_comparison"] = comp
    out["t4_backend_breakdown"] = bb
    out["mape"] = mape

    # ---- 표5: 프로토콜·기준선 ----
    t5 = {}
    for label, cols, cv_kind in [
            ("full_random", NUMERIC, "kfold"),
            ("full_group", NUMERIC, "group"),
            ("no_hw_group", [c for c in NUMERIC if c not in HW_COLS], "group"),
            ("flops_group", ["flops"], "group"),
            ("params_group", ["total_params"], "group")]:
        vals = {}
        for tgt in TARGETS:
            X, y = xy(rows, cols, tgt)
            cv = (KFold(5, shuffle=True, random_state=42) if cv_kind == "kfold"
                  else GroupKFold(5))
            g = None if cv_kind == "kfold" else groups
            pl = cross_val_predict(xgb_fixed(), X, y, cv=cv, groups=g, n_jobs=1)
            vals[tgt] = round(r2_score(y, pl), 3)
        t5[label] = vals
        log(f"[표5] {label} {vals}")
    # 분석적 추정 (Paleo식) — eval_fill_v4.analytic과 동일 절차
    for tgt in TARGETS:
        mult = 3.0 if tgt == "avg_train" else 1.0
        est_bk, y_bk = {}, {}
        for r in rows:
            bk = backend_key(r)
            if bk not in SPECS:
                continue
            tf, bw = SPECS[bk]
            fl = float(r.get("flops", 0) or 0) * mult
            byt = (float(r.get("total_op_memory_read", 0) or 0)
                   + float(r.get("total_op_memory_write", 0) or 0)) * mult
            est_bk.setdefault(bk, []).append(fl / (tf * 1e12) + byt / (bw * 1e9))
            y_bk.setdefault(bk, []).append(float(r.get(tgt, 0) or 0))
        yl, pl = [], []
        for bk in est_bk:
            e = np.array(est_bk[bk]); yv = np.array(y_bk[bk])
            ratio = yv / np.clip(e, 1e-12, None)
            ppp = np.median(ratio[np.isfinite(ratio) & (ratio > 0)])
            yl.append(np.log1p(yv))
            pl.append(np.log1p(np.clip(e * ppp, 0, None)))
        t5.setdefault("analytic", {})[tgt] = round(
            r2_score(np.concatenate(yl), np.concatenate(pl)), 2)
    out["t5_protocols"] = t5
    log(f"[표5] analytic {t5['analytic']}")

    # LOBO / LODO (§4.4 프로즈)
    def holdout(tr, te, tgt):
        Xtr, ytr = xy(tr, NUMERIC, tgt)
        Xte, yte = xy(te, NUMERIC, tgt)
        return round(r2_score(yte, xgb_fixed().fit(Xtr, ytr).predict(Xte)), 3)

    lobo = {b: {t: holdout([e for e in rows if e["_backend"] != b],
                           [e for e in rows if e["_backend"] == b], t)
                for t in TARGETS} for b in BACKENDS}
    lodo = {}
    for dv, lb in [("mac", "desktop→mac"), ("desktop", "mac→desktop")]:
        te_is = (lambda e: (e["_src"] == "mac") == (dv == "mac"))
        lodo[lb] = {t: holdout([e for e in rows if not te_is(e)],
                               [e for e in rows if te_is(e)], t) for t in TARGETS}
    out["lobo"], out["lodo"] = lobo, lodo
    log(f"[LOBO] {lobo}\n[LODO] {lodo}")

    # ---- 표6: 백엔드별 중요도 (RF, 학습시간) ----
    t6 = {}
    for b in BACKENDS:
        sub = [e for e in rows if e["_backend"] == b]
        X, y = xy(sub, NUMERIC, "avg_train")
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1).fit(X, y)
        imp = sorted(zip(NUMERIC, rf.feature_importances_), key=lambda t: -t[1])
        rank_f = next(i + 1 for i, (c, _) in enumerate(imp) if c == "flops")
        v_f = dict(imp)["flops"]
        t6[b] = {"top3": [(c, round(float(v), 3)) for c, v in imp[:3]],
                 "flops_rank": rank_f, "flops_imp": round(float(v_f), 3)}
        log(f"[표6] {b} {t6[b]}")
    out["t6_importance"] = t6

    # ---- 표7: 절제·permutation (RF, 학습시간, 백엔드별) ----
    idx_w = NUMERIC.index("total_op_memory_write")
    idx_f = NUMERIC.index("flops")

    def cv_r2log(sub, cols):
        X, y = xy(sub, cols, "avg_train")
        kf = KFold(5, shuffle=True, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
        return round(r2_score(y, cross_val_predict(rf, X, y, cv=kf, n_jobs=1)), 4)

    t7 = {}
    for b in BACKENDS:
        sub = [e for e in rows if e["_backend"] == b]
        full = cv_r2log(sub, NUMERIC)
        abl_a = cv_r2log(sub, [c for c in NUMERIC if c not in ABLATE_A])
        abl_b = cv_r2log(sub, [c for c in NUMERIC if c not in ABLATE_B])
        X, y = xy(sub, NUMERIC, "avg_train")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1).fit(Xtr, ytr)
        pi = permutation_importance(rf, Xte, yte, n_repeats=20,
                                    random_state=42, n_jobs=1)
        t7[b] = {"full": full, "ablate_a": abl_a, "ablate_b": abl_b,
                 "perm_write": round(float(pi.importances_mean[idx_w]), 3),
                 "perm_flops": round(float(pi.importances_mean[idx_f]), 3)}
        log(f"[표7] {b} {t7[b]}")
    out["t7_ablation"] = t7

    # ---- 표8: leave-one-family-out (XGBoost) ----
    t8 = {}
    for fam in FAMILIES:
        tr = [e for e in rows if e["model_type"] != fam]
        te = [e for e in rows if e["model_type"] == fam]
        est_kw = dict(n_estimators=400, max_depth=6, learning_rate=0.1,
                      random_state=42, n_jobs=1, verbosity=0)
        t8[fam] = {}
        for tgt in TARGETS:
            Xtr, ytr = xy(tr, NUMERIC, tgt)
            Xte, yte = xy(te, NUMERIC, tgt)
            est = XGBRegressor(**est_kw).fit(Xtr, ytr)
            t8[fam][tgt] = round(r2_score(yte, est.predict(Xte)), 2)
        log(f"[표8] {fam} {t8[fam]}")
    out["t8_lofo"] = t8

    path = os.path.join(BASE, "paper/v4_measured_f.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    log("저장:", path)


if __name__ == "__main__":
    main()
