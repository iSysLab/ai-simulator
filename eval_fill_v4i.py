"""오염 정리판(632셀) 전체 재계산 — 논문 모든 표·수치의 최종 소스.

데이터 정리:
  - 데스크톱 파일의 CPU 행 중 cpu_cores==10(=M4 혼입) 48행 → Mac CPU로 재분류,
    mac 파일의 동일 구성과 평균
  - 하드웨어 기록이 없는(cpu_cores==0) ViT 12행 → 측정 기기 불명이므로 제거
  - x86 기록이 불완전한 ViT_d256_l4_h8_p8·ViT_d256_l6_h8_p4 → 전 백엔드 제외
  → 158개 구성 × 4개 백엔드 = 632셀

산출: paper/v4_measured_i.json
사용: python eval_fill_v4i.py   (export_onnx_configs.py 선행 필요)
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_val_predict, train_test_split)
from xgboost import XGBRegressor

from eval_fill_v4 import models, SPECS
from eval_fill_v4e import config_key, onnx_feats_for, to_onnx_row, ONNX_DIR
from eval_fill_v4h import load_mapping, sample_feats
from train_merged import enrich, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"ViT_d256_l4_h8_p8", "ViT_d256_l6_h8_p4"}
BACKENDS = ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]
FAMILIES = ["simple_ann", "simple_cnn", "resnet_mnist",
            "mobilenet_mnist", "transformer", "gan"]
TARGETS = ["avg_train", "avg_infer"]
GRID = {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}

import re
HW_PAT = re.compile(r"cpu_|gpu_|memory_bandwidth|unified|shared_memory"
                    r"|memory_channels|device_type")
HW_EXTRA = ["ram_total_gb", "dedicated_vram_gb", "peak_bandwidth_gbs",
            "tflops_fp32", "tflops_fp16", "host_to_device_bandwidth_gbs",
            "batch_size"]
HW_COLS = [c for c in NUMERIC if HW_PAT.search(c)] + \
          [c for c in HW_EXTRA if c in NUMERIC]


def load_clean():
    """정리된 632행 (raw, ONNX 피처 적용 전). _backend/_src 라벨 포함."""
    desk = json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
    mac = json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json")))

    out = []
    m4_mixed = []
    for r in desk:
        if r["model_name"] in EXCLUDE:
            continue
        if r["device"] == "CPU":
            cores = r.get("cpu_cores") or 0
            if cores == 16:
                out.append({**r, "_backend": "Desktop CPU", "_src": "win"})
            elif cores == 10:
                m4_mixed.append(r)          # M4 혼입 → Mac CPU로 재분류
            # cores==0 (기기 불명 ViT) → 제거
        else:
            out.append({**r, "_backend": "CUDA", "_src": "win"})

    # Mac CPU: mac 파일 + 재분류된 48행을 구성별 평균
    groups = defaultdict(list)
    for r in mac:
        if r["device"] == "CPU" and r["model_name"] not in EXCLUDE:
            groups[r["model_name"]].append(r)
    for r in m4_mixed:
        groups[r["model_name"]].append(r)
    for name, rs in groups.items():
        base = {**rs[0], "_backend": "Mac CPU", "_src": "mac"}
        for t in TARGETS:
            base[t] = float(np.mean([float(x.get(t, 0) or 0) for x in rs]))
        out.append(base)

    out += [{**r, "_backend": "MPS", "_src": "mac"} for r in mac
            if r["device"] == "GPU(MPS)" and r["model_name"] not in EXCLUDE]

    # 메타데이터 정정: 실제 실행은 전 백엔드 학습 배치 64로 동일
    # (run_benchmark.py의 get_optimal_batch_size 결과는 출력만 되고 미사용,
    #  DataManager 기본값 batch_size=64 사용. 데스크톱 행의 None → 0 오염 수정)
    for e in out:
        e["batch_size"] = 64
    return out


def onnx_rows_of(raw_rows):
    manifest = json.load(open(os.path.join(ONNX_DIR, "manifest.json")))
    cache = {}
    out = []
    for r in raw_rows:
        of = onnx_feats_for(manifest, config_key(r["model_type"], r.get("config", {}) or {}), cache)
        assert of is not None, r["model_name"]
        e = enrich(to_onnx_row(r, of))
        e["_backend"], e["_src"] = r["_backend"], r["_src"]
        out.append(e)
    return out


def xy(subset, cols, target):
    X = np.array([[float(e.get(c, 0) or 0) for c in cols] for e in subset])
    y = np.log1p(np.array([float(e.get(target, 0) or 0) for e in subset]))
    return X, y


def grid_cvp(X, y, cv, groups=None):
    gs = GridSearchCV(XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
                      GRID, cv=cv, scoring="r2", n_jobs=1)
    gs.fit(X, y, groups=groups) if groups is not None else gs.fit(X, y)
    return cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=1)


def xgb_fixed():
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                        random_state=42, n_jobs=1, verbosity=0)


def main():
    raw = load_clean()
    print("정리 후:", {b: sum(1 for r in raw if r["_backend"] == b) for b in BACKENDS},
          "합", len(raw), flush=True)
    rows = onnx_rows_of(raw)
    groups = np.array([f'{e["model_type"]}/{e["model_name"]}' for e in rows])
    out = {"cells": len(rows)}

    # ---- 표2: 플랫폼 간 상대 학습시간 (정리된 데이터로 재계산) ----
    by = {b: {e["model_name"]: float(e["avg_train"]) for e in raw if e["_backend"] == b}
          for b in BACKENDS}
    fam_of = {e["model_name"]: e["model_type"] for e in raw}
    t2 = {}
    for label, num, den in [("mac_over_desktop", "Mac CPU", "Desktop CPU"),
                            ("mps_speedup", "Mac CPU", "MPS")]:
        agg = defaultdict(list)
        for n in set(by[num]) & set(by[den]):
            agg[fam_of[n]].append(by[num][n] / by[den][n])
        t2[label] = {f: round(float(np.mean(v)), 1) for f, v in agg.items()}
    out["t2_ratios"] = t2
    print("표2:", t2, flush=True)

    # ---- 표4(모델 비교)·표5(백엔드 분해)·MAPE ----
    comp, bb, mape = {}, {}, {}
    for tgt in TARGETS:
        X, y0 = xy(rows, NUMERIC, tgt)
        y = np.expm1(y0)
        defs, kf = models()
        comp[tgt] = {}
        for name, (est, grid) in defs.items():
            if grid:
                gs = GridSearchCV(est, grid, cv=kf, scoring="r2", n_jobs=1)
                gs.fit(X, y0)
                est = gs.best_estimator_
            pl = cross_val_predict(est, X, y0, cv=kf, n_jobs=1)
            comp[tgt][name] = round(r2_score(y0, pl), 3)
            if name == "XGBoost":
                bks = np.array([e["_backend"] for e in rows])
                bb[tgt] = {b: round(r2_score(y0[bks == b], pl[bks == b]), 3)
                           for b in BACKENDS}
                pred = np.expm1(pl)
                pe = np.abs(y - pred) / np.clip(np.abs(y), 1e-9, None)
                m = y >= (0.01 if tgt == "avg_infer" else 0)
                mape[tgt] = {"mape_ge10ms": round(float(np.mean(pe[m]) * 100), 1),
                             "w20_ge10ms": round(float(np.mean(pe[m] < 0.2) * 100), 1)}
        print(f"표4[{tgt}]", comp[tgt], flush=True)
    out["t4_model_comparison"], out["t5_backend"], out["mape"] = comp, bb, mape

    # ---- 표6: 프로토콜·기준선 (GridSearchCV 통일 절차) ----
    t6 = {}
    for label, cols, kind in [("full_random", NUMERIC, "kfold"),
                              ("full_group", NUMERIC, "group"),
                              ("no_hw_group", [c for c in NUMERIC if c not in HW_COLS], "group"),
                              ("flops_group", ["flops"], "group"),
                              ("params_group", ["total_params"], "group")]:
        vals = {}
        for tgt in TARGETS:
            X, y = xy(rows, cols, tgt)
            cv = (KFold(5, shuffle=True, random_state=42) if kind == "kfold"
                  else GroupKFold(5))
            g = None if kind == "kfold" else groups
            vals[tgt] = round(r2_score(y, grid_cvp(X, y, cv, g)), 3)
        t6[label] = vals
        print("표6", label, vals, flush=True)
    for tgt in TARGETS:  # 분석적 추정
        mult = 3.0 if tgt == "avg_train" else 1.0
        est_bk, y_bk = defaultdict(list), defaultdict(list)
        for r in rows:
            bk = (r["_src"], r.get("device"))
            if bk not in SPECS:
                continue
            tf, bw = SPECS[bk]
            fl = float(r.get("flops", 0) or 0) * mult
            byt = (float(r.get("total_op_memory_read", 0) or 0)
                   + float(r.get("total_op_memory_write", 0) or 0)) * mult
            est_bk[bk].append(fl / (tf * 1e12) + byt / (bw * 1e9))
            y_bk[bk].append(float(r.get(tgt, 0) or 0))
        yl, pl = [], []
        for bk in est_bk:
            e = np.array(est_bk[bk]); yv = np.array(y_bk[bk])
            ratio = yv / np.clip(e, 1e-12, None)
            ppp = np.median(ratio[np.isfinite(ratio) & (ratio > 0)])
            yl.append(np.log1p(yv)); pl.append(np.log1p(np.clip(e * ppp, 0, None)))
        t6.setdefault("analytic", {})[tgt] = round(
            r2_score(np.concatenate(yl), np.concatenate(pl)), 2)
    out["t6_protocols"] = t6
    print("표6 analytic", t6["analytic"], flush=True)

    # ---- LOBO / LODO ----
    def holdout(tr, te, tgt):
        Xtr, ytr = xy(tr, NUMERIC, tgt)
        Xte, yte = xy(te, NUMERIC, tgt)
        return round(r2_score(yte, xgb_fixed().fit(Xtr, ytr).predict(Xte)), 3)

    out["lobo"] = {b: {t: holdout([e for e in rows if e["_backend"] != b],
                                  [e for e in rows if e["_backend"] == b], t)
                       for t in TARGETS} for b in BACKENDS}
    out["lodo"] = {}
    for dv, lb in [("mac", "desktop→mac"), ("desktop", "mac→desktop")]:
        te_is = lambda e: (e["_src"] == "mac") == (dv == "mac")  # noqa: E731
        out["lodo"][lb] = {t: holdout([e for e in rows if not te_is(e)],
                                      [e for e in rows if te_is(e)], t)
                           for t in TARGETS}
    print("LOBO", out["lobo"], "\nLODO", out["lodo"], flush=True)

    # ---- 표7: 중요도 / 표8: 절제·permutation ----
    idx_w = NUMERIC.index("total_op_memory_write")
    idx_f = NUMERIC.index("flops")
    t7, t8 = {}, {}
    for b in BACKENDS:
        sub = [e for e in rows if e["_backend"] == b]
        X, y = xy(sub, NUMERIC, "avg_train")
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1).fit(X, y)
        imp = sorted(zip(NUMERIC, rf.feature_importances_), key=lambda t: -t[1])
        t7[b] = {"top3": [(c, round(float(v), 3)) for c, v in imp[:3]],
                 "flops_rank": next(i + 1 for i, (c, _) in enumerate(imp) if c == "flops"),
                 "flops_imp": round(float(dict(imp)["flops"]), 3)}

        def cvr(cols):
            Xc, yc = xy(sub, cols, "avg_train")
            kf = KFold(5, shuffle=True, random_state=42)
            rfc = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
            return round(r2_score(yc, cross_val_predict(rfc, Xc, yc, cv=kf, n_jobs=1)), 4)

        AA = ["total_op_memory_read", "total_op_memory_write"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        rf2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1).fit(Xtr, ytr)
        pi = permutation_importance(rf2, Xte, yte, n_repeats=20, random_state=42, n_jobs=1)
        t8[b] = {"full": cvr(NUMERIC),
                 "ablate_a": cvr([c for c in NUMERIC if c not in AA]),
                 "ablate_b": cvr([c for c in NUMERIC if c not in AA + ["activation_memory_mb"]]),
                 "perm_write": round(float(pi.importances_mean[idx_w]), 3),
                 "perm_flops": round(float(pi.importances_mean[idx_f]), 3)}
        print("표7/8", b, t7[b], t8[b], flush=True)
    out["t7_importance"], out["t8_ablation"] = t7, t8

    # ---- 표8 보강: 메모리 vs FLOPs 주장의 그룹 단위 재검증 ----
    # (a) 통합 모델 + 구성 단위 홀드아웃, (b) 통합 모델 + 계열 단위 홀드아웃
    from sklearn.model_selection import GroupShuffleSplit
    grp_perm = {}
    for glabel, gvals in [("config", groups),
                          ("family", np.array([e["model_type"] for e in rows]))]:
        X, y = xy(rows, NUMERIC, "avg_train")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_i, te_i = next(gss.split(X, y, groups=gvals))
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
        rf.fit(X[tr_i], y[tr_i])
        pi = permutation_importance(rf, X[te_i], y[te_i], n_repeats=20,
                                    random_state=42, n_jobs=1)
        grp_perm[glabel] = {
            "base_r2": round(float(r2_score(y[te_i], rf.predict(X[te_i]))), 3),
            "perm_write": round(float(pi.importances_mean[idx_w]), 3),
            "perm_flops": round(float(pi.importances_mean[idx_f]), 3),
            "held_groups": sorted(set(np.array(gvals)[te_i]))[:6] if glabel == "family" else len(set(np.array(gvals)[te_i]))}
        print("그룹 perm", glabel, grp_perm[glabel], flush=True)
    out["t8_group_perm"] = grp_perm

    # ---- 표9: LOFO ----
    t9 = {}
    for fam in FAMILIES:
        tr = [e for e in rows if e["model_type"] != fam]
        te = [e for e in rows if e["model_type"] == fam]
        t9[fam] = {t: holdout(tr, te, t) for t in TARGETS}
        print("표9", fam, t9[fam], flush=True)
    out["t9_lofo"] = t9

    # ---- 표10: ONNX vs PyTorch 훅 ----
    pt_rows = [enrich(dict(r)) for r in raw]  # PyTorch(enriched 원본) 피처
    t10 = {}
    for label, rs in [("onnx", rows), ("pytorch", pt_rows)]:
        vals = {}
        for tgt in TARGETS:
            X, y = xy(rs, NUMERIC, tgt)
            kf = KFold(5, shuffle=True, random_state=42)
            vals[tgt] = round(r2_score(y, grid_cvp(X, y, kf)), 3)
        t10[label] = vals
        print("표10", label, vals, flush=True)
    out["t10_path_comparison"] = t10

    # ---- 표11: 외부 ONNX 파일 (leave-14-configs-out) ----
    mapping = load_mapping()
    held = set(mapping.values()) - EXCLUDE
    name2feats = {mapping[s]: sample_feats(s) for s in mapping if mapping[s] in held}
    tr = [e for e in rows if e["model_name"] not in held]
    te_raw = [r for r in raw if r["model_name"] in held]
    te = []
    for r in te_raw:
        e = enrich(to_onnx_row(r, name2feats[r["model_name"]]))
        e["_backend"] = r["_backend"]
        te.append(e)
    t11 = {"n_train": len(tr), "n_test": len(te), "per_target": {}, "cases": []}
    preds = {}
    for tgt in TARGETS:
        Xtr, ytr = xy(tr, NUMERIC, tgt)
        Xte, yte = xy(te, NUMERIC, tgt)
        est = xgb_fixed().fit(Xtr, ytr)
        pl = est.predict(Xte)
        y, p = np.expm1(yte), np.expm1(pl)
        pe = np.abs(y - p) / np.clip(np.abs(y), 1e-9, None)
        novit = np.array([not r["model_name"].startswith("ViT") for r in te_raw])
        m = (y >= 0.01)
        t11["per_target"][tgt] = {
            "r2log": round(r2_score(yte, pl), 3),
            "mape_novit": round(float(np.mean(pe[novit & m]) * 100), 1),
            "w20_novit": round(float(np.mean(pe[novit & m] < 0.2) * 100), 1),
            "mape_vit": round(float(np.mean(pe[~novit & m]) * 100), 1),
        }
        preds[tgt] = p
        print("표11", tgt, t11["per_target"][tgt], flush=True)
    for i, r in enumerate(te_raw):
        t11["cases"].append({
            "model_name": r["model_name"], "backend": r["_backend"],
            "train_meas": round(float(r["avg_train"]), 3),
            "train_pred": round(float(preds["avg_train"][i]), 3),
            "infer_meas": round(float(r["avg_infer"]), 4),
            "infer_pred": round(float(preds["avg_infer"][i]), 4)})
    out["t11_external"] = t11

    path = os.path.join(BASE, "paper/v4_measured_i.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    main()
