"""심사 대응 재계산(v4j) — nested CV·전체 오차·일반화 검증.

eval_fill_v4i.load_clean()의 632셀을 그대로 쓰되 평가 절차를 보강한다.
  1) nested CV: 외부 5-겹 / 내부 4-겹 GridSearchCV, 전체 OOF 합산 R²(log)
  2) 분석적 기준선: 효율 상수를 학습 폴드에서만 보정 (구성 단위 5-겹)
  3) LOBO에 MAPE 병기
  4) 추론 오차 전체 공개: 전체 / 10ms 이상 / GAN 제외
  5) 메모리 그룹 vs FLOPs 그룹 '공동' permutation (구성·계열 홀드아웃 × 시드 5)
  6) 하드웨어 절제: 실제 수치 하드웨어 피처 30개만 제거(배치 크기 유지)
  7) LOFO: 계열 식별자 없이 미지 계열 외삽 평가
  8) ONNX·PyTorch 피처 경로를 동일한 nested CV로 비교

산출: paper/v4_measured_j.json
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, cross_val_predict)
from xgboost import XGBRegressor

from eval_fill_v4 import SPECS
from eval_fill_v4i import (load_clean, onnx_rows_of, xy, BACKENDS, TARGETS,
                           FAMILIES)
from train_merged import NUMERIC, enrich

BASE = os.path.dirname(os.path.abspath(__file__))

XGB_GRID = {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}
RF_GRID = {"n_estimators": [100, 200], "max_depth": [None, 12]}

MEM_GROUP = ["total_op_memory_read", "total_op_memory_write", "activation_memory_mb"]
FLOPS_GROUP = (["flops", "total_op_flops", "flops_per_sample", "num_mult_adds",
                "max_op_flops", "avg_op_flops", "std_op_flops"]
               + [c for c in NUMERIC if c.startswith("flops_ratio_")])

# merged_schema의 하드웨어 구간에서 문자열 6개를 제외한 27개와, v3에서
# 추가된 GPU 사양 3개다. batch_size는 입력 설정이므로 절제 대상에 넣지 않는다.
HW_NUMERIC_COLS = [
    "cpu_cores", "cpu_cores_logical", "cpu_perf_cores",
    "cpu_efficiency_cores", "cpu_freq_base_ghz", "cpu_freq_boost_ghz",
    "cpu_cache_l2_mb", "cpu_cache_l3_mb", "ram_total_gb",
    "memory_bandwidth_gbs", "is_unified_memory", "shared_memory_gb",
    "dedicated_vram_gb", "gpu_count", "gpu_memory_gb", "gpu_cores",
    "peak_bandwidth_gbs", "tflops_fp32", "tflops_fp16", "fp16_support",
    "bf16_support", "host_to_device_bandwidth_gbs", "is_discrete_gpu",
    "is_integrated_gpu", "device_type_encoded", "cpu_freq_ghz",
    "memory_channels", "gpu_tensor_core_count", "gpu_compute_capability",
    "gpu_clock_ghz",
]
FAMILY_ID_COLS = {"model_family_encoded", "model_family", "model_arch"}
LOFO_COLS = [c for c in NUMERIC if c not in FAMILY_ID_COLS]

assert len(HW_NUMERIC_COLS) == 30
assert set(HW_NUMERIC_COLS) <= set(NUMERIC)
assert "batch_size" not in HW_NUMERIC_COLS and "batch_size" in NUMERIC
assert "model_family_encoded" not in LOFO_COLS


def nested_oof(X, y, make_est, grid, groups=None, seed=42):
    """외부 5-겹 / 내부 4-겹 nested CV의 전체 OOF 예측."""
    outer = (KFold(5, shuffle=True, random_state=seed) if groups is None
             else GroupKFold(5))
    oof = np.zeros(len(y))
    for tr, te in outer.split(X, y, groups):
        inner = (KFold(4, shuffle=True, random_state=seed) if groups is None
                 else GroupKFold(4))
        gs = GridSearchCV(make_est(), grid, cv=inner, scoring="r2", n_jobs=1)
        gs.fit(X[tr], y[tr], groups=None if groups is None else groups[tr])
        oof[te] = gs.best_estimator_.predict(X[te])
    return oof


def mk_xgb():
    return XGBRegressor(random_state=42, n_jobs=1, verbosity=0)


def mk_rf():
    return RandomForestRegressor(random_state=42, n_jobs=1)


def err_stats(y, pred, mask=None):
    if mask is None:
        mask = np.ones(len(y), bool)
    pe = np.abs(y[mask] - pred[mask]) / np.clip(np.abs(y[mask]), 1e-9, None)
    return {"n": int(mask.sum()), "mape": round(float(np.mean(pe) * 100), 1),
            "w20": round(float(np.mean(pe < 0.2) * 100), 1)}


def main():
    raw = load_clean()
    rows = onnx_rows_of(raw)
    groups = np.array([e["model_name"] for e in rows])
    fams = np.array([e["model_type"] for e in rows])
    bks = np.array([e["_backend"] for e in rows])
    is_gan = fams == "gan"
    out = {}

    # ---- 1) nested CV: 표4 모델 비교 + 표5 백엔드 + 오차 공개 ----
    comp, bb, errors, onnx_nested = {}, {}, {}, {}
    for tgt in TARGETS:
        X, y0 = xy(rows, NUMERIC, tgt)
        y = np.expm1(y0)
        kf = KFold(5, shuffle=True, random_state=42)
        comp[tgt] = {}
        comp[tgt]["LinearRegression"] = round(r2_score(y0, cross_val_predict(
            LinearRegression(), X, y0, cv=kf, n_jobs=1)), 3)
        comp[tgt]["GradientBoosting"] = round(r2_score(y0, cross_val_predict(
            GradientBoostingRegressor(random_state=42), X, y0, cv=kf, n_jobs=1)), 3)
        oof_rf = nested_oof(X, y0, mk_rf, RF_GRID)
        comp[tgt]["RandomForest"] = round(r2_score(y0, oof_rf), 3)
        oof = nested_oof(X, y0, mk_xgb, XGB_GRID)
        comp[tgt]["XGBoost"] = round(r2_score(y0, oof), 3)
        onnx_nested[tgt] = comp[tgt]["XGBoost"]
        bb[tgt] = {b: round(r2_score(y0[bks == b], oof[bks == b]), 3)
                   for b in BACKENDS}
        pred = np.expm1(oof)
        m10 = y >= 0.01
        errors[tgt] = {"all": err_stats(y, pred),
                       "ge10ms": err_stats(y, pred, m10),
                       "ge10ms_nogan": err_stats(y, pred, m10 & ~is_gan),
                       "gan_only": err_stats(y, pred, is_gan)}
        print(f"[표4 nested {tgt}]", comp[tgt], "\n  백엔드:", bb[tgt],
              "\n  오차:", errors[tgt], flush=True)
    out["t4_nested"], out["t5_backend"], out["errors"] = comp, bb, errors

    # ---- 표6: 프로토콜·기준선 (nested) ----
    t6 = {}
    for label, cols, kind in [("full_random", NUMERIC, "kfold"),
                              ("full_group", NUMERIC, "group"),
                              ("no_hw_group", [c for c in NUMERIC
                                               if c not in HW_NUMERIC_COLS], "group"),
                              ("flops_group", ["flops"], "group"),
                              ("params_group", ["total_params"], "group")]:
        vals = {}
        for tgt in TARGETS:
            X, y = xy(rows, cols, tgt)
            g = None if kind == "kfold" else groups
            vals[tgt] = round(r2_score(y, nested_oof(X, y, mk_xgb, XGB_GRID, g)), 3)
        t6[label] = vals
        print("표6", label, vals, flush=True)

    # 분석적 기준선: 효율 상수를 학습 폴드에서만 보정
    for tgt in TARGETS:
        mult = 3.0 if tgt == "avg_train" else 1.0
        y_raw = np.array([float(r.get(tgt, 0) or 0) for r in rows])
        t_hat = np.zeros(len(rows))
        bk_of = []
        for i, r in enumerate(rows):
            bk = (r["_src"], r.get("device"))
            bk_of.append(bk)
            tf, bw = SPECS[bk]
            fl = float(r.get("flops", 0) or 0) * mult
            byt = (float(r.get("total_op_memory_read", 0) or 0)
                   + float(r.get("total_op_memory_write", 0) or 0)) * mult
            t_hat[i] = fl / (tf * 1e12) + byt / (bw * 1e9)
        bk_of = np.array([f"{a}/{b}" for a, b in bk_of])
        pred = np.zeros(len(rows))
        for tr, te in GroupKFold(5).split(t_hat, y_raw, groups):
            for bk in set(bk_of):
                trb = tr[bk_of[tr] == bk]
                teb = te[bk_of[te] == bk]
                if len(trb) == 0 or len(teb) == 0:
                    continue
                ratio = y_raw[trb] / np.clip(t_hat[trb], 1e-12, None)
                ppp = np.median(ratio[np.isfinite(ratio) & (ratio > 0)])
                pred[teb] = t_hat[teb] * ppp
        t6.setdefault("analytic_foldwise", {})[tgt] = round(
            r2_score(np.log1p(y_raw), np.log1p(np.clip(pred, 0, None))), 2)
    out["t6_protocols"] = t6
    print("표6 analytic(fold)", t6["analytic_foldwise"], flush=True)

    # ---- LOBO: R² + MAPE ----
    lobo = {}
    for b in BACKENDS:
        tr = [e for e in rows if e["_backend"] != b]
        te = [e for e in rows if e["_backend"] == b]
        lobo[b] = {}
        for tgt in TARGETS:
            Xtr, ytr = xy(tr, NUMERIC, tgt)
            Xte, yte = xy(te, NUMERIC, tgt)
            est = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                               random_state=42, n_jobs=1, verbosity=0).fit(Xtr, ytr)
            pl = est.predict(Xte)
            yv, pv = np.expm1(yte), np.expm1(pl)
            m = yv >= (0.01 if tgt == "avg_infer" else 0)
            lobo[b][tgt] = {"r2log": round(r2_score(yte, pl), 3),
                            **err_stats(yv, pv, m)}
        print("LOBO", b, lobo[b], flush=True)
    out["lobo"] = lobo

    # ---- LOFO: 계열명/아키텍처 식별자 없이 구조 피처만으로 외삽 ----
    lofo = {}
    for fam in FAMILIES:
        tr = [e for e in rows if e["model_type"] != fam]
        te = [e for e in rows if e["model_type"] == fam]
        lofo[fam] = {}
        for tgt in TARGETS:
            Xtr, ytr = xy(tr, LOFO_COLS, tgt)
            Xte, yte = xy(te, LOFO_COLS, tgt)
            est = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1,
                               random_state=42, n_jobs=1, verbosity=0).fit(Xtr, ytr)
            pl = est.predict(Xte)
            yv, pv = np.expm1(yte), np.expm1(pl)
            m = yv >= (0.01 if tgt == "avg_infer" else 0)
            lofo[fam][tgt] = {"r2log": round(r2_score(yte, pl), 3),
                              **err_stats(yv, pv, m)}
        print("LOFO(no family id)", fam, lofo[fam], flush=True)
    out["lofo_no_family_id"] = {
        "excluded_columns": sorted(FAMILY_ID_COLS),
        "numeric_feature_count": len(LOFO_COLS),
        "results": lofo,
    }

    # ---- ONNX vs PyTorch 훅: 같은 행·피처열·외부/내부 폴드의 nested CV ----
    pt_rows = [enrich(dict(r)) for r in raw]
    pytorch_nested = {}
    for tgt in TARGETS:
        X, y = xy(pt_rows, NUMERIC, tgt)
        oof = nested_oof(X, y, mk_xgb, XGB_GRID)
        pytorch_nested[tgt] = round(r2_score(y, oof), 3)
        print("경로 비교 nested", tgt,
              {"onnx": onnx_nested[tgt], "pytorch": pytorch_nested[tgt]},
              flush=True)
    out["t12_path_nested"] = {
        "protocol": "outer 5-fold / inner 4-fold nested CV, random_state=42",
        "n_rows": len(rows),
        "numeric_feature_count": len(NUMERIC),
        "onnx": onnx_nested,
        "pytorch": pytorch_nested,
    }

    # ---- 5) 그룹 공동 permutation: 메모리 그룹 vs FLOPs 그룹 ----
    idx_mem = [NUMERIC.index(c) for c in MEM_GROUP if c in NUMERIC]
    idx_flo = [NUMERIC.index(c) for c in FLOPS_GROUP if c in NUMERIC]
    X, y = xy(rows, NUMERIC, "avg_train")

    def joint_perm(gvals, seeds=(0, 1, 2, 3, 4), n_rep=10):
        drops = {"mem": [], "flops": []}
        bases = []
        for s in seeds:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s)
            tr, te = next(gss.split(X, y, groups=gvals))
            rf = RandomForestRegressor(n_estimators=200, random_state=42,
                                       n_jobs=1).fit(X[tr], y[tr])
            base = r2_score(y[te], rf.predict(X[te]))
            bases.append(base)
            rng = np.random.default_rng(s)
            for key, idxs in [("mem", idx_mem), ("flops", idx_flo)]:
                for _ in range(n_rep):
                    perm = rng.permutation(len(te))
                    Xp = X[te].copy()
                    Xp[:, idxs] = X[te][perm][:, idxs]   # 그룹 열을 함께 셔플
                    drops[key].append(base - r2_score(y[te], rf.predict(Xp)))
        return {"base_r2": f"{np.mean(bases):.3f}±{np.std(bases):.3f}",
                **{k: f"{np.mean(v):.3f}±{np.std(v):.3f}" for k, v in drops.items()}}

    out["group_joint_perm"] = {"config": joint_perm(groups),
                               "family": joint_perm(fams)}
    print("공동 perm:", out["group_joint_perm"], flush=True)

    path = os.path.join(BASE, "paper/v4_measured_j.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    main()
