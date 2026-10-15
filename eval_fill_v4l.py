"""앙상블 예측기 평가(v4l) — 설계 paper/70_ensemble_design.md, 추정기 benchmark/predictor/ensemble.py.

v4k의 Ctx(632셀, 구성 단위 nested CV: 외부 5-겹 / 내부 4-겹 GridSearchCV, scoring=r2 in z)를 그대로 쓴다.
모든 후보는 로그 공간 z=log1p(초) 입출력이라 nested_oof()로 같은 절차를 돈다.

  baseline           현재 통합 XGB(이 기기에서 재계산) + oracle 전문 모델 상한 (v4k와 같은 정의)
  gate_analysis      피처 분류기 AUC, base 예측 라우팅 정밀도·재현율 — B의 관문
  log_target         (F) 타깃 변환 log(y)·log1p(ms)
  weighted_baseline  (D) 소형 셀 sample_weight
  residual_baseline  (E) 소형 구간 잔차 보정
  regime_ensemble    (B) 게이트 앙상블 hard/soft/clf × τ 그리드
  algo_average       (A) XGB·GB·RF 평균
  per_backend_control(C) 백엔드별 4모델 대조군
  ladder_check       폭·깊이·최대 사분위 외삽(v4k 사다리 3~5단계)에서 후보별 성능
  seed_stability     시드 10개

사용:
  python eval_fill_v4l.py                       # 전체 (~50분)
  python eval_fill_v4l.py --only baseline,gate_analysis,log_target
산출: paper/v4_measured_l.json (절 단위 병합 저장)
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from xgboost import XGBClassifier

from eval_fill_v4i import TARGETS, FAMILIES
from eval_fill_v4j import nested_oof, mk_xgb, XGB_GRID
from eval_fill_v4k import Ctx, err_stats, nested_oof_seeded, fit_best, width_depth, FAM_LABEL, SEEDS
from train_merged import NUMERIC
from benchmark.predictor import (RegimeEnsembleRegressor, WeightedXGBRegressor,
                                 ResidualCorrectedRegressor, LogTargetXGBRegressor,
                                 AlgoAverageRegressor, PerBackendRegressor)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, "paper/v4_measured_l.json")

T = lambda sec: float(np.log1p(sec))           # 초 → z
SMALL_CUT = {"avg_infer": 0.01, "avg_train": 1.0}  # "소형" 보고 구간 (v4k와 동일: <10 ms / <1 s)
# 라우팅 임계 τ 와 전문 모델 학습 임계 τ_s 후보 (초)
TAU = {"avg_infer": [0.02, 0.05], "avg_train": [0.5, 1.0]}
TAU_S = {"avg_infer": [0.05, 0.1], "avg_train": [1.0, 2.0]}
BETA = {"avg_infer": [0.005, 0.02], "avg_train": [0.05, 0.2]}


# ----------------------------------------------------------------------------- 공통
def summarize(ctx, tgt, y, oof, mask=None):
    """z 공간 R² + 원 단위 오차(전체 / ≥10ms / 소형 / GAN). mask는 평가 대상 셀."""
    m = np.ones(len(y), bool) if mask is None else mask
    yy, pp = np.expm1(y), np.expm1(oof)
    cut = SMALL_CUT[tgt]
    s = {"r2log": round(float(r2_score(y[m], oof[m])), 4),
         "all": err_stats(yy, pp, m),
         "small": err_stats(yy, pp, m & (yy < cut)),
         "large": err_stats(yy, pp, m & (yy >= cut)),
         "gan": err_stats(yy, pp, m & ctx.is_gan)}
    if tgt == "avg_infer":
        s["ge10ms"] = s["large"]
    return s


def run_nested(ctx, tgt, make_est, grid, cols=NUMERIC, X=None):
    """구성 단위 nested CV OOF → 요약. X를 주면 그대로 쓴다(백엔드 열 추가용)."""
    Xd, y = ctx.X(cols, tgt)
    if X is not None:
        Xd = X
    t0 = time.time()
    oof = nested_oof(Xd, y, make_est, grid, ctx.groups)
    s = summarize(ctx, tgt, y, oof)
    s["_elapsed_s"] = round(time.time() - t0, 1)
    return s, oof


def chosen_params(ctx, tgt, make_est, grid, cols=NUMERIC, X=None):
    """전체 632셀에 구성 단위 4-겹 GridSearchCV 1회 → best_params (사다리·시드 절에서 그리드 고정용)."""
    Xd, y = ctx.X(cols, tgt)
    if X is not None:
        Xd = X
    gs = GridSearchCV(make_est(), grid, cv=GroupKFold(4), scoring="r2", n_jobs=1)
    gs.fit(Xd, y, groups=ctx.groups)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}


def single_grid(params):
    return {k: [v] for k, v in params.items()}


def log_line(tag, tgt, s):
    sm = s["small"]
    print(f"[{tag}] {tgt}: R2 {s['r2log']}  all MAPE {s['all']['mape']} MdAPE {s['all']['mdape']}  "
          f"small(n={sm['n']}) MAPE {sm['mape']} MdAPE {sm['mdape']} ±20% {sm['w20']}  "
          f"large MAPE {s['large']['mape']}  ({s.get('_elapsed_s', '?')}s)", flush=True)


# ----------------------------------------------------------------------------- baseline
def baseline(ctx):
    """현재 통합 XGB(Ctx.base와 동일 절차)와 oracle 전문 모델(실제 y로 부분집합 선택 — 상한)."""
    out = {}
    for tgt in TARGETS:
        y, oof, _ = ctx.base(tgt)
        s = summarize(ctx, tgt, y, oof)
        # oracle 전문 모델: v4k overhead_features.specialist_small과 같은 정의 (τ_s = 50 ms / 1 s)
        cut_s = 0.05 if tgt == "avg_infer" else 1.0
        X, _ = ctx.X(NUMERIC, tgt)
        small = np.expm1(y) < cut_s
        oof_s = np.full(len(y), np.nan)
        oof_s[small] = nested_oof(X[small], y[small], mk_xgb, XGB_GRID, ctx.groups[small])
        yy, pp = np.expm1(y), np.expm1(oof_s)
        s["oracle_specialist"] = {"tau_s_sec": cut_s, "n_train_cells": int(small.sum()),
                                  "small": err_stats(yy, np.nan_to_num(pp), small & (yy < SMALL_CUT[tgt]))}
        s["chosen_params"] = chosen_params(ctx, tgt, mk_xgb, XGB_GRID)
        out[tgt] = s
        log_line("base", tgt, s)
        print(f"       oracle specialist small: {s['oracle_specialist']['small']}  chosen {s['chosen_params']}", flush=True)
    return out


# ----------------------------------------------------------------------------- gate_analysis
def gate_analysis(ctx):
    """소형 구간이 (a) 피처만으로, (b) base 예측으로 얼마나 갈리는가."""
    out = {}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        _, zb, _ = ctx.base(tgt)
        res = {}
        taus = sorted(set(TAU[tgt] + [SMALL_CUT[tgt]] + TAU_S[tgt]))
        for tau_sec in taus:
            tau = T(tau_sec)
            true_small = y < tau
            if true_small.sum() < 5 or (~true_small).sum() < 5:
                continue
            # (a) 피처 분류기, 구성 단위 5-겹
            clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                random_state=42, n_jobs=1, verbosity=0)
            proba = cross_val_predict(clf, X, true_small.astype(int), groups=ctx.groups,
                                      cv=GroupKFold(5), method="predict_proba")[:, 1]
            pred_c = proba >= 0.5
            # (b) base 예측 라우팅
            pred_b = zb < tau
            def pr(pred):
                tp = int((pred & true_small).sum()); fp = int((pred & ~true_small).sum()); fn = int((~pred & true_small).sum())
                return {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
            r = {"tau_sec": tau_sec, "n_small": int(true_small.sum()),
                 "clf_auc": round(float(roc_auc_score(true_small, proba)), 4),
                 "clf_acc": round(float((pred_c == true_small).mean()), 4),
                 "clf": pr(pred_c),
                 "base_route_auc": round(float(roc_auc_score(true_small, -zb)), 4),
                 "base_route": pr(pred_b)}
            # 라우팅된 셀 중 전문 모델 학습 범위(τ_s) 안인 비율
            for ts in TAU_S[tgt]:
                in_range = y < T(ts)
                r[f"routed_in_range_tau_s={ts}"] = round(float((in_range & pred_b).sum() / max(pred_b.sum(), 1)), 3)
            res[str(tau_sec)] = r
            print(f"[gate] {tgt} τ={tau_sec}s n_small={r['n_small']}: clf AUC {r['clf_auc']} P/R {r['clf']['precision']}/{r['clf']['recall']} | "
                  f"base-route AUC {r['base_route_auc']} P/R {r['base_route']['precision']}/{r['base_route']['recall']}", flush=True)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- F
def log_target(ctx):
    out = {}
    for tgt in TARGETS:
        res = {}
        for tr in ["log", "log1p_ms"]:
            grid = {"transform": [tr], **XGB_GRID}
            s, _ = run_nested(ctx, tgt, LogTargetXGBRegressor, grid)
            s["chosen_params"] = chosen_params(ctx, tgt, LogTargetXGBRegressor, grid)
            res[tr] = s
            log_line(f"F:{tr}", tgt, s)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- D
def weighted_baseline(ctx):
    out = {}
    for tgt in TARGETS:
        grid = {"tau_s": [T(v) for v in TAU_S[tgt]], "alpha": [3.0, 10.0, 30.0],
                "scheme": ["step", "linear"], **XGB_GRID}
        s, _ = run_nested(ctx, tgt, WeightedXGBRegressor, grid)
        s["chosen_params"] = chosen_params(ctx, tgt, WeightedXGBRegressor, grid)
        out[tgt] = s
        log_line("D", tgt, s)
        print(f"       chosen {s['chosen_params']}", flush=True)
    return out


# ----------------------------------------------------------------------------- E
def residual_baseline(ctx):
    out = {}
    for tgt in TARGETS:
        grid = {"tau": [T(v) for v in TAU[tgt]], "tau_s": [T(v) for v in TAU_S[tgt]],
                "beta": BETA[tgt], **XGB_GRID}
        s, _ = run_nested(ctx, tgt, ResidualCorrectedRegressor, grid)
        s["chosen_params"] = chosen_params(ctx, tgt, ResidualCorrectedRegressor, grid)
        out[tgt] = s
        log_line("E", tgt, s)
        print(f"       chosen {s['chosen_params']}", flush=True)
    return out


# ----------------------------------------------------------------------------- B
def regime_grid(tgt):
    taus = [T(v) for v in TAU[tgt]]
    taus_s = [T(v) for v in TAU_S[tgt]]
    return [{"gate": ["hard"], "tau": taus, "tau_s": taus_s, **XGB_GRID},
            {"gate": ["soft"], "tau": taus, "tau_s": taus_s, "beta": BETA[tgt], **XGB_GRID},
            {"gate": ["clf"], "tau": taus, "tau_s": taus_s, **XGB_GRID}]


def regime_ensemble(ctx):
    out = {}
    for tgt in TARGETS:
        res = {}
        # 게이트별 따로 (어느 게이트가 이기는지 보이도록) + 전체 그리드
        for g in regime_grid(tgt):
            name = g["gate"][0]
            s, oof = run_nested(ctx, tgt, RegimeEnsembleRegressor, g)
            s["chosen_params"] = chosen_params(ctx, tgt, RegimeEnsembleRegressor, g)
            # 라우팅 통계: 최종 모델(chosen)로 전체 fit 후 가중 분포
            X, y = ctx.X(NUMERIC, tgt)
            est = RegimeEnsembleRegressor(**s["chosen_params"]).fit(X, y)
            w = est.gate_weights(X)
            s["gate_weight"] = {"mean": round(float(w.mean()), 3),
                                "frac_w_gt_0.5": round(float((w > 0.5).mean()), 3),
                                "n_specialist_train_cells": est.n_small_}
            res[name] = s
            log_line(f"B:{name}", tgt, s)
            print(f"       chosen {s['chosen_params']}  w>0.5: {s['gate_weight']['frac_w_gt_0.5']}", flush=True)
        s_all, _ = run_nested(ctx, tgt, RegimeEnsembleRegressor, regime_grid(tgt))
        s_all["chosen_params"] = chosen_params(ctx, tgt, RegimeEnsembleRegressor, regime_grid(tgt))
        res["all_gates"] = s_all
        log_line("B:all", tgt, s_all)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- A
def algo_average(ctx):
    out = {}
    for tgt in TARGETS:
        grid = {"members": ["xgb,gb,rf"], **XGB_GRID}
        s, _ = run_nested(ctx, tgt, AlgoAverageRegressor, grid)
        s["chosen_params"] = chosen_params(ctx, tgt, AlgoAverageRegressor, grid)
        out[tgt] = s
        log_line("A", tgt, s)
    return out


# ----------------------------------------------------------------------------- C
def per_backend_control(ctx):
    out = {}
    for tgt in TARGETS:
        X, _ = ctx.X(NUMERIC, tgt)
        Xb = np.hstack([X, ctx.bid[:, None].astype(float)])
        grid = {"backend_col": [-1], **XGB_GRID}
        s, _ = run_nested(ctx, tgt, PerBackendRegressor, grid, X=Xb)
        out[tgt] = s
        log_line("C", tgt, s)
    return out


# ----------------------------------------------------------------------------- F + C
def log_per_backend(ctx):
    """F와 C의 결합: 백엔드별 4모델, 각 구성원이 log(y) 타깃. (C의 학습시간 우위가 log 타깃과 겹치는지)"""
    out = {}
    for tgt in TARGETS:
        X, _ = ctx.X(NUMERIC, tgt)
        Xb = np.hstack([X, ctx.bid[:, None].astype(float)])
        res = {}
        for tr in ["log", "log1p"]:
            grid = {"backend_col": [-1], "transform": [tr], **XGB_GRID}
            s, _ = run_nested(ctx, tgt, PerBackendRegressor, grid, X=Xb)
            s["chosen_params"] = chosen_params(ctx, tgt, PerBackendRegressor, grid, X=Xb)
            res[tr] = s
            log_line(f"F+C:{tr}", tgt, s)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- 기여 2 재검증 (log 타깃)
def _nested_models_log(ctx, tgt, transform):
    """ctx._nested_models와 같은 무셔플 GroupKFold nested CV, 추정기만 LogTargetXGBRegressor."""
    X, y = ctx.X(NUMERIC, tgt)
    grid = {"transform": [transform], **XGB_GRID}
    oof, models = np.zeros(len(y)), []
    for tr, te in GroupKFold(5).split(X, y, ctx.groups):
        gs = GridSearchCV(LogTargetXGBRegressor(), grid, cv=GroupKFold(4), scoring="r2", n_jobs=1)
        gs.fit(X[tr], y[tr], groups=ctx.groups[tr])
        oof[te] = gs.best_estimator_.predict(X[te])
        models.append((te, gs.best_estimator_))
    return X, y, oof, models


def log_target_claims(ctx, n_rep=20, seed=42):
    """v4k unified_importance + symmetric_ablation을 log(y) 타깃으로 재실행. 같은 기기에서 log1p도 같이 돌려 비교."""
    from collections import defaultdict
    from eval_fill_v4j import MEM_GROUP, FLOPS_GROUP, HW_NUMERIC_COLS
    rng = np.random.default_rng(seed)
    singles = ["total_op_memory_write", "total_op_memory_read", "flops", "total_params", "model_family_encoded"]
    groups_def = {"MEM_GROUP": MEM_GROUP, "FLOPS_GROUP": FLOPS_GROUP, "HW_30": HW_NUMERIC_COLS}
    idx = {c: NUMERIC.index(c) for c in singles}
    gidx = {g: [NUMERIC.index(c) for c in cols if c in NUMERIC] for g, cols in groups_def.items()}
    out = {}
    for transform in ["log1p", "log"]:
        res_t = {}
        for tgt in TARGETS:
            X, y, oof, models = _nested_models_log(ctx, tgt, transform)
            drops, gain_rank = defaultdict(list), defaultdict(list)
            for te, est in models:
                Xt, yt = X[te], y[te]
                base = r2_score(yt, est.predict(Xt))
                for name, cols in list(idx.items()) + list(gidx.items()):
                    cols = [cols] if isinstance(cols, int) else cols
                    d = []
                    for _ in range(n_rep):
                        Xp = Xt.copy(); perm = rng.permutation(len(te))
                        Xp[:, cols] = Xt[perm][:, cols]
                        d.append(base - r2_score(yt, est.predict(Xp)))
                    drops[name].append(float(np.mean(d)))
                order = np.argsort(-est.model_.feature_importances_)
                for c in singles:
                    gain_rank[c].append(int(np.where(order == idx[c])[0][0]) + 1)
                gain_rank["_top5"].append([NUMERIC[i] for i in order[:5]])
            imp = {"permutation_drop": {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in drops.items()},
                   "gain_rank": dict(gain_rank), "r2log_oof": round(float(r2_score(y, oof)), 4)}
            # 대칭 절제 + 단독 기준선
            conds = {"full": NUMERIC,
                     "minus_MEM": [c for c in NUMERIC if c not in MEM_GROUP],
                     "minus_FLOPS": [c for c in NUMERIC if c not in FLOPS_GROUP],
                     "minus_both": [c for c in NUMERIC if c not in MEM_GROUP and c not in FLOPS_GROUP],
                     "only_mem_write": ["total_op_memory_write"], "only_flops": ["flops"],
                     "mem_write+backend_id": ["total_op_memory_write", "_backend_id"],
                     "flops+backend_id": ["flops", "_backend_id"],
                     "mem_write+flops+backend_id": ["total_op_memory_write", "flops", "_backend_id"]}
            abl = {}
            for label, cols in conds.items():
                Xc, yc = ctx.X(cols, tgt)
                o = nested_oof(Xc, yc, LogTargetXGBRegressor, {"transform": [transform], **XGB_GRID}, ctx.groups)
                s = summarize(ctx, tgt, yc, o)
                abl[label] = {"n_features": len(cols), "r2log": s["r2log"], "all_mape": s["all"]["mape"],
                              "large_mape": s["large"]["mape"], "small_mape": s["small"]["mape"]}
            res_t[tgt] = {"importance": imp, "ablation": abl}
            pd_ = imp["permutation_drop"]
            print(f"[claim2:{transform}] {tgt}: Δ mem_write {pd_['total_op_memory_write']['mean']}  flops {pd_['flops']['mean']}  "
                  f"MEM {pd_['MEM_GROUP']['mean']}  FLOPS {pd_['FLOPS_GROUP']['mean']}  HW {pd_['HW_30']['mean']}  "
                  f"| gain rank mem_write {gain_rank['total_op_memory_write']} flops {gain_rank['flops']}", flush=True)
            print(f"        ablation R2: full {abl['full']['r2log']} −MEM {abl['minus_MEM']['r2log']} −FLOPS {abl['minus_FLOPS']['r2log']} −both {abl['minus_both']['r2log']} "
                  f"| mem_write+id {abl['mem_write+backend_id']['r2log']} flops+id {abl['flops+backend_id']['r2log']} "
                  f"| MAPE full {abl['full']['all_mape']} −MEM {abl['minus_MEM']['all_mape']} −FLOPS {abl['minus_FLOPS']['all_mape']}", flush=True)
        out[transform] = res_t
    return out


# ----------------------------------------------------------------------------- LOFO / LOBO (log 타깃)
def log_target_lofo(ctx):
    """v4k 사다리 6단계(LOFO, 계열 식별자 제외)·7단계(LOBO)를 log1p와 log으로 나란히. 내부 튜닝 포함(fit_best)."""
    from sklearn.model_selection import LeaveOneGroupOut
    from eval_fill_v4j import LOFO_COLS
    out = {}
    for transform in ["log1p", "log"]:
        mk = (lambda t=transform: LogTargetXGBRegressor(transform=t))
        grid = {"transform": [transform], **XGB_GRID}
        res_t = {}
        for label, og, cols in [("L6_family_out", ctx.fams, LOFO_COLS), ("L7_backend_out", ctx.bks, NUMERIC)]:
            res = {}
            for tgt in TARGETS:
                X, y = ctx.X(cols, tgt)
                oof, per = np.zeros(len(y)), {}
                for tr, te in LeaveOneGroupOut().split(X, y, og):
                    est = fit_best(X[tr], y[tr], ctx.groups[tr], make_est=mk, grid=grid)
                    oof[te] = est.predict(X[te])
                    g = og[te][0]
                    mask = np.zeros(len(y), bool); mask[te] = True
                    s = summarize(ctx, tgt, y, oof, mask)
                    per[FAM_LABEL.get(g, g)] = {"r2log": s["r2log"], "mape": s["all"]["mape"], "mdape": s["all"]["mdape"], "w20": s["all"]["w20"]}
                pooled = summarize(ctx, tgt, y, oof)
                pooled["per_group"] = per
                res[tgt] = pooled
                print(f"[lofo:{transform}] {label} {tgt}: pooled R2 {pooled['r2log']} MAPE {pooled['all']['mape']} | "
                      + " ".join(f"{k} {v['r2log']}/{v['mape']}" for k, v in per.items()), flush=True)
            res_t[label] = res
        out[transform] = res_t
    return out


# ----------------------------------------------------------------------------- ladder_check
CANDIDATES = {
    # 이름: (클래스, 절 이름, JSON에서 chosen_params를 꺼내는 경로)
    "xgb": (mk_xgb, "baseline", None),
    "F_log": (LogTargetXGBRegressor, "log_target", ("log",)),
    "D_weighted": (WeightedXGBRegressor, "weighted_baseline", ()),
    "E_residual": (ResidualCorrectedRegressor, "residual_baseline", ()),
    "B_regime": (RegimeEnsembleRegressor, "regime_ensemble", ("all_gates",)),
}


def _chosen(existing, tgt, section, path):
    node = existing.get(section, {}).get(tgt)
    if node is None:
        return None
    for p in (path or ()):
        node = node.get(p)
        if node is None:
            return None
    return node.get("chosen_params")


def _make(cls, params):
    if cls is mk_xgb:
        from xgboost import XGBRegressor
        return lambda: XGBRegressor(random_state=42, n_jobs=1, verbosity=0)
    return cls


def ladder_check(ctx, existing):
    """v4k 사다리 3·4·5단계(폭·깊이·최대 사분위)를 후보별로. 그리드는 각 절의 chosen_params로 고정."""
    wd = np.array([width_depth(e) for e in ctx.rows], dtype=object)
    width_g = np.array([f"{f}|w={w}" for f, (w, _) in zip(ctx.fams, wd)])
    depth_g = np.array([f"{f}|d={d}" for f, (_, d) in zip(ctx.fams, wd)])
    params = np.array([float(e.get("total_params", 0) or 0) for e in ctx.rows])
    top = np.zeros(len(ctx.rows), bool)
    for f in FAMILIES:
        idx = np.where(ctx.fams == f)[0]
        cfg = {ctx.groups[i]: params[i] for i in idx}
        names = sorted(cfg, key=lambda n: cfg[n])
        held = set(names[-max(1, int(round(len(names) * 0.25))):])
        top[[i for i in idx if ctx.groups[i] in held]] = True

    out = {}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        res = {}
        for name, (cls, section, path) in CANDIDATES.items():
            cp = _chosen(existing, tgt, section, path)
            if cp is None:
                print(f"[ladder] {tgt} {name}: chosen_params 없음 — 절 먼저 실행", flush=True)
                continue
            make_est, grid = _make(cls, cp), single_grid(cp)
            r = {"chosen_params": cp}
            for label, og in [("L3_width", width_g), ("L4_depth", depth_g)]:
                oof = nested_oof_seeded(X, y, ctx.groups, 42, make_est=make_est, grid=grid, outer_groups=og)
                r[label] = summarize(ctx, tgt, y, oof)
            est = fit_best(X[~top], y[~top], ctx.groups[~top], make_est=make_est, grid=grid)
            full = np.zeros(len(y)); full[top] = est.predict(X[top])
            r["L5_top_quartile"] = summarize(ctx, tgt, y, full, top)
            r["L5_top_quartile"]["median_pred_over_true"] = round(float(np.median(np.expm1(full[top]) / np.expm1(y[top]))), 3)
            res[name] = r
            print(f"[ladder] {tgt} {name}: L3 {r['L3_width']['r2log']}/{r['L3_width']['all']['mape']}  "
                  f"L4 {r['L4_depth']['r2log']}/{r['L4_depth']['all']['mape']}  "
                  f"L5 {r['L5_top_quartile']['r2log']}/{r['L5_top_quartile']['all']['mape']} p/t {r['L5_top_quartile']['median_pred_over_true']}", flush=True)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- seed_stability
def seed_stability(ctx, existing):
    out = {}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        res = {}
        for name, (cls, section, path) in CANDIDATES.items():
            cp = _chosen(existing, tgt, section, path)
            if cp is None:
                continue
            make_est, grid = _make(cls, cp), single_grid(cp)
            per = [summarize(ctx, tgt, y, nested_oof_seeded(X, y, ctx.groups, s, make_est=make_est, grid=grid)) for s in SEEDS]
            agg = {}
            for key, get in [("r2log", lambda p: p["r2log"]), ("all_mape", lambda p: p["all"]["mape"]),
                             ("small_mape", lambda p: p["small"]["mape"]), ("small_mdape", lambda p: p["small"]["mdape"]),
                             ("large_mape", lambda p: p["large"]["mape"])]:
                v = [get(p) for p in per]
                agg[key] = {"mean": round(float(np.mean(v)), 4 if key == "r2log" else 1),
                            "std": round(float(np.std(v)), 4 if key == "r2log" else 1)}
            agg["per_seed"] = per
            res[name] = agg
            print(f"[seeds] {tgt} {name}: R2 {agg['r2log']['mean']}±{agg['r2log']['std']}  "
                  f"small MAPE {agg['small_mape']['mean']}±{agg['small_mape']['std']}  large MAPE {agg['large_mape']['mean']}±{agg['large_mape']['std']}", flush=True)
        out[tgt] = res
    return out


# ----------------------------------------------------------------------------- main
SECTIONS = [
    ("baseline", baseline),
    ("gate_analysis", gate_analysis),
    ("log_target", log_target),
    ("weighted_baseline", weighted_baseline),
    ("residual_baseline", residual_baseline),
    ("regime_ensemble", regime_ensemble),
    ("algo_average", algo_average),
    ("per_backend_control", per_backend_control),
    ("log_per_backend", log_per_backend),
    ("log_target_claims", log_target_claims),
    ("log_target_lofo", log_target_lofo),
    ("ladder_check", ladder_check),
    ("seed_stability", seed_stability),
]
NEEDS_EXISTING = {"ladder_check", "seed_stability"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="쉼표로 구분한 절 이름")
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else {n for n, _ in SECTIONS}

    existing = {}
    if os.path.exists(OUT_PATH):
        existing = json.load(open(OUT_PATH, encoding="utf-8"))

    ctx = Ctx()
    print(f"행 {len(ctx.rows)} | 피처 {len(NUMERIC)} | 절: {[n for n, _ in SECTIONS if n in wanted]}\n", flush=True)
    for name, fn in SECTIONS:
        if name not in wanted:
            continue
        t0 = time.time()
        existing[name] = fn(ctx, existing) if name in NEEDS_EXISTING else fn(ctx)
        existing[name]["_elapsed_s"] = round(time.time() - t0, 1)
        existing["_meta"] = {"rows": len(ctx.rows), "numeric_features": len(NUMERIC),
                             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=1)
        print(f"   ↳ {name} 저장 ({existing[name]['_elapsed_s']}s)\n", flush=True)
    print("완료:", OUT_PATH)


if __name__ == "__main__":
    main()
