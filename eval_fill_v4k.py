"""심사 대응 재계산(v4k) — RACS 2026 camera-ready, 리뷰어 1 지적별 보충 실험.

eval_fill_v4i.load_clean()의 632셀과 v4j의 nested CV 절차를 그대로 쓰되, 지적마다
새 증거를 계산한다. 설계는 paper/60_racs_camera_ready_plan.md §2·§3.

  backend_values        R1-3   백엔드별 하드웨어 기술자 값 (무엇이 백엔드를 구분하는가)
  base_oof              (공통) 구성 단위 nested CV의 OOF 예측 — 아래 여러 절이 공유
  precision_check       R1-9   R²(log) 4자리 + 부트스트랩 95% CI + RMSE(log)
  measurement_noise     R1-11  반복 표준편차 기반 CV, 잡음 하한 대 예측 오차, 2σ 이내 비율
  errors_by_regime      R1-1   백엔드 × 구간(전체/≥10ms/<10ms/GAN) 오차 표 (MAPE·MdAPE·±20%)
  protocol_ladder       R1-2/4 검증 프로토콜 사다리 7단계 (무작위→구성→폭→깊이→최대사분위→계열→백엔드)
  label_vs_descriptors  R1-5   하드웨어 기술자 30개 vs 백엔드 라벨 1개
  unified_importance    R1-6   통합 XGBoost의 폴드별 홀드아웃 permutation 중요도 (단일·그룹)
  symmetric_ablation    R1-7   메모리 그룹 / FLOPs 그룹 대칭 절제 + 단독 피처 기준선 + 상관
  tuned_comparison      R1-8   네 알고리즘 동일 nested CV·동일 그리드 (GB 튜닝, Ridge)
  overhead_features     R1-1   커널 실행 오버헤드 피처·Huber 목적함수·소형 전문 모델로 <10ms 오차 개선 시도

사용:
  python eval_fill_v4k.py                 # 전체
  python eval_fill_v4k.py --only precision_check,measurement_noise
산출: paper/v4_measured_k.json (절 단위로 병합 저장)
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from eval_fill_v4i import load_clean, onnx_rows_of, xy, BACKENDS, TARGETS, FAMILIES
from eval_fill_v4j import (nested_oof, mk_xgb, mk_rf, XGB_GRID, RF_GRID,
                           HW_NUMERIC_COLS, MEM_GROUP, FLOPS_GROUP, LOFO_COLS,
                           FAMILY_ID_COLS)
from train_merged import NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, "paper/v4_measured_k.json")
SEEDS = list(range(10))
FAM_LABEL = {"simple_ann": "ANN", "simple_cnn": "CNN", "resnet_mnist": "ResNet",
             "mobilenet_mnist": "MobileNet", "transformer": "ViT", "gan": "GAN"}
MNIST_FAMILIES = {"simple_ann", "simple_cnn", "resnet_mnist", "mobilenet_mnist"}


# ----------------------------------------------------------------------------- 공통
def err_stats(y, pred, mask=None):
    """원 단위 오차: MAPE, MdAPE(중앙 APE), ±20% 이내 비율."""
    if mask is None:
        mask = np.ones(len(y), bool)
    if mask.sum() == 0:
        return {"n": 0, "mape": None, "mdape": None, "w20": None}
    pe = np.abs(y[mask] - pred[mask]) / np.clip(np.abs(y[mask]), 1e-9, None)
    return {"n": int(mask.sum()), "mape": round(float(np.mean(pe) * 100), 1),
            "mdape": round(float(np.median(pe) * 100), 1),
            "w20": round(float(np.mean(pe < 0.2) * 100), 1)}


def nested_oof_seeded(X, y, groups, seed, make_est=mk_xgb, grid=XGB_GRID,
                      outer_groups=None, return_models=False):
    """v4j.nested_oof와 같은 절차. 차이: 외부·내부 폴드를 seed로 셔플하고,
    outer_groups를 주면 외부는 그 그룹으로, 내부는 항상 구성(groups) 단위로 묶는다."""
    og = groups if outer_groups is None else outer_groups
    outer = GroupKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    models = []
    for tr, te in outer.split(X, y, og):
        inner = GroupKFold(4, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_est(), grid, cv=inner, scoring="r2", n_jobs=1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = gs.best_estimator_.predict(X[te])
        if return_models:
            models.append((te, gs.best_estimator_))
    return (oof, models) if return_models else oof


def fit_best(Xtr, ytr, gtr, make_est=mk_xgb, grid=XGB_GRID, seed=42):
    gs = GridSearchCV(make_est(), grid, cv=GroupKFold(4, shuffle=True, random_state=seed),
                      scoring="r2", n_jobs=1)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.best_estimator_


def width_depth(e):
    c, t = e["config"], e["model_type"]
    if t == "simple_ann":
        return c["hidden_size"], c["num_layers"]
    if t == "simple_cnn":
        return c["num_filters"], c["num_conv_layers"]
    if t == "resnet_mnist":
        return c["base_width"], sum(c["layers"])
    if t == "mobilenet_mnist":
        return c["width_mult"], c["num_blocks"]
    if t == "transformer":
        return c["embed_dim"], c["num_layers"]
    if t == "gan":
        return max(c["g_hidden_dims"]), len(c["g_hidden_dims"])
    raise KeyError(t)


class Ctx:
    """데이터·라벨·공유 OOF를 한 번만 만든다."""

    def __init__(self):
        self.raw = load_clean()
        self.rows = onnx_rows_of(self.raw)
        self.groups = np.array([e["model_name"] for e in self.rows])
        self.fams = np.array([e["model_type"] for e in self.rows])
        self.bks = np.array([e["_backend"] for e in self.rows])
        self.bid = np.array([e["_backend_id"] for e in self.rows])
        self.is_gan = self.fams == "gan"
        self._base = {}

    def X(self, cols, tgt):
        return xy(self.rows, cols, tgt)

    def base(self, tgt):
        """구성 단위 nested CV OOF (v4j와 동일한 무셔플 GroupKFold, seed 42). 로그 공간."""
        if tgt not in self._base:
            X, y = self.X(NUMERIC, tgt)
            oof, models = self._nested_models(X, y)
            self._base[tgt] = (y, oof, models)
        return self._base[tgt]

    def _nested_models(self, X, y):
        outer = GroupKFold(5)
        oof = np.zeros(len(y))
        models = []
        for tr, te in outer.split(X, y, self.groups):
            gs = GridSearchCV(mk_xgb(), XGB_GRID, cv=GroupKFold(4), scoring="r2", n_jobs=1)
            gs.fit(X[tr], y[tr], groups=self.groups[tr])
            oof[te] = gs.best_estimator_.predict(X[te])
            models.append((te, gs.best_estimator_))
        return oof, models


# ----------------------------------------------------------------------------- R1-3
def backend_values(ctx):
    """하드웨어 수치 30개의 백엔드별 값. 백엔드 안에서 상수인지도 확인."""
    out = {"columns": HW_NUMERIC_COLS, "per_backend": {}, "non_constant_within_backend": {}}
    for b in BACKENDS:
        sub = [e for e in ctx.rows if e["_backend"] == b]
        vals, nonconst = {}, []
        for c in HW_NUMERIC_COLS:
            v = sorted({float(e.get(c, 0) or 0) for e in sub})
            vals[c] = v[0] if len(v) == 1 else v
            if len(v) > 1:
                nonconst.append(c)
        out["per_backend"][b] = vals
        out["non_constant_within_backend"][b] = nonconst
    # 백엔드 쌍을 구분하는 피처
    pairs = {}
    for i, a in enumerate(BACKENDS):
        for b in BACKENDS[i + 1:]:
            va, vb = out["per_backend"][a], out["per_backend"][b]
            pairs[f"{a} vs {b}"] = [c for c in HW_NUMERIC_COLS if va[c] != vb[c]]
    out["distinguishing_features"] = pairs
    print("[R1-3] 백엔드 구분 피처:", {k: len(v) for k, v in pairs.items()})
    print("   Mac CPU vs MPS:", pairs["Mac CPU vs MPS"])
    print("   Desktop CPU vs Mac CPU:", pairs["Desktop CPU vs Mac CPU"])
    return out


# ----------------------------------------------------------------------------- R1-9
def precision_check(ctx, n_boot=1000, seed=42):
    """전체·백엔드별 R²(log) 4자리, 부트스트랩 95% CI, RMSE(log).
    리뷰어가 본 표 5·6은 무작위 분할이므로 무작위·구성 단위 둘 다 낸다."""
    rng = np.random.default_rng(seed)

    def table(y, oof):
        res = {}
        for label, mask in [("all", np.ones(len(y), bool))] + [(b, ctx.bks == b) for b in BACKENDS]:
            yy, pp = y[mask], oof[mask]
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(yy), len(yy))
                boots.append(r2_score(yy[idx], pp[idx]))
            res[label] = {"n": int(mask.sum()),
                          "r2log": round(float(r2_score(yy, pp)), 4),
                          "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                   round(float(np.percentile(boots, 97.5)), 4)],
                          "rmse_log": round(float(np.sqrt(np.mean((yy - pp) ** 2))), 4),
                          "var_log_y": round(float(np.var(yy)), 4)}
        return res

    out = {"random_split": {}, "config_split": {}}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        oof_r = nested_oof(X, y, mk_xgb, XGB_GRID, None, seed=42)
        out["random_split"][tgt] = table(y, oof_r)
        _, oof_c, _ = ctx.base(tgt)
        out["config_split"][tgt] = table(y, oof_c)
        for k in ("random_split", "config_split"):
            print(f"[R1-9] {k:12s} {tgt}: " + ", ".join(f"{b} {v['r2log']}" for b, v in out[k][tgt].items()))
    return out


# ----------------------------------------------------------------------------- R1-11
def measurement_noise(ctx):
    """반복 표준편차(10회, ddof=0; Mac CPU 병합 셀은 풀링)로 측정 잡음을 정량화하고
    예측 오차와 견준다."""
    out = {"note": "cv = std/avg of the 10 repeats per cell; Mac CPU merged cells use pooled std"}
    for tgt in TARGETS:
        y_log, oof, _ = ctx.base(tgt)
        y, pred = np.expm1(y_log), np.expm1(oof)
        std = np.array([float(e.get("std_" + tgt[4:], 0) or 0) for e in ctx.rows])
        cv = std / np.clip(y, 1e-12, None)
        ape = np.abs(pred - y) / np.clip(y, 1e-12, None)
        within = np.abs(pred - y) <= 2 * std
        res = {}
        masks = [("all", np.ones(len(y), bool))] + [(b, ctx.bks == b) for b in BACKENDS]
        if tgt == "avg_infer":
            masks += [("lt10ms", y < 0.01), ("ge10ms", y >= 0.01)]
            for b in BACKENDS:
                masks.append((f"{b} lt10ms", (ctx.bks == b) & (y < 0.01)))
        for label, m in masks:
            if m.sum() == 0:
                continue
            res[label] = {"n": int(m.sum()),
                          "cv_median_pct": round(float(np.median(cv[m]) * 100), 1),
                          "cv_p90_pct": round(float(np.percentile(cv[m], 90) * 100), 1),
                          "cv_max_pct": round(float(cv[m].max() * 100), 1),
                          "ape_median_pct": round(float(np.median(ape[m]) * 100), 1),
                          "ape_over_cv_median": round(float(np.median(ape[m] / np.clip(cv[m], 1e-4, None))), 1),
                          "within_2sigma_pct": round(float(within[m].mean() * 100), 1)}
        out[tgt] = res
        print(f"[R1-11] {tgt} all: cv med {res['all']['cv_median_pct']}%  ape med {res['all']['ape_median_pct']}%  "
              f"within2σ {res['all']['within_2sigma_pct']}%")
        if tgt == "avg_infer":
            print(f"        <10ms: cv med {res['lt10ms']['cv_median_pct']}%  ape med {res['lt10ms']['ape_median_pct']}%  "
                  f"(ape/cv 중앙 {res['lt10ms']['ape_over_cv_median']}배)")
    return out


# ----------------------------------------------------------------------------- R1-1
def errors_by_regime(ctx):
    """백엔드 × 구간 오차 표. 구성 단위 nested CV OOF 기준."""
    out = {}
    for tgt in TARGETS:
        y_log, oof, _ = ctx.base(tgt)
        y, pred = np.expm1(y_log), np.expm1(oof)
        std = np.array([float(e.get("std_" + tgt[4:], 0) or 0) for e in ctx.rows])
        cv = std / np.clip(y, 1e-12, None)
        regimes = [("all", np.ones(len(y), bool)), ("ge10ms", y >= 0.01),
                   ("lt10ms", y < 0.01), ("gan", ctx.is_gan), ("nongan_ge10ms", (~ctx.is_gan) & (y >= 0.01))]
        table = {}
        for b in ["all"] + BACKENDS:
            bm = np.ones(len(y), bool) if b == "all" else ctx.bks == b
            table[b] = {}
            for rl, rm in regimes:
                m = bm & rm
                s = err_stats(y, pred, m)
                s["r2log"] = round(float(r2_score(y_log[m], oof[m])), 3) if m.sum() >= 3 else None
                s["noise_cv_median_pct"] = round(float(np.median(cv[m]) * 100), 1) if m.sum() else None
                table[b][rl] = s
        out[tgt] = table
        print(f"[R1-1] {tgt}: all MAPE {table['all']['all']['mape']}%  ≥10ms {table['all']['ge10ms']['mape']}%  "
              f"<10ms {table['all']['lt10ms']['mape']}% (n={table['all']['lt10ms']['n']})")
    return out


# ----------------------------------------------------------------------------- R1-2 / R1-4
def protocol_ladder(ctx):
    """검증 프로토콜 사다리. 각 단계는 R²(log)·MAPE·±20%를 같은 형식으로 보고한다."""
    out = {}
    wd = np.array([width_depth(e) for e in ctx.rows], dtype=object)
    width_g = np.array([f"{f}|w={w}" for f, (w, _) in zip(ctx.fams, wd)])
    depth_g = np.array([f"{f}|d={d}" for f, (_, d) in zip(ctx.fams, wd)])
    params = np.array([float(e.get("total_params", 0) or 0) for e in ctx.rows])

    def summarize(y_log, oof, tgt, mask=None):
        m = np.ones(len(y_log), bool) if mask is None else mask
        y, p = np.expm1(y_log[m]), np.expm1(oof[m])
        s = {"r2log": round(float(r2_score(y_log[m], oof[m])), 4)}
        s.update(err_stats(y, p, (y >= 0.01) if tgt == "avg_infer" else None))
        return s

    # 1·2) 무작위 / 구성 단위 — 시드 10개
    for label, kind in [("L1_random", "random"), ("L2_config", "config")]:
        res = {}
        for tgt in TARGETS:
            X, y = ctx.X(NUMERIC, tgt)
            per_seed = []
            for s in SEEDS:
                if kind == "random":
                    oof = nested_oof(X, y, mk_xgb, XGB_GRID, None, seed=s)
                else:
                    oof = nested_oof_seeded(X, y, ctx.groups, s)
                per_seed.append(summarize(y, oof, tgt))
            r2s = [p["r2log"] for p in per_seed]
            mapes = [p["mape"] for p in per_seed]
            res[tgt] = {"r2log_mean": round(float(np.mean(r2s)), 4), "r2log_std": round(float(np.std(r2s)), 4),
                        "r2log_min": min(r2s), "r2log_max": max(r2s),
                        "mape_mean": round(float(np.mean(mapes)), 1), "mape_std": round(float(np.std(mapes)), 1),
                        "w20_mean": round(float(np.mean([p["w20"] for p in per_seed])), 1),
                        "per_seed": per_seed}
            print(f"[R1-2] {label} {tgt}: R2 {res[tgt]['r2log_mean']}±{res[tgt]['r2log_std']}  "
                  f"MAPE {res[tgt]['mape_mean']}±{res[tgt]['mape_std']}", flush=True)
        out[label] = res
    # 짝지은 차이 (구성 − 무작위), 시드별
    out["L2_minus_L1"] = {}
    for tgt in TARGETS:
        d = [c["r2log"] - r["r2log"] for c, r in zip(out["L2_config"][tgt]["per_seed"], out["L1_random"][tgt]["per_seed"])]
        out["L2_minus_L1"][tgt] = {"mean": round(float(np.mean(d)), 4), "std": round(float(np.std(d)), 4),
                                   "n_positive": int(sum(1 for x in d if x > 0))}

    # 3·4) 폭 수준 / 깊이 수준 GroupKFold (외부 그룹 = 계열|폭, 내부는 구성 단위)
    for label, og in [("L3_width_level", width_g), ("L4_depth_level", depth_g)]:
        res = {"n_groups": int(len(set(og)))}
        for tgt in TARGETS:
            X, y = ctx.X(NUMERIC, tgt)
            oof = nested_oof_seeded(X, y, ctx.groups, 42, outer_groups=og)
            res[tgt] = summarize(y, oof, tgt)
            res[tgt]["per_family_r2log"] = {FAM_LABEL[f]: round(float(r2_score(y[ctx.fams == f], oof[ctx.fams == f])), 3)
                                            for f in FAMILIES}
            print(f"[R1-2] {label} {tgt}: {res[tgt]['r2log']}  MAPE {res[tgt]['mape']}", flush=True)
        out[label] = res

    # 5) 계열별 최대 사분위(파라미터 수 상위 25%) 홀드아웃 — 단일 분할
    top = np.zeros(len(ctx.rows), bool)
    for f in FAMILIES:
        idx = np.where(ctx.fams == f)[0]
        cfg_params = {}
        for i in idx:
            cfg_params[ctx.groups[i]] = params[i]
        names = sorted(cfg_params, key=lambda n: cfg_params[n])
        k = max(1, int(round(len(names) * 0.25)))
        held = set(names[-k:])
        for i in idx:
            if ctx.groups[i] in held:
                top[i] = True
    res = {"n_test": int(top.sum()), "n_train": int((~top).sum())}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        est = fit_best(X[~top], y[~top], ctx.groups[~top])
        pred = est.predict(X[top])
        full = np.zeros(len(y)); full[top] = pred
        res[tgt] = summarize(y, full, tgt, top)
        res[tgt]["per_family_r2log"] = {FAM_LABEL[f]: round(float(r2_score(y[top & (ctx.fams == f)], full[top & (ctx.fams == f)])), 3)
                                        for f in FAMILIES if (top & (ctx.fams == f)).sum() >= 3}
        # 과소/과대 추정 방향
        res[tgt]["median_pred_over_true"] = round(float(np.median(np.expm1(pred) / np.expm1(y[top]))), 3)
        print(f"[R1-2] L5_top_quartile {tgt}: {res[tgt]['r2log']}  MAPE {res[tgt]['mape']}  "
              f"pred/true 중앙 {res[tgt]['median_pred_over_true']}", flush=True)
    out["L5_top_quartile"] = res

    # 6) 계열 제외 (LOFO, 계열 식별자 제외) / 7) 백엔드 제외 (LOBO) — 내부 튜닝 포함
    for label, og, cols in [("L6_family_out", ctx.fams, LOFO_COLS), ("L7_backend_out", ctx.bks, NUMERIC)]:
        res = {}
        for tgt in TARGETS:
            X, y = ctx.X(cols, tgt)
            oof = np.zeros(len(y))
            per = {}
            for tr, te in LeaveOneGroupOut().split(X, y, og):
                est = fit_best(X[tr], y[tr], ctx.groups[tr])
                oof[te] = est.predict(X[te])
                g = og[te][0]
                s = summarize(y, oof, tgt, te)
                per[FAM_LABEL.get(g, g)] = s
            pooled = summarize(y, oof, tgt)
            pooled["per_group"] = per
            pooled["per_group_r2log_min"] = min(v["r2log"] for v in per.values())
            pooled["per_group_r2log_max"] = max(v["r2log"] for v in per.values())
            res[tgt] = pooled
            print(f"[R1-2] {label} {tgt}: pooled {pooled['r2log']}  MAPE {pooled['mape']}  "
                  f"per-group R2 {pooled['per_group_r2log_min']}~{pooled['per_group_r2log_max']}", flush=True)
        out[label] = res
    return out


# ----------------------------------------------------------------------------- R1-5
def label_vs_descriptors(ctx):
    """하드웨어 기술자 30개 vs 백엔드 라벨. 구성 단위 nested CV."""
    no_hw = [c for c in NUMERIC if c not in HW_NUMERIC_COLS]
    conds = {
        "a_full_120": NUMERIC,
        "b_no_hw_90": no_hw,
        "c_no_hw_plus_backend_id": no_hw + ["_backend_id"],
        "d_no_hw_plus_device_type_only": no_hw + ["device_type_encoded"],
        "e_no_hw_plus_backend_onehot": no_hw + ["_bk0", "_bk1", "_bk2", "_bk3"],
    }
    for e in ctx.rows:
        for i in range(4):
            e[f"_bk{i}"] = 1 if e["_backend_id"] == i else 0
    out = {}
    for label, cols in conds.items():
        out[label] = {"n_features": len(cols)}
        for tgt in TARGETS:
            X, y = ctx.X(cols, tgt)
            oof = nested_oof(X, y, mk_xgb, XGB_GRID, ctx.groups)
            yy, pp = np.expm1(y), np.expm1(oof)
            out[label][tgt] = {"r2log": round(float(r2_score(y, oof)), 4),
                               **err_stats(yy, pp, (yy >= 0.01) if tgt == "avg_infer" else None)}
        print(f"[R1-5] {label:32s} train {out[label]['avg_train']['r2log']}  infer {out[label]['avg_infer']['r2log']}", flush=True)

    # LOBO 기준선: 다른 세 백엔드 OOF 예측 평균 / 같은 기기 종류의 최근접 백엔드 예측
    nearest = {"Desktop CPU": "Mac CPU", "Mac CPU": "Desktop CPU", "CUDA": "MPS", "MPS": "CUDA"}
    lobo = {}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        lobo[tgt] = {}
        for b in BACKENDS:
            te, tr = ctx.bks == b, ctx.bks != b
            est = fit_best(X[tr], y[tr], ctx.groups[tr])
            pred_desc = est.predict(X[te])
            # 기준선 1: 학습 세 백엔드의 같은 구성 실측 평균(로그)
            names_te = ctx.groups[te]
            by_name = defaultdict(list)
            for n, yy, bb in zip(ctx.groups, y, ctx.bks):
                if bb != b:
                    by_name[n].append(yy)
            base_mean = np.array([np.mean(by_name[n]) for n in names_te])
            # 기준선 2: 최근접 백엔드의 같은 구성 실측
            nb = nearest[b]
            near = {n: yy for n, yy, bb in zip(ctx.groups, y, ctx.bks) if bb == nb}
            base_near = np.array([near[n] for n in names_te])
            yt = y[te]
            lobo[tgt][b] = {
                "descriptor_model": {"r2log": round(float(r2_score(yt, pred_desc)), 3),
                                     **err_stats(np.expm1(yt), np.expm1(pred_desc))},
                "baseline_mean_of_others": {"r2log": round(float(r2_score(yt, base_mean)), 3),
                                            **err_stats(np.expm1(yt), np.expm1(base_mean))},
                f"baseline_nearest({nb})": {"r2log": round(float(r2_score(yt, base_near)), 3),
                                            **err_stats(np.expm1(yt), np.expm1(base_near))},
            }
            print(f"[R1-5] LOBO {b:11s} {tgt}: model {lobo[tgt][b]['descriptor_model']['r2log']}  "
                  f"mean-others {lobo[tgt][b]['baseline_mean_of_others']['r2log']}  "
                  f"nearest {lobo[tgt][b][f'baseline_nearest({nb})']['r2log']}", flush=True)
    out["lobo_vs_baselines"] = lobo
    return out


# ----------------------------------------------------------------------------- R1-6
def unified_importance(ctx, n_rep=20, seed=42):
    """통합 XGBoost(구성 단위 nested CV의 각 외부 폴드 best 모델)의 홀드아웃 permutation 중요도."""
    rng = np.random.default_rng(seed)
    singles = ["total_op_memory_write", "total_op_memory_read", "flops", "total_params", "model_family_encoded"]
    groups_def = {"MEM_GROUP": MEM_GROUP, "FLOPS_GROUP": FLOPS_GROUP, "HW_30": HW_NUMERIC_COLS,
                  "FAMILY_ID": [c for c in FAMILY_ID_COLS if c in NUMERIC]}
    idx = {c: NUMERIC.index(c) for c in singles}
    gidx = {g: [NUMERIC.index(c) for c in cols if c in NUMERIC] for g, cols in groups_def.items()}
    out = {"n_repeats": n_rep, "protocol": "config-wise nested CV, per-fold holdout permutation, mean±std over 5 folds"}
    for tgt in TARGETS:
        y, _, models = ctx.base(tgt)
        X, _ = ctx.X(NUMERIC, tgt)
        drops = defaultdict(list)
        gain_rank = defaultdict(list)
        for te, est in models:
            Xt, yt = X[te], y[te]
            base = r2_score(yt, est.predict(Xt))
            for name, cols in list(idx.items()) + list(gidx.items()):
                cols = [cols] if isinstance(cols, int) else cols
                d = []
                for _ in range(n_rep):
                    Xp = Xt.copy()
                    perm = rng.permutation(len(te))
                    Xp[:, cols] = Xt[perm][:, cols]
                    d.append(base - r2_score(yt, est.predict(Xp)))
                drops[name].append(float(np.mean(d)))
            imp = est.feature_importances_
            order = np.argsort(-imp)
            for c in singles:
                gain_rank[c].append(int(np.where(order == idx[c])[0][0]) + 1)
            gain_rank["_top5"].append([NUMERIC[i] for i in order[:5]])
        res = {"permutation_drop": {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                                    for k, v in drops.items()},
               "gain_rank": {k: v for k, v in gain_rank.items()},
               "group_sizes": {g: len(v) for g, v in gidx.items()}}
        out[tgt] = res
        pd_ = res["permutation_drop"]
        print(f"[R1-6] {tgt}: Δ mem_write {pd_['total_op_memory_write']['mean']}  flops {pd_['flops']['mean']}  "
              f"MEM {pd_['MEM_GROUP']['mean']}  FLOPS {pd_['FLOPS_GROUP']['mean']}  HW {pd_['HW_30']['mean']}  "
              f"| gain rank mem_write {gain_rank['total_op_memory_write']} flops {gain_rank['flops']}", flush=True)
    return out


# ----------------------------------------------------------------------------- R1-7
def symmetric_ablation(ctx):
    """대칭 그룹 절제 + 단독 피처 기준선 + 상관."""
    conds = {
        "full": NUMERIC,
        "minus_MEM": [c for c in NUMERIC if c not in MEM_GROUP],
        "minus_FLOPS": [c for c in NUMERIC if c not in FLOPS_GROUP],
        "minus_both": [c for c in NUMERIC if c not in MEM_GROUP and c not in FLOPS_GROUP],
    }
    singles = {
        "only_mem_write": ["total_op_memory_write"],
        "only_flops": ["flops"],
        "only_params": ["total_params"],
        "mem_write+backend_id": ["total_op_memory_write", "_backend_id"],
        "flops+backend_id": ["flops", "_backend_id"],
        "params+backend_id": ["total_params", "_backend_id"],
        "mem_write+flops+backend_id": ["total_op_memory_write", "flops", "_backend_id"],
    }
    out = {"ablation": {}, "single_feature": {}, "group_sizes": {"MEM": len([c for c in MEM_GROUP if c in NUMERIC]),
                                                                    "FLOPS": len([c for c in FLOPS_GROUP if c in NUMERIC])}}
    for section, table in [("ablation", conds), ("single_feature", singles)]:
        for label, cols in table.items():
            out[section][label] = {"n_features": len(cols)}
            for tgt in TARGETS:
                X, y = ctx.X(cols, tgt)
                oof = nested_oof(X, y, mk_xgb, XGB_GRID, ctx.groups)
                yy, pp = np.expm1(y), np.expm1(oof)
                out[section][label][tgt] = {"r2log": round(float(r2_score(y, oof)), 4),
                                            **err_stats(yy, pp, (yy >= 0.01) if tgt == "avg_infer" else None)}
            print(f"[R1-7] {section:14s} {label:28s} train {out[section][label]['avg_train']['r2log']}  "
                  f"infer {out[section][label]['avg_infer']['r2log']}", flush=True)
    # 상관 (로그 공간 Spearman): 전체·계열별
    mw = np.log1p(np.array([float(e.get("total_op_memory_write", 0) or 0) for e in ctx.rows]))
    fl = np.log1p(np.array([float(e.get("flops", 0) or 0) for e in ctx.rows]))
    tp = np.log1p(np.array([float(e.get("total_params", 0) or 0) for e in ctx.rows]))
    corr = {"all": {"mem_write_vs_flops": round(float(spearmanr(mw, fl).correlation), 3),
                    "mem_write_vs_params": round(float(spearmanr(mw, tp).correlation), 3),
                    "flops_vs_params": round(float(spearmanr(fl, tp).correlation), 3)}}
    for f in FAMILIES:
        m = ctx.fams == f
        corr[FAM_LABEL[f]] = {"mem_write_vs_flops": round(float(spearmanr(mw[m], fl[m]).correlation), 3)}
    # 타깃과의 상관(백엔드 내부)
    tcorr = {}
    for tgt in TARGETS:
        _, y = ctx.X(NUMERIC, tgt)
        tcorr[tgt] = {b: {"mem_write": round(float(spearmanr(mw[ctx.bks == b], y[ctx.bks == b]).correlation), 3),
                          "flops": round(float(spearmanr(fl[ctx.bks == b], y[ctx.bks == b]).correlation), 3),
                          "params": round(float(spearmanr(tp[ctx.bks == b], y[ctx.bks == b]).correlation), 3)}
                      for b in BACKENDS}
    out["spearman_features"] = corr
    out["spearman_with_target_within_backend"] = tcorr
    print(f"[R1-7] Spearman(mem_write, flops) all {corr['all']['mem_write_vs_flops']}; "
          f"per family {[(k, v['mem_write_vs_flops']) for k, v in corr.items() if k != 'all']}")
    return out


# ----------------------------------------------------------------------------- R1-8
def tuned_comparison(ctx):
    """표 5 재산출: 네 알고리즘을 같은 무작위 5-겹 nested CV로, 트리 앙상블은 같은 크기 그리드로."""
    GB_GRID = {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.1]}
    RIDGE_GRID = {"ridge__alpha": [0.1, 1.0, 10.0]}
    defs = {
        "LinearRegression(default)": (lambda: LinearRegression(), {}),
        "Ridge(scaled, α∈{0.1,1,10})": (lambda: make_pipeline(StandardScaler(), Ridge()), RIDGE_GRID),
        "RandomForest(grid)": (mk_rf, RF_GRID),
        "GradientBoosting(default)": (lambda: GradientBoostingRegressor(random_state=42), {}),
        "GradientBoosting(grid=XGB)": (lambda: GradientBoostingRegressor(random_state=42), GB_GRID),
        "XGBoost(grid)": (mk_xgb, XGB_GRID),
    }
    out = {"protocol": "outer 5-fold / inner 4-fold nested CV, seed 42, identical folds for all models; "
                       "random_split = KFold (원고 표 5 기준), config_split = GroupKFold(구성)"}
    for split in ("random_split", "config_split"):
        out[split] = {}
        g = None if split == "random_split" else ctx.groups
        for name, (mk, grid) in defs.items():
            out[split][name] = {}
            for tgt in TARGETS:
                X, y = ctx.X(NUMERIC, tgt)
                if grid:
                    oof = nested_oof(X, y, mk, grid, g, seed=42)
                else:
                    oof = np.zeros(len(y))
                    cv = KFold(5, shuffle=True, random_state=42) if g is None else GroupKFold(5)
                    for tr, te in cv.split(X, y, g):
                        oof[te] = mk().fit(X[tr], y[tr]).predict(X[te])
                yy, pp = np.expm1(y), np.expm1(oof)
                out[split][name][tgt] = {"r2log": round(float(r2_score(y, oof)), 4),
                                         **err_stats(yy, pp, (yy >= 0.01) if tgt == "avg_infer" else None)}
            print(f"[R1-8] {split:13s} {name:30s} train {out[split][name]['avg_train']['r2log']}  "
                  f"infer {out[split][name]['avg_infer']['r2log']}", flush=True)
    return out


# ----------------------------------------------------------------------------- R1-1 (b)
def overhead_features(ctx):
    """<10ms 구간 오차 개선 시도: (a) 커널 실행 오버헤드 피처, (b) Huber 목적함수, (c) 소형 전문 모델."""
    for e in ctx.rows:
        n_ops = float(e.get("num_ops", 0) or 0)
        nb_tr = 938.0 if e["model_type"] in MNIST_FAMILIES else 782.0
        e["_n_batches_train"] = nb_tr
        e["_ops_x_batches_train"] = n_ops * nb_tr
        e["_ops_x_batches_infer"] = n_ops * 10.0
        e["_log_num_ops"] = float(np.log1p(n_ops))
        e["_log_ops_x_batches_train"] = float(np.log1p(n_ops * nb_tr))
    OVH = ["_n_batches_train", "_ops_x_batches_train", "_ops_x_batches_infer", "_log_num_ops", "_log_ops_x_batches_train"]

    def evaluate(cols, tgt, make_est=mk_xgb, grid=XGB_GRID, subset=None):
        X, y = ctx.X(cols, tgt)
        if subset is None:
            oof = nested_oof(X, y, make_est, grid, ctx.groups)
            m_all = np.ones(len(y), bool)
        else:
            oof = np.full(len(y), np.nan)
            Xs, ys, gs = X[subset], y[subset], ctx.groups[subset]
            oof[subset] = nested_oof(Xs, ys, make_est, grid, gs)
            m_all = subset
        yy, pp = np.expm1(y), np.expm1(oof)
        res = {"r2log": round(float(r2_score(y[m_all], oof[m_all])), 4),
               "all": err_stats(yy, pp, m_all),
               "lt10ms": err_stats(yy, pp, m_all & (yy < 0.01)),
               "ge10ms": err_stats(yy, pp, m_all & (yy >= 0.01))}
        if tgt == "avg_train":
            res["lt1s"] = err_stats(yy, pp, m_all & (yy < 1.0))
        return res

    mk_huber = lambda: XGBRegressor(random_state=42, n_jobs=1, verbosity=0, objective="reg:pseudohubererror")
    out = {"overhead_columns": OVH}
    for tgt in TARGETS:
        res = {}
        res["baseline_120"] = evaluate(NUMERIC, tgt)
        res["plus_overhead_125"] = evaluate(NUMERIC + OVH, tgt)
        res["baseline_huber"] = evaluate(NUMERIC, tgt, mk_huber)
        res["plus_overhead_huber"] = evaluate(NUMERIC + OVH, tgt, mk_huber)
        # 소형 전문 모델: 추론 <50ms 셀만으로 학습·검증 (구성 단위)
        _, y = ctx.X(NUMERIC, tgt)
        small = np.expm1(y) < (0.05 if tgt == "avg_infer" else 1.0)
        res["specialist_small"] = {"n_cells": int(small.sum()), **evaluate(NUMERIC + OVH, tgt, subset=small)}
        out[tgt] = res
        key = "lt10ms" if tgt == "avg_infer" else "lt1s"
        print(f"[R1-1b] {tgt}: {key} MAPE base {res['baseline_120'][key]['mape']} → +ovh {res['plus_overhead_125'][key]['mape']} "
              f"→ huber {res['baseline_huber'][key]['mape']} → +ovh+huber {res['plus_overhead_huber'][key]['mape']} "
              f"→ specialist {res['specialist_small'][key]['mape']}  | R2 base {res['baseline_120']['r2log']} +ovh {res['plus_overhead_125']['r2log']}", flush=True)
    return out


# ----------------------------------------------------------------------------- main
SECTIONS = [
    ("backend_values", backend_values),
    ("precision_check", precision_check),
    ("measurement_noise", measurement_noise),
    ("errors_by_regime", errors_by_regime),
    ("protocol_ladder", protocol_ladder),
    ("label_vs_descriptors", label_vs_descriptors),
    ("unified_importance", unified_importance),
    ("symmetric_ablation", symmetric_ablation),
    ("tuned_comparison", tuned_comparison),
    ("overhead_features", overhead_features),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="쉼표로 구분한 절 이름")
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else {n for n, _ in SECTIONS}

    existing = {}
    if os.path.exists(OUT_PATH):
        existing = json.load(open(OUT_PATH, encoding="utf-8"))

    ctx = Ctx()
    print(f"행 {len(ctx.rows)} | 피처 {len(NUMERIC)} | 절: {sorted(wanted)}\n", flush=True)
    for name, fn in SECTIONS:
        if name not in wanted:
            continue
        t0 = time.time()
        existing[name] = fn(ctx)
        existing[name]["_elapsed_s"] = round(time.time() - t0, 1)
        existing["_meta"] = {"rows": len(ctx.rows), "numeric_features": len(NUMERIC),
                             "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=1)
        print(f"   ↳ {name} 저장 ({existing[name]['_elapsed_s']}s)\n", flush=True)
    print("완료:", OUT_PATH)


if __name__ == "__main__":
    main()
