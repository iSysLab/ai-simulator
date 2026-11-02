"""스마트미디어저널 원고 실험 — XGBoost·GradientBoosting·RandomForest와 세 가지 결합 방법 비교.

비교 대상(기반 학습기는 모두 네 플랫폼을 함께 학습, 타깃 ln y):
  xgb, gb, rf      단일 모델
  xgb_log1p        XGBoost, log1p 타깃(선행 연구 설정, 시간 타깃만)
  select           플랫폼마다 내부 교차검증 오차가 가장 작은 단일 모델 하나를 고름(사람이 고르는 방식)
  average          세 모델의 단순 평균
  global_w         모든 샘플에 같은 결합 비율(비율 ≥ 0, 합 1)
  platform_w       (방법 1) 플랫폼마다 결합 비율을 따로 학습
  tree_gate        (방법 2) 얕은 결정 트리가 조건(플랫폼·계열·규모)으로 샘플을 나누고, 나뉜 구간마다 결합 비율 학습
  meta_stack       (방법 3) 세 예측과 "예측 × 조건"을 입력으로 하는 Ridge 메타 모델(feature-weighted linear stacking)

절차(원고 그림 2): 바깥 GroupKFold(5) × 내부 GroupKFold(4), 둘 다 구성(model_name) 단위, 시드마다 셔플.
결합기는 내부 4겹 OOF 예측으로만 학습하고, 결합기의 하이퍼파라미터도 OOF 위에서 구성 단위 4겹으로 고른다.
바깥 검증 폴드는 하이퍼파라미터·결합기·재학습 어디에도 쓰지 않는다.

사용(저장소 루트에서):
  .venv/bin/python -B paper/smj/run_smj_experiments.py                                 # 시드 10개, 타깃 3종
  .venv/bin/python -B paper/smj/run_smj_experiments.py --seeds 1 --targets avg_infer   # 빠른 확인
산출: paper/smj/results/smj_results.json, paper/smj/results/preds_<타깃>.npz
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from eval_fill_v4k import Ctx, FAM_LABEL  # noqa: E402
from train_merged import NUMERIC  # noqa: E402

OUT_DIR = os.path.join(ROOT, "paper", "smj", "results")
TARGETS = ["avg_train", "avg_infer", "peak_mem"]
SMALL_CUT = {"avg_train": 1.0, "avg_infer": 0.01}     # 소형 구간: 학습 1 s 미만, 추론 10 ms 미만
BASE_NAMES = ["xgb", "gb", "rf"]
LEARNERS = {
    "xgb": (lambda: XGBRegressor(learning_rate=0.1, random_state=42, n_jobs=1, verbosity=0),
            {"n_estimators": [200, 400], "max_depth": [3, 6]}),
    "gb": (lambda: GradientBoostingRegressor(learning_rate=0.1, random_state=42),
           {"n_estimators": [200, 400], "max_depth": [3, 6]}),
    "rf": (lambda: RandomForestRegressor(random_state=42, n_jobs=1),
           {"n_estimators": [100, 200], "max_depth": [None, 12]}),
}
GATE_GRID = [(d, m) for d in (1, 2, 3) for m in (40, 80, 120)]  # 결정 트리 깊이 × 잎 최소 샘플 수
GATE_SHRINK = 30                                                 # 잎의 결합 비율을 전체 비율 쪽으로 당기는 강도(샘플 수 단위)
META_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]               # 메타 모델 Ridge 정규화 강도
METHODS_COMB = ["select", "average", "global_w", "platform_w", "tree_gate", "meta_stack"]
PROPOSED = ["platform_w", "tree_gate", "meta_stack"]


# ----------------------------------------------------------------------------- 데이터
def conditions(d):
    """결합기가 보는 조건: 플랫폼·모델 계열(원-핫), 파라미터 수와 FLOPs(로그)."""
    plats, fams = sorted(set(d["plat"])), sorted(set(d["fam"]))
    cols = [(d["plat"] == p).astype(float) for p in plats] + [(d["fam"] == f).astype(float) for f in fams]
    cols += [d["X"][:, NUMERIC.index("log_total_params")], np.log10(1.0 + d["X"][:, NUMERIC.index("flops")])]
    names = [f"platform={p}" for p in plats] + [f"family={FAM_LABEL[f]}" for f in fams] + ["log_params", "log_flops"]
    return np.column_stack(cols), names


def load_data():
    """학습·추론 시간은 632셀(eval_fill_v4i.load_clean), 메모리는 CUDA·MPS 316셀(프로토콜 v2 peak_mem_mb)."""
    ctx = Ctx()
    X = np.array([[float(e.get(c, 0) or 0) for c in NUMERIC] for e in ctx.rows])
    base = {"X": X, "groups": ctx.groups.astype(str), "plat": ctx.bks.astype(str), "fam": ctx.fams.astype(str)}
    data = {t: dict(base, y=np.array([float(e[t]) for e in ctx.rows])) for t in ("avg_train", "avg_infer")}
    mem = {}
    for bk, name in (("CUDA", "remeasure_cuda_v2.json"), ("MPS", "remeasure_mps_v2.json")):
        for r in json.load(open(os.path.join(ROOT, "results", name))):
            if r.get("peak_mem_mb"):
                mem[(bk, r["model_name"])] = float(r["peak_mem_mb"])
    idx = np.array([i for i, e in enumerate(ctx.rows) if (str(ctx.bks[i]), e["model_name"]) in mem])
    data["peak_mem"] = {k: v[idx] for k, v in base.items()}
    data["peak_mem"]["y"] = np.array([mem[(str(ctx.bks[i]), ctx.rows[i]["model_name"])] for i in idx])
    for d in data.values():
        d["C"], d["C_names"] = conditions(d)
    return data


# ----------------------------------------------------------------------------- 결합기
def simplex_ls(O, z):
    """min ||z − O·w||², w ≥ 0, Σw = 1 — 세 학습기의 결합 비율."""
    k = O.shape[1]
    if len(z) < k:
        return np.full(k, 1.0 / k)
    res = minimize(lambda w: np.sum((z - O @ w) ** 2), np.full(k, 1.0 / k),
                   jac=lambda w: -2.0 * O.T @ (z - O @ w), method="SLSQP", bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}])
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


class Average:
    def fit(self, O, z, C, plat, groups, seed):
        return self

    def predict(self, P, C, plat):
        return P.mean(axis=1)


class GlobalWeights:
    def fit(self, O, z, C, plat, groups, seed):
        self.w_ = simplex_ls(O, z)
        return self

    def predict(self, P, C, plat):
        return P @ self.w_


class PlatformSelect:
    """플랫폼마다 OOF 오차(로그 공간 MSE)가 가장 작은 단일 모델 하나를 고른다."""

    def fit(self, O, z, C, plat, groups, seed):
        mse = lambda m: ((O[m] - z[m, None]) ** 2).mean(axis=0)
        self.choice_ = {p: int(np.argmin(mse(plat == p))) for p in np.unique(plat)}
        self.fallback_ = int(np.argmin(mse(np.ones(len(z), bool))))
        return self

    def predict(self, P, C, plat):
        idx = np.array([self.choice_.get(p, self.fallback_) for p in plat])
        return P[np.arange(len(P)), idx]


class PlatformWeights:
    """(방법 1) 플랫폼마다 결합 비율을 따로 학습한다."""

    def fit(self, O, z, C, plat, groups, seed):
        self.global_ = simplex_ls(O, z)
        self.w_ = {p: simplex_ls(O[plat == p], z[plat == p]) for p in np.unique(plat)}
        return self

    def predict(self, P, C, plat):
        W = np.array([self.w_.get(p, self.global_) for p in plat])
        return np.sum(P * W, axis=1)


class TreeGate:
    """(방법 2) 얕은 결정 트리가 조건 C로 샘플을 나누고, 나뉜 구간(잎)마다 결합 비율을 학습한다.
    트리의 학습 목표는 학습기별 OOF 절대 오차에서 샘플 평균을 뺀 값(어느 학습기가 상대적으로 나은가)이고,
    잎의 비율은 샘플 수에 따라 전체 비율 쪽으로 당긴다(GATE_SHRINK). 깊이·잎 최소 샘플 수는 OOF 위에서
    구성 단위 4겹으로 고른다."""

    def _fit_once(self, O, z, C, depth, min_leaf):
        err = np.abs(O - z[:, None])
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=0)
        tree.fit(C, err - err.mean(axis=1, keepdims=True))
        leaves = tree.apply(C)
        glob = simplex_ls(O, z)
        W = {}
        for lf in np.unique(leaves):
            m = leaves == lf
            W[int(lf)] = (m.sum() * simplex_ls(O[m], z[m]) + GATE_SHRINK * glob) / (m.sum() + GATE_SHRINK)
        return tree, W, glob

    @staticmethod
    def _predict_with(tree, W, glob, P, C):
        Wm = np.array([W.get(int(lf), glob) for lf in tree.apply(C)])
        return np.sum(P * Wm, axis=1)

    def fit(self, O, z, C, plat, groups, seed):
        cv = GroupKFold(4, shuffle=True, random_state=seed)
        best, best_err = None, np.inf
        for depth, min_leaf in GATE_GRID:
            err = 0.0
            for a, b in cv.split(O, z, groups):
                t, W, g = self._fit_once(O[a], z[a], C[a], depth, min_leaf)
                err += np.sum((self._predict_with(t, W, g, O[b], C[b]) - z[b]) ** 2)
            if err < best_err:
                best, best_err = (depth, min_leaf), err
        self.params_ = best
        self.tree_, self.W_, self.glob_ = self._fit_once(O, z, C, *best)
        return self

    def predict(self, P, C, plat):
        return self._predict_with(self.tree_, self.W_, self.glob_, P, C)


class MetaStack:
    """(방법 3) feature-weighted linear stacking: 세 예측의 평균을 기준으로, [평균과의 차이, 차이 × 각 조건]을
    입력으로 잔차를 Ridge(표준화 후)로 학습한다. 결합 비율이 조건의 선형 함수가 되고, 정규화가 강하면
    단순 평균으로 돌아간다. α는 OOF 위에서 구성 단위 4겹으로 고른다."""

    @staticmethod
    def _feats(P, C):
        D = P - P.mean(axis=1, keepdims=True)
        return np.hstack([D] + [D * C[:, [k]] for k in range(C.shape[1])])

    def fit(self, O, z, C, plat, groups, seed):
        gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": META_ALPHAS},
                          cv=GroupKFold(4, shuffle=True, random_state=seed),
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(self._feats(O, C), z - O.mean(axis=1), groups=groups)
        self.model_, self.alpha_ = gs.best_estimator_, float(gs.best_params_["ridge__alpha"])
        return self

    def predict(self, P, C, plat):
        return P.mean(axis=1) + self.model_.predict(self._feats(P, C))


COMBINERS = {"select": PlatformSelect, "average": Average, "global_w": GlobalWeights,
             "platform_w": PlatformWeights, "tree_gate": TreeGate, "meta_stack": MetaStack}


# ----------------------------------------------------------------------------- 기반 학습기
def fit_base(X, z, groups, cv):
    """학습기마다 내부 GridSearchCV → 최적 모델(전체 재학습)과 같은 내부 폴드의 OOF 예측."""
    models, O, params = [], np.zeros((len(z), len(BASE_NAMES))), {}
    for j, b in enumerate(BASE_NAMES):
        mk, grid = LEARNERS[b]
        gs = GridSearchCV(mk(), grid, cv=cv, scoring="r2", n_jobs=1).fit(X, z, groups=groups)
        O[:, j] = cross_val_predict(clone(gs.best_estimator_), X, z, cv=cv, groups=groups, n_jobs=1)
        models.append(gs.best_estimator_)
        params[b] = {k: (None if v is None else int(v)) for k, v in gs.best_params_.items()}
    return models, O, params


def run_seed(tgt, seed, d):
    """시드 하나: 바깥 5겹마다 기반 학습기·결합기를 학습해 모든 샘플의 바깥 폴드 예측(원 단위)을 만든다."""
    X, y, groups, plat, C = d["X"], d["y"], d["groups"], d["plat"], d["C"]
    z = np.log(y)
    names = BASE_NAMES + (["xgb_log1p"] if tgt in SMALL_CUT else []) + METHODS_COMB
    pred = {m: np.zeros(len(y)) for m in names}
    info = {k: [] for k in ("base_params", "select", "global_w", "platform_w", "gate_params", "meta_alpha")}
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        inner = GroupKFold(4, shuffle=True, random_state=seed)
        models, O, params = fit_base(X[tr], z[tr], groups[tr], inner)
        P = np.column_stack([m.predict(X[te]) for m in models])
        for j, b in enumerate(BASE_NAMES):
            pred[b][te] = P[:, j]
        if "xgb_log1p" in pred:
            mk, grid = LEARNERS["xgb"]
            gs = GridSearchCV(mk(), grid, cv=inner, scoring="r2", n_jobs=1)
            gs.fit(X[tr], np.log1p(y[tr]), groups=groups[tr])
            pred["xgb_log1p"][te] = np.log(np.clip(np.expm1(gs.best_estimator_.predict(X[te])), 1e-6, None))
        for m in METHODS_COMB:
            comb = COMBINERS[m]().fit(O, z[tr], C[tr], plat[tr], groups[tr], seed)
            pred[m][te] = comb.predict(P, C[te], plat[te])
            if m == "select":
                info["select"].append({p: BASE_NAMES[i] for p, i in comb.choice_.items()})
            elif m == "global_w":
                info["global_w"].append(np.round(comb.w_, 4).tolist())
            elif m == "platform_w":
                info["platform_w"].append({p: np.round(w, 4).tolist() for p, w in comb.w_.items()})
            elif m == "tree_gate":
                info["gate_params"].append(list(comb.params_))
            elif m == "meta_stack":
                info["meta_alpha"].append(comb.alpha_)
        info["base_params"].append(params)
    return tgt, seed, {m: np.exp(v) for m, v in pred.items()}, info


def full_fit(tgt, d):
    """해석용: 전체 데이터로 기반 학습기 OOF를 만들고 결합기를 한 번 학습해 학습된 규칙을 기록한다."""
    X, y, groups, plat, C = d["X"], d["y"], d["groups"], d["plat"], d["C"]
    z = np.log(y)
    _, O, params = fit_base(X, z, groups, GroupKFold(4, shuffle=True, random_state=0))
    mse = {p: dict(zip(BASE_NAMES, np.round(((O[plat == p] - z[plat == p, None]) ** 2).mean(axis=0), 4).tolist()))
           for p in sorted(set(plat))}
    pw = PlatformWeights().fit(O, z, C, plat, groups, 0)
    tg = TreeGate().fit(O, z, C, plat, groups, 0)
    leaves = tg.tree_.apply(C)
    leaf_info = {}
    for lf in np.unique(leaves):
        m = leaves == lf
        leaf_info[int(lf)] = {"n": int(m.sum()), "weights": dict(zip(BASE_NAMES, np.round(tg.W_[int(lf)], 3).tolist())),
                              "platform": dict(Counter(plat[m].tolist())),
                              "family": {FAM_LABEL[f]: c for f, c in Counter(d["fam"][m].tolist()).items()}}
    ms = MetaStack().fit(O, z, C, plat, groups, 0)
    return {"base_params": params, "oof_mse_by_platform": mse,
            "global_w": dict(zip(BASE_NAMES, np.round(GlobalWeights().fit(O, z, C, plat, groups, 0).w_, 3).tolist())),
            "platform_w": {p: dict(zip(BASE_NAMES, np.round(w, 3).tolist())) for p, w in pw.w_.items()},
            "select": {p: BASE_NAMES[i] for p, i in PlatformSelect().fit(O, z, C, plat, groups, 0).choice_.items()},
            "tree_gate": {"params": list(tg.params_), "leaves": leaf_info,
                          "tree": export_text(tg.tree_, feature_names=d["C_names"], decimals=2)},
            "meta_stack": {"alpha": ms.alpha_}}


# ----------------------------------------------------------------------------- 지표
def metric_block(y, p, plat, fam, tgt):
    ape = np.abs(p - y) / y
    out = {"r2log": float(r2_score(np.log1p(y), np.log1p(p))), "mape": float(ape.mean() * 100),
           "mdape": float(np.median(ape) * 100), "w20": float((ape < 0.2).mean() * 100)}
    if tgt in SMALL_CUT:
        m = y < SMALL_CUT[tgt]
        out["small_mape"], out["small_n"] = float(ape[m].mean() * 100), int(m.sum())
    out["by_platform"] = {q: float(ape[plat == q].mean() * 100) for q in sorted(set(plat))}
    out["by_family"] = {FAM_LABEL[f]: float(ape[fam == f].mean() * 100) for f in sorted(set(fam))}
    out["family_macro"] = float(np.mean(list(out["by_family"].values())))
    return out


def aggregate(blocks):
    """시드별 지표 dict 목록 → 같은 구조의 {mean, sd}."""
    if isinstance(blocks[0], dict):
        return {k: aggregate([b[k] for b in blocks]) for k in blocks[0]}
    a = np.array(blocks, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4)}


def paired(y, preds, groups, m, b):
    """시드별 MAPE 차이(m − b), m이 더 낮았던 시드 수, 구성별 |ln 오차|(시드·플랫폼 평균)의 Wilcoxon 검정."""
    mape = lambda p: np.mean(np.abs(p - y) / y, axis=1) * 100
    diff = mape(preds[m]) - mape(preds[b])
    em = np.abs(np.log(preds[m]) - np.log(y)).mean(axis=0)
    eb = np.abs(np.log(preds[b]) - np.log(y)).mean(axis=0)
    cfg = np.unique(groups)
    cm = np.array([em[groups == g].mean() for g in cfg])
    cb = np.array([eb[groups == g].mean() for g in cfg])
    p = float(wilcoxon(cm, cb).pvalue) if np.any(cm != cb) else None
    return {"delta_mape_mean": round(float(diff.mean()), 3),
            "delta_mape_sd": round(float(diff.std(ddof=1)) if len(diff) > 1 else 0.0, 3),
            "wins": int((diff < 0).sum()), "n_seeds": int(len(diff)), "wilcoxon_p": p,
            "cfg_better": int((cm < cb).sum()), "n_cfg": int(len(cfg))}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()
    targets = args.targets.split(",")
    data = load_data()
    print("샘플 수:", {t: len(data[t]["y"]) for t in targets}, "| 시드", args.seeds, flush=True)
    t0 = time.time()
    tasks = [(t, s) for t in targets for s in range(args.seeds)]
    res = Parallel(n_jobs=args.jobs, verbose=10)(delayed(run_seed)(t, s, data[t]) for t, s in tasks)
    full = dict(zip(targets, Parallel(n_jobs=min(args.jobs, len(targets)))(
        delayed(full_fit)(t, data[t]) for t in targets)))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "smj_results.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out["_meta"] = {"seeds": args.seeds, "elapsed_s": round(time.time() - t0, 1),
                    "learners": {b: LEARNERS[b][1] for b in BASE_NAMES}, "gate_grid": GATE_GRID,
                    "meta_alphas": META_ALPHAS, "conditions": data[targets[0]]["C_names"]}
    for tgt in targets:
        d = data[tgt]
        runs = sorted([r for r in res if r[0] == tgt], key=lambda r: r[1])
        preds = {m: np.array([r[2][m] for r in runs]) for m in runs[0][2]}
        summary = {m: aggregate([metric_block(d["y"], preds[m][s], d["plat"], d["fam"], tgt)
                                 for s in range(len(runs))]) for m in preds}
        best_single = min(BASE_NAMES, key=lambda m: summary[m]["mape"]["mean"])
        comps = {}
        for m in METHODS_COMB:
            comps[f"{m}_vs_{best_single}"] = paired(d["y"], preds, d["groups"], m, best_single)
        for m in PROPOSED:
            for b in ("select", "global_w", "average"):
                comps[f"{m}_vs_{b}"] = paired(d["y"], preds, d["groups"], m, b)
        for a, b in (("tree_gate", "platform_w"), ("meta_stack", "platform_w"), ("tree_gate", "meta_stack")):
            comps[f"{a}_vs_{b}"] = paired(d["y"], preds, d["groups"], a, b)
        if "xgb_log1p" in preds:
            comps["xgb_vs_xgb_log1p"] = paired(d["y"], preds, d["groups"], "xgb", "xgb_log1p")
        out[tgt] = {"n": int(len(d["y"])), "best_single": best_single, "summary": summary, "paired": comps,
                    "fold_info": [r[3] for r in runs], "full_fit": full[tgt]}
        np.savez_compressed(os.path.join(OUT_DIR, f"preds_{tgt}.npz"), y=d["y"], plat=d["plat"], fam=d["fam"],
                            groups=d["groups"], **preds)
        print(f"\n=== {tgt} (n={len(d['y'])}, 최고 단일 모델 {best_single}) ===")
        for m, s in summary.items():
            sm = f" 소형 {s['small_mape']['mean']:6.1f}" if "small_mape" in s else ""
            print(f"  {m:11s} R² {s['r2log']['mean']:.4f}  MAPE {s['mape']['mean']:6.2f}±{s['mape']['sd']:.2f}{sm}"
                  f"  계열평균 {s['family_macro']['mean']:6.1f}  ±20% {s['w20']['mean']:5.1f}")
        for k, v in comps.items():
            print(f"  {k:28s} ΔMAPE {v['delta_mape_mean']:+7.2f}±{v['delta_mape_sd']:.2f}  "
                  f"승 {v['wins']}/{v['n_seeds']}  구성 {v['cfg_better']}/{v['n_cfg']}  p={v['wilcoxon_p']}")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1, default=_json_default)
    print(f"\n저장: {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
