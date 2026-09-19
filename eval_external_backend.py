"""외부(5번째) 백엔드 평가 — 4백엔드로 학습한 모델이 미지 하드웨어를 얼마나 맞히는가.

절차: paper/72_external_backend_runbook.md. 배경: 리뷰어 R1-5·R2(미지 하드웨어 일반화).
632셀에서 하드웨어 기술자는 백엔드마다 상수라 라벨과 정보량이 같았다(v4k (c)행). 같은 장치 종류
(예: CUDA)에 다른 GPU가 들어오면 이 동치가 처음 깨진다 — 기술자가 전이 정보를 담는지 보는 첫 실험.

사용:
  # 외부 기기 측정(run_benchmark v2 출력) + 환경 기록으로 평가
  python eval_external_backend.py --external results/external_<host>.json --env paper/env_<host>.json --name <host>
  # 코드 점검: 기존 백엔드 하나를 외부로 간주 (학습에서 빼고 예측)
  python eval_external_backend.py --simulate MPS
옵션:
  --transform log|log1p   학습 타깃 (기본 log — paper/71_log_target_findings.md)
  --no-refresh-hw         4백엔드 하드웨어 열을 env_desktop/env_mac.json으로 갱신하지 않음
                          (기본은 갱신: v1 데이터는 tflops·gpu_cores 등이 비어 있어 갱신 없이는 새 GPU를 구별 못 한다)
산출: paper/external_<name>.json, 콘솔 표
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from eval_fill_v4e import config_key
from eval_fill_v4i import load_clean, onnx_rows_of, xy, TARGETS, FAMILIES
from eval_fill_v4j import XGB_GRID, HW_NUMERIC_COLS
from eval_fill_v4k import err_stats, FAM_LABEL
from train_merged import NUMERIC
from benchmark.predictor import LogTargetXGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_OF_BACKEND = {"Desktop CPU": ("paper/env_desktop.json", "cpu"), "CUDA": ("paper/env_desktop.json", "cuda"),
                  "Mac CPU": ("paper/env_mac.json", "cpu"), "MPS": ("paper/env_mac.json", "mps")}
# 수집기(v2 이름) → 스키마 NUMERIC(v3 이름) 별칭
HW_ALIAS = {"gpu_cores": "gpu_core_count", "cpu_cores": "cpu_cores_physical", "cpu_freq_ghz": "cpu_freq_boost_ghz"}
DEVICE_OF = {"cpu": "CPU", "cuda": "GPU(CUDA)", "mps": "GPU(MPS)"}
SMALL_CUT = {"avg_infer": 0.01, "avg_train": 1.0}


def hw_from_env(env_path, key):
    """env_<host>.json의 hardware[key] → NUMERIC 하드웨어 열 dict."""
    hw = json.load(open(os.path.join(BASE, env_path) if not os.path.isabs(env_path) else env_path))["hardware"][key]
    out = {c: hw[c] for c in HW_NUMERIC_COLS if c in hw}
    for schema_name, coll_name in HW_ALIAS.items():
        if coll_name in hw:
            out[schema_name] = hw[coll_name]
    out["device"] = DEVICE_OF[key]
    return out


def apply_hw(rows, hw):
    for r in rows:
        r.update(hw)
    return rows


def structural_templates(raw):
    """구성 키 → 구조·입력 메타(하드웨어·측정 필드 제외). load_clean의 스키마 정합과 같은 목적."""
    skip = set(HW_NUMERIC_COLS) | set(TARGETS) | {"device", "std_train", "std_infer", "train_times", "infer_times",
                                                   "peak_mem_mb", "final_accuracy", "avg_accuracy", "_backend", "_src", "_n_merged"}
    tpl = {}
    for r in raw:
        k = config_key(r["model_type"], r.get("config", {}) or {})
        if k not in tpl:
            tpl[k] = {c: r[c] for c in NUMERIC if c in r and c not in skip and r[c] is not None}
    return tpl


def load_external(path, env_path, tpl):
    rows = json.load(open(path))
    if isinstance(rows, dict) and "results" in rows:
        rows = rows["results"]
    dev = rows[0].get("device", "")
    key = "cuda" if "CUDA" in dev else "mps" if "MPS" in dev else "cpu"
    hw = hw_from_env(env_path, key)
    out, missing = [], 0
    for r in rows:
        e = dict(r)
        k = config_key(e["model_type"], e.get("config", {}) or {})
        t = tpl.get(k)
        if t is None:
            missing += 1
            continue
        for c, v in t.items():                       # 외부 행에 없는 구조 메타는 템플릿으로
            if e.get(c) is None:
                e[c] = v
        e.update(hw)
        e["_backend"], e["_src"], e["_backend_id"] = "external", "ext", 4
        out.append(e)
    if missing:
        print(f"[경고] manifest/632셀에 없는 구성 {missing}행 제외")
    return out, key, hw


def fit_predict(X_tr, y_tr, g_tr, X_te, transform):
    gs = GridSearchCV(LogTargetXGBRegressor(), {"transform": [transform], **XGB_GRID},
                      cv=GroupKFold(4), scoring="r2", n_jobs=1)
    gs.fit(X_tr, y_tr, groups=g_tr)
    return gs.best_estimator_.predict(X_te), gs.best_params_


def stats(y, p, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    yy, pp = np.expm1(y[m]), np.expm1(p[m])
    s = {"r2log": round(float(r2_score(y[m], p[m])), 4) if m.sum() > 2 else None, **err_stats(yy, pp),
         "spearman": round(float(spearmanr(yy, pp).correlation), 3) if m.sum() > 2 else None}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external"); ap.add_argument("--env"); ap.add_argument("--name", default=None)
    ap.add_argument("--simulate", choices=["Desktop CPU", "CUDA", "Mac CPU", "MPS"])
    ap.add_argument("--transform", default="log", choices=["log", "log1p"])
    ap.add_argument("--no-refresh-hw", action="store_true")
    args = ap.parse_args()
    if not args.simulate and not (args.external and args.env):
        raise SystemExit("--external와 --env 를 주거나 --simulate <백엔드>")

    raw = load_clean()
    if not args.no_refresh_hw:
        for b, (env, key) in ENV_OF_BACKEND.items():
            hw = hw_from_env(env, key)
            apply_hw([r for r in raw if r["_backend"] == b], hw)
        print("하드웨어 열 갱신: " + ", ".join(f"{b}(tflops {hw_from_env(e, k).get('tflops_fp32')}, gpu_cores {hw_from_env(e, k).get('gpu_cores')})"
                                          for b, (e, k) in ENV_OF_BACKEND.items()))
    tpl = structural_templates(raw)

    if args.simulate:
        name = f"sim_{args.simulate.replace(' ', '_')}"
        ext_raw = [dict(r, _backend="external", _src="ext", _backend_id=4) for r in raw if r["_backend"] == args.simulate]
        train_raw = [r for r in raw if r["_backend"] != args.simulate]
        ext_key = "cuda" if args.simulate == "CUDA" else "mps" if args.simulate == "MPS" else "cpu"
        ext_hw = {c: ext_raw[0].get(c) for c in HW_NUMERIC_COLS}
    else:
        name = args.name or os.path.splitext(os.path.basename(args.external))[0]
        ext_raw, ext_key, ext_hw = load_external(args.external, args.env, tpl)
        train_raw = raw
    print(f"학습 {len(train_raw)}셀 / 외부 {len(ext_raw)}셀 ({name}, device={ext_key}) / 타깃 {args.transform}")

    rows_tr = onnx_rows_of(train_raw)
    rows_ext = onnx_rows_of(ext_raw)
    for e in rows_tr + rows_ext:
        e.setdefault("_backend_id", 4)
    g_tr = np.array([e["model_name"] for e in rows_tr])
    fams_ext = np.array([e["model_type"] for e in rows_ext])
    names_ext = [e["model_name"] for e in rows_ext]

    # 기준선용: 외부 셀과 같은 구성의 각 백엔드 실측
    by_bk = {}
    for e in rows_tr:
        by_bk.setdefault(e["_backend"], {})[e["model_name"]] = {t: float(e[t]) for t in TARGETS}
    nearest = {"cuda": "CUDA", "mps": "MPS", "cpu": "Desktop CPU"}[ext_key]
    if nearest not in by_bk:                         # 시뮬레이션에서 그 백엔드를 뺐을 때
        nearest = {"cuda": "MPS", "mps": "CUDA", "cpu": "Mac CPU"}[ext_key]
    ref_tflops = {"CUDA": hw_from_env(*ENV_OF_BACKEND["CUDA"]).get("tflops_fp32", 0),
                  "MPS": hw_from_env(*ENV_OF_BACKEND["MPS"]).get("tflops_fp32", 0)}.get(nearest, 0)
    ext_tflops = ext_hw.get("tflops_fp32", 0) or 0

    out = {"name": name, "device": ext_key, "transform": args.transform, "n_train": len(rows_tr), "n_external": len(rows_ext),
           "nearest_backend": nearest, "external_hw": {k: ext_hw.get(k) for k in ["gpu_cores", "gpu_memory_gb", "tflops_fp32", "gpu_clock_ghz", "cpu_cores", "ram_total_gb"]}}
    for tgt in TARGETS:
        X_tr, y_tr = xy(rows_tr, NUMERIC, tgt)
        X_te, y_te = xy(rows_ext, NUMERIC, tgt)
        pred, params = fit_predict(X_tr, y_tr, g_tr, X_te, args.transform)
        yy = np.expm1(y_te)
        res = {"params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
               "descriptor_model": stats(y_te, pred),
               "small": stats(y_te, pred, yy < SMALL_CUT[tgt]), "large": stats(y_te, pred, yy >= SMALL_CUT[tgt]),
               "per_family": {FAM_LABEL[f]: stats(y_te, pred, fams_ext == f) for f in FAMILIES if (fams_ext == f).sum() >= 3}}
        # 기준선 1: 최근접 백엔드 실측 복사, 2: 4백엔드 로그 평균, 3: 최근접 × 사양 비율(tflops)
        near = np.array([by_bk[nearest].get(n, {}).get(tgt, np.nan) for n in names_ext])
        allm = np.array([np.mean([np.log1p(by_bk[b][n][tgt]) for b in by_bk if n in by_bk[b]]) if any(n in by_bk[b] for b in by_bk) else np.nan for n in names_ext])
        ok = ~np.isnan(near)
        res["baseline_nearest_copy"] = stats(y_te[ok], np.log1p(near[ok]))
        res["baseline_all_backend_mean"] = stats(y_te[~np.isnan(allm)], allm[~np.isnan(allm)])
        if ref_tflops and ext_tflops:
            res["baseline_nearest_spec_scaled"] = stats(y_te[ok], np.log1p(near[ok] * ref_tflops / ext_tflops))
            res["spec_ratio_tflops"] = round(ref_tflops / ext_tflops, 3)
        out[tgt] = res
        d = res["descriptor_model"]
        print(f"\n[{tgt}] 기술자 모델: R2(log) {d['r2log']}  MAPE {d['mape']}  MdAPE {d['mdape']}  ±20% {d['w20']}  Spearman {d['spearman']}"
              f"  | 소형 MAPE {res['small']['mape']}  대형 MAPE {res['large']['mape']}")
        for k in ["baseline_nearest_copy", "baseline_all_backend_mean", "baseline_nearest_spec_scaled"]:
            if k in res:
                b = res[k]
                print(f"   {k:28s}: R2(log) {b['r2log']}  MAPE {b['mape']}  MdAPE {b['mdape']}  Spearman {b['spearman']}")
        print("   계열별 MAPE: " + ", ".join(f"{f} {v['mape']}" for f, v in res["per_family"].items()))
    path = os.path.join(BASE, f"paper/external_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n저장:", path)


if __name__ == "__main__":
    main()
