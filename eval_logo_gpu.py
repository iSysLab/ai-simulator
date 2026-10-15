"""Leave-one-GPU-out — CUDA 기기가 여러 대일 때 하드웨어 기술자가 전이되는가.

학습: 632셀(4백엔드, 기술자는 env_*.json으로 갱신) + 외부 기기 측정(GPU·CPU 모두).
평가: CUDA 기기 하나(4060 Ti / GTX 1050 / RTX 3060 …)를 통째로 빼고 학습 → 그 기기 예측.
비교: (a) 남은 GPU 중 TFLOPS 최근접 기기 실측 복사, (b) 남은 GPU 실측을 log(TFLOPS)에 대해
      구성별 선형 보간·외삽, (c) 기술자 민감도 — 뺀 기기의 사양을 최근접 기기 값으로 바꿨을 때 예측 변화.
사양을 쓰는 모델이라면 (a)를 이기고, (c)에서 tflops·gpu_cores가 예측을 움직여야 한다.

사용:
  python eval_logo_gpu.py --external gtx1050:results/external_gtx1050.json:paper/env_gtx1050.json \
                          --external gtx1050_cpu:results/external_gtx1050_cpu.json:paper/env_gtx1050.json \
                          --external rtx3060:results/external_rtx3060.json:paper/env_rtx3060.json \
                          --external rtx3060_cpu:results/external_rtx3060_cpu.json:paper/env_rtx3060.json
산출: paper/logo_gpu.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from eval_fill_v4i import load_clean, onnx_rows_of, xy, TARGETS
from eval_fill_v4j import XGB_GRID
from eval_fill_v4k import err_stats
from train_merged import NUMERIC
from benchmark.predictor import LogTargetXGBRegressor
import eval_external_backend as E

BASE = os.path.dirname(os.path.abspath(__file__))
SENS_COLS = ["tflops_fp32", "gpu_cores", "gpu_clock_ghz", "dedicated_vram_gb", "gpu_memory_gb",
             "gpu_compute_capability", "cpu_cores", "ram_total_gb", "cpu_freq_ghz"]


def stats(y, p):
    yy, pp = np.expm1(y), np.expm1(p)
    return {"r2log": round(float(r2_score(y, p)), 4), **err_stats(yy, pp),
            "spearman": round(float(spearmanr(yy, pp).correlation), 3),
            "pred_over_true": round(float(np.median(pp / yy)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", action="append", required=True, help="name:results.json:env.json")
    args = ap.parse_args()

    raw = load_clean()
    for b, (env, key) in E.ENV_OF_BACKEND.items():
        E.apply_hw([r for r in raw if r["_backend"] == b], E.hw_from_env(env, key))
    tpl = E.structural_templates(raw)
    for r in raw:
        r["_device_name"] = {"CUDA": "rtx4060ti", "Desktop CPU": "r7800x3d", "Mac CPU": "m4_cpu", "MPS": "m4_mps"}[r["_backend"]]
    ext_all, dev_hw = [], {}
    for spec in args.external:
        name, rpath, epath = spec.split(":")
        rows, key, hw = E.load_external(rpath, epath, tpl)
        for r in rows:
            r["_device_name"], r["_backend"] = name, ("CUDA" if key == "cuda" else "CPU_ext")
        ext_all += rows
        dev_hw[name] = (key, hw)
    dev_hw["rtx4060ti"] = ("cuda", E.hw_from_env(*E.ENV_OF_BACKEND["CUDA"]))
    allraw = raw + ext_all
    rows = onnx_rows_of(allraw)
    for e, r in zip(rows, allraw):
        e["_device_name"] = r["_device_name"]
    dev = np.array([e["_device_name"] for e in rows])
    names = np.array([e["model_name"] for e in rows])
    groups = np.array([f"{n}@{d}" if d not in ("rtx4060ti", "r7800x3d", "m4_cpu", "m4_mps") else n for n, d in zip(names, dev)])
    gpus = [d for d, (k, _) in dev_hw.items() if k == "cuda"]
    tfl = {d: float(dev_hw[d][1].get("tflops_fp32") or 0) for d in gpus}
    print("CUDA 기기:", {d: tfl[d] for d in gpus}, "| 전체 셀", len(rows))

    out = {"gpus": {d: {c: dev_hw[d][1].get(c) for c in SENS_COLS} for d in gpus}, "targets": {}}
    for tgt in TARGETS:
        X, y = xy(rows, NUMERIC, tgt)
        res = {}
        for held in gpus:
            te = dev == held; tr = ~te
            gs = GridSearchCV(LogTargetXGBRegressor(), {"transform": ["log"], **XGB_GRID},
                              cv=GroupKFold(4), scoring="r2", n_jobs=1).fit(X[tr], y[tr], groups=groups[tr])
            est = gs.best_estimator_
            p = est.predict(X[te])
            r = {"n": int(te.sum()), "model": stats(y[te], p)}
            others = [d for d in gpus if d != held]
            near = min(others, key=lambda d: abs(np.log(tfl[d]) - np.log(tfl[held])))
            # (a) 최근접 복사, (b) log-TFLOPS 보간 — 구성별로 남은 GPU 실측 사용
            lookup = {(n, d): yv for n, d, yv in zip(names, dev, y)}
            te_names = names[te]
            a = np.array([lookup.get((n, near), np.nan) for n in te_names])
            b = []
            for n in te_names:
                pts = [(np.log(tfl[d]), lookup[(n, d)]) for d in others if (n, d) in lookup]
                if len(pts) >= 2:
                    xs, ys = zip(*pts); k, c = np.polyfit(xs, ys, 1); b.append(k * np.log(tfl[held]) + c)
                else:
                    b.append(np.nan)
            b = np.array(b)
            ok = ~np.isnan(a); okb = ~np.isnan(b)
            r["baseline_nearest_copy"] = {"device": near, **stats(y[te][ok], a[ok])} if ok.sum() > 2 else None
            r["baseline_logtflops_interp"] = {**stats(y[te][okb], b[okb]), "n": int(okb.sum())} if okb.sum() > 2 else None
            # (c) 민감도: 뺀 기기 사양을 최근접 기기 값으로 하나씩 교체
            base_ratio = float(np.median(np.expm1(p) / np.expm1(y[te])))
            sens = {}
            for col in SENS_COLS:
                Xs = X[te].copy(); v = dev_hw[near][1].get(col)
                if v is None:
                    continue
                Xs[:, NUMERIC.index(col)] = float(v)
                sens[col] = round(float(np.median(np.expm1(est.predict(Xs)) / np.expm1(y[te]))) / base_ratio, 3)
            Xall = X[te].copy()
            for col in SENS_COLS:
                v = dev_hw[near][1].get(col)
                if v is not None:
                    Xall[:, NUMERIC.index(col)] = float(v)
            sens["ALL"] = round(float(np.median(np.expm1(est.predict(Xall)) / np.expm1(y[te]))) / base_ratio, 3)
            r["sensitivity_swap_to_nearest"] = sens
            imp = est.model_.feature_importances_
            r["gain_hw"] = {c: round(float(imp[NUMERIC.index(c)]), 4) for c in SENS_COLS}
            res[held] = r
            m, bn, bi = r["model"], r["baseline_nearest_copy"], r["baseline_logtflops_interp"]
            print(f"\n[{tgt}] 제외 {held} (TFLOPS {tfl[held]}) n={r['n']}")
            print(f"   모델        R2 {m['r2log']:.3f} MAPE {m['mape']:5.1f} MdAPE {m['mdape']:5.1f} pred/true {m['pred_over_true']:.2f}")
            if bn: print(f"   복사({near:9s}) R2 {bn['r2log']:.3f} MAPE {bn['mape']:5.1f} MdAPE {bn['mdape']:5.1f} pred/true {bn['pred_over_true']:.2f}")
            if bi: print(f"   logTFLOPS보간 R2 {bi['r2log']:.3f} MAPE {bi['mape']:5.1f} MdAPE {bi['mdape']:5.1f} pred/true {bi['pred_over_true']:.2f}")
            print("   사양→최근접 교체 시 예측 배율:", {k: v for k, v in sens.items() if abs(v - 1) > 0.02} or "변화 없음", "| ALL", sens["ALL"])
        out["targets"][tgt] = res
    json.dump(out, open(os.path.join(BASE, "paper/logo_gpu.json"), "w"), ensure_ascii=False, indent=1)
    print("\n저장: paper/logo_gpu.json")


if __name__ == "__main__":
    main()
