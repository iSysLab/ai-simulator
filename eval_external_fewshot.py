"""외부 기기 few-shot 추가 — 새 기기 구성 k개를 632셀에 더해 학습하고 나머지 구성으로 평가.

paper/73_external_gtx1050_results.md §3.2 의 실험을 스크립트로. 기술자는 env_*.json으로 갱신, 타깃 log(y).
사용:
  python eval_external_fewshot.py --external results/external_<host>.json --env paper/env_<host>.json --name <host>
  옵션 --ks 0,6,12,24  --seeds 5  --reference-backend "Desktop CPU"|CUDA|MPS|"Mac CPU" (실측 비 계산용, 기본은 장치 종류로 자동)
산출: paper/external_<name>_fewshot.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from eval_fill_v4i import load_clean, onnx_rows_of, xy, TARGETS
from eval_fill_v4j import XGB_GRID
from eval_fill_v4k import err_stats, FAM_LABEL
from train_merged import NUMERIC
from benchmark.predictor import LogTargetXGBRegressor
import eval_external_backend as E

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True); ap.add_argument("--env", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ks", default="0,6,12,24"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reference-backend", default=None)
    args = ap.parse_args()

    raw = load_clean()
    for b, (env, key) in E.ENV_OF_BACKEND.items():
        E.apply_hw([r for r in raw if r["_backend"] == b], E.hw_from_env(env, key))
    tpl = E.structural_templates(raw)
    ext_raw, ext_key, _ = E.load_external(args.external, args.env, tpl)
    rows, ext = onnx_rows_of(raw), onnx_rows_of(ext_raw)
    for e in rows + ext:
        e.setdefault("_backend_id", 4)
    g_base = np.array([e["model_name"] for e in rows])
    fam_ext = np.array([e["model_type"] for e in ext]); n_ext = len(ext)
    fams = sorted(set(fam_ext))
    ref = args.reference_backend or {"cuda": "CUDA", "mps": "MPS", "cpu": "Desktop CPU"}[ext_key]
    ref_rows = {e["model_name"]: e for e in rows if e["_backend"] == ref}

    out = {"name": args.name, "device": ext_key, "n_external": n_ext, "reference_backend": ref, "targets": {}}
    # 참조 백엔드 대비 실측 비
    for tgt in TARGETS:
        r = np.array([float(e[tgt]) / float(ref_rows[e["model_name"]][tgt]) for e in ext if e["model_name"] in ref_rows])
        yt = np.array([float(ref_rows[e["model_name"]]["avg_train"]) for e in ext if e["model_name"] in ref_rows])
        out.setdefault("ratio_vs_reference", {})[tgt] = {
            "median": round(float(np.median(r)), 2),
            "small_lt1s": round(float(np.median(r[yt < 1])), 2) if (yt < 1).any() else None,
            "large_gt10s": round(float(np.median(r[yt > 10])), 2) if (yt > 10).any() else None}
    print(f"{args.name} vs {ref} 실측 비:", out["ratio_vs_reference"])

    for tgt in TARGETS:
        Xb, yb = xy(rows, NUMERIC, tgt); Xe, ye = xy(ext, NUMERIC, tgt); yy = np.expm1(ye)
        res = {}
        for k in [int(x) for x in args.ks.split(",")]:
            stats = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                per = k // len(fams)
                pick = []
                for f in fams:
                    idx = np.where(fam_ext == f)[0]
                    pick += list(rng.choice(idx, min(per, len(idx)), replace=False))
                pick = np.array(pick, int); rest = np.setdiff1d(np.arange(n_ext), pick)
                if k:
                    X_tr = np.vstack([Xb, Xe[pick]]); y_tr = np.concatenate([yb, ye[pick]])
                    g_tr = np.concatenate([g_base, np.array([ext[i]["model_name"] + "@" + args.name for i in pick])])
                else:
                    X_tr, y_tr, g_tr = Xb, yb, g_base
                gs = GridSearchCV(LogTargetXGBRegressor(), {"transform": ["log"], **XGB_GRID},
                                  cv=GroupKFold(4), scoring="r2", n_jobs=1).fit(X_tr, y_tr, groups=g_tr)
                p = gs.best_estimator_.predict(Xe[rest])
                s = err_stats(yy[rest], np.expm1(p))
                s["r2log"] = float(r2_score(ye[rest], p)); s["ratio"] = float(np.median(np.expm1(p) / yy[rest])); s["n_eval"] = int(len(rest))
                stats.append(s)
            agg = {m: {"mean": round(float(np.mean([s[m] for s in stats])), 3), "std": round(float(np.std([s[m] for s in stats])), 3)}
                   for m in ["mape", "mdape", "w20", "r2log", "ratio"]}
            agg["n_eval"] = stats[0]["n_eval"]; agg["n_train_added"] = int(len(pick))
            res[str(k)] = agg
            print(f"[{tgt}] {args.name} 구성 {len(pick):2d}개 학습 포함 → 나머지 {agg['n_eval']}개: MAPE {agg['mape']['mean']:.1f}±{agg['mape']['std']:.1f}  "
                  f"MdAPE {agg['mdape']['mean']:.1f}  ±20% {agg['w20']['mean']:.0f}%  R2(log) {agg['r2log']['mean']:.3f}  pred/true {agg['ratio']['mean']:.2f}")
        out["targets"][tgt] = res
    path = os.path.join(BASE, f"paper/external_{args.name}_fewshot.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    main()
