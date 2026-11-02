"""스마트미디어저널 원고 추가 분석 — run_smj_experiments.py의 외부 폴드 예측만 읽는다(모델 재학습 없음).

1) 실행 시간 구간별 오차: [2](RACS 2026)처럼 추론 10 ms·학습 1 s를 기준으로 하위/상위 구간으로 나누고,
   추론 작업의 정의와 처리 이미지 수가 분류와 다른 GAN(생성기가 약 5만 장 생성)은 상위 구간에서 따로 본다.
2) 플랫폼별 결합 방법 비교: 플랫폼별 가중 결합이 공통 가중 결합·단순 평균·최고 단일 모델보다
   어느 플랫폼에서 나은지(시드별 MAPE 차이, 승리 수, 구성별 |ln 오차|의 Wilcoxon 검정).

사용: .venv/bin/python -B paper/smj/analyze_smj_results.py
입력: paper/smj/results/preds_<타깃>.npz  →  산출: paper/smj/results/smj_analysis.json
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TARGETS = ["avg_train", "avg_infer", "peak_mem"]
SEG_CUT = {"avg_train": 1.0, "avg_infer": 0.01}   # run_smj_experiments.SMALL_CUT와 같은 기준(초)
BASE_NAMES = ["xgb", "gb", "rf"]
PAIRS = [("platform_w", "global_w"), ("platform_w", "average"), ("platform_w", "best_single")]
META_KEYS = {"y", "plat", "fam", "groups"}


def mape_by_seed(pred, y, m):
    """시드별 MAPE(%) — pred는 (시드, 샘플)."""
    return np.mean(np.abs(pred[:, m] - y[m]) / y[m], axis=1) * 100


def seg_block(pred, y, m):
    s = mape_by_seed(pred, y, m)
    w20 = float((np.abs(pred[:, m] - y[m]) / y[m] < 0.2).mean() * 100)
    return {"n": int(m.sum()), "mape": round(float(s.mean()), 4),
            "mape_sd": round(float(s.std(ddof=1)), 4), "w20": round(w20, 2)}


def paired_in(pred_a, pred_b, y, m, groups):
    """마스크 m 안에서 a − b: 시드별 MAPE 차이, a가 더 낮았던 시드 수, 구성별 |ln 오차|(시드 평균)의 Wilcoxon."""
    diff = mape_by_seed(pred_a, y, m) - mape_by_seed(pred_b, y, m)
    ea = np.abs(np.log(pred_a[:, m]) - np.log(y[m])).mean(axis=0)
    eb = np.abs(np.log(pred_b[:, m]) - np.log(y[m])).mean(axis=0)
    g = groups[m]
    cfg = np.unique(g)
    ca = np.array([ea[g == c].mean() for c in cfg])
    cb = np.array([eb[g == c].mean() for c in cfg])
    p = float(wilcoxon(ca, cb).pvalue) if np.any(ca != cb) else None
    return {"delta_mape_mean": round(float(diff.mean()), 3), "delta_mape_sd": round(float(diff.std(ddof=1)), 3),
            "wins": int((diff < 0).sum()), "n_seeds": int(len(diff)), "wilcoxon_p": p,
            "cfg_better": int((ca < cb).sum()), "n_cfg": int(len(cfg))}


def analyze(tgt):
    d = np.load(os.path.join(RES_DIR, f"preds_{tgt}.npz"))
    y, plat, fam, groups = d["y"], d["plat"], d["fam"], d["groups"]
    preds = {k: d[k] for k in d.files if k not in META_KEYS}
    allm = np.ones(len(y), bool)
    best = min(BASE_NAMES, key=lambda b: mape_by_seed(preds[b], y, allm).mean())
    preds["best_single"] = preds[best]
    out = {"best_single": best}

    if tgt in SEG_CUT:
        lo = y < SEG_CUT[tgt]
        gan = fam == "gan"
        masks = {"all": allm, "lower": lo, "upper": ~lo, "upper_non_gan": ~lo & ~gan, "upper_gan": ~lo & gan}
        out["segments"] = {"cut_s": SEG_CUT[tgt],
                           "lower_n_by_family": {str(f): int((lo & (fam == f)).sum()) for f in sorted(set(fam))},
                           "methods": {k: {s: seg_block(p, y, m) for s, m in masks.items()}
                                       for k, p in preds.items() if k != "best_single"}}

    out["by_platform"] = {}
    for q in sorted(set(plat)):
        m = plat == q
        out["by_platform"][q] = {
            "mape": {k: round(float(mape_by_seed(p, y, m).mean()), 4) for k, p in preds.items() if k != "best_single"},
            "paired": {f"{a}_vs_{b}": paired_in(preds[a], preds[b], y, m, groups) for a, b in PAIRS}}
    out["paired_all"] = {f"{a}_vs_{b}": paired_in(preds[a], preds[b], y, allm, groups) for a, b in PAIRS}
    return out


def main():
    res = {t: analyze(t) for t in TARGETS}
    path = os.path.join(RES_DIR, "smj_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    for t, r in res.items():
        print(f"\n=== {t} (최고 단일 모델 {r['best_single']}) ===")
        if "segments" in r:
            print(f"  구간 기준 {r['segments']['cut_s']} s, 하위 구간 계열 구성 {r['segments']['lower_n_by_family']}")
            for k in ("xgb_log1p", r["best_single"], "average", "global_w", "platform_w", "tree_gate"):
                s = r["segments"]["methods"][k]
                print(f"  {k:11s} " + "  ".join(f"{name} {s[name]['mape']:6.2f}(n={s[name]['n']})"
                                                for name in ("all", "lower", "upper", "upper_non_gan", "upper_gan")))
        for q, b in r["by_platform"].items():
            pw = b["paired"]
            print(f"  {q:12s} " + "  ".join(
                f"{key.replace('platform_w_vs_', 'pw−')} {v['delta_mape_mean']:+.2f}±{v['delta_mape_sd']:.2f} "
                f"승 {v['wins']}/{v['n_seeds']} p={v['wilcoxon_p']:.3g}" for key, v in pw.items()))
        for key, v in r["paired_all"].items():
            print(f"  전체 {key:28s} {v['delta_mape_mean']:+.3f}±{v['delta_mape_sd']:.3f} 승 {v['wins']}/{v['n_seeds']} "
                  f"구성 {v['cfg_better']}/{v['n_cfg']} p={v['wilcoxon_p']:.3g}")
    print("\n저장:", path)


if __name__ == "__main__":
    main()
