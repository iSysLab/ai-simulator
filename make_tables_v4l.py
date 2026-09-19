"""v4l 결과(paper/v4_measured_l.json) → 표 초안 paper/v4l_tables.md.

표 1  후보별 성능 (타깃 × 후보): R²(log), 전체/소형/대형 MAPE·MdAPE·±20%, GAN
표 2  게이트 분석: τ별 분류기 AUC·P/R, base 라우팅 AUC·P/R
표 3  외삽 사다리 3~5단계, 후보별
표 4  시드 10개 안정성
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "paper/v4_measured_l.json")
OUT = os.path.join(BASE, "paper/v4l_tables.md")
TGT_LABEL = {"avg_train": "학습시간", "avg_infer": "추론시간"}
SMALL_LABEL = {"avg_train": "<1 s", "avg_infer": "<10 ms"}


def f(x, nd=1):
    if x is None:
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def row_of(name, s):
    if s is None:
        return None
    a, sm, lg, g = s["all"], s["small"], s["large"], s.get("gan", {})
    return (f"| {name} | {f(s['r2log'], 4)} | {f(a['mape'])} | {f(a['mdape'])} | {f(a['w20'])} "
            f"| {f(sm['mape'])} | {f(sm['mdape'])} | {f(sm['w20'])} | {f(lg['mape'])} | {f(g.get('mape'))} |")


def candidates(d, tgt):
    """(표시 이름, 요약 dict) 목록 — 있는 절만."""
    out = []
    b = d.get("baseline", {}).get(tgt)
    if b:
        out.append(("통합 XGB (현재)", b))
        o = b.get("oracle_specialist")
        if o:
            out.append((f"oracle 전문 모델 (τ_s={o['tau_s_sec']} s, 상한)",
                        {"r2log": None, "all": {"mape": None, "mdape": None, "w20": None},
                         "small": o["small"], "large": {"mape": None}, "gan": {}}))
    lt = d.get("log_target", {}).get(tgt, {})
    for k, lab in [("log", "F: 타깃 log(y)"), ("log1p_ms", "F: 타깃 log1p(y·1000)")]:
        if k in lt:
            out.append((lab, lt[k]))
    for sec, lab in [("weighted_baseline", "D: 소형 가중"), ("residual_baseline", "E: 잔차 보정")]:
        if d.get(sec, {}).get(tgt):
            out.append((lab, d[sec][tgt]))
    re_ = d.get("regime_ensemble", {}).get(tgt, {})
    for k, lab in [("hard", "B: 게이트 hard"), ("soft", "B: 게이트 soft"), ("clf", "B: 게이트 clf"), ("all_gates", "B: 게이트 자동 선택")]:
        if k in re_:
            out.append((lab, re_[k]))
    for sec, lab in [("algo_average", "A: XGB·GB·RF 평균"), ("per_backend_control", "C: 백엔드별 4모델 (대조군)")]:
        if d.get(sec, {}).get(tgt):
            out.append((lab, d[sec][tgt]))
    fc = d.get("log_per_backend", {}).get(tgt, {})
    for k, lab in [("log", "F+C: 백엔드별 4모델 + log(y)"), ("log1p", "C(재확인): 백엔드별 4모델 + log1p")]:
        if k in fc:
            out.append((lab, fc[k]))
    return out


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    L = [f"# v4l 표 초안 — 앙상블·타깃 변환 후보 비교\n",
         f"소스: `paper/v4_measured_l.json` (갱신 {d.get('_meta', {}).get('updated', '?')}). "
         "구성 단위 5-겹 nested CV(내부 4-겹, scoring=R² in z=log1p(초)). 오차는 원 단위. "
         "이 기기(MacBook M4)에서 재계산한 값이며 v4k(데스크톱)와 소형 구간 수치가 다를 수 있다.\n"]

    L.append("## 표 1 — 후보별 성능\n")
    for tgt in ["avg_infer", "avg_train"]:
        cands = candidates(d, tgt)
        if not cands:
            continue
        L.append(f"**{TGT_LABEL[tgt]}** (소형 = {SMALL_LABEL[tgt]})\n")
        L.append("| 후보 | R²(log) | 전체 MAPE | 전체 MdAPE | 전체 ±20% | 소형 MAPE | 소형 MdAPE | 소형 ±20% | 대형 MAPE | GAN MAPE |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, s in cands:
            r = row_of(name, s)
            if r:
                L.append(r)
        L.append("")
        # 선택된 하이퍼파라미터
        picks = []
        for name, s in cands:
            cp = s.get("chosen_params") if isinstance(s, dict) else None
            if cp:
                picks.append(f"- {name}: `{cp}`")
        if picks:
            L.append("선택된 하이퍼파라미터(전체 632셀 4-겹 GridSearchCV):")
            L.extend(picks)
            L.append("")

    ga = d.get("gate_analysis", {})
    if ga:
        L.append("## 표 2 — 게이트 분석 (소형 구간 분리 가능성)\n")
        L.append("clf = XGB 분류기(피처만, 구성 단위 5-겹). base 라우팅 = 통합 모델 예측 ẑ < τ. P/R = 소형 클래스 정밀도/재현율.\n")
        L.append("| 타깃 | τ (s) | n 소형 | clf AUC | clf P | clf R | base AUC | base P | base R |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tgt in ["avg_infer", "avg_train"]:
            for k, r in sorted(ga.get(tgt, {}).items(), key=lambda kv: float(kv[0])):
                L.append(f"| {TGT_LABEL[tgt]} | {r['tau_sec']} | {r['n_small']} | {f(r['clf_auc'], 4)} | {f(r['clf']['precision'], 3)} | {f(r['clf']['recall'], 3)} "
                         f"| {f(r['base_route_auc'], 4)} | {f(r['base_route']['precision'], 3)} | {f(r['base_route']['recall'], 3)} |")
        L.append("")

    lad = d.get("ladder_check", {})
    if lad:
        L.append("## 표 3 — 외삽 사다리 (v4k 3·4·5단계), 후보별\n")
        L.append("그리드는 표 1의 선택 파라미터로 고정. L3 = 폭 수준 홀드아웃, L4 = 깊이 수준, L5 = 계열별 파라미터 상위 25%. p/t = L5 예측/실측 중앙값(1 미만이면 과소 추정).\n")
        L.append("| 타깃 | 후보 | L3 R²(log) | L3 MAPE | L4 R²(log) | L4 MAPE | L5 R²(log) | L5 MAPE | L5 p/t |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for tgt in ["avg_infer", "avg_train"]:
            for name, r in lad.get(tgt, {}).items():
                if not isinstance(r, dict) or "L3_width" not in r:
                    continue
                L.append(f"| {TGT_LABEL[tgt]} | {name} | {f(r['L3_width']['r2log'], 4)} | {f(r['L3_width']['all']['mape'])} "
                         f"| {f(r['L4_depth']['r2log'], 4)} | {f(r['L4_depth']['all']['mape'])} "
                         f"| {f(r['L5_top_quartile']['r2log'], 4)} | {f(r['L5_top_quartile']['all']['mape'])} | {f(r['L5_top_quartile']['median_pred_over_true'], 3)} |")
        L.append("")

    sd = d.get("seed_stability", {})
    if sd:
        L.append("## 표 4 — 시드 10개 안정성 (구성 단위 nested CV, 외부·내부 셔플)\n")
        L.append("| 타깃 | 후보 | R²(log) | 전체 MAPE | 소형 MAPE | 소형 MdAPE | 대형 MAPE |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for tgt in ["avg_infer", "avg_train"]:
            for name, a in sd.get(tgt, {}).items():
                if not isinstance(a, dict) or "r2log" not in a:
                    continue
                pm = lambda k, nd=1: f"{a[k]['mean']:.{nd}f} ± {a[k]['std']:.{nd}f}"
                L.append(f"| {TGT_LABEL[tgt]} | {name} | {pm('r2log', 4)} | {pm('all_mape')} | {pm('small_mape')} | {pm('small_mdape')} | {pm('large_mape')} |")
        L.append("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("저장:", OUT)


if __name__ == "__main__":
    main()
