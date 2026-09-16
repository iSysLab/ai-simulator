"""paper/v4_measured_k.json → paper/v4k_tables.md (camera-ready용 표 초안).

각 표에 어느 지적에 답하는지, 원고의 어느 표를 대체·확장하는지 적는다.
사용: python make_tables_v4k.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(BASE, "paper/v4_measured_k.json"), encoding="utf-8"))
BACKENDS = ["Desktop CPU", "CUDA", "Mac CPU", "MPS"]
FAMS = ["ANN", "CNN", "ResNet", "MobileNet", "ViT", "GAN"]
L = []


def h(title, note=""):
    L.append(f"\n### {title}\n")
    if note:
        L.append(note + "\n")


def row(cells):
    L.append("| " + " | ".join(str(c) for c in cells) + " |")


def hdr(cells, align=None):
    row(cells)
    align = align or (["---"] + ["---:"] * (len(cells) - 1))
    L.append("|" + "|".join(align) + "|")


def f(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def pct(x):
    return "–" if x is None else f"{x:.1f}%"


L.append("# v4k 표 초안 (camera-ready)\n")
L.append(f"소스: `paper/v4_measured_k.json` (갱신 {K.get('_meta', {}).get('updated', '?')}). "
         "모든 XGBoost 결과는 외부 5-겹 / 내부 4-겹 nested CV, 별도 표기 없으면 구성 단위 분할.\n")

# ------------------------------------------------------------------ R1-3
if "backend_values" in K:
    bv = K["backend_values"]
    h("표 2 확장 — 백엔드를 구분하는 하드웨어 기술자 (R1-3)",
      "수치 하드웨어 피처 30개 중 백엔드 쌍을 실제로 가르는 열. 나머지는 두 백엔드에서 같은 값(모델 분기에 기여하지 않음).")
    hdr(["백엔드 쌍", "구분 피처 수", "구분 피처"], ["---", "---:", "---"])
    for k, v in bv["distinguishing_features"].items():
        row([k, len(v), ", ".join(f"`{c}`" for c in v)])
    L.append("")
    show = ["device_type_encoded", "cpu_cores", "ram_total_gb", "gpu_cores", "gpu_memory_gb", "tflops_fp32", "is_integrated_gpu", "cpu_freq_ghz"]
    hdr(["피처"] + BACKENDS, ["---"] + ["---:"] * 4)
    for c in show:
        row([f"`{c}`"] + [bv["per_backend"][b].get(c, "–") for b in BACKENDS])
    nc = {b: v for b, v in bv["non_constant_within_backend"].items() if v}
    L.append(f"\n백엔드 안에서 상수가 아닌 열: {nc if nc else '없음 (모두 상수)'}\n")

# ------------------------------------------------------------------ R1-9
if "precision_check" in K:
    pc = K["precision_check"]
    for split, title in [("random_split", "무작위 분할 (원고 표 5·6 기준)"), ("config_split", "구성 단위 분할")]:
        h(f"표 6 정밀화 — 백엔드별 R²(log), 4자리 + 부트스트랩 95% CI, {title} (R1-9)")
        hdr(["부분집합", "n", "학습 R²(log)", "95% CI", "RMSE(log)", "추론 R²(log)", "95% CI", "RMSE(log)"])
        for b in ["all"] + BACKENDS:
            a, i = pc[split]["avg_train"][b], pc[split]["avg_infer"][b]
            row([b, a["n"], f(a["r2log"], 4), f"[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]", f(a["rmse_log"], 3),
                 f(i["r2log"], 4), f"[{i['ci95'][0]:.4f}, {i['ci95'][1]:.4f}]", f(i["rmse_log"], 3)])
    L.append("\n각주 초안: 백엔드별 R²는 해당 부분집합의 OOF 예측을 그 부분집합 자체 평균 기준으로 계산한 값이며 전체 R²의 분해가 아니다. "
             "무작위 분할에서 Desktop CPU와 전체의 3자리 일치는 우연으로, 4자리와 신뢰구간에서 서로 다른 양임을 확인할 수 있다.\n")

# ------------------------------------------------------------------ R1-1 / R1-11
if "errors_by_regime" in K and "measurement_noise" in K:
    er, mn = K["errors_by_regime"], K["measurement_noise"]
    h("새 표 — 구간별 예측 오차와 측정 잡음 (R1-1, R1-11)",
      "구성 단위 nested CV OOF. MAPE·MdAPE·±20%는 원 단위. 잡음 CV = 셀별 10회 반복의 std/avg 중앙값. "
      "APE/CV = 셀별 예측 상대오차를 그 셀의 측정 CV로 나눈 값의 중앙값(1이면 예측 오차가 측정 잡음 수준).")
    for tgt, name in [("avg_train", "학습시간"), ("avg_infer", "추론시간")]:
        L.append(f"\n**{name}**\n")
        hdr(["부분집합", "구간", "n", "R²(log)", "MAPE", "MdAPE", "±20% 이내", "잡음 CV 중앙", "APE/CV", "2σ 이내"])
        regimes = [("all", "전체"), ("ge10ms", "≥10 ms"), ("lt10ms", "<10 ms"), ("gan", "GAN")] if tgt == "avg_infer" \
            else [("all", "전체"), ("gan", "GAN"), ("nongan_ge10ms", "비-GAN")]
        for b in ["all"] + BACKENDS:
            for rk, rl in regimes:
                s = er[tgt][b][rk]
                if not s["n"]:
                    continue
                key = b if rk == "all" else (f"{b} lt10ms" if rk == "lt10ms" and b != "all" else ("lt10ms" if rk == "lt10ms" else None))
                m = mn[tgt].get(key) if key else None
                row([b if rk == regimes[0][0] else "", rl, s["n"], f(s["r2log"]), pct(s["mape"]), pct(s["mdape"]), pct(s["w20"]),
                     pct(s["noise_cv_median_pct"]), f(m["ape_over_cv_median"], 1) if m else "–",
                     pct(m["within_2sigma_pct"]) if m else "–"])
    a = mn["avg_infer"]
    L.append(f"\n해석 초안: 추론 10 ms 미만 {a['lt10ms']['n']}셀의 측정 CV 중앙값은 {a['lt10ms']['cv_median_pct']}%인 반면 예측 상대오차 중앙값은 "
             f"{a['lt10ms']['ape_median_pct']}%로, 셀 단위로 예측 오차가 측정 잡음의 {a['lt10ms']['ape_over_cv_median']}배다. "
             f"즉 이 구간의 오차는 측정 분산이 아니라 모델이 설명하지 못하는 실행 오버헤드다.\n")
    L.append("주의: 부분집합 안의 R²(log)는 그 부분집합의 로그 분산에 좌우된다. GAN 8셀이나 <10 ms 구간처럼 분산이 작은 부분집합에서는 "
             "음수가 나올 수 있으며 이는 정확도가 아니라 지표의 성질이다(R1-1의 지적 그대로). 부분집합 비교는 MAPE·MdAPE·±20%로 한다. "
             "MAPE는 소수 이상치에 민감하므로 <10 ms 구간은 MdAPE를 함께 본다.\n")

# ------------------------------------------------------------------ R1-2 / R1-4
if "protocol_ladder" in K:
    pl = K["protocol_ladder"]
    h("표 7 교체 — 검증 프로토콜 사다리 (R1-2, R1-4)",
      "위에서 아래로 홀드아웃 대상이 커진다. 1·2단계는 시드 10개 평균±표준편차. MAPE는 학습 전체·추론 ≥10 ms. "
      "6·7단계의 R²(log)는 전 폴드 OOF 합산이며, 괄호는 그룹별 R²(log)의 최소~최대.")
    hdr(["단계", "홀드아웃 단위", "학습 R²(log)", "학습 MAPE", "추론 R²(log)", "추론 MAPE"])
    def seeded(k, label):
        a, i = pl[k]["avg_train"], pl[k]["avg_infer"]
        row([label.split("|")[0], label.split("|")[1], f"{a['r2log_mean']:.3f} ± {a['r2log_std']:.3f}", f"{a['mape_mean']:.1f} ± {a['mape_std']:.1f}%",
             f"{i['r2log_mean']:.3f} ± {i['r2log_std']:.3f}", f"{i['mape_mean']:.1f} ± {i['mape_std']:.1f}%"])
    def single(k, label, per=False):
        a, i = pl[k]["avg_train"], pl[k]["avg_infer"]
        ra = f"{a['r2log']:.3f}" + (f" ({a['per_group_r2log_min']:.2f}~{a['per_group_r2log_max']:.2f})" if per else "")
        ri = f"{i['r2log']:.3f}" + (f" ({i['per_group_r2log_min']:.2f}~{i['per_group_r2log_max']:.2f})" if per else "")
        row([label.split("|")[0], label.split("|")[1], ra, pct(a["mape"]), ri, pct(i["mape"])])
    seeded("L1_random", "1|무작위 (셀)")
    seeded("L2_config", "2|구성 (4백엔드 묶음)")
    single("L3_width_level", f"3|폭 수준 (계열별 한 폭 값, {pl['L3_width_level']['n_groups']}그룹)")
    single("L4_depth_level", f"4|깊이 수준 (계열별 한 깊이 값, {pl['L4_depth_level']['n_groups']}그룹)")
    single("L5_top_quartile", f"5|계열별 최대 사분위 ({pl['L5_top_quartile']['n_test']}셀 외삽)")
    single("L6_family_out", "6|계열 전체 (LOFO, 계열 식별자 제외)", per=True)
    single("L7_backend_out", "7|백엔드 전체 (LOBO)", per=True)
    d = pl["L2_minus_L1"]
    L.append(f"\n1↔2단계 짝지은 차이(구성 − 무작위, 시드 10개): 학습 {d['avg_train']['mean']:+.4f} ± {d['avg_train']['std']:.4f}, "
             f"추론 {d['avg_infer']['mean']:+.4f} ± {d['avg_infer']['std']:.4f}. "
             "시드 표준편차 안에 들어가므로 두 분할은 통계적으로 구별되지 않는다 → 구성 단위 묶음은 실질적 제약이 아니다.\n")
    L.append("\n**6단계 계열별 (LOFO)**\n")
    hdr(["R²(log)"] + FAMS)
    for tgt, name in [("avg_train", "학습"), ("avg_infer", "추론")]:
        per = pl["L6_family_out"][tgt]["per_group"]
        row([name] + [f(per[fm]["r2log"], 2) if fm in per else "–" for fm in FAMS])
    for tgt, name in [("avg_train", "학습 MAPE"), ("avg_infer", "추론 MAPE(≥10ms)")]:
        per = pl["L6_family_out"][tgt]["per_group"]
        row([name] + [pct(per[fm]["mape"]) if fm in per else "–" for fm in FAMS])
    L.append("\n**5단계 계열별 (최대 사분위 외삽)**\n")
    hdr(["R²(log)"] + FAMS)
    for tgt, name in [("avg_train", "학습"), ("avg_infer", "추론")]:
        per = pl["L5_top_quartile"][tgt]["per_family_r2log"]
        row([name] + [f(per.get(fm), 2) for fm in FAMS])
    L.append(f"\n5단계 예측/실측 중앙값: 학습 {pl['L5_top_quartile']['avg_train']['median_pred_over_true']}, "
             f"추론 {pl['L5_top_quartile']['avg_infer']['median_pred_over_true']} (1 미만이면 큰 구성을 과소 추정).\n")

# ------------------------------------------------------------------ R1-5
if "label_vs_descriptors" in K:
    lv = K["label_vs_descriptors"]
    h("표 7 추가 행 — 하드웨어 기술자 30개 vs 백엔드 라벨 1개 (R1-5)",
      "구성 단위 nested CV. 632셀 안에서 하드웨어 기술자는 백엔드마다 상수이므로, 라벨 1개와 정보량이 같다면 (c)≈(a)여야 한다.")
    hdr(["조건", "피처 수", "학습 R²(log)", "학습 MAPE", "추론 R²(log)", "추론 MAPE(≥10ms)"])
    names = {"a_full_120": "(a) 전체 피처", "b_no_hw_90": "(b) 하드웨어 30개 제거", "c_no_hw_plus_backend_id": "(c) 제거 + 백엔드 라벨 1개(4수준)",
             "d_no_hw_plus_device_type_only": "(d) 제거 + 장치 종류만(3수준, 두 CPU 미구분)", "e_no_hw_plus_backend_onehot": "(e) 제거 + 백엔드 원-핫 4개"}
    for k, nm in names.items():
        v = lv[k]
        row([nm, v["n_features"], f(v["avg_train"]["r2log"]), pct(v["avg_train"]["mape"]), f(v["avg_infer"]["r2log"]), pct(v["avg_infer"]["mape"])])
    L.append("\n**미지 백엔드(LOBO)에서 기술자 모델 대 무학습 기준선**\n")
    L.append("기준선 1 = 나머지 세 백엔드의 같은 구성 실측 평균(로그). 기준선 2 = 같은 기기 종류의 최근접 백엔드 실측. "
             "기술자 모델이 기준선을 이겨야 코어 수·클럭·대역폭이 정체성 이상의 전이 가능한 정보를 가진 것이다.\n")
    hdr(["제외 백엔드", "타깃", "기술자 모델 R²(log)", "MAPE", "기준선1 평균 R²(log)", "MAPE", "기준선2 최근접 R²(log)", "MAPE"])
    for b in BACKENDS:
        for tgt, name in [("avg_train", "학습"), ("avg_infer", "추론")]:
            e = lv["lobo_vs_baselines"][tgt][b]
            near_key = [k for k in e if k.startswith("baseline_nearest")][0]
            row([b if tgt == "avg_train" else "", name, f(e["descriptor_model"]["r2log"], 2), pct(e["descriptor_model"]["mape"]),
                 f(e["baseline_mean_of_others"]["r2log"], 2), pct(e["baseline_mean_of_others"]["mape"]),
                 f(e[near_key]["r2log"], 2) + f" ({near_key[17:-1]})", pct(e[near_key]["mape"])])

# ------------------------------------------------------------------ R1-6
if "unified_importance" in K:
    ui = K["unified_importance"]
    h("표 8 교체 — 통합 XGBoost의 permutation 중요도 (R1-6)",
      f"구성 단위 nested CV의 외부 폴드 5개 각각에서 best 모델로 홀드아웃 셔플({ui['n_repeats']}회) 시 R²(log) 하락. 폴드 평균 ± 표준편차. "
      "그룹은 열을 함께 셔플. gain 순위는 각 폴드 best 모델의 feature_importances_ 순위.")
    hdr(["피처 / 그룹", "열 수", "학습 Δ R²(log)", "추론 Δ R²(log)", "학습 gain 순위(5폴드)", "추론 gain 순위(5폴드)"])
    items = [("total_op_memory_write", "`total_op_memory_write`", 1), ("total_op_memory_read", "`total_op_memory_read`", 1),
             ("flops", "`flops`", 1), ("total_params", "`total_params`", 1), ("model_family_encoded", "`model_family_encoded`", 1),
             ("MEM_GROUP", "메모리 그룹", ui["avg_train"]["group_sizes"]["MEM_GROUP"]),
             ("FLOPS_GROUP", "FLOPs 그룹", ui["avg_train"]["group_sizes"]["FLOPS_GROUP"]),
             ("HW_30", "하드웨어 30개", ui["avg_train"]["group_sizes"]["HW_30"]),
             ("FAMILY_ID", "계열 식별자", ui["avg_train"]["group_sizes"]["FAMILY_ID"])]
    for k, nm, n in items:
        a, i = ui["avg_train"]["permutation_drop"][k], ui["avg_infer"]["permutation_drop"][k]
        ga = ui["avg_train"]["gain_rank"].get(k, ""); gi = ui["avg_infer"]["gain_rank"].get(k, "")
        row([nm, n, f"{a['mean']:.3f} ± {a['std']:.3f}", f"{i['mean']:.3f} ± {i['std']:.3f}", ga, gi])
    L.append(f"\n학습 모델 폴드별 gain 상위 5: {ui['avg_train']['gain_rank']['_top5']}\n")
    L.append(f"추론 모델 폴드별 gain 상위 5: {ui['avg_infer']['gain_rank']['_top5']}\n")

# ------------------------------------------------------------------ R1-7
if "symmetric_ablation" in K:
    sa = K["symmetric_ablation"]
    h("표 9 재구성 — 대칭 그룹 절제와 단독 피처 기준선 (R1-7)",
      f"구성 단위 nested CV로 재학습. 메모리 그룹 {sa['group_sizes']['MEM']}열, FLOPs 그룹 {sa['group_sizes']['FLOPS']}열. "
      "절제는 '그 정보를 다른 피처가 대체할 수 있는가', 단독은 '그 피처 하나가 얼마나 설명하는가'를 본다.")
    hdr(["조건", "피처 수", "학습 R²(log)", "학습 MAPE", "추론 R²(log)", "추론 MAPE(≥10ms)"])
    for sec, names in [("ablation", {"full": "전체", "minus_MEM": "− 메모리 그룹", "minus_FLOPS": "− FLOPs 그룹", "minus_both": "− 둘 다"}),
                       ("single_feature", {"only_mem_write": "`mem_write` 단독", "only_flops": "`flops` 단독", "only_params": "`total_params` 단독",
                                           "mem_write+backend_id": "`mem_write` + 백엔드 라벨", "flops+backend_id": "`flops` + 백엔드 라벨",
                                           "params+backend_id": "`total_params` + 백엔드 라벨", "mem_write+flops+backend_id": "`mem_write` + `flops` + 라벨"})]:
        for k, nm in names.items():
            v = sa[sec][k]
            row([nm, v["n_features"], f(v["avg_train"]["r2log"]), pct(v["avg_train"]["mape"]), f(v["avg_infer"]["r2log"]), pct(v["avg_infer"]["mape"])])
    c = sa["spearman_features"]
    L.append(f"\n로그 공간 Spearman 상관: mem_write–flops 전체 {c['all']['mem_write_vs_flops']}, 계열별 "
             + ", ".join(f"{k} {v['mem_write_vs_flops']}" for k, v in c.items() if k != "all") + ".\n")
    L.append("\n**백엔드 내부에서 타깃과의 Spearman 상관 (단변량)**\n")
    hdr(["타깃", "피처"] + BACKENDS, ["---", "---"] + ["---:"] * 4)
    for tgt, name in [("avg_train", "학습"), ("avg_infer", "추론")]:
        for feat in ["mem_write", "flops", "params"]:
            row([name if feat == "mem_write" else "", feat] + [f(sa["spearman_with_target_within_backend"][tgt][b][feat], 3) for b in BACKENDS])

# ------------------------------------------------------------------ R1-8
if "tuned_comparison" in K:
    tc = K["tuned_comparison"]
    h("표 5 재산출 — 동일 nested CV·동일 그리드 (R1-8)", tc["protocol"])
    hdr(["회귀 모델", "학습 R²(log)", "추론 R²(log)"])
    for k, v in tc.items():
        if k.startswith("_") or k == "protocol":
            continue
        row([k, f(v["avg_train"], 4), f(v["avg_infer"], 4)])

# ------------------------------------------------------------------ R1-1 (b)
if "overhead_features" in K:
    ov = K["overhead_features"]
    h("새 실험 — 소형 구간 오차 개선 시도 (R1-1)",
      "오버헤드 피처 = 배치 수, op 수 × 배치 수(학습·추론), log op 수 (" + ", ".join(f"`{c}`" for c in ov["overhead_columns"]) + "). "
      "Huber = XGBoost `reg:pseudohubererror`. 전문 모델 = 소형 셀(추론 <50 ms, 학습 <1 s)만으로 학습·검증.")
    for tgt, name, key, klabel in [("avg_train", "학습시간", "lt1s", "<1 s"), ("avg_infer", "추론시간", "lt10ms", "<10 ms")]:
        L.append(f"\n**{name}**\n")
        hdr(["모델", "R²(log)", "전체 MAPE", "전체 ±20%", f"{klabel} n", f"{klabel} MAPE", f"{klabel} MdAPE", "≥10ms MAPE"])
        for k, nm in [("baseline_120", "기준 (120 피처)"), ("plus_overhead_125", "+ 오버헤드 5개"), ("baseline_huber", "기준 + Huber"),
                      ("plus_overhead_huber", "+ 오버헤드 + Huber"), ("specialist_small", "소형 전문 모델")]:
            v = ov[tgt][k]
            row([nm + (f" ({v['n_cells']}셀)" if k == "specialist_small" else ""), f(v["r2log"]), pct(v["all"]["mape"]), pct(v["all"]["w20"]),
                 v[key]["n"], pct(v[key]["mape"]), pct(v[key]["mdape"]), pct(v["ge10ms"]["mape"])])

out = os.path.join(BASE, "paper/v4k_tables.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("저장:", out, f"({len(L)}줄)")
