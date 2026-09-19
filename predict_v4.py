"""논문 파이프라인용 배포 스크립트 — ONNX 파일 + 백엔드 → 학습(1에폭)·추론(테스트셋) 시간 예측.

구판 predict_from_onnx.py(45피처)와 별개다. 여기서는 논문 모델과 같은 경로를 쓴다:
  merged_schema NUMERIC 120열 = ONNX 구조·op-level 39열 + 하드웨어 30열 + enrich 파생 14열
                              + 입력·구성 메타 37열(데이터셋 기본값 + 구성 템플릿)
타깃은 z = log1p(초). 추정기는 benchmark/predictor/ensemble.py의 클래스 또는 XGBRegressor.

학습(번들 저장):
  python predict_v4.py --train --estimator log --model-dir results/trained_models_v4
      --estimator {xgb, log, log1p_ms, regime}  (chosen_params는 paper/v4_measured_l.json에서 읽음, 없으면 기본값)
예측:
  python predict_v4.py --onnx results/onnx_configs/<file>.onnx --backend MPS --model-dir results/trained_models_v4
      --backend 는 --list-backends 로 확인. 알려진 구성(manifest에 있는 ONNX)은 구성 템플릿을 자동으로 채운다.
      새 ONNX는 --dataset {mnist,cifar10} 와 --config-json '{"embed_dim":64,...}' 로 메타를 준다(없으면 0).
점검:
  python predict_v4.py --demo --model-dir results/trained_models_v4     # 160구성 × 4백엔드 in-sample MAPE
"""
import argparse
import json
import os
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE, "results/trained_models_v4")
BUNDLE = "bundle_v4.pkl"
TARGETS = ["avg_train", "avg_infer"]
DATASET_FIELDS = ["input_height", "input_width", "input_channels", "num_classes", "dataset_encoded", "seq_length"]
BACKEND_META = ["device", "batch_size", "test_batch_size", "device_type", "os_type", "accelerator_brand",
                "accelerator_name", "memory_type", "interconnect_type"]


def _lazy():
    from eval_fill_v4e import ONNX_DIR, REPLACE_DERIVED, config_key
    from eval_fill_v4i import load_clean
    from eval_fill_v4j import HW_NUMERIC_COLS
    from eval_fill_v4k import Ctx
    from train_merged import NUMERIC, enrich
    from benchmark.features.onnx_oplevel import get_op_level_features_onnx, get_structural_features_onnx
    from benchmark.predictor import (LogTargetXGBRegressor, RegimeEnsembleRegressor, bundle_versions)
    return locals()


def onnx_features(path, L):
    f = dict(L["get_structural_features_onnx"](path))
    f.update(L["get_op_level_features_onnx"](path))
    f["flops"] = f["total_op_flops"]
    return f


def make_estimator(name, params, L):
    from xgboost import XGBRegressor
    p = {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.1, **(params or {})}
    if name == "xgb":
        return XGBRegressor(random_state=42, n_jobs=1, verbosity=0, **p)
    if name in ("log", "log1p_ms"):
        p.pop("transform", None)
        return L["LogTargetXGBRegressor"](transform=name, **p)
    if name == "regime":
        return L["RegimeEnsembleRegressor"](**p)
    raise ValueError(name)


def chosen_params_for(name, tgt):
    path = os.path.join(BASE, "paper/v4_measured_l.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    try:
        if name == "xgb":
            return d["baseline"][tgt]["chosen_params"]
        if name in ("log", "log1p_ms"):
            return d["log_target"][tgt][name]["chosen_params"]
        if name == "regime":
            return d["regime_ensemble"][tgt]["all_gates"]["chosen_params"]
    except KeyError:
        return None


# ----------------------------------------------------------------------------- train
def train(args):
    import joblib
    L = _lazy()
    ctx = L["Ctx"]()
    raw = ctx.raw
    NUMERIC, HW = L["NUMERIC"], L["HW_NUMERIC_COLS"]

    # 백엔드 템플릿: 하드웨어 30열 + 측정 설정 메타 (백엔드 안에서 상수)
    templates = {}
    for r in raw:
        b = r["_backend"]
        if b not in templates:
            templates[b] = {c: r.get(c) for c in HW + BACKEND_META if c in r}
    # 데이터셋 기본값 (입력 메타는 데이터셋의 함수)
    ds = {}
    for r in raw:
        name = "mnist" if r["model_type"] in {"simple_ann", "simple_cnn", "resnet_mnist", "mobilenet_mnist"} else "cifar10"
        if name not in ds:
            ds[name] = {c: r.get(c, 0) for c in DATASET_FIELDS}
    # 구성 템플릿: ONNX가 못 주는 구성·구조 메타 (구성의 함수, 백엔드 무관)
    onnx_keys = set(onnx_features(os.path.join(L["ONNX_DIR"], json.load(open(os.path.join(L["ONNX_DIR"], "manifest.json")))
                                               [next(iter(json.load(open(os.path.join(L["ONNX_DIR"], "manifest.json")))))]["file"]), L))
    skip = set(HW) | set(BACKEND_META) | set(DATASET_FIELDS) | onnx_keys | set(L["REPLACE_DERIVED"]) | set(TARGETS) \
        | {"std_train", "std_infer", "train_times", "infer_times", "peak_mem_mb", "final_accuracy", "model_name"}
    cfg_rows = {}
    for r in raw:
        k = L["config_key"](r["model_type"], r.get("config", {}) or {})
        if k not in cfg_rows:
            cfg_rows[k] = {c: r.get(c, 0) for c in NUMERIC if c not in skip and c in r}
            cfg_rows[k]["model_type"] = r["model_type"]

    models, info = {}, {}
    for tgt in TARGETS:
        X, y = ctx.X(NUMERIC, tgt)
        params = chosen_params_for(args.estimator, tgt)
        est = make_estimator(args.estimator, params, L)
        t0 = time.time()
        est.fit(X, y)
        models[tgt] = est
        info[tgt] = {"params": params, "fit_s": round(time.time() - t0, 1), "n": int(len(y))}
        print(f"[train] {tgt}: {args.estimator} params={params} n={len(y)} ({info[tgt]['fit_s']}s)", flush=True)

    bundle = {"estimator": args.estimator, "models": models, "numeric_cols": NUMERIC,
              "backend_templates": templates, "dataset_defaults": ds, "config_rows": cfg_rows,
              "target_space": "log1p_seconds", "versions": L["bundle_versions"](),
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "info": info}
    os.makedirs(args.model_dir, exist_ok=True)
    out = os.path.join(args.model_dir, BUNDLE)
    joblib.dump(bundle, out)
    print(f"번들 저장: {out}  백엔드 {sorted(templates)}  구성 템플릿 {len(cfg_rows)}")


# ----------------------------------------------------------------------------- predict
def load_bundle(model_dir):
    import joblib
    return joblib.load(os.path.join(model_dir, BUNDLE))


def build_row(bundle, onnx_path, backend, L, dataset=None, config_json=None):
    if backend not in bundle["backend_templates"]:
        raise SystemExit(f"알 수 없는 백엔드 {backend!r} — 가능: {sorted(bundle['backend_templates'])}")
    row = dict(bundle["backend_templates"][backend])
    of = onnx_features(onnx_path, L)
    # 알려진 구성이면 manifest → 구성 템플릿
    man_path = os.path.join(L["ONNX_DIR"], "manifest.json")
    cfg_key, model_type = None, None
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        base = os.path.basename(onnx_path)
        for k, e in man.items():
            if e["file"] == base:
                cfg_key, model_type = k, e["model_type"]
                break
    if cfg_key and cfg_key in bundle["config_rows"]:
        row.update(bundle["config_rows"][cfg_key])
        src = "manifest 구성 템플릿"
    else:
        src = "구성 템플릿 없음(0)"
        if config_json:
            cfg = json.loads(config_json)
            model_type = cfg.pop("model_type", model_type)
            row.update({k: v for k, v in cfg.items() if k in bundle["numeric_cols"]})
            src = "--config-json"
    if model_type:
        row["model_type"] = model_type
    ds_name = dataset or ("mnist" if model_type in {"simple_ann", "simple_cnn", "resnet_mnist", "mobilenet_mnist"} else "cifar10" if model_type else None)
    if ds_name:
        row.update(bundle["dataset_defaults"][ds_name])
    row.update(of)
    row = L["enrich"](row)
    x = np.array([[float(row.get(c, 0) or 0) for c in bundle["numeric_cols"]]])
    return x, {"config_source": src, "model_type": model_type, "dataset": ds_name, "num_ops": of.get("num_ops")}


def predict_one(bundle, onnx_path, backend, L, dataset=None, config_json=None):
    x, meta = build_row(bundle, onnx_path, backend, L, dataset, config_json)
    out = {tgt: float(np.expm1(bundle["models"][tgt].predict(x)[0])) for tgt in TARGETS}
    return out, meta


def predict(args):
    L = _lazy()
    bundle = load_bundle(args.model_dir)
    pred, meta = predict_one(bundle, args.onnx, args.backend, L, args.dataset, args.config_json)
    print(f"ONNX: {args.onnx}\n백엔드: {args.backend}  추정기: {bundle['estimator']}  구성 출처: {meta['config_source']}  "
          f"model_type={meta['model_type']} dataset={meta['dataset']} ops={meta['num_ops']}")
    print(f"예측 학습시간(1에폭): {pred['avg_train']:.4f} s\n예측 추론시간(테스트셋): {pred['avg_infer']:.4f} s")


def demo(args):
    """manifest의 전 구성 × 전 백엔드를 예측해 실측과 대조 (in-sample 점검, 일반화 지표 아님)."""
    L = _lazy()
    bundle = load_bundle(args.model_dir)
    raw = L["load_clean"]()
    man = json.load(open(os.path.join(L["ONNX_DIR"], "manifest.json")))
    by_key = {}
    for r in raw:
        by_key[(L["config_key"](r["model_type"], r.get("config", {}) or {}), r["_backend"])] = r
    errs = {t: [] for t in TARGETS}
    n = 0
    for k, e in man.items():
        path = os.path.join(L["ONNX_DIR"], e["file"])
        for b in bundle["backend_templates"]:
            r = by_key.get((k, b))
            if r is None:
                continue
            pred, _ = predict_one(bundle, path, b, L)
            for t in TARGETS:
                errs[t].append(abs(pred[t] - r[t]) / max(r[t], 1e-9))
            n += 1
    for t in TARGETS:
        a = np.array(errs[t]) * 100
        print(f"[demo] {t}: n={len(a)} in-sample MAPE {a.mean():.1f}%  MdAPE {np.median(a):.1f}%  ±20% {np.mean(a < 20) * 100:.1f}%")
    print(f"셀 {n}개 (학습에 쓴 셀과 동일 — 파이프라인 정합 점검용)")


def list_backends(args):
    bundle = load_bundle(args.model_dir)
    for b, t in bundle["backend_templates"].items():
        print(f"{b:12s} device={t.get('device')} cpu_cores={t.get('cpu_cores')} gpu_cores={t.get('gpu_cores')} "
              f"gpu_memory_gb={t.get('gpu_memory_gb')} ram_total_gb={t.get('ram_total_gb')}")
    print("추정기:", bundle["estimator"], "| 학습:", bundle["trained_at"], "| 버전:", bundle["versions"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--estimator", default="log", choices=["xgb", "log", "log1p_ms", "regime"])
    ap.add_argument("--onnx")
    ap.add_argument("--backend")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"])
    ap.add_argument("--config-json")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--list-backends", action="store_true")
    args = ap.parse_args()
    if args.train:
        train(args)
    elif args.demo:
        demo(args)
    elif args.list_backends:
        list_backends(args)
    elif args.onnx:
        if not args.backend:
            raise SystemExit("--backend 필요 (--list-backends 로 확인)")
        predict(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
