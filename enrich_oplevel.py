"""벤치마크 JSON에 op-level 피처를 채워 넣는 보강(enrichment) 스크립트.

배경:
  벤치 JSON에는 op-level 피처(total_op_memory_write/read, flops_ratio_* 등)가
  저장돼 있지 않다(0으로 채워짐). 이 피처들은 논문의 "메모리바운드" 분석 근거이므로,
  각 행의 config로 모델을 재구성해 op_profiler로 계산해 채운다.

특징:
  - op-level 피처는 디바이스 무관(구조적)이므로 unique (model_type, config)당 1번만 계산.
  - 원본은 보존하고 `*_enriched.json`으로 저장.
  - 프로파일 실패(예: GAN forward 특수성) 시 해당 config만 건너뛰고 로그.

사용:
  python enrich_oplevel.py
  python enrich_oplevel.py --input results/benchmark_results.json --output results/benchmark_results_enriched.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
# 모델 모듈 import → @register_model 데코레이터 실행(레지스트리 등록)
from benchmark.models import (  # noqa: E402,F401
    simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist, transformer, gan,
)
from benchmark.models.registry import create_model  # noqa: E402
from benchmark.features.op_profiler import (  # noqa: E402
    decompose_model, get_op_level_features,
)


def input_shape_for(model_type, cfg):
    """모델 타입·config에서 (N, C, H, W) 입력 shape 결정."""
    if model_type == "transformer":
        s = cfg.get("img_size", 32)
        return (1, cfg.get("in_channels", 3), s, s)
    if model_type == "gan":
        s = cfg.get("img_size", 32)
        return (1, cfg.get("img_channels", 3), s, s)
    # MNIST 계열 (ann/cnn/resnet/mobilenet)
    return (1, 1, 28, 28)


def config_key(model_type, cfg):
    return model_type + "::" + json.dumps(cfg, sort_keys=True)


def enrich_file(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    cache = {}          # config_key -> op-level feature dict (or None)
    failures = {}       # config_key -> error message

    for r in data:
        mt = r["model_type"]
        cfg = r.get("config", {}) or {}
        key = config_key(mt, cfg)

        if key not in cache:
            try:
                model = create_model(mt, **cfg)
                model.eval()
                shape = input_shape_for(mt, cfg)
                with torch.no_grad():
                    ops = decompose_model(model, input_shape=shape)
                feats = get_op_level_features(ops)
                cache[key] = feats
            except Exception as e:  # noqa: BLE001
                failures[key] = f"{type(e).__name__}: {e}"
                cache[key] = None

        if cache[key]:
            r.update(cache[key])

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    n_ok = sum(1 for v in cache.values() if v)
    print(f"[{os.path.basename(input_path)}] 행 {len(data)} | "
          f"unique config {len(cache)} | 프로파일 성공 {n_ok} | 실패 {len(failures)}")
    for k, msg in failures.items():
        print(f"   ✗ {k.split('::')[0]}  {msg}")
    print(f"   → 저장: {output_path}")
    return len(data), len(cache), failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.input:
        out = args.output or args.input.replace(".json", "_enriched.json")
        enrich_file(args.input, out)
    else:
        # 기본: 두 플랫폼 파일 모두 보강
        pairs = [
            ("results/benchmark_results.json",
             "results/benchmark_results_enriched.json"),
            ("results/benchmark_results_mac.json",
             "results/benchmark_results_mac_enriched.json"),
        ]
        for src, dst in pairs:
            if os.path.exists(src):
                enrich_file(src, dst)
            else:
                print(f"(건너뜀) 없음: {src}")


if __name__ == "__main__":
    main()
