"""벤치마크 전체 unique config를 ONNX로 export하는 스크립트.

enrich_oplevel.py와 동일한 방식으로 각 행의 config에서 모델을 재구성하되,
op_profiler 대신 torch.onnx.export로 그래프를 저장한다.
LayerNorm/GELU가 단일 노드로 남도록 가능한 한 높은 opset을 시도한다.

산출:
  results/onnx_configs/<model_type>__<sha1(config)[:12]>.onnx
  results/onnx_configs/manifest.json  (config_key -> 파일·model_type·config)

사용: python export_onnx_configs.py
"""
import hashlib
import json
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

logging.getLogger("torch.onnx").setLevel(logging.ERROR)

from benchmark.models import (  # noqa: E402,F401
    simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist, transformer, gan,
)
from benchmark.models.registry import create_model  # noqa: E402
from enrich_oplevel import input_shape_for, config_key  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "results/onnx_configs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = (json.load(open(os.path.join(BASE, "results/benchmark_results_enriched.json")))
            + json.load(open(os.path.join(BASE, "results/benchmark_results_mac_enriched.json"))))

    manifest, failures = {}, {}
    for r in data:
        mt = r["model_type"]
        cfg = r.get("config", {}) or {}
        key = config_key(mt, cfg)
        if key in manifest or key in failures:
            continue

        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        path = os.path.join(OUT_DIR, f"{mt}__{h}.onnx")
        try:
            model = create_model(mt, **cfg)
            model.eval()
            if mt == "gan":  # SimpleGAN.forward는 잠재벡터 z를 받음
                dummy = torch.zeros(1, cfg.get("latent_dim", 128))
            else:
                dummy = torch.zeros(*input_shape_for(mt, cfg))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                err = None
                for opset in (17, 13, 11):  # LayerNorm 단일 노드는 opset 17+
                    try:
                        torch.onnx.export(model, dummy, path, dynamo=False,
                                          export_params=True, opset_version=opset,
                                          do_constant_folding=True,
                                          input_names=["input"], output_names=["output"])
                        err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        err = e
                if err is not None:
                    raise err
            manifest[key] = {"file": os.path.basename(path), "opset": opset,
                             "model_type": mt, "config": cfg}
            print(f"OK  {mt:16s} opset{opset} {os.path.basename(path)}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"FAIL {mt:16s} {e}", flush=True)

    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\nunique config {len(manifest) + len(failures)} | 성공 {len(manifest)} | 실패 {len(failures)}")
    for k, msg in failures.items():
        print(f"  ✗ {k.split('::')[0]}  {msg}")


if __name__ == "__main__":
    main()
