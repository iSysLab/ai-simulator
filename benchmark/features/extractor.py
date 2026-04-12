"""111차원 피처 추출 — repo 루트 `features/extractor.py`(dal-merge)에 위임.

`run_benchmark.py`는 ijunsoo 스타일 인자
`(model, model_type, input_shape, device_str, config, batch_size)`를 사용한다.
collect 스크립트는 루트 `features.extractor.extract_features` 직접 호출.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from features.extractor import extract_features as extract_features_dal  # noqa: E402

# ijunsoo 벤치마크 6종 → model_family_encoded (train_predictor와 동일)
MODEL_FAMILY_MAP = {
    "simple_ann": 0,
    "simple_cnn": 1,
    "resnet_mnist": 2,
    "mobilenet_mnist": 3,
    "transformer": 4,
    "gan": 5,
}

# 데이터셋 인코딩: MNIST 계열 0, CIFAR-10 계열 1
_DATASET_ENCODED = {
    "simple_ann": 0,
    "simple_cnn": 0,
    "resnet_mnist": 0,
    "mobilenet_mnist": 0,
    "transformer": 1,
    "gan": 1,
}


def _dal_family(model_type: str) -> str:
    if model_type == "simple_ann":
        return "ann"
    if model_type in ("simple_cnn", "resnet_mnist", "mobilenet_mnist"):
        return "cnn"
    if model_type == "transformer":
        return "transformer"
    if model_type == "gan":
        return "gan"
    raise ValueError(f"unknown model_type: {model_type}")


def _build_model_config(model_type: str, config: dict) -> dict:
    """ijunsoo config → dal `features.extractor`용 model_config."""
    if model_type == "simple_ann":
        return {
            "hidden_size": config.get("hidden_size", 0),
            "num_hidden_layers": config.get("num_layers", 1),
        }
    if model_type == "simple_cnn":
        return {
            "num_filters": config.get("num_filters", 0),
            "num_conv_layers": config.get("num_conv_layers", 0),
            "has_batchnorm": 1 if config.get("use_batchnorm") else 0,
            "has_pooling": 1,
            "kernel_size": 3,
            "num_fc_layers": 1,
            "has_residual": 0,
        }
    if model_type == "resnet_mnist":
        layers = config.get("layers", [2, 2, 2, 2])
        base_w = config.get("base_width", 16)
        n_conv = sum(l * 2 for l in layers)
        return {
            "num_filters": base_w,
            "num_conv_layers": n_conv,
            "has_batchnorm": 1,
            "has_pooling": 1,
            "kernel_size": 3,
            "num_fc_layers": 1,
            "has_residual": 1,
        }
    if model_type == "mobilenet_mnist":
        wm = config.get("width_mult", 1.0)
        nb = config.get("num_blocks", 5)
        return {
            "num_filters": max(8, int(32 * wm)),
            "num_conv_layers": nb * 3,
            "has_batchnorm": 1,
            "has_pooling": 1,
            "kernel_size": 3,
            "num_fc_layers": 1,
            "has_residual": 1,
        }
    if model_type == "transformer":
        return {
            "embed_dim": config.get("embed_dim", 0),
            "num_transformer_layers": config.get("num_layers", 0),
            "num_heads": config.get("num_heads", 0),
            "patch_size": config.get("patch_size", 0),
        }
    if model_type == "gan":
        return {
            "latent_dim": config.get("latent_dim", 0),
            "g_hidden_dims": list(config.get("g_hidden_dims", [])),
        }
    return {}


def _build_input_config(model_type: str, input_shape: tuple, batch_size: int) -> dict:
    _, ch, h, w = input_shape
    return {
        "batch_size": batch_size,
        "input_channels": ch,
        "input_height": h,
        "input_width": w,
        "num_classes": 10,
        "dataset_encoded": _DATASET_ENCODED.get(model_type, 0),
    }


def extract_features(
    model,
    model_type: str,
    input_shape=(1, 1, 28, 28),
    device_str: str = "cpu",
    config=None,
    batch_size: int = 64,
):
    """ijunsoo 벤치마크용: 111차원 dict, `model_family_encoded`는 6종 체계로 덮어씀.

    Args:
        model: PyTorch 모듈
        model_type: simple_ann | simple_cnn | resnet_mnist | mobilenet_mnist | transformer | gan
        input_shape: (N, C, H, W)
        device_str: cpu | cuda | mps
        config: generator에서 온 하이퍼파라미터 dict
        batch_size: 배치 크기 (입력 피처용)
    """
    if config is None:
        config = {}

    dal_type = _dal_family(model_type)
    mconf = _build_model_config(model_type, config)
    iconf = _build_input_config(model_type, input_shape, batch_size)

    ds = str(device_str).lower()
    if ds in ("mps:0", "cuda:0"):
        ds = ds.split(":")[0]

    feat = extract_features_dal(model, dal_type, mconf, ds, iconf)
    feat["model_family_encoded"] = MODEL_FAMILY_MAP[model_type]
    return feat


# collect / 외부 스크립트 호환용 별칭
extract_features_collect = extract_features_dal

__all__ = [
    "extract_features",
    "extract_features_collect",
    "extract_features_dal",
    "MODEL_FAMILY_MAP",
]
