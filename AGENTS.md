# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

PyTorch DNN execution-time prediction across heterogeneous hardware. Six model families (ANN, CNN, ResNet, MobileNet on MNIST; ViT and GAN on CIFAR-10) are measured on four hardware platforms: x86 CPU and CUDA GPU (desktop: Ryzen 7 7800X3D + RTX 4060 Ti) and Apple ARM CPU and MPS (MacBook, Apple M4). The paper pipeline (v4, ACM RACS 2026) combines ONNX-graph architectural and op-level features with configuration and hardware metadata (130-column schema, 120 numeric inputs) and trains one XGBoost regressor per target (training time, inference time) that covers all four platforms. The canonical dataset is 158 configurations × 4 platforms = 632 cells. Memory is not a prediction target (the recorded value is just `total_params × 4`).

The older v3 pipeline (96 PyTorch-hook features, per-device models: `train_predictor.py`, `visualize_results.py`, `predict_from_onnx.py`) is kept as legacy; `run_benchmark.py` is still the measurement tool. README.md has the results, the changes made while writing the paper, and the reproduction steps.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Tested with Python 3.13.5, torch 2.10.0, scikit-learn 1.8.0 (≥1.6 required for `GroupKFold(shuffle=...)`), xgboost 3.2.0, numpy 2.4.3, scipy 1.18.1, onnx 1.20.1. Run every script from the repo root (several use CWD-relative paths).

## Commands

```bash
# Paper pipeline (v4). Steps 1-2 regenerate gitignored inputs.
python enrich_oplevel.py           # results/benchmark_results{,_mac}.json → results/*_enriched.json
python export_onnx_configs.py      # → results/onnx_configs/ (160 ONNX, opset 17) + manifest.json
python eval_fill_v4i.py            # 632-cell load_clean, Tables 3/9/10 → paper/v4_measured_i.json
python eval_fill_v4j.py            # nested CV, Tables 6/7/8, error stats → paper/v4_measured_j.json
python eval_fill_v4k.py [--only a,b]   # camera-ready analyses → paper/v4_measured_k.json
python make_tables_v4k.py          # → paper/v4k_tables.md
python eval_fill_v4l.py [--only a,b]   # log-target / ensemble study → paper/v4_measured_l.json
python make_tables_v4l.py          # → paper/v4l_tables.md

# Predictor CLI (default --estimator log = log-target XGBoost; --estimator xgb = the paper's log1p XGBoost)
python predict_v4.py --train       # → results/trained_models_v4/bundle_v4.pkl (gitignored)
python predict_v4.py --onnx model.onnx --backend MPS --dataset mnist   # backends: "Desktop CPU", CUDA, "Mac CPU", MPS
python predict_v4.py --list-backends

# Measurement (protocol v2). Always pass --output: the default appends to the tracked raw dataset.
python check_env.py --json > paper/env_<host>.json
python run_benchmark.py --device cuda --repeats 10 --require-idle --tag remeasure_v2 --output results/remeasure_cuda_v2.json
python run_benchmark.py --quick --output results/quick_test.json
# other flags: --model --resume --discard-first --subset stratified:N|every:K|names:... --batch-size --idle-cpu-max --idle-gpu-max --idle-wait --profile-ops

# External-device analyses (paper/72-77)
python eval_external_backend.py --external results/external_<name>.json --env paper/env_<host>.json --name <name>
python eval_external_fewshot.py --external results/external_<name>.json --env paper/env_<host>.json --name <name>
python eval_logo_gpu.py --external <name>:results/external_<name>.json:paper/env_<host>.json   # repeat per GPU

# Legacy (v3, 96 features)
python train_predictor.py --features core|full|merged --cv 10   # --save-models overwrites results/trained_models/
python visualize_results.py --input-mac results/benchmark_results_mac.json --output-dir <dir>   # default overwrites results/figures/
python predict_from_onnx.py [model.onnx]   # no argument = all results/onnx_samples
```

## Architecture

### Pipeline Flow

```
run_benchmark.py → results/benchmark_results.json (desktop), results/benchmark_results_mac.json
        ↓ enrich_oplevel.py (PyTorch op-level features)
results/*_enriched.json → export_onnx_configs.py → results/onnx_configs/*.onnx
        ↓ eval_fill_v4i.load_clean() (632 cells) + ONNX features (eval_fill_v4e.onnx_feats_for)
eval_fill_v4i/j/k/l.py → paper/v4_measured_*.json → make_tables_v4k/l.py → paper/v4k_tables.md, v4l_tables.md
predict_v4.py --train → results/trained_models_v4/bundle_v4.pkl
```

### benchmark/ Package

- **models/registry.py**: Decorator-based model factory (`@register_model("name")` → `create_model("name", **kwargs)`). `models/__init__.py` is empty, so scripts import the six model modules explicitly to register them — those "unused" imports are required.
- **models/**: `simple_ann.py`, `simple_cnn.py`, `resnet_mnist.py`, `mobilenet_mnist.py` (MNIST, 28×28×1), `transformer.py` (ViT), `gan.py` (CIFAR-10, 32×32×3)
- **configs/generator.py**: 160 configurations (ANN 42, CNN 60, ResNet 18, MobileNet 20, ViT 12, GAN 8)
- **features/merged_schema.py**: v4 schema — `MERGED_FEATURE_COLUMNS` (130), `STRING_FEATURES` (10), `OP_LEVEL_FEATURES` (14), `NUMERIC_FEATURE_COLUMNS` (120). `train_merged.NUMERIC` is the same list and is what the v4 scripts import.
- **features/onnx_oplevel.py**: torch-free op-level features from ONNX shape inference. Memory read = input + weight bytes, write = output bytes, from a static batch-1 graph (a proxy, not measured traffic).
- **features/op_profiler.py**: PyTorch-hook op decomposition used by `enrich_oplevel.py`. GAN forward fails with the image-shaped dummy input, so GAN op features stay 0 there; the ONNX path recomputes them.
- **features/extractor.py, onnx_extractor.py**: legacy 96-feature schema (`CORE_FEATURE_COLUMNS` 50 + `EXTENDED_FEATURE_COLUMNS` 46) plus hardware via `benchmark/platform`
- **platform/**: hardware auto-detection (`detector`, `cpu_info`, `gpu_info`, `memory_info`, `compatibility`) and `software_env.py` (`sw_*` versions, `load_*` pre-run load, idle check)
- **runner/device.py, data.py, experiment.py**: device sync and peak memory; `MNISTDataManager`/`CIFAR10DataManager` preload batches to device memory; `ExperimentRunner` with protocol `v2-uniform-warmup` (one real training step + one eval pass for every backend and family), per-repeat times, `peak_mem_mb`, `--discard-first`
- **predictor/ensemble.py**: log-target and other regressors used by `eval_fill_v4l.py` and `predict_v4.py`
- **results/io.py**: incremental save with resume, atomic `os.replace` with retries

### Key Design Decisions

- **Canonical loader** `eval_fill_v4i.load_clean()`: moves 48 M4 rows found in the desktop file to Mac CPU and averages them, drops 12 ViT rows without hardware info and 2 ViT configurations incomplete on x86 (→ 158 × 4 = 632), sets `batch_size` 64, and (since 2026-09-16, f054d66) fills 24 structural fields missing on desktop rows from the matching Mac rows. Paper tables were produced before that fix (`paper/v4_measured_j.json` from Jul 13, `paper/v4_measured_k_before_schema_fix.json`); current code reproduces `paper/v4_measured_k.json`.
- **Targets and validation**: `log1p(seconds)`, metric R²(log). Nested CV: outer 5-fold (random `KFold`, or configuration-wise `GroupKFold` grouped by `model_name`), inner 4-fold GridSearchCV, seed 42. XGBoost grid: n_estimators {200, 400} × max_depth {3, 6}, learning_rate 0.1.
- **log1p flaw** (paper/71): log1p(y) ≈ y below 0.1 s, so sub-10 ms cells barely affect the loss; a log(y) target (`LogTargetXGBRegressor`) cuts inference MAPE 58% → 16%. Not reflected in the paper tables.
- **Hardware features act as platform identifiers**: replacing the 30 numeric hardware columns with a platform label gives the same accuracy, and the external devices (GTX 1050, i3-8100, RTX 3060, i7-10700F) showed the model does not use specs as a continuous speed axis.
- **Measurement**: training = one epoch over the training set (batch 64), inference = the full test set (batch 1000; for GAN, the generator produces ~50k images in batches of 64), mean of 10 repeats. Batches are preloaded to device memory; GAN uses `ExperimentRunner.run_gan()`.
- **MNIST vs CIFAR-10 split**: `MNIST_MODELS`/`CIFAR10_MODELS` sets in `run_benchmark.py`.
- **CNN AdaptiveAvgPool2d**: `SimpleCNN` uses `AdaptiveAvgPool2d((1,1))` before the classifier to fix the parameter-inversion bug where deeper models had fewer params.
- **Merge-save**: `eval_fill_v4k.py`/`eval_fill_v4l.py` overwrite only the sections they run, so one JSON can mix runs.

## Data Files

- `results/benchmark_results.json`: desktop raw data (protocol v1, March 2026), 378 rows — x86 CPU 218 (158 desktop + 48 mixed-in M4 + 12 ViT without hardware info) and CUDA 160. `results/benchmark_results_mac.json`: Mac raw data, 320 rows (CPU 160, MPS 160). Both are force-added despite `.gitignore` `*.json`.
- `results/*_enriched.json`, `results/onnx_configs/`, `results/trained_models_v4/`: gitignored; regenerate with the commands above.
- `results/remeasure_{cuda,mps}_v2.json`: protocol-v2 re-measurements (160 rows each). `results/external_*.json`: external devices (3 repeats). `results/probe_*.json`, `results/diag_*`: desktop drift probes.
- `paper/v4_measured_*.json`: sources of the paper and analysis numbers (`_j` headline and Tables 6–8, `_i` Tables 3/9/10, `_k` and `_k_before_schema_fix` camera-ready, `_l` log target; `v4_measured{,_b,_d,_e,_f,_g,_h}` are earlier 698-row iterations). `paper/env_*.json`: `check_env.py --json` per host.
- `results/trained_models/`, `results/figures/`: legacy models and figures built from March 2026 data.
- `data/`: auto-downloaded MNIST/CIFAR-10 (gitignored)

## Language

Code comments, print output, and commit messages are in Korean (한국어). Variable/function names are in English.

### Commit Convention

`[TAG] 설명` — tags: `[feat]`, `[fix]`, `[chore]`, `[add]`, `[docs]`, `[refactor]`, `[improve]`
