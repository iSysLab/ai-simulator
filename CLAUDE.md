# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based DNN benchmarking and execution time/memory prediction framework. Benchmarks 6 model architectures (ANN, CNN, ResNet, MobileNet, Transformer, GAN) on MNIST/CIFAR-10, extracts **111** structural + hardware features (**Feature Schema v2.0**, `scripts/train_predictor.py` → `FEATURE_COLUMNS`), and trains ML regressors (XGBoost, RandomForest, etc.) to predict training time, inference time, and memory usage. Supports multi-platform hardware auto-detection (macOS/Windows/Linux).

## Documentation

| File | Purpose |
|------|---------|
| `README.md` (root) | **Single unified report** — overview, repo layout, quick start, 111-dim feature schema, methodology, results, ONNX path, YAML schema draft, Stage 1–5 history, branch merge, troubleshooting |
| `CLAUDE.md` (this file) | Short English cheat-sheet for AI assistants |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `scikit-learn`, `xgboost`, `onnx`, `joblib`, `psutil` (matplotlib for visualization)

## Commands

```bash
# All CLI entry points live under scripts/ (run from repo root ai-simulator/)

# Full benchmark (6 models × 160 configs × 10 repeats, hours on CPU)
python scripts/run_benchmark.py

# Single model type
python scripts/run_benchmark.py --model simple_ann
python scripts/run_benchmark.py --model transformer

# Quick test (3 repeats, CPU only)
python scripts/run_benchmark.py --repeats 3 --device cpu

# Resume interrupted run
python scripts/run_benchmark.py --resume

# With op-level profiling
python scripts/run_benchmark.py --profile-ops

# Train prediction models (XGBoost + GridSearchCV, 5-fold CV)
python scripts/train_predictor.py
python scripts/train_predictor.py --cv 10 --save-models

# Generate 9 visualization figures (includes cross-platform if Mac data present)
python scripts/visualize_results.py
python scripts/visualize_results.py --input-mac results/benchmark_results_mac.json

# ONNX pipeline
python scripts/export_onnx.py
python scripts/predict_from_onnx.py path/to/model.onnx --device cpu
python scripts/predict_from_onnx.py

# One-click ONNX predict + JSON/CSV save
python scripts/test.py model.onnx
python scripts/test.py --all
python scripts/test.py --benchmark --model simple_ann --repeats 3
```

Legacy standalone scripts (`ann.py`, `cnn.py`, `cnn_remaining.py`) have been removed — use `scripts/run_benchmark.py` instead.

## Architecture

### Pipeline Flow

```
scripts/run_benchmark.py → benchmark/ package → results/benchmark_results.json
                                              ↓
                                     scripts/train_predictor.py → results/trained_models/*.pkl
                                              ↓
                                     scripts/visualize_results.py → results/figures/*.png
```

Parallel ONNX path: `scripts/export_onnx.py` → `.onnx` → `scripts/predict_from_onnx.py` / `scripts/test.py`

### benchmark/ Package

- **models/registry.py**: Decorator-based model factory (`@register_model("name")` → `create_model("name", **kwargs)`). All 6 model files import-register themselves.
- **models/**: `simple_ann.py`, `simple_cnn.py`, `resnet_mnist.py`, `mobilenet_mnist.py` (MNIST, 28×28×1), `transformer.py`, `gan.py` (CIFAR-10, 32×32×3)
- **configs/generator.py**: Generates 160 hyperparameter combinations across all 6 model types (ANN 42 + CNN 60 + ResNet 18 + MobileNet 20 + ViT 12 + GAN 8)
- **`benchmark/dal/extractor.py`**: Canonical **111** `FEATURE_COLUMNS` PyTorch extraction (`named_modules()`, hooks, op-level stats).
- **`benchmark/dal/op_profiler.py`**: FLOPs/memory op decomposition for the dal feature schema.
- **`benchmark/support/`**: `hardware_info.py`, `device_utils.py`, `timer.py`, legacy `onnx_feature_extractor.py` (Stage 5 ONNX path).
- **`benchmark/features/extractor.py`**: **Wrapper** mapping ijunsoo `model_type` → dal `ann/cnn/...` then calling `benchmark.dal.extractor`.
- **`benchmark/features/onnx_extractor.py`**: ONNX graph → feature dict aligned with `feature_columns.pkl` (missing keys default to 0 when predicting).
- **`benchmark/features/op_profiler.py`**: ijunsoo-style op timing profiler (used by `scripts/run_benchmark.py --profile-ops`), distinct from `benchmark/dal/op_profiler.py`.
- **runner/device.py**: Auto-detects CPU/CUDA/MPS, handles sync and warmup
- **runner/data.py**: `MNISTDataManager` and `CIFAR10DataManager` — preloads batches to device memory
- **runner/experiment.py**: `ExperimentRunner` — timed training/inference loops (classification + GAN), 10 repeats with avg/std
- **results/io.py**: Atomic JSON save (`os.replace`) + corrupted file recovery + CSV export

### Prediction Pipeline (`scripts/train_predictor.py`)

- Loads `results/benchmark_results.json`, builds feature matrix from **111** columns defined in `FEATURE_COLUMNS` (with `enrich_result` for legacy rows)
- Trains 4 ML models: LinearRegression, RandomForest+GridSearchCV, GradientBoosting, XGBoost+GridSearchCV
- Predicts 3 targets per device (CPU/GPU separately): training time, inference time, memory
- Uses `log1p` transform on targets for stability across wide ranges (ms to hundreds of seconds)
- Key metric: R²(log) — evaluated in log space for balanced accuracy across scales

### Key Design Decisions

- **MNIST vs CIFAR-10 split**: ANN/CNN/ResNet/MobileNet use MNIST (28×28×1), Transformer/GAN use CIFAR-10 (32×32×3). Determined by `MNIST_MODELS`/`CIFAR10_MODELS` sets in `scripts/run_benchmark.py`.
- **Data preloading**: All batches are loaded to device memory before timing to exclude data transfer overhead
- **GAN handled separately**: `ExperimentRunner.run_gan()` measures adversarial training time + generator inference, distinct from classification `run()`
- **Incremental save**: Results are appended after each config completes, enabling `--resume` after interruption
- **CNN AdaptiveAvgPool2d**: `SimpleCNN` uses `AdaptiveAvgPool2d((1,1))` before classifier to fix parameter inversion bug where deeper models had fewer params

## Legacy archive

- **`past/`**: Stage별 `experiments/`, `collect/`, `data/stage*`, `reports/`(figures only), 루트 구 `models/` 등. Stage 1–5 narrative is now summarized in `README.md` §10. 메인 코드는 여기를 import하지 않음.
- To run a legacy script from `past/`, add the folder to `PYTHONPATH` (from the repo root):
  - PowerShell: `$env:PYTHONPATH = "$PWD\past"; python past/collect/ann_collector.py`
  - bash: `export PYTHONPATH="$(pwd)/past" && python past/collect/ann_collector.py`

## Data Files

- `results/benchmark_results.json` / `.csv`: benchmark samples (tracked in git despite .gitignore `*.json` — force-added)
- `results/trained_models/`: Saved `.pkl` predictor models (joblib)
- `results/figures/`: 9 visualization PNGs (7 base + 2 cross-platform)
- `data/`: Auto-downloaded MNIST/CIFAR-10 (gitignored)

## Language

Code comments, print output, and commit messages are in Korean (한국어). Variable/function names are in English.

### Commit Convention

`[TAG] 설명` — tags: `[feat]`, `[fix]`, `[chore]`, `[add]`, `[docs]`, `[refactor]`, `[improve]`
