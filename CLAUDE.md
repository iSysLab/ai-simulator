# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based DNN benchmarking and execution time/memory prediction framework. Benchmarks 6 model architectures (ANN, CNN, ResNet, MobileNet, Transformer, GAN) on MNIST/CIFAR-10, extracts 45 structural + hardware features, and trains ML regressors (XGBoost, RandomForest, etc.) to predict training time, inference time, and memory usage. Supports multi-platform hardware auto-detection (macOS/Windows/Linux).

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
# Full benchmark (6 models × 160 configs × 10 repeats, hours on CPU)
python run_benchmark.py

# Single model type
python run_benchmark.py --model simple_ann
python run_benchmark.py --model transformer

# Quick test (3 repeats, CPU only)
python run_benchmark.py --repeats 3 --device cpu

# Resume interrupted run
python run_benchmark.py --resume

# With op-level profiling
python run_benchmark.py --profile-ops

# Train prediction models (XGBoost + GridSearchCV, 5-fold CV)
python train_predictor.py
python train_predictor.py --cv 10 --save-models

# Generate 9 visualization figures (includes cross-platform if Mac data present)
python visualize_results.py
python visualize_results.py --input-mac results/benchmark_results_mac.json

# ONNX pipeline
python export_onnx.py                              # PyTorch → ONNX
python predict_from_onnx.py --onnx model.onnx --device cpu
python predict_from_onnx.py --demo                 # all sample predictions
```

Legacy standalone scripts (`ann.py`, `cnn.py`, `cnn_remaining.py`) have been removed — use `run_benchmark.py` instead.

## Architecture

### Pipeline Flow

```
run_benchmark.py → benchmark/ package → results/benchmark_results.json
                                              ↓
                                     train_predictor.py → results/trained_models/*.pkl
                                              ↓
                                     visualize_results.py → results/figures/*.png
```

Parallel ONNX path: `export_onnx.py` → `.onnx` files → `predict_from_onnx.py` (uses trained models to predict execution time from ONNX graph structure)

### benchmark/ Package

- **models/registry.py**: Decorator-based model factory (`@register_model("name")` → `create_model("name", **kwargs)`). All 6 model files import-register themselves.
- **models/**: `simple_ann.py`, `simple_cnn.py`, `resnet_mnist.py`, `mobilenet_mnist.py` (MNIST, 28×28×1), `transformer.py`, `gan.py` (CIFAR-10, 32×32×3)
- **configs/generator.py**: Generates 160 hyperparameter combinations across all 6 model types (cartesian products of size/depth/width params)
- **features/extractor.py**: Single source of truth for all 45 FEATURE_COLUMNS. Extracts structural features via `named_modules()` introspection + forward hooks for FLOPs + OS-auto-detected hardware info (macOS `sysctl`/`system_profiler`, Windows `wmic`/CUDA props, Linux `psutil`)
- **features/onnx_extractor.py**: Same 45-feature schema but from ONNX graph (weight shapes → FLOPs). Imports constants from extractor.py to stay in sync.
- **features/op_profiler.py**: Decomposes model into individual ops, measures per-op time/memory, simulates total
- **runner/device.py**: Auto-detects CPU/CUDA/MPS, handles sync and warmup
- **runner/data.py**: `MNISTDataManager` and `CIFAR10DataManager` — preloads batches to device memory
- **runner/experiment.py**: `ExperimentRunner` — timed training/inference loops (classification + GAN), 10 repeats with avg/std
- **results/io.py**: Atomic JSON save (`os.replace`) + corrupted file recovery + CSV export

### Prediction Pipeline (train_predictor.py)

- Loads `results/benchmark_results.json`, builds feature matrix from 45 columns defined in `FEATURE_COLUMNS`
- Trains 4 ML models: LinearRegression, RandomForest+GridSearchCV, GradientBoosting, XGBoost+GridSearchCV
- Predicts 3 targets per device (CPU/GPU separately): training time, inference time, memory
- Uses `log1p` transform on targets for stability across wide ranges (ms to hundreds of seconds)
- Key metric: R²(log) — evaluated in log space for balanced accuracy across scales

### Key Design Decisions

- **MNIST vs CIFAR-10 split**: ANN/CNN/ResNet/MobileNet use MNIST (28×28×1), Transformer/GAN use CIFAR-10 (32×32×3). Determined by `MNIST_MODELS`/`CIFAR10_MODELS` sets in `run_benchmark.py`.
- **Data preloading**: All batches are loaded to device memory before timing to exclude data transfer overhead
- **GAN handled separately**: `ExperimentRunner.run_gan()` measures adversarial training time + generator inference, distinct from classification `run()`
- **Incremental save**: Results are appended after each config completes, enabling `--resume` after interruption
- **CNN AdaptiveAvgPool2d**: `SimpleCNN` uses `AdaptiveAvgPool2d((1,1))` before classifier to fix parameter inversion bug where deeper models had fewer params

## Data Files

- `results/benchmark_results.json` / `.csv`: 330 benchmark samples (tracked in git despite .gitignore `*.json` — force-added)
- `results/trained_models/`: Saved `.pkl` predictor models (joblib)
- `results/figures/`: 9 visualization PNGs (7 base + 2 cross-platform)
- `data/`: Auto-downloaded MNIST/CIFAR-10 (gitignored)

## Language

Code comments, print output, and commit messages are in Korean (한국어). Variable/function names are in English.

### Commit Convention

`[TAG] 설명` — tags: `[feat]`, `[fix]`, `[chore]`, `[add]`, `[docs]`, `[refactor]`, `[improve]`
