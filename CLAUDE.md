# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch benchmarking project that compares ANN (Artificial Neural Network) and CNN (Convolutional Neural Network) performance on the MNIST dataset across CPU and GPU (CUDA/MPS) devices. The project measures training time, inference time, and accuracy across different model configurations, repeating each experiment 10 times for statistical reliability.

## Setup

```bash
# Python 3.11 venv
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`

## Running Experiments

```bash
python ann.py            # ANN benchmark (all 9 configs × 10 repeats per device)
python cnn.py            # CNN benchmark (all 9 configs × 10 repeats per device)
python cnn_remaining.py  # CNN partial re-run (MPS-focused, CPU only runs (128,5))
```

Each script auto-detects available devices (CPU always, plus CUDA on Windows or MPS on Mac) and runs experiments on all of them. Results are saved to `ann_results.json` and `cnn_results.json`. Large CNN configs (e.g., 128 filters, 5 layers) can take 10+ minutes per repeat on CPU.

## Architecture

All three scripts follow the same pattern and are self-contained (no shared modules):

1. **Hyperparameters** — batch size 64, LR 0.01, 1 epoch per run, 10 repeats
2. **Device setup** — auto-detect CPU/CUDA/MPS with sync and cache helpers
3. **Data loading** — MNIST downloaded to `./data/`, pre-loaded to device memory to exclude data transfer from timing
4. **Model definition** — configurable by (neurons/filters count, layer count)
5. **Experiment loop** — iterates over 9 configs: `{32,64,128} × {2,3,5}` layers
6. **Results** — JSON output with avg/std for train time, inference time, and accuracy

### Model Configs

Both ANN and CNN test the same 9 (size, depth) combinations. The size parameter means hidden neurons for ANN and conv filters for CNN.

| Config Param | ANN (`SimpleANN`) | CNN (`SimpleCNN`) |
|---|---|---|
| Size | hidden layer neuron count | base filter count (doubles per layer, capped at ×4) |
| Depth | number of hidden `Linear` layers | number of `Conv2d` layers (pooling every 2nd) |
| Classifier | single output Linear | flatten → 128-unit FC → output |

### Key Differences: `cnn_remaining.py`

This is a partial re-run variant of `cnn.py` designed for completing experiments that were interrupted. It hardcodes CPU to only run `(128, 5)` and runs all 8 configs on MPS. It also omits accuracy tracking in its result dict (only times).

## Language

Code comments and print output are in Korean (한국어). The `ANN 실행시간 이미지/` directory contains benchmark result screenshots.
