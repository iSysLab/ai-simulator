# GPU 메타모델로 results/onnx_samples 전부 예측 (프로젝트 루트에서 실행)
# 사용: powershell -ExecutionPolicy Bypass -File scripts/run_onnx_predictions_gpu.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$common = @(
    "--device", "cuda",
    "--model-dir", "results/trained_models_gpu"
)

function Run-Predict($onnxRel, $extra) {
    $onnx = Join-Path $root $onnxRel
    $args = @("scripts/predict_from_onnx.py", $onnx) + $common + $extra
    Write-Host "`n>>> $onnxRel" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "실패: $onnxRel" }
}

# ANN
Run-Predict "results/onnx_samples/ann_h128_l1.onnx" @()
Run-Predict "results/onnx_samples/ann_h256_l2.onnx" @()
Run-Predict "results/onnx_samples/ann_h512_l3.onnx" @()

# CNN
Run-Predict "results/onnx_samples/cnn_f32_l2.onnx" @()
Run-Predict "results/onnx_samples/cnn_f64_l3.onnx" @()
Run-Predict "results/onnx_samples/cnn_f128_l4.onnx" @()

# ResNet
Run-Predict "results/onnx_samples/resnet_2222_w16.onnx" @()
Run-Predict "results/onnx_samples/resnet_2222_w32.onnx" @()
Run-Predict "results/onnx_samples/resnet_2222_w64.onnx" @()

# ViT
Run-Predict "results/onnx_samples/vit_d64_l2_h4.onnx" @("--embed-dim", "64", "--num-heads", "4", "--patch-size", "4")
Run-Predict "results/onnx_samples/vit_d128_l4_h4.onnx" @("--embed-dim", "128", "--num-heads", "4", "--patch-size", "4")
Run-Predict "results/onnx_samples/vit_d256_l4_h8.onnx" @("--embed-dim", "256", "--num-heads", "8", "--patch-size", "4")

# GAN
Run-Predict "results/onnx_samples/gan_z64_G128_256.onnx" @("--latent-dim", "64")
Run-Predict "results/onnx_samples/gan_z128_G256_512_1024.onnx" @("--latent-dim", "128")

Write-Host "`nDone: all ONNX samples (GPU meta-models, cuda)." -ForegroundColor Green
