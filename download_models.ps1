# Pixelle-Video model downloads (FLUX 3 Image Klein + Wan2.2 I2V A14B)
# Run: powershell -ExecutionPolicy Bypass -File .\download_models.ps1
# Requires: ComfyUI portable python (from setup_comfyui.ps1)
# Downloads only the exact files the workflows reference (no wildcards).
# Re-runs are resumable: already-downloaded files are skipped, and
# huggingface_hub continues partial downloads from its cache.
$ErrorActionPreference = "Stop"

$Portable = "C:\AI\ComfyUI\ComfyUI_windows_portable"
$Cu       = "$Portable\ComfyUI"
$Py       = "$Portable\python_embeded\python.exe"
$Models   = "$Cu\models"
$Staging = "C:\AI\ComfyUI\.hf-staging"

Write-Host "=== Pixelle-Video model downloads (~36 GB) ===" -ForegroundColor Cyan

# Disk space check (~36 GB needed)
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($freeGB -lt 40) { Write-Warning "Only ${freeGB}GB free on C:. This download needs ~36GB." }

# HF token: only the FLUX 3 repo is gated (accept the license at
# https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy).
# If no token is in the environment, prompt for it securely. The token is
# kept in the process environment only: never printed, never logged,
# never written to disk.
if (-not $env:HF_TOKEN) {
    Write-Host "No HF_TOKEN in environment. The FLUX 3 repo is gated."
    $secure = Read-Host "Paste your Hugging Face token (input is hidden)" -AsSecureString
    $env:HF_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    Clear-Variable -Name secure
}

# Download one exact file from a repo into a ComfyUI models subfolder.
function Download-Exact {
    param([string]$Repo, [string]$Include, [string]$DestDir, [string]$DestName)
    $dstFile = Join-Path $DestDir $DestName
    if (Test-Path $dstFile) {
        Write-Host "  (already present) $DestName"
        return
    }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    New-Item -ItemType Directory -Path $Staging -Force | Out-Null
    & $Py -m huggingface_hub download $Repo --include $Include --local-dir $Staging
    $srcFile = Join-Path $Staging $Include
    if (-not (Test-Path $srcFile)) {
        throw "Download did not produce expected file: $Repo/$Include"
    }
    Move-Item $srcFile $dstFile -Force
}

# 1. FLUX 3 Image Klein (~10.3 GB, gated repo)
Write-Host "[1/4] FLUX 3 Image Klein (~10.3 GB, gated repo)..." -ForegroundColor Yellow
Download-Exact "black-forest-labs/flux-3-image-klein-comfy" `
    "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors" "$Models\diffusion_models" "flux3_image_klein_sd_int8_convrot.safetensors"
Download-Exact "black-forest-labs/flux-3-image-klein-comfy" `
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors" "$Models\text_encoders" "qwen3vl_4b_int8_convrot.safetensors"
Download-Exact "black-forest-labs/flux-3-image-klein-comfy" `
    "vae/flux2-vae.safetensors" "$Models\vae" "flux2-vae.safetensors"

# 2. Wan2.2 I2V A14B GGUF Q4_K_M (~19.3 GB, public). The repo keeps the two
#    checkpoints in HighNoise/ and LowNoise/ subfolders; they are flattened
#    into models/unet, which is where ComfyUI-GGUF looks (models/unet).
Write-Host "[2/4] Wan2.2 I2V A14B GGUF Q4_K_M (~19.3 GB)..." -ForegroundColor Yellow
Download-Exact "QuantStack/Wan2.2-I2V-A14B-GGUF" `
    "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf" "$Models\unet" "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
Download-Exact "QuantStack/Wan2.2-I2V-A14B-GGUF" `
    "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf" "$Models\unet" "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"

# 3. umt5-xxl fp8 text encoder (~5.7 GB, public)
Write-Host "[3/4] umt5-xxl fp8 (~5.7 GB)..." -ForegroundColor Yellow
Download-Exact "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$Models\text_encoders" "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# 4. Wan 2.1 VAE (~0.3 GB, public)
Write-Host "[4/4] wan_2.1 VAE (~0.3 GB)..." -ForegroundColor Yellow
Download-Exact "Comfy-Org/Wan_2.1_ComfyUI_repackaged" `
    "split_files/vae/wan_2.1_vae.safetensors" "$Models\vae" "wan_2.1_vae.safetensors"

# Verify all expected files landed
$expected = @(
    "$Models\diffusion_models\flux3_image_klein_sd_int8_convrot.safetensors",
    "$Models\text_encoders\qwen3vl_4b_int8_convrot.safetensors",
    "$Models\vae\flux2-vae.safetensors",
    "$Models\unet\Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
    "$Models\unet\Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    "$Models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "$Models\vae\wan_2.1_vae.safetensors"
)
$missing = @($expected | Where-Object { -not (Test-Path $_) })
if ($missing.Count) {
    Write-Warning "Missing after download:"
    $missing | ForEach-Object { Write-Warning "  $_" }
} else {
    Write-Host "All 7 model files present." -ForegroundColor Green
}