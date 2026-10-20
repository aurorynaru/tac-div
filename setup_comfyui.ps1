# Pixelle-Video ComfyUI setup (Option A: local ComfyUI)
# Run: powershell -ExecutionPolicy Bypass -File .\setup_comfyui.ps1
$ErrorActionPreference = "Stop"

$Base     = "C:\AI\ComfyUI"
$Portable = "$Base\ComfyUI_windows_portable"
$Cu       = "$Portable\ComfyUI"
$Py       = "$Portable\python_embeded\python.exe"
# Pinned to the official v0.41.0 release (tag sha 98cc0bf): first release with
# FLUX 3 Image support (CLIPLoader type "flux3") and ReferenceMultiLatent.
$Release = "v0.41.0"
$Url = "https://github.com/Comfy-Org/ComfyUI/releases/download/$Release/ComfyUI_windows_portable_nvidia.7z"

Write-Host "=== Pixelle-Video ComfyUI setup ===" -ForegroundColor Cyan

# Disk space check (~25GB needed)
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($freeGB -lt 30) { Write-Warning "Only ${freeGB}GB free on C:. This setup needs ~25GB." }

# 1. 7zr (standalone console, no install). The old 7-zip.org/a/7za.exe URL is
# dead; the current standalone console that reads .7z is 7zr.exe from GitHub.
$7za = Join-Path $env:TEMP "7zr.exe"
if (-not (Test-Path $7za)) {
    Write-Host "[1/6] Downloading 7zr.exe..." -ForegroundColor Yellow
    curl.exe -L --fail -sSL -o $7za "https://github.com/ip7z/7zip/releases/download/26.04/7zr.exe"
}

# 2. ComfyUI portable (resumable)
New-Item -ItemType Directory -Path $Base -Force | Out-Null
$zip = "$Base\ComfyUI_windows_portable_nvidia.7z"
if (-not (Test-Path $zip)) {
    Write-Host "[2/6] Downloading ComfyUI portable ($Release, ~1.9GB)..." -ForegroundColor Yellow
    curl.exe -L --fail -C - -o $zip $Url
} else {
    Write-Host "[2/6] ComfyUI archive already present, skipping download."
}

# 3. Extract
if (-not (Test-Path "$Cu\main.py")) {
    Write-Host "[3/6] Extracting..." -ForegroundColor Yellow
    & $7za x -y -o"$Base" "$zip" | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$Cu\main.py")) {
        throw "Extraction failed (7zr exit $LASTEXITCODE, main.py missing)."
    }
} else {
    Write-Host "[3/6] ComfyUI already extracted, skipping."
}

# 4. Custom nodes
$nodes = [ordered]@{
    "ComfyUI-Manager"     = "https://github.com/Comfy-Org/ComfyUI-Manager.git"
    "ComfyUI-Easy-Use"    = "https://github.com/Comfy-Org/ComfyUI-Easy-Use.git"
    "ComfyUI-Impact-Pack" = "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
    "ComfyUI-QwenVL"      = "https://github.com/1038lab/ComfyUI-QwenVL.git"
    "ComfyUI-GGUF"        = "https://github.com/city96/ComfyUI-GGUF.git"
    "VideoHelperSuite"    = "https://github.com/Kosinkadink/VideoHelperSuite.git"
}
Write-Host "[4/6] Installing custom nodes..." -ForegroundColor Yellow
foreach ($name in $nodes.Keys) {
    $dir = "$Cu\custom_nodes\$name"
    if (-not (Test-Path $dir)) {
        Write-Host "  - $name"
        git clone --depth 1 $nodes[$name] $dir
    } else {
        Write-Host "  - $name (already present)"
    }
}

# 5. Node dependencies (portable python)
Write-Host "[5/6] Installing node dependencies (pip)..." -ForegroundColor Yellow
& $Py -m pip install -q -r "$Cu\custom_nodes\ComfyUI-QwenVL\requirements.txt"
& $Py -m pip install -q -r "$Cu\custom_nodes\ComfyUI-Manager\requirements.txt"
& $Py -m pip install -q -r "$Cu\custom_nodes\ComfyUI-GGUF\requirements.txt"
& $Py -m pip install -q -r "$Cu\custom_nodes\ComfyUI-Impact-Pack\requirements.txt"
& $Py -m pip install -q -U "huggingface_hub[cli]"

# 6. Qwen3-VL-8B-Instruct model (~17GB, resumable)
$modelDir = "$Cu\models\LLM\Qwen\Qwen3-VL-8B-Instruct"
if (-not (Test-Path (Join-Path $modelDir "config.json"))) {
    Write-Host "[6/6] Downloading Qwen3-VL-8B-Instruct (~17GB)..." -ForegroundColor Yellow
    & $Py -m huggingface_hub download "Qwen/Qwen3-VL-8B-Instruct" --local-dir $modelDir
} else {
    Write-Host "[6/6] Model already present, skipping."
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Start ComfyUI with:  & '$Portable\run_nvidia_gpu.bat'"
Write-Host "Then open http://127.0.0.1:8188 in your browser."