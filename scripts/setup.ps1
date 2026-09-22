$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Python312 {
    param([string[]]$PythonArgs)
    if ($env:PYTHON_BIN) {
        & $env:PYTHON_BIN @PythonArgs
    } else {
        & py -3.12 @PythonArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Python312 @('-c', 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"')
Invoke-Python312 @('-m', 'venv', '.venv')
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed with exit code $LASTEXITCODE"
}
Write-Host 'Classical pipeline Ready. Run .venv\Scripts\python.exe scripts\process_batch.py --help'

Write-Host ''
Write-Host 'Fetching neural repos/weights (skip clones that already exist)...'
& (Join-Path $PSScriptRoot 'fetch_neural_weights.ps1')
if ($LASTEXITCODE -ne 0) {
    Write-Host '[warn] fetch_neural_weights.ps1 reported a problem; classical looks still work'
}

# Neural looks (spectroformer / nu2net) require WSL2 + micromamba env uw_eval.
$spectro = Join-Path (Get-Location) 'eval\repos\spectroformer\checkpoints\best.pth'
$nu2 = Join-Path (Get-Location) 'eval\repos\uie_benchmark\checkpoints\UIEB\NU2Net.ckpt'
$runner = Join-Path (Get-Location) 'eval\scripts\run_uie_look.py'

Write-Host ''
Write-Host 'Neural GPU looks checklist (default CLI look: auto):'
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    Write-Host '  [ok] wsl.exe on PATH'
} else {
    Write-Host '  [missing] wsl.exe — install WSL2 for spectroformer/nu2net'
}
if (Test-Path $runner) {
    Write-Host "  [ok] runner $runner"
} else {
    Write-Host "  [missing] $runner"
}
if (Test-Path $spectro) {
    Write-Host "  [ok] Spectroformer weights"
} else {
    Write-Host "  [missing] $spectro — see docs/neural-setup.md"
}
if (Test-Path $nu2) {
    Write-Host "  [ok] NU2Net weights"
} else {
    Write-Host "  [missing] $nu2 — optional unless -Look nu2net"
}

$envCheck = @'
eval "$(/home/jkowall/micromamba/bin/micromamba shell hook -s bash)" && micromamba activate uw_eval && python -c "import torch; print(\"uw_eval torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
'@
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    $out = & wsl -e bash -lc $envCheck 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [ok] $out"
    } else {
        Write-Host '  [missing] micromamba env uw_eval with torch+CUDA'
        Write-Host "           $out"
    }
}

Write-Host ''
Write-Host 'Default: .\scripts\process.ps1 <NEF-folder>  (-Look auto: UW→spectroformer, topside→natural)'
Write-Host 'UW-only GPU: .\scripts\process.ps1 <NEF-folder> -Look spectroformer'
Write-Host 'Classical CPU: .\scripts\process.ps1 <NEF-folder> -Look vivid'
