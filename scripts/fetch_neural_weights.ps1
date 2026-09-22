# Fetch neural repos + checkpoints for Spectroformer / NU2Net looks.
# Safe to re-run; skips work that is already present.
#
# Usage (from repo root):
#   .\scripts\fetch_neural_weights.ps1
#   .\scripts\fetch_neural_weights.ps1 -SkipClone

[CmdletBinding()]
param(
    [switch]$SkipClone
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$spectroRepo = Join-Path $root 'eval\repos\spectroformer'
$nu2Repo = Join-Path $root 'eval\repos\uie_benchmark'
$spectroBest = Join-Path $spectroRepo 'checkpoints\best.pth'
$nu2Ckpt = Join-Path $nu2Repo 'checkpoints\UIEB\NU2Net.ckpt'

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Clone-IfMissing([string]$Url, [string]$Dest, [string]$Name) {
    if (Test-Path (Join-Path $Dest '.git')) {
        Write-Host "[ok] $Name already cloned: $Dest"
        return
    }
    if (Test-Path $Dest) {
        Write-Host "[warn] $Dest exists but is not a git clone; leave as-is"
        return
    }
    Ensure-Dir (Split-Path $Dest)
    Write-Host "[fetch] cloning $Name ..."
    git clone --depth 1 $Url $Dest
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed for $Url"
    }
}

if (-not $SkipClone) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'git is required on PATH to clone neural repos'
    }
    Clone-IfMissing 'https://github.com/Mdraqibkhan/Spectroformer.git' $spectroRepo 'Spectroformer'
    Clone-IfMissing 'https://github.com/ddz16/Underwater-Image-Enhancement-Benchmark.git' $nu2Repo 'uie_benchmark (NU2Net)'
}

# Spectroformer: promote a dataset checkpoint to checkpoints/best.pth if needed
Ensure-Dir (Join-Path $spectroRepo 'checkpoints')
if (Test-Path $spectroBest) {
    Write-Host "[ok] Spectroformer weights: $spectroBest"
} else {
    $candidates = @()
    if (Test-Path (Join-Path $spectroRepo 'checkpoints')) {
        $candidates = Get-ChildItem -Path (Join-Path $spectroRepo 'checkpoints') -Recurse -Filter '*.pth' -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne 'best.pth' }
    }
    if ($candidates -and $candidates.Count -gt 0) {
        $pick = $candidates | Sort-Object Length -Descending | Select-Object -First 1
        Copy-Item $pick.FullName $spectroBest -Force
        Write-Host "[ok] copied $($pick.FullName) -> checkpoints\best.pth"
    } else {
        Write-Host "[missing] $spectroBest"
        Write-Host "         Upstream does not always ship weights in-tree. Place the validated"
        Write-Host "         UIEB (or bakeoff) checkpoint at that path, then re-run this script."
        Write-Host "         See docs/neural-setup.md and THIRD_PARTY.md (no upstream LICENSE yet)."
    }
}

# NU2Net
if (Test-Path $nu2Ckpt) {
    Write-Host "[ok] NU2Net weights: $nu2Ckpt"
} else {
    Write-Host "[missing] $nu2Ckpt"
    Write-Host "         After cloning uie_benchmark, download NU2Net.ckpt into"
    Write-Host "         eval\repos\uie_benchmark\checkpoints\UIEB\ (see that repo's README)."
}

Write-Host ''
Write-Host 'Next: .\scripts\setup.ps1   # classical venv + neural checklist'
Write-Host 'Then: .\scripts\process.ps1 <NEF-folder>   # default -Look auto'
