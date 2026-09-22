[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$InputPath,
    [string]$Output,
    [ValidateSet('natural', 'pop', 'vivid', 'auto', 'spectroformer', 'nu2net')]
    [string]$Look = 'spectroformer'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
}

$batchArgs = @(
    'scripts\process_batch.py',
    '--input', $InputPath,
    '--look', $Look,
    '--resume'
)
if ($Output) {
    $batchArgs += @('--output', $Output)
}

& $python @batchArgs
exit $LASTEXITCODE
