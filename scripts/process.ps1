[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$InputPath,
    [string]$Output,
    [ValidateSet('natural', 'pop', 'vivid', 'auto', 'spectroformer', 'nu2net')]
    [string]$Look = 'auto',
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
}

$inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
if ($Output) {
    $outputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
} else {
    $item = Get-Item -LiteralPath $inputResolved
    $parent = if ($item.PSIsContainer) { $item.FullName } else { $item.DirectoryName }
    $baseName = if ($item.PSIsContainer) { $item.Name } else { [IO.Path]::GetFileNameWithoutExtension($item.Name) }
    $outputDir = Join-Path $parent "$baseName-$Look"
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
$code = $LASTEXITCODE

if (($code -eq 0) -and (-not $NoOpen) -and (Test-Path -LiteralPath $outputDir)) {
    Write-Host "Opening output folder: $outputDir"
    Start-Process explorer.exe -ArgumentList $outputDir
}

exit $code
