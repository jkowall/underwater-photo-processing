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
Write-Host 'Ready. Run .venv\Scripts\python.exe scripts\process_batch.py --help'
