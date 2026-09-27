# NetForecast quick start (Windows PowerShell).
#   .\run.ps1            # install deps if needed, then serve
#   .\run.ps1 -Test      # run the test suite instead
param([switch]$Test, [switch]$NoInstall, [int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $NoInstall) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    python -m pip install -r requirements-dev.txt --quiet
}

if (-not (Test-Path "server/static/vendor/plotly.min.js")) {
    Write-Host "Fetching offline dashboard assets..." -ForegroundColor Cyan
    python tools/fetch_vendor.py
}

if ($Test) {
    python -m pytest -q
    exit $LASTEXITCODE
}

Write-Host "NetForecast -> http://127.0.0.1:$Port" -ForegroundColor Green
python -m uvicorn server.main:app --host 127.0.0.1 --port $Port
