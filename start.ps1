# FedLiverNet PowerShell Single-Command Startup Script
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  Starting FedLiverNet Telemedicine Platform..." -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython run.py
} else {
    python run.py
}
