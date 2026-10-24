# FedLiverNet PowerShell Single-Command Startup Script
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  FEDLIVERNET: PRIVACY-PRESERVING FEDERATED TELEMEDICINE PLATFORM" -ForegroundColor Green
Write-Host "  AI-Assisted Liver Disease Screening (CSE Final Year Project 23CS8701)" -ForegroundColor Yellow
Write-Host "  Presenter: YUVARAJAN D (421123102175) | Supervisor: Mrs. MANIMEGALAI (AP/CSE)" -ForegroundColor White
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "[*] Using Python interpreter: $venvPython" -ForegroundColor Green
    & $venvPython run.py
} else {
    Write-Host "[*] Using System Python interpreter..." -ForegroundColor Yellow
    python run.py
}
