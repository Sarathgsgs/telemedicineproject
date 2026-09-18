@echo off
title FedLiverNet Telemedicine Platform
cd /d "%~dp0"

echo ===================================================================
echo   Starting FedLiverNet Telemedicine Platform...
echo ===================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py
) else (
    python run.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Server encountered an error or was terminated.
    pause
)
