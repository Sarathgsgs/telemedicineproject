@echo off
title FedLiverNet Telemedicine Platform (Review-II)
cd /d "%~dp0"

echo ===============================================================================
echo   FEDLIVERNET: PRIVACY-PRESERVING FEDERATED TELEMEDICINE PLATFORM
echo   AI-Assisted Liver Disease Screening (CSE Final Year Project 23CS8701)
echo   Presenter: YUVARAJAN D (421123102175) ^| Supervisor: Mrs. MANIMEGALAI (AP/CSE)
echo ===============================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python not found! Please install Python or create the virtualenv.
        echo.
        pause
        exit /b 1
    )
)

echo [*] Using Python interpreter: %PYTHON_EXE%
echo [*] Launching FastAPI backend and clinical interface...
echo [*] Web browser will open automatically at: http://127.0.0.1:8000
echo.

"%PYTHON_EXE%" run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Platform stopped with exit code %ERRORLEVEL%.
    pause
)
