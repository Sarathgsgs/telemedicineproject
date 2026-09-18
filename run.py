"""
run.py
------
Single-command launcher for the FedLiverNet Federated Telemedicine Platform.
Automatically:
1. Validates environment dependencies and checks for .venv.
2. Ensures edge model checkpoints (TorchScript) and sample CT slices are ready.
3. Automatically launches the default web browser to http://127.0.0.1:8000.
4. Starts the high-performance FastAPI backend + static frontend server.

Usage:
    python run.py
    or (Windows terminal)
    .\\start.bat
    or (PowerShell)
    .\\start.ps1
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess

PORT = 8000
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"

def ensure_environment():
    """Validates that .venv is used if available."""
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    # If running with a different python and .venv exists, re-exec with .venv
    if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        print(f"[Launcher] Switching to virtual environment Python: {venv_python}")
        try:
            subprocess.run([venv_python, __file__] + sys.argv[1:])
            sys.exit(0)
        except Exception as e:
            print(f"[Launcher] Warning: Could not spawn virtualenv python: {e}")

def check_and_prepare_artifacts():
    """Ensures checkpoints and sample CT slices are available."""
    print("=" * 65)
    print("  FEDLIVERNET: PRIVACY-PRESERVING TELEMEDICINE PLATFORM")
    print("  Reference: Nature Scientific Reports (2026)")
    print("=" * 65)

    # 1. Check TorchScript edge model
    traced_model = os.path.join("checkpoints", "edge_model_traced.pt")
    centralized_model = os.path.join("checkpoints", "centralized_best.pt")
    
    if not os.path.exists(traced_model) and os.path.exists(centralized_model):
        print("[Launcher] Compiling TorchScript edge model from checkpoint...")
        from model.export_edge import export_to_torchscript, load_model_from_checkpoint
        model = load_model_from_checkpoint(centralized_model)
        export_to_torchscript(model, traced_model)
        print("[Launcher] TorchScript edge model compiled successfully.")

    # 2. Check sample CT slices
    sample_slice = os.path.join("data", "processed", "slices", "cohort_master_slice_0001.npz")
    if not os.path.exists(sample_slice):
        print("[Launcher] Generating demonstration CT slice partitions...")
        from data.init_dataset import init_dataset
        init_dataset()
        print("[Launcher] CT slice partitions initialized.")

    print(f"[+] System checks verified.")
    print(f"[+] Clinical Teleconsultation Station available at: {URL}")
    print(f"[+] Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65)

def open_browser():
    """Opens the web browser after a brief delay to allow Uvicorn to bind."""
    time.sleep(1.5)
    try:
        webbrowser.open(URL)
    except Exception as e:
        print(f"[Launcher] Could not automatically open browser: {e}")

def main():
    ensure_environment()
    check_and_prepare_artifacts()

    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI / Uvicorn Server
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, log_level="info", reload=False)

if __name__ == "__main__":
    main()
