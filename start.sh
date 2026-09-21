#!/usr/bin/env bash
# start.sh - Launch FedLiverNet on Linux / macOS
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "==================================================================="
echo "  Starting FedLiverNet Telemedicine Platform..."
echo "==================================================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python run.py
else
    python3 run.py
fi
