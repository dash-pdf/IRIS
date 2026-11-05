#!/usr/bin/env bash
# ==============================================================================
# IRIS — Intelligent Reasoning & Interaction System
# One-Command Launcher
# ==============================================================================
# Usage:
#   ./run.sh             # Hands-free voice assistant mode
#   ./run.sh --text      # Interactive text chat mode (no mic needed)
#   ./run.sh --help      # Show all available options and flags
# ==============================================================================

set -euo pipefail

# Resolve repository root directory regardless of working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# 1. Ensure virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Virtual environment not found at .venv"
    echo "Creating virtual environment and installing dependencies..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# 2. Ensure .env exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        echo "ℹ️  Creating .env from .env.example..."
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    fi
fi

# 3. Launch IRIS forwarding all command-line arguments
exec "$VENV_PYTHON" "$SCRIPT_DIR/iris_voice.py" "$@"
