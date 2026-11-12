#!/usr/bin/env bash
# ==============================================================================
# IRIS — Intelligent Reasoning & Interaction System
# One-Command Launcher
# ==============================================================================
# Usage:
#   ./run.sh --desktop   # Launch PySide6 IRIS Desktop Application (Phase 2)
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

# 3. Ensure native macOS Location Helper is built if on macOS
if [ ! -f "$SCRIPT_DIR/bin/IrisLocationHelper.app/Contents/MacOS/IrisLocationHelper" ] && [ "$(uname)" = "Darwin" ]; then
    if command -v swiftc >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/native/location_helper/build.sh" ]; then
        "$SCRIPT_DIR/native/location_helper/build.sh" >/dev/null 2>&1 || true
    fi
fi

# 4. Launch IRIS Desktop UI if --desktop is passed; otherwise launch CLI/voice runner
if [ "${1:-}" = "--desktop" ]; then
    shift
    exec "$VENV_PYTHON" "$SCRIPT_DIR/iris_desktop.py" "$@"
fi
exec "$VENV_PYTHON" "$SCRIPT_DIR/iris_voice.py" "$@"


