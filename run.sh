#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-5000}"
DISABLE_HTTPS="${DISABLE_HTTPS:-0}"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Ensure cryptography when HTTPS enabled
if [ "$DISABLE_HTTPS" != "1" ]; then
  python -c "import cryptography" 2>/dev/null || {
    echo "cryptography not found; installing…"
    pip install cryptography
  }
fi

export PORT DISABLE_HTTPS
python app.py
