#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Create a sorted, fully pinned lock file for reproducible installs
TMP_FILE="$(mktemp)"
pip freeze > "$TMP_FILE"

# Optionally, you can keep torch channel hints by not rewriting index URLs here.
# If you want a separate lock, change the output name below (e.g., requirements.lock.txt)
sort "$TMP_FILE" > requirements.txt
rm -f "$TMP_FILE"

echo "✔ Wrote pinned requirements to requirements.txt"
