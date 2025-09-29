#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/install.sh
#   TORCH_SELECTOR=cuda121 scripts/install.sh   # force CUDA 12.1 wheels
#   TORCH_SELECTOR=cpu     scripts/install.sh   # force CPU wheels
#   TORCH_SELECTOR=mps     scripts/install.sh   # force macOS MPS wheels

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_SELECTOR="${TORCH_SELECTOR:-auto}"

echo "→ Python: ${PYTHON_BIN}"
echo "→ Venv:   ${VENV_DIR}"
echo "→ Torch:  ${TORCH_SELECTOR}"

# 1) make venv
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# 2) base tooling
python -m pip install --upgrade pip wheel setuptools

# 3) choose PyTorch channel
uname_s="$(uname -s 2>/dev/null || echo Unknown)"
uname_m="$(uname -m 2>/dev/null || echo Unknown)"

if [ "$TORCH_SELECTOR" = "auto" ]; then
  if [ "$uname_s" = "Darwin" ] && [[ "$uname_m" == arm64* ]]; then
    TORCH_SELECTOR="mps"         # Apple Silicon
  else
    # If nvidia-smi exists, assume CUDA 12.1 wheels; else CPU
    if command -v nvidia-smi >/dev/null 2>&1; then
      TORCH_SELECTOR="cuda121"
    else
      TORCH_SELECTOR="cpu"
    fi
  fi
fi

echo "→ Resolved Torch selector: $TORCH_SELECTOR"

case "$TORCH_SELECTOR" in
  mps|cpu)
    # Default PyPI (CPU wheels; on macOS arm64 this includes MPS support)
    pip install torch torchvision torchaudio
    ;;
  cuda121)
    # CUDA 12.1 wheels from PyTorch index
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    ;;
  cuda122)
    # CUDA 12.2 wheels (if available for your platform)
    pip install --index-url https://download.pytorch.org/whl/cu122 torch torchvision torchaudio
    ;;
  *)
    echo "Unknown TORCH_SELECTOR: $TORCH_SELECTOR" >&2
    exit 1
    ;;
esac

# 4) project deps
pip install -r requirements.in

# 5) helpful runtime env (optional)
if [ "$uname_s" = "Darwin" ] && [[ "$uname_m" == arm64* ]]; then
  # Improve stability for some ops on Apple MPS
  if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" "$VENV_DIR/bin/activate"; then
    {
      echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1'
    } >> "$VENV_DIR/bin/activate"
  fi
fi

echo
echo "✔ Install complete."
echo "Next:"
echo "  source $VENV_DIR/bin/activate"
echo "  python app.py"
echo
echo "For HTTPS camera access, a self-signed cert is generated automatically."
echo "If you prefer HTTP only (desktop localhost), run: DISABLE_HTTPS=1 python app.py"
