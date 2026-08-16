#!/usr/bin/env bash
# =============================================================================
#  ATRIA EchoTrace - one-command launcher for macOS and Linux
#
#      ./run.sh            review tier (dataset browsing, tracing, exports)
#      ./run.sh --ai       additionally install the local MedGemma inference tier
#
#  Creates an isolated environment on first run, installs the package, starts
#  the server and opens the workstation in a browser. Requires only a Python
#  3.11+ interpreter. Uses `uv` when available because it is far faster.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
EXTRAS=""
WANT_AI=0

for arg in "$@"; do
    case "$arg" in
        --ai) WANT_AI=1; EXTRAS="[ai]" ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

echo
echo "  ATRIA EchoTrace - starting up"
echo "  ---------------------------------------------------------------"
if [ "$WANT_AI" -eq 1 ]; then
    echo "  Tier: review + AI  (torch/transformers/peft will be installed)"
else
    echo "  Tier: review       (add --ai for local model inference)"
fi
echo

# ---- Prefer uv, which manages the interpreter and environment itself --------
if command -v uv >/dev/null 2>&1; then
    echo "  Using uv."
    uv venv --allow-existing "$VENV"
    if [ "$WANT_AI" -eq 1 ]; then
        # PyPI serves CPU-only torch wheels on Windows and macOS (Linux gets CUDA), so a
        # plain install can leave an NVIDIA GPU unused. --torch-backend probes the driver
        # and picks the matching CUDA index, falling back to CPU when there is no GPU.
        # Override with ATRIA_TORCH_BACKEND=cu118 (older drivers) or =cpu (force CPU).
        backend="${ATRIA_TORCH_BACKEND:-auto}"
        echo "  Selecting a PyTorch build (--torch-backend=$backend) ..."
        uv pip install --python "$VENV/bin/python" --torch-backend="$backend" -e ".${EXTRAS}"
    else
        uv pip install --python "$VENV/bin/python" -e ".${EXTRAS}"
    fi
    exec "$VENV/bin/python" -m atria_echotrace serve
fi

# ---- Fall back to the standard library venv ---------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
        major="${version%%.*}"
        minor="${version##*.}"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "  ERROR: No Python 3.11+ interpreter found." >&2
    echo "  Install one from https://www.python.org/downloads/ (or your package manager)." >&2
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "  Creating environment in $VENV ..."
    "$PY" -m venv "$VENV"
fi

VPY="$VENV/bin/python"

needs_install=0
[ -x "$VENV/bin/atria" ] || needs_install=1
if [ "$WANT_AI" -eq 1 ] && ! "$VPY" -c 'import torch' >/dev/null 2>&1; then
    needs_install=1
fi

if [ "$needs_install" -eq 1 ]; then
    echo "  Installing dependencies (first run may take a few minutes) ..."
    "$VPY" -m pip install --quiet --upgrade pip
    if [ "$WANT_AI" -eq 1 ]; then
        # Same trap, without uv to solve it. Install torch from the PyTorch index first
        # when a driver is present, so the editable install below finds it satisfied.
        # Linux PyPI already ships CUDA wheels, so this mainly rescues Windows/WSL.
        backend="${ATRIA_TORCH_BACKEND:-cu126}"
        if [ "$backend" != "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
            echo "  NVIDIA driver detected; installing CUDA PyTorch ($backend) ..."
            "$VPY" -m pip install torch --index-url "https://download.pytorch.org/whl/$backend" || {
                echo "  WARNING: CUDA PyTorch install failed; falling back to the default wheel." >&2
            }
        elif [ "$backend" = "cpu" ]; then
            echo "  ATRIA_TORCH_BACKEND=cpu; installing the CPU PyTorch build."
        else
            echo "  No NVIDIA driver detected; installing the CPU PyTorch build."
        fi
    fi
    if ! "$VPY" -m pip install -e ".${EXTRAS}"; then
        echo >&2
        echo "  ERROR: Dependency installation failed. See the messages above." >&2
        if [ "$WANT_AI" -eq 1 ]; then
            echo "  The AI tier downloads PyTorch, which is large; check disk space and network." >&2
        fi
        exit 1
    fi
fi

echo
exec "$VPY" -m atria_echotrace serve
