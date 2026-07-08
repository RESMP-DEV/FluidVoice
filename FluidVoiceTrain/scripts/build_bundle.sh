#!/bin/bash
# Build a self-contained, relocatable fluidvoice-finetune bundle for embedding
# in FluidVoice.app (so end users don't need Python installed).
#
# Output: dist/FluidVoiceTrain/
#   bin/fluidvoice-finetune       # entrypoint
#   bin/llama-quantize            # copied from LLAMA_CPP_DIR/build/bin
#   bin/convert_hf_to_gguf.py     # copied from LLAMA_CPP_DIR
#   lib/python3.x/site-packages/  # mlx, mlx_lm, transformers, torch, etc.
#
# Why a relocatable venv (not PyInstaller): mlx_lm imports model architecture
# modules dynamically at runtime (importlib in mlx_lm/utils.py), which makes
# PyInstaller hidden-imports fragile across model additions. A relocatable venv
# is what Apple's mlx ecosystem documents and survives mlx-lm version bumps.
#
# Usage:
#   ./scripts/build_bundle.sh [--llama-cpp /path/to/llama.cpp]
#
# Requirements on the build host:
#   - uv (https://docs.astral.sh/uv/)
#   - a llama.cpp checkout built with: cmake -B build && cmake --build build --target llama-quantize
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
BUNDLE_NAME="FluidVoiceTrain"
BUNDLE_DIR="$DIST_DIR/$BUNDLE_NAME"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/../llama.cpp}"
# Parse --llama-cpp override.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-cpp) LLAMA_CPP_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "Building $BUNDLE_DIR ..."
echo "  project:   $PROJECT_DIR"
echo "  llama.cpp: $LLAMA_CPP_DIR"

# Sanity: llama.cpp artifacts must exist.
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
CONV_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [[ ! -x "$QUANTIZE_BIN" ]]; then
    echo "ERROR: $QUANTIZE_BIN not found. Build llama.cpp first:" >&2
    echo "  cd $LLAMA_CPP_DIR && cmake -B build -DLLAMA_NATIVE=OFF && cmake --build build --target llama-quantize" >&2
    exit 1
fi
if [[ ! -f "$CONV_SCRIPT" ]]; then
    echo "ERROR: $CONV_SCRIPT not found." >&2
    exit 1
fi

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/bin"

# 1. Create a relocatable venv with all deps (idle + gguf extras).
echo "[1/4] Creating relocatable venv..."
uv venv "$BUNDLE_DIR" --python 3.11 --clear
# Install the package + the heavy extras into the bundle's own site-packages.
uv pip install --python "$BUNDLE_DIR/bin/python" \
    -e "$PROJECT_DIR[idle,gguf]"

# 2. Copy the llama.cpp binaries into bin/.
echo "[2/4] Copying llama.cpp artifacts..."
cp "$QUANTIZE_BIN" "$BUNDLE_DIR/bin/llama-quantize"
cp "$CONV_SCRIPT" "$BUNDLE_DIR/bin/convert_hf_to_gguf.py"
# convert_hf_to_gguf.py imports from the gguf package (in llama.cpp/gguf-py).
# Copy gguf-py so the script works standalone.
cp -R "$LLAMA_CPP_DIR/gguf-py" "$BUNDLE_DIR/bin/gguf-py"
uv pip install --python "$BUNDLE_DIR/bin/python" "$LLAMA_CPP_DIR/gguf-py" 2>/dev/null || true

# 3. Write a launcher that sets LLAMA_CPP_DIR to the bundle's bin/.
echo "[3/4] Writing launcher..."
cat > "$BUNDLE_DIR/bin/fluidvoice-finetune" <<'LAUNCHER'
#!/bin/bash
# Relocatable launcher: resolves the bundle root and points LLAMA_CPP_DIR at it.
SCRIPT_PATH="${BASH_SOURCE[0]}"
BUNDLE_BIN="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
BUNDLE_ROOT="$(cd "$BUNDLE_BIN/.." && pwd)"
export LLAMA_CPP_DIR="$BUNDLE_BIN"
export PYTHONPATH="$BUNDLE_BIN:${PYTHONPATH:-}"
exec "$BUNDLE_ROOT/bin/python" -m fluidvoice_finetune.cli "$@"
LAUNCHER
chmod +x "$BUNDLE_DIR/bin/fluidvoice-finetune"

# 4. Report.
echo "[4/4] Bundle ready."
SIZE=$(du -sh "$BUNDLE_DIR" | cut -f1)
echo "  → $BUNDLE_DIR  ($SIZE)"
echo ""
echo "Smoke test:"
echo "  $BUNDLE_DIR/bin/fluidvoice-finetune --help"
echo ""
echo "Embed in FluidVoice.app:"
echo "  cp -R $BUNDLE_DIR /path/to/FluidVoice.app/Contents/Resources/"
echo "  # then set the venv path in Settings → Fluid Intelligence Training"
echo "  # to: /path/to/FluidVoice.app/Contents/Resources/FluidVoiceTrain"
