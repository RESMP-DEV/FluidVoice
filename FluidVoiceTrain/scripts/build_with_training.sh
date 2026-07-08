#!/bin/bash
# Release-build wrapper: builds the FluidVoiceTrain bundle, runs the private-FI
# build, and embeds the bundle into FluidVoice.app/Contents/Resources/.
#
# This is the maintainers' release entrypoint for builds that include on-device
# training. It does NOT replace build.sh (the public OSS build) — it layers on
# top of the private Fluid Intelligence build (build_with_FI_incremental.sh).
#
# Usage:
#   sh FluidVoiceTrain/scripts/build_with_training.sh
#
# Requires (on the build host):
#   - uv (https://docs.astral.sh/uv/)
#   - a llama.cpp checkout at $LLAMA_CPP_DIR (default: ../llama.cpp) built with
#     `cmake --build build --target llama-quantize`
#   - the private Fluid Intelligence build setup (build_with_FI_incremental.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_SCRIPT="$SCRIPT_DIR/build_bundle.sh"
FI_BUILD_SCRIPT="$REPO_ROOT/build_with_FI_incremental.sh"

echo "==> [1/4] Building FluidVoiceTrain bundle..."
sh "$BUNDLE_SCRIPT"
BUNDLE_DIR="$REPO_ROOT/FluidVoiceTrain/dist/FluidVoiceTrain"
if [[ ! -d "$BUNDLE_DIR" ]]; then
    echo "ERROR: bundle not produced at $BUNDLE_DIR" >&2
    exit 1
fi

echo "==> [2/4] Running private Fluid Intelligence build..."
if [[ ! -x "$FI_BUILD_SCRIPT" ]]; then
    echo "ERROR: $FI_BUILD_SCRIPT not found. The private FI build setup is required for a training-enabled release." >&2
    exit 1
fi
sh "$FI_BUILD_SCRIPT"

# Locate the built .app. The FI build writes to DerivedData; this is a best-effort
# search. Maintain: this is intentionally not auto-run by build.sh (public OSS).
APP_PATH="$(find ~/Library/Developer/Xcode/DerivedData -maxdepth 4 -name 'FluidVoice.app' -type d 2>/dev/null | head -1)"
if [[ -z "$APP_PATH" ]]; then
    echo "ERROR: could not locate the built FluidVoice.app in DerivedData." >&2
    exit 1
fi
RESOURCES_DIR="$APP_PATH/Contents/Resources"
echo "==> [3/4] Embedding bundle into $RESOURCES_DIR/FluidVoiceTrain/"
rm -rf "$RESOURCES_DIR/FluidVoiceTrain"
cp -R "$BUNDLE_DIR" "$RESOURCES_DIR/FluidVoiceTrain"

echo "==> [4/4] Re-signing (adding resources invalidates the signature)..."
# Re-sign with the same identity the FI build used. Ad-hoc if unconfigured.
SIGN_IDENTITY="${FV_SIGN_IDENTITY:--}"
codesign --force --deep --sign "$SIGN_IDENTITY" "$APP_PATH" 2>/dev/null || \
    echo "  (re-sign skipped or run manually)"

echo ""
echo "Done. $APP_PATH now includes the FluidVoiceTrain bundle."
echo "The app will auto-discover it at runtime via FluidIntelligenceTrainer."
