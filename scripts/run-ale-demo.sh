#!/usr/bin/env bash
# Build (with ALE_ROOT) and run ALERandomAgent, ensuring MLX metallib is discoverable.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${ALE_ROOT:-}" ]]; then
  echo "error: set ALE_ROOT (e.g. export ALE_ROOT=\$HOME/.local/ale)" >&2
  exit 2
fi
if [[ -z "${ALE_ROM_PATH:-}" && -z "${ALE_ROM_DIR:-}" ]]; then
  echo "error: set ALE_ROM_PATH or ALE_ROM_DIR" >&2
  exit 2
fi

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
swift build --product ALERandomAgent

BIN_DIR=".build/arm64-apple-macosx/debug"
if [[ ! -x "$BIN_DIR/ALERandomAgent" ]]; then
  BIN_DIR=".build/debug"
fi

# SPM CLI often omits Cmlx metallib next to the binary; copy from Xcode DerivedData if needed.
if [[ ! -f "$BIN_DIR/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]]; then
  SRC=$(find "$HOME/Library/Developer/Xcode/DerivedData" -path '*/Debug/mlx-swift_Cmlx.bundle' -type d 2>/dev/null | head -1 || true)
  if [[ -z "$SRC" ]]; then
    # trigger an xcodebuild once so the bundle exists
    echo "note: building once with xcodebuild to materialize mlx metallib bundle..."
    xcodebuild -scheme rlx-swift-Package -destination 'platform=macOS' build -quiet || true
    SRC=$(find "$HOME/Library/Developer/Xcode/DerivedData" -path '*/Debug/mlx-swift_Cmlx.bundle' -type d 2>/dev/null | head -1 || true)
  fi
  if [[ -n "$SRC" ]]; then
    rm -rf "$BIN_DIR/mlx-swift_Cmlx.bundle"
    cp -R "$SRC" "$BIN_DIR/"
    echo "staged metallib from $SRC"
  fi
fi

export ALE_FRAME_OUT="${ALE_FRAME_OUT:-/tmp/rlx_ale_frame.ppm}"
exec "$BIN_DIR/ALERandomAgent"
