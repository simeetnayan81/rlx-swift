#!/usr/bin/env bash
# Capture frames from ALEEnvironment and encode a GIF with ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${ALE_ROOT:-}" ]]; then
  echo "error: export ALE_ROOT (e.g. \$HOME/.local/ale)" >&2
  exit 2
fi
if [[ -z "${ALE_ROM_PATH:-}" && -z "${ALE_ROM_DIR:-}" ]]; then
  echo "error: export ALE_ROM_PATH=/path/to/game.bin" >&2
  exit 2
fi

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
export ALE_GIF_DIR="${ALE_GIF_DIR:-/tmp/rlx_ale_gif}"
export ALE_GIF_STEPS="${ALE_GIF_STEPS:-150}"
export ALE_GIF_EVERY="${ALE_GIF_EVERY:-2}"

GAME_HINT=$(basename "${ALE_ROM_PATH:-${ALE_GAME:-atari}}" .bin)
OUT_GIF="${ALE_GIF_OUT:-/tmp/rlx_ale_${GAME_HINT}.gif}"

swift build --product ALEGifDemo

BIN_DIR=".build/arm64-apple-macosx/debug"
[[ -x "$BIN_DIR/ALEGifDemo" ]] || BIN_DIR=".build/debug"

# Stage MLX metallib for CLI (same as run-ale-demo.sh)
if [[ ! -f "$BIN_DIR/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]]; then
  SRC=$(find "$HOME/Library/Developer/Xcode/DerivedData" -path '*/Debug/mlx-swift_Cmlx.bundle' -type d 2>/dev/null | head -1 || true)
  if [[ -n "$SRC" ]]; then
    rm -rf "$BIN_DIR/mlx-swift_Cmlx.bundle"
    cp -R "$SRC" "$BIN_DIR/"
  fi
fi

"$BIN_DIR/ALEGifDemo"

if ! command -v ffmpeg >/dev/null; then
  echo "error: ffmpeg not found (brew install ffmpeg)" >&2
  exit 1
fi

# PPM sequence → GIF (palette for decent quality)
ffmpeg -y -loglevel error -framerate 15 \
  -i "$ALE_GIF_DIR/frame_%04d.ppm" \
  -vf "scale=320:-1:flags=neighbor,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse" \
  "$OUT_GIF"

echo "GIF written: $OUT_GIF"
ls -la "$OUT_GIF"
