#!/usr/bin/env bash
# Build and install Farama ALE for use with rlx-swift (ALE_ROOT).
# Usage: ./scripts/build-ale.sh [install_prefix]
set -euo pipefail

PREFIX="${1:-$HOME/.local/ale}"
SRC_DIR="${ALE_SRC_DIR:-/tmp/Arcade-Learning-Environment}"
REPO="${ALE_REPO:-https://github.com/Farama-Foundation/Arcade-Learning-Environment.git}"
TAG="${ALE_TAG:-v0.10.2}"

echo "Installing ALE $TAG → $PREFIX"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  rm -rf "$SRC_DIR"
  git clone --depth 1 --branch "$TAG" "$REPO" "$SRC_DIR"
fi

cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DSDL_SUPPORT=OFF \
  -DBUILD_PYTHON_MODULE=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build "$SRC_DIR/build" --config Release -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
cmake --install "$SRC_DIR/build"

echo ""
echo "Done. Use:"
echo "  export ALE_ROOT=$PREFIX"
echo "  export ALE_ROM_PATH=/path/to/rom.bin"
echo "  swift build --product RLXALE"
