#!/usr/bin/env bash
# Build and run XCTest via xcodebuild so mlx-swift Cmlx Metal shaders are compiled.
# SwiftPM command-line (`swift test`) cannot produce default.metallib.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

SCHEME="${SCHEME:-rlx-swift-Package}"
DESTINATION="${DESTINATION:-platform=macOS}"

# Xcode 16+/26+ may require the standalone Metal Toolchain for mlx-swift shaders.
if ! xcrun --find metal >/dev/null 2>&1; then
  echo "==> Metal compiler missing; downloading MetalToolchain component..."
  xcodebuild -downloadComponent MetalToolchain || true
fi

echo "==> xcodebuild test -scheme ${SCHEME} -destination '${DESTINATION}'"
xcodebuild test \
  -scheme "${SCHEME}" \
  -destination "${DESTINATION}" \
  "$@"
