#!/usr/bin/env bash
# Local/CI helper: build+test with xcodebuild (required for mlx-swift Metal shaders).
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer full Xcode when the active developer dir is only Command Line Tools.
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
  if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  fi
fi

xcodebuild -downloadComponent MetalToolchain 2>/dev/null || true
xcodebuild test -scheme rlx-swift-Package -destination 'platform=macOS' "$@"
