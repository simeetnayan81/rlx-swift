#!/usr/bin/env bash
# Local/CI helper: build+test with xcodebuild (required for mlx-swift Metal shaders).
set -euo pipefail
cd "$(dirname "$0")/.."

xcodebuild -downloadComponent MetalToolchain 2>/dev/null || true
xcodebuild test -scheme rlx-swift-Package -destination 'platform=macOS' "$@"
