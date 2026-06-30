#!/usr/bin/env bash
# Full local verification matrix (run before push for any library change).
#
# 1) swift build
# 2) RLXCoreSmoke (CLI / Linux-parity executable)
# 3) macOS xcodebuild test (tier-1 XCTest + Metal where needed)
# 4) Linux Docker smoke (tier-2 CI parity)
# 5) iOS Simulator compile (optional CI smoke)
#
# Usage: ./scripts/verify-all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

fail=0
pass() { echo "PASS: $*"; }
fail_msg() { echo "FAIL: $*"; fail=1; }

echo "=== 1/5 swift build ==="
if swift build; then pass "swift build"; else fail_msg "swift build"; fi

echo "=== 2/5 RLXCoreSmoke ==="
if swift run RLXCoreSmoke; then pass "RLXCoreSmoke"; else fail_msg "RLXCoreSmoke"; fi

echo "=== 3/5 macOS xcodebuild test ==="
if ./scripts/xcodebuild-test.sh; then pass "xcodebuild test"; else fail_msg "xcodebuild test"; fi

echo "=== 4/5 Linux Docker smoke ==="
if ./scripts/linux-smoke-docker.sh; then pass "linux-smoke-docker"; else fail_msg "linux-smoke-docker"; fi

echo "=== 5/5 iOS Simulator build ==="
if xcodebuild build -scheme rlx-swift-Package -destination 'generic/platform=iOS Simulator'; then
  pass "iOS Simulator build"
else
  fail_msg "iOS Simulator build"
fi

echo
if [[ "$fail" -ne 0 ]]; then
  echo "verify-all: FAILED"
  exit 1
fi
echo "verify-all: all checks passed"
