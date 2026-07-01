#!/usr/bin/env bash
# Generate DocC static sites for all library targets under .build/docc-site (gitignored via .build).
# Requires full Xcode (docc tool), not Command Line Tools only.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

if ! xcrun --find docc >/dev/null 2>&1; then
  echo "error: docc not found. Install Xcode and set DEVELOPER_DIR." >&2
  exit 1
fi

OUT="${1:-.build/docc-site}"
mkdir -p "$OUT"

targets=(RLXCore RLXWrappers RLXEnvs RLXTesting RLXVector)

for t in "${targets[@]}"; do
  echo "=== DocC: $t → $OUT/$t ==="
  xcrun swift package --allow-writing-to-directory "$OUT" \
    generate-documentation \
    --target "$t" \
    --output-path "$OUT/$t" \
    --disable-indexing \
    --transform-for-static-hosting \
    --hosting-base-path "rlx-swift/$t"
done

echo "Done. Static sites under $OUT/<Target>/."
echo "Xcode: Product → Build Documentation."
