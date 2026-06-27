#!/usr/bin/env bash
# Entrypoint for the rlx-swift Linux Dockerfile.
# Prefers a bind-mounted host tree at /workspace (Mac live edit); falls back to /src
# baked into the image. Always builds in a writable /work tree so host Darwin .build
# is never reused.
set -euo pipefail

WORK="${RLX_LINUX_WORK:-/work}"
mkdir -p "$WORK"

if [[ -f /workspace/Package.swift ]]; then
  SRC=/workspace
  echo "==> using bind-mounted sources at /workspace"
elif [[ -f /src/Package.swift ]]; then
  SRC=/src
  echo "==> using image-baked sources at /src"
else
  echo "error: no Package.swift in /workspace or /src" >&2
  echo "Mount the repo: docker run --rm -v \"\$PWD\":/workspace:ro rlx-swift-linux" >&2
  exit 1
fi

# --delete keeps /work aligned with the mount; exclude VCS and prior build trees.
rsync -a --delete \
  --exclude .build \
  --exclude .git \
  --exclude .swiftpm \
  --exclude DerivedData \
  "$SRC"/ "$WORK"/

cd "$WORK"
exec ./scripts/linux-smoke.sh "$@"
