#!/usr/bin/env bash
# Linux tier-2: resolve, build, and run RLXCoreSmoke (CPU mlx-swift; no Metal).
# Usage:
#   ./scripts/linux-smoke.sh              # build debug + smoke
#   ./scripts/linux-smoke.sh --release    # release configuration
#   ./scripts/linux-smoke.sh --install-deps  # apt-get install BLAS/LAPACK (needs sudo)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=debug
INSTALL_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --release) CONFIG=release ;;
    --install-deps) INSTALL_DEPS=1 ;;
    -h|--help)
      sed -n '2,7p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warning: this script is intended for Linux (uname=$(uname -s)); continuing anyway" >&2
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y \
      build-essential \
      libblas-dev \
      liblapacke-dev \
      libopenblas-dev \
      gfortran
  else
    echo "error: --install-deps only supports apt-get; install blas/lapack/openblas/gfortran manually" >&2
    exit 1
  fi
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: swift not on PATH; install Swift 6.0+ from https://www.swift.org/install/linux/" >&2
  exit 1
fi

# Soft check for cblas.h (mlx-swift CPU backend on Linux)
if ! (echo '#include <cblas.h>' | cc -E -x c - -o /dev/null 2>/dev/null); then
  echo "error: compiler cannot find <cblas.h>." >&2
  echo "Install OpenBLAS/LAPACK dev packages, e.g. on Debian/Ubuntu:" >&2
  echo "  sudo apt-get install -y build-essential libblas-dev liblapacke-dev libopenblas-dev gfortran" >&2
  echo "Or re-run: $0 --install-deps" >&2
  exit 1
fi

echo "==> swift --version"
swift --version

echo "==> swift package resolve"
swift package resolve

echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

SMOKE_BIN=".build/${CONFIG}/RLXCoreSmoke"
if [[ ! -x "$SMOKE_BIN" ]]; then
  # Product name path can vary slightly; fall back to swift run
  echo "==> swift run -c $CONFIG RLXCoreSmoke"
  swift run -c "$CONFIG" RLXCoreSmoke
else
  echo "==> $SMOKE_BIN"
  "$SMOKE_BIN"
fi

echo "linux-smoke: OK ($CONFIG)"
