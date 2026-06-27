#!/usr/bin/env bash
# Build the Linux Dockerfile and run tier-2 smoke (useful from macOS).
#
# Usage:
#   ./scripts/linux-smoke-docker.sh              # build image (if needed) + smoke (debug)
#   ./scripts/linux-smoke-docker.sh --release    # release configuration
#   ./scripts/linux-smoke-docker.sh --rebuild    # force docker build --no-cache
#   IMAGE_TAG=rlx-swift-linux:local ./scripts/linux-smoke-docker.sh
#   SWIFT_IMAGE=swift:6.0.3 ./scripts/linux-smoke-docker.sh   # base image for docker build
#
# Pass-through: any args other than --rebuild are forwarded to linux-smoke.sh inside
# the container (e.g. --release). Host sources are bind-mounted at /workspace so you
# do not need to rebuild the image after every source edit (only after Dockerfile /
# apt dependency changes).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:-rlx-swift-linux:local}"
SWIFT_IMAGE="${SWIFT_IMAGE:-swift:6.0}"
REBUILD=0
SMOKE_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      SMOKE_ARGS+=("$arg")
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not on PATH; install Docker Desktop (Mac) or Docker Engine" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: cannot talk to Docker daemon — is Docker Desktop running?" >&2
  exit 1
fi

BUILD_FLAGS=( -t "$IMAGE_TAG" --build-arg "SWIFT_IMAGE=$SWIFT_IMAGE" -f Dockerfile . )
if [[ "$REBUILD" -eq 1 ]]; then
  BUILD_FLAGS=( --no-cache "${BUILD_FLAGS[@]}" )
fi

echo "==> docker build ${BUILD_FLAGS[*]}"
docker build "${BUILD_FLAGS[@]}"

echo "==> docker run --rm -v \$PWD:/workspace:ro $IMAGE_TAG ${SMOKE_ARGS[*]:-}"
docker run --rm \
  -v "$PWD":/workspace:ro \
  "$IMAGE_TAG" \
  ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}

echo "linux-smoke-docker: OK ($IMAGE_TAG)"
