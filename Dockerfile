# Linux tier-2 image for rlx-swift: build + RLXCoreSmoke (CPU mlx-swift, no Metal).
#
# Use from macOS (or any Docker host) to exercise the same gate as CI `linux-smoke`
# without a Linux VM.
#
# Build:
#   docker build -t rlx-swift-linux .
#
# Run (baked sources from the build context):
#   docker run --rm rlx-swift-linux
#   docker run --rm rlx-swift-linux --release
#
# Run with live host tree (typical Mac workflow — no image rebuild on every edit):
#   docker run --rm -v "$PWD":/workspace:ro rlx-swift-linux
#   ./scripts/linux-smoke-docker.sh
#
# Override base Swift image (must be 6.0+ Linux):
#   docker build --build-arg SWIFT_IMAGE=swift:6.0.3 -t rlx-swift-linux .

ARG SWIFT_IMAGE=swift:6.0
FROM ${SWIFT_IMAGE}

LABEL org.opencontainers.image.title="rlx-swift-linux"
LABEL org.opencontainers.image.description="Linux CPU build + RLXCoreSmoke for rlx-swift (mlx-swift backend)"
LABEL org.opencontainers.image.source="https://github.com/ml-explore/rlx-swift"

ENV DEBIAN_FRONTEND=noninteractive \
    # Prefer CPU for any accidental MLX default-device probes inside tools.
    MLX_DEFAULT_DEVICE=cpu

# mlx-swift CPU backend needs OpenBLAS/LAPACK headers + gfortran (see design.md / README).
# rsync: copy bind-mounted sources into a writable workdir (avoids Darwin .build conflicts).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libblas-dev \
        liblapacke-dev \
        libopenblas-dev \
        gfortran \
        ca-certificates \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# Snapshot of the repo at image build time (used when no /workspace mount is provided).
WORKDIR /src
COPY . /src

COPY scripts/docker-linux-entrypoint.sh /usr/local/bin/rlx-linux-entrypoint
RUN chmod +x /usr/local/bin/rlx-linux-entrypoint \
    && test -x /src/scripts/linux-smoke.sh

ENTRYPOINT ["/usr/local/bin/rlx-linux-entrypoint"]
# Extra args are forwarded to scripts/linux-smoke.sh (e.g. --release).
CMD []
