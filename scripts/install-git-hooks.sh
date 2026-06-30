#!/usr/bin/env bash
# Configure this clone to use .githooks/ for git hooks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit scripts/pre-commit.sh
echo "git hooksPath -> .githooks (pre-commit active)"
echo "  Skip once:     SKIP_PRECOMMIT=1 git commit ..."
echo "  Skip tests:    PRECOMMIT_SKIP_TESTS=1 git commit ...  # no smoke / XCTest"
echo "  Skip smoke:    PRECOMMIT_SKIP_SMOKE=1 git commit ..."
echo "  Manual run:    ./scripts/pre-commit.sh"
