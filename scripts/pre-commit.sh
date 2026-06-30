#!/usr/bin/env bash
# Pre-commit checks for rlx-swift (also invoked by .githooks/pre-commit).
#
# Runs:
#   1) Lightweight lint (SwiftLint if installed; otherwise basic file checks)
#   2) swift build
#   3) RLXCoreSmoke (local CLI smoke — not Linux Docker; CI covers Linux)
#   4) Unit tests via xcodebuild on macOS (skipped on Linux — no Metal/XCTest gate here)
#
# Does NOT run Linux Docker smoke or iOS Simulator (GitHub Actions / verify-all).
#
# Skip once:   SKIP_PRECOMMIT=1 git commit ...
# Skip tests:  PRECOMMIT_SKIP_TESTS=1 git commit ...   # also skips smoke
# Skip smoke:  PRECOMMIT_SKIP_SMOKE=1 git commit ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_PRECOMMIT:-}" == "1" ]]; then
  echo "pre-commit: skipped (SKIP_PRECOMMIT=1)"
  exit 0
fi

# Only enforce when Swift sources / package manifest change (faster commits for docs-only).
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  CHANGED=$(git diff --cached --name-only --diff-filter=ACM || true)
else
  CHANGED=$(git diff --cached --name-only --diff-filter=ACM || true)
fi

needs_swift_checks=0
if [[ -z "${CHANGED}" ]]; then
  # Allow empty staged set edge cases; still run if forced
  needs_swift_checks=1
else
  while IFS= read -r f; do
    case "$f" in
      Package.swift|Package.resolved|Sources/*|Tests/*|scripts/xcodebuild-test.sh|scripts/pre-commit.sh|.swiftlint.yml)
        needs_swift_checks=1
        break
        ;;
    esac
  done <<< "$CHANGED"
fi

if [[ "$needs_swift_checks" -eq 0 ]]; then
  echo "pre-commit: no Swift/package changes staged — skipping build & tests"
  # Still run basic whitespace on staged non-binary text
fi

echo "pre-commit: lint"

# --- Basic staged-file checks (always) ---
if [[ -n "${CHANGED:-}" ]]; then
  while IFS= read -r f; do
    [[ -z "$f" || ! -f "$f" ]] && continue
    # Skip binaries / large assets
    case "$f" in
      *.png|*.jpg|*.jpeg|*.gif|*.pdf|*.xcresult) continue ;;
    esac
    # Disallow tabs in Swift sources
    if [[ "$f" == *.swift ]] && grep -q $'\t' "$f" 2>/dev/null; then
      echo "pre-commit: tabs found in $f (use spaces)" >&2
      exit 1
    fi
    # Trailing whitespace on staged Swift / markdown / scripts
    case "$f" in
      *.swift|*.md|*.sh|Package.swift)
        if grep -n '[[:space:]]$' "$f" 2>/dev/null | head -5; then
          echo "pre-commit: trailing whitespace in $f" >&2
          exit 1
        fi
        ;;
    esac
  done <<< "$CHANGED"
fi

# --- SwiftLint (optional) ---
if command -v swiftlint >/dev/null 2>&1; then
  echo "pre-commit: SwiftLint"
  # Lint only staged Swift files when possible
  staged_swift=$(git diff --cached --name-only --diff-filter=ACM | grep '\.swift$' || true)
  if [[ -n "$staged_swift" ]]; then
    # shellcheck disable=SC2086
    swiftlint lint --strict --quiet $staged_swift || {
      echo "pre-commit: SwiftLint failed (install config: .swiftlint.yml)" >&2
      exit 1
    }
  else
    swiftlint lint --strict --quiet || exit 1
  fi
else
  echo "pre-commit: SwiftLint not installed — skipping (brew install swiftlint)"
fi

if [[ "$needs_swift_checks" -eq 0 ]]; then
  echo "pre-commit: ok (lint only)"
  exit 0
fi

echo "pre-commit: swift build"
swift build

if [[ "${PRECOMMIT_SKIP_TESTS:-}" == "1" ]]; then
  echo "pre-commit: smoke + unit tests skipped (PRECOMMIT_SKIP_TESTS=1)"
  echo "pre-commit: ok"
  exit 0
fi

# Local CLI smoke (same binary Linux CI runs, but on the host — no Docker).
if [[ "${PRECOMMIT_SKIP_SMOKE:-}" == "1" ]]; then
  echo "pre-commit: RLXCoreSmoke skipped (PRECOMMIT_SKIP_SMOKE=1)"
else
  echo "pre-commit: RLXCoreSmoke"
  swift run RLXCoreSmoke
fi

# Unit tests (macOS + Xcode). Linux Docker / iOS left to GitHub Actions.
if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app/Contents/Developer ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  fi
  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "pre-commit: xcodebuild not found — cannot run unit tests" >&2
    exit 1
  fi
  # Fail if only CLT (no full Xcode) — XCTest needs Xcode
  if [[ ! -d "${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}/Platforms/MacOSX.platform" ]]; then
    echo "pre-commit: set DEVELOPER_DIR to full Xcode to run tests" >&2
    exit 1
  fi
  echo "pre-commit: unit tests (xcodebuild, macOS)"
  ./scripts/xcodebuild-test.sh
else
  echo "pre-commit: non-macOS — skipping xcodebuild unit tests (CI runs Linux smoke + build)"
fi

echo "pre-commit: ok"
