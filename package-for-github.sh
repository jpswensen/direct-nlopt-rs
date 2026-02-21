#!/usr/bin/env bash
#
# package-for-github.sh
#
# Packages the direct-nlopt-rs crate for upload to GitHub.
# This script:
#   1. Commits any pending changes on the feature branch
#   2. Merges the feature branch into main
#   3. Runs the full test suite to validate
#   4. Adds a GitHub remote (if not already set)
#   5. Pushes to GitHub
#
# Usage:
#   ./package-for-github.sh <github-repo-url>
#
# Example:
#   ./package-for-github.sh https://github.com/username/direct-nlopt-rs.git
#   ./package-for-github.sh git@github.com:username/direct-nlopt-rs.git
#

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Configuration ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRATE_DIR="$SCRIPT_DIR"
FEATURE_BRANCH="feature/nlopt-direct-faithful-implementation"
MAIN_BRANCH="main"

# ── Preflight checks ───────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo ""
    echo "Usage: $0 <github-repo-url>"
    echo ""
    echo "Examples:"
    echo "  $0 https://github.com/username/direct-nlopt-rs.git"
    echo "  $0 git@github.com:username/direct-nlopt-rs.git"
    echo ""
    exit 1
fi

REPO_URL="$1"

command -v cargo >/dev/null 2>&1 || fail "cargo not found. Install Rust: https://rustup.rs"
command -v git   >/dev/null 2>&1 || fail "git not found."

[[ -d "$CRATE_DIR" ]] || fail "Crate directory not found: $CRATE_DIR"
[[ -f "$CRATE_DIR/Cargo.toml" ]] || fail "Cargo.toml not found in $CRATE_DIR"

cd "$CRATE_DIR"
info "Working in: $CRATE_DIR"

# ── Step 1: Ensure we're on the feature branch ─────────────────────────
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$FEATURE_BRANCH" ]]; then
    fail "Expected branch '$FEATURE_BRANCH', but on '$CURRENT_BRANCH'. Please switch branches first."
fi
ok "On branch: $FEATURE_BRANCH"

# ── Step 2: Clean build artifacts from tests/ ──────────────────────────
info "Cleaning compiled test binaries..."
rm -f tests/test_c_ffi tests/test_cpp_ffi
ok "Test binaries cleaned"

# ── Step 3: Stage and commit any pending changes ───────────────────────
if [[ -n "$(git status --porcelain)" ]]; then
    info "Staging uncommitted changes..."
    git add -A
    git commit -m "chore: final polish before packaging for GitHub"
    ok "Changes committed on $FEATURE_BRANCH"
else
    ok "Working tree is clean — nothing to commit"
fi

# ── Step 4: Run the full test suite ────────────────────────────────────
info "Running cargo test (release)..."
if cargo test --release 2>&1; then
    ok "All tests pass"
else
    fail "Tests failed. Fix issues before packaging."
fi

# ── Step 5: Build release artifacts ────────────────────────────────────
info "Building release (with FFI)..."
cargo build --release --features ffi 2>&1
ok "Release build complete"

# ── Step 6: Verify library artifacts exist ─────────────────────────────
STATIC_LIB="target/release/libdirect_nlopt.a"
if [[ "$(uname)" == "Darwin" ]]; then
    SHARED_LIB="target/release/libdirect_nlopt.dylib"
else
    SHARED_LIB="target/release/libdirect_nlopt.so"
fi

[[ -f "$STATIC_LIB" ]]  || fail "Static library not found: $STATIC_LIB"
[[ -f "$SHARED_LIB" ]]  || fail "Shared library not found: $SHARED_LIB"
ok "Libraries verified: $(du -h "$STATIC_LIB" | cut -f1) static, $(du -h "$SHARED_LIB" | cut -f1) shared"

# ── Step 7: Merge feature branch into main ─────────────────────────────
info "Merging $FEATURE_BRANCH into $MAIN_BRANCH..."

# Create main branch if it doesn't exist
if ! git show-ref --verify --quiet refs/heads/"$MAIN_BRANCH"; then
    info "Creating $MAIN_BRANCH branch..."
    git checkout -b "$MAIN_BRANCH"
else
    git checkout "$MAIN_BRANCH"
fi

git merge "$FEATURE_BRANCH" --no-ff -m "Merge $FEATURE_BRANCH: complete NLOPT DIRECT algorithm implementation

- 100% faithful port of both NLOPT DIRECT implementations (Gablonsky + SGJ cdirect)
- All 8 algorithm variants: Original, LocallyBiased, Randomized + unscaled + Gablonsky
- Bit-identical results to NLOPT C in serial mode
- Rayon parallelization for Gablonsky backend
- C FFI via cbindgen
- Comprehensive test suite (unit, integration, regression, edge cases)
- Built using Claude Opus 4.6 and the ralph-wiggum method"
ok "Merged into $MAIN_BRANCH"

# ── Step 8: Add GitHub remote and push ─────────────────────────────────
REMOTE_NAME="origin"
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    EXISTING_URL="$(git remote get-url "$REMOTE_NAME")"
    if [[ "$EXISTING_URL" != "$REPO_URL" ]]; then
        warn "Remote '$REMOTE_NAME' points to '$EXISTING_URL'"
        info "Updating remote to: $REPO_URL"
        git remote set-url "$REMOTE_NAME" "$REPO_URL"
    fi
    ok "Remote '$REMOTE_NAME' -> $REPO_URL"
else
    info "Adding remote '$REMOTE_NAME': $REPO_URL"
    git remote add "$REMOTE_NAME" "$REPO_URL"
    ok "Remote added"
fi

info "Pushing $MAIN_BRANCH to $REMOTE_NAME..."
git push -u "$REMOTE_NAME" "$MAIN_BRANCH"
ok "Pushed $MAIN_BRANCH"

info "Pushing $FEATURE_BRANCH to $REMOTE_NAME..."
git push -u "$REMOTE_NAME" "$FEATURE_BRANCH"
ok "Pushed $FEATURE_BRANCH"

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  direct-nlopt-rs successfully packaged and pushed!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo "  Repository: $REPO_URL"
echo "  Branches:   $MAIN_BRANCH, $FEATURE_BRANCH"
echo ""
echo "  Next steps:"
echo "    • Create a GitHub Release with tag v0.1.0"
echo "    • Publish to crates.io: cd direct-nlopt-rs && cargo publish"
echo ""
