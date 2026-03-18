#!/usr/bin/env bash
#
# package-for-github.sh
#
# Commits and pushes the current state of the direct-nlopt-rs crate
# to the GitHub repository at https://github.com/jpswensen/direct-nlopt-rs.
#
# This script:
#   1. Detects uncommitted changes
#   2. Runs the full test suite to validate
#   3. Stages and commits with a message (provided or auto-generated)
#   4. Ensures the GitHub remote is configured
#   5. Pushes the current branch to GitHub
#
# Usage:
#   ./package-for-github.sh                        # auto-generated commit message
#   ./package-for-github.sh "feat: add constraints" # custom commit message
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
GITHUB_URL="https://github.com/jpswensen/direct-nlopt-rs"
REMOTE_NAME="origin"

cd "$CRATE_DIR"
info "Working in: $CRATE_DIR"

# ── Preflight checks ───────────────────────────────────────────────────
command -v cargo >/dev/null 2>&1 || fail "cargo not found. Install Rust: https://rustup.rs"
command -v git   >/dev/null 2>&1 || fail "git not found."
[[ -f "$CRATE_DIR/Cargo.toml" ]] || fail "Cargo.toml not found in $CRATE_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || fail "Not a git repository: $CRATE_DIR"

# ── Step 1: Show current state ─────────────────────────────────────────
CURRENT_BRANCH="$(git branch --show-current)"
ok "On branch: $CURRENT_BRANCH"

CHANGED_FILES="$(git status --porcelain)"
if [[ -z "$CHANGED_FILES" ]]; then
    warn "Working tree is clean — nothing to commit."
    info "Checking if there are unpushed commits..."
    UNPUSHED="$(git log "$REMOTE_NAME/$CURRENT_BRANCH..$CURRENT_BRANCH" --oneline 2>/dev/null || true)"
    if [[ -z "$UNPUSHED" ]]; then
        ok "Everything is up-to-date with $REMOTE_NAME/$CURRENT_BRANCH"
        exit 0
    else
        info "Found unpushed commits:"
        echo "$UNPUSHED"
    fi
else
    info "Changed files:"
    echo "$CHANGED_FILES"
    echo ""
fi

# ── Step 2: Run the full test suite ────────────────────────────────────
info "Running cargo test --release --lib ..."
if cargo test --release --lib 2>&1; then
    ok "Library tests pass"
else
    fail "Library tests failed. Fix issues before pushing."
fi

info "Running cargo test --release --tests ..."
if cargo test --release --tests 2>&1; then
    ok "Integration tests pass"
else
    fail "Integration tests failed. Fix issues before pushing."
fi

# ── Step 3: Stage and commit ───────────────────────────────────────────
if [[ -n "$CHANGED_FILES" ]]; then
    # Build commit message
    if [[ $# -ge 1 ]]; then
        COMMIT_MSG="$1"
    else
        # Auto-generate from changed files
        NUM_MODIFIED="$(echo "$CHANGED_FILES" | grep -c '^ M\|^M ' || true)"
        NUM_NEW="$(echo "$CHANGED_FILES" | grep -c '^??\|^A ' || true)"
        NUM_DELETED="$(echo "$CHANGED_FILES" | grep -c '^ D\|^D ' || true)"

        PARTS=()
        [[ "$NUM_MODIFIED" -gt 0 ]] && PARTS+=("${NUM_MODIFIED} modified")
        [[ "$NUM_NEW" -gt 0 ]]      && PARTS+=("${NUM_NEW} new")
        [[ "$NUM_DELETED" -gt 0 ]]   && PARTS+=("${NUM_DELETED} deleted")
        SUMMARY="$(IFS=', '; echo "${PARTS[*]}")"

        COMMIT_MSG="chore: update (${SUMMARY} files)"
    fi

    info "Staging all changes..."
    git add -A
    info "Committing: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
    ok "Changes committed"
fi

# ── Step 4: Ensure GitHub remote is configured ─────────────────────────
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    EXISTING_URL="$(git remote get-url "$REMOTE_NAME")"
    ok "Remote '$REMOTE_NAME' -> $EXISTING_URL"
else
    info "Adding remote '$REMOTE_NAME': $GITHUB_URL"
    git remote add "$REMOTE_NAME" "$GITHUB_URL"
    ok "Remote added"
fi

# ── Step 5: Push ───────────────────────────────────────────────────────
info "Pushing $CURRENT_BRANCH to $REMOTE_NAME..."
git push -u "$REMOTE_NAME" "$CURRENT_BRANCH"
ok "Pushed $CURRENT_BRANCH"

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  direct-nlopt-rs successfully pushed to GitHub!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo "  Repository: $GITHUB_URL"
echo "  Branch:     $CURRENT_BRANCH"
echo "  Commit:     $(git log --oneline -1)"
echo ""
