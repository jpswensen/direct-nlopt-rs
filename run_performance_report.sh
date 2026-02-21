#!/usr/bin/env bash
#
# run_performance_report.sh
#
# Runs the comprehensive performance benchmark suite and writes results
# to PERFORMANCE.md in the crate root.
#
# Usage:
#   ./run_performance_report.sh
#
# Environment variables:
#   RAYON_NUM_THREADS  - Control parallel thread count (default: all cores)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building performance_report example (release) ==="
RUSTFLAGS="-C target-cpu=native" cargo build --example performance_report --release 2>&1

echo ""
echo "=== Running performance benchmarks ==="
echo "This may take a few minutes..."
echo ""

RUSTFLAGS="-C target-cpu=native" cargo run --example performance_report --release

echo ""
echo "=== Done ==="
echo "Results written to PERFORMANCE.md"
