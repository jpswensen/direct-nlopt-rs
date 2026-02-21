#!/usr/bin/env bash
# run_all_comparisons.sh — Automated CI-friendly test runner for NLOPT-Rust equivalence.
#
# Compiles NLOPT C, runs all Rust comparison tests, and generates a summary report.
# Exits with failure if any comparison shows non-identical results (for sequential mode).
#
# Usage: ./tests/run_all_comparisons.sh [--release] [--perf] [--report FILE]
#   --release   Build and test in release mode (default: debug)
#   --perf      Include performance comparison in the report
#   --report    Write summary report to FILE (default: stdout)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Parse arguments ──
RELEASE_FLAG=""
BUILD_MODE="debug"
INCLUDE_PERF=false
REPORT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            RELEASE_FLAG="--release"
            BUILD_MODE="release"
            shift
            ;;
        --perf)
            INCLUDE_PERF=true
            shift
            ;;
        --report)
            REPORT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--release] [--perf] [--report FILE]"
            exit 1
            ;;
    esac
done

# ── Setup output ──
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SUITE_RESULTS=""

# Timestamps
START_TIME=$(date +%s)

report() {
    if [[ -n "$REPORT_FILE" ]]; then
        echo "$@" >> "$REPORT_FILE"
    fi
    echo "$@"
}

# Initialize report file
if [[ -n "$REPORT_FILE" ]]; then
    > "$REPORT_FILE"
fi

report "============================================================"
report "  NLOPT C ↔ Rust DIRECT Equivalence Test Report"
report "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
report "  Build mode: $BUILD_MODE"
report "============================================================"
report ""

# ── Step 1: Build with nlopt-compare feature ──
report "--- Step 1: Building Rust crate with nlopt-compare feature ---"
if cargo build $RELEASE_FLAG --features nlopt-compare 2>"$TMPDIR/build_stderr.txt"; then
    report "✅ Build succeeded"
else
    report "❌ Build FAILED"
    cat "$TMPDIR/build_stderr.txt"
    exit 1
fi
report ""

# ── Step 2: Run test suites ──
# Each suite is a group of related comparison tests.
# We run them individually to report per-suite pass/fail status.

run_test_suite() {
    local suite_name="$1"
    local test_filter="$2"
    local description="$3"

    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    local output_file="$TMPDIR/suite_${suite_name}.txt"

    # Run tests, capture all output to file
    cargo test $RELEASE_FLAG --features nlopt-compare -- $test_filter > "$output_file" 2>&1
    local exit_code=$?

    # Parse test counts from output (sum across all test binaries)
    local suite_passed=0
    local suite_failed=0
    while IFS= read -r result_line; do
        local p f
        p=$(echo "$result_line" | grep -o '[0-9]* passed' | grep -o '[0-9]*' || echo 0)
        f=$(echo "$result_line" | grep -o '[0-9]* failed' | grep -o '[0-9]*' || echo 0)
        suite_passed=$((suite_passed + ${p:-0}))
        suite_failed=$((suite_failed + ${f:-0}))
    done < <(grep -E '^test result:' "$output_file" || true)

    if [[ $exit_code -eq 0 && "$suite_failed" -eq 0 ]]; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + suite_passed))
        PASSED_TESTS=$((PASSED_TESTS + suite_passed))
        SUITE_RESULTS="${SUITE_RESULTS}✅ ${suite_name} (${suite_passed} tests) — ${description}\n"
        report "  ✅ ${suite_name}: ${suite_passed} tests passed"
    else
        FAILED_SUITES=$((FAILED_SUITES + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + suite_passed + suite_failed))
        PASSED_TESTS=$((PASSED_TESTS + suite_passed))
        FAILED_TESTS=$((FAILED_TESTS + suite_failed))
        SUITE_RESULTS="${SUITE_RESULTS}❌ ${suite_name} (${suite_passed} passed, ${suite_failed} failed) — ${description}\n"
        report "  ❌ ${suite_name}: ${suite_passed} passed, ${suite_failed} failed"
        # Show failure details
        grep -E 'FAILED|panicked|thread.*panicked' "$output_file" | head -10 || true
    fi
}

report "--- Step 2: Running comparison test suites ---"
report ""

# Unit-level comparison suites
run_test_suite "scaling" \
    "test_dirpreprc_ test_dirinfcn_ test_scaling_" \
    "Scaling/unscaling (dirpreprc_ + dirinfcn_)"

run_test_suite "levels" \
    "test_level_ test_levels_" \
    "Level computation (dirgetlevel_) for both variants"

run_test_suite "thirds" \
    "test_thirds_" \
    "Thirds/levels precomputation"

run_test_suite "longest_dims" \
    "test_dirget_i_" \
    "Get longest dimensions (dirget_i__)"

run_test_suite "linked_list" \
    "test_init_lists test_insert_ test_removal_ test_walk_list_" \
    "Linked list operations (dirinitlist_, dirinsertlist_)"

run_test_suite "dirchoose" \
    "test_dirchoose_ test_dirdoubleinsert_" \
    "PotentiallyOptimal selection (dirchoose_)"

run_test_suite "divide" \
    "test_divide_" \
    "Rectangle division (dirdivide_)"

run_test_suite "convex_hull" \
    "test_convex_hull_ test_rect_diameter_" \
    "CDirect convex hull and rect diameter"

# Integration-level comparison suites
run_test_suite "sphere_gablonsky" \
    "test_sphere_gablonsky" \
    "Sphere function — DIRECT_GABLONSKY (DIRECT-L)"

run_test_suite "sphere_original" \
    "test_sphere_original" \
    "Sphere function — DIRECT_ORIGINAL"

run_test_suite "rosenbrock" \
    "test_rosenbrock_" \
    "Rosenbrock function — both variants"

run_test_suite "rastrigin" \
    "test_rastrigin_" \
    "Rastrigin function — both variants"

run_test_suite "ackley" \
    "test_ackley_" \
    "Ackley function — both variants"

run_test_suite "styblinski_tang" \
    "test_styblinski_tang_" \
    "Styblinski-Tang function — both variants"

run_test_suite "cdirect_comparison" \
    "test_cdirect_ test_cross_implementation_" \
    "CDirect vs Gablonsky implementation consistency"

run_test_suite "maxfeval_termination" \
    "test_maxfeval_" \
    "maxfeval termination"

run_test_suite "fglobal_termination" \
    "test_fglobal_" \
    "fglobal termination"

run_test_suite "voltol_sigmatol" \
    "test_voltol_ test_sigmatol_" \
    "volume_reltol and sigma_reltol termination"

run_test_suite "force_stop" \
    "test_force_stop_" \
    "Force stop / callback behavior"

run_test_suite "highdim" \
    "test_sphere_5d_ test_sphere_10d_ test_sphere_20d_" \
    "Higher-dimensional problems (5D, 10D, 20D)"

run_test_suite "hidden_constraints" \
    "test_hidden_constraint_" \
    "Hidden constraints (infeasible regions)"

run_test_suite "1d_edge_cases" \
    "test_1d_" \
    "1D optimization edge cases"

run_test_suite "bounds_edge_cases" \
    "test_asymmetric_ test_narrow_ test_wide_ test_near_zero_ test_positive_asymmetric_ test_skewed_" \
    "Asymmetric and extreme bounds"

run_test_suite "flat_degenerate" \
    "test_constant_ test_maxiter_0_ test_maxiter_1_ test_maxfeval_1_" \
    "Flat objective and degenerate cases"

run_test_suite "golden_regression" \
    "test_golden_" \
    "Golden-file regression tests"

run_test_suite "trace_comparison" \
    "test_compare_" \
    "Step-by-step trace comparison"

run_test_suite "parallel_verification" \
    "test_parallel_" \
    "Parallel mode produces correct results"

run_test_suite "ffi" \
    "test_ffi_ nlopt_compare::" \
    "FFI bindings and NLOPT C interop"

report ""

# ── Step 3: Optional performance comparison ──
if $INCLUDE_PERF; then
    report "--- Step 3: Performance comparison ---"
    report ""

    run_test_suite "benchmark_comparison" \
        "test_benchmark_comparison" \
        "Performance benchmark: Rust vs NLOPT C"

    # Run the benchmark comparison test with --nocapture to show the table
    PERF_OUTPUT="$TMPDIR/perf_output.txt"
    if cargo test $RELEASE_FLAG --features nlopt-compare -- test_benchmark_comparison --nocapture 2>&1 > "$PERF_OUTPUT"; then
        # Extract the markdown table from the output
        if grep -q '|' "$PERF_OUTPUT"; then
            report ""
            report "### Performance Comparison Table"
            report ""
            grep -E '^\|' "$PERF_OUTPUT" | while IFS= read -r line; do
                report "$line"
            done
            report ""
        fi
    fi
fi

# ── Step 4: Generate summary ──
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

report "============================================================"
report "  SUMMARY"
report "============================================================"
report ""
report "Test suites: $PASSED_SUITES passed, $FAILED_SUITES failed, $TOTAL_SUITES total"
report "Tests:       $PASSED_TESTS passed, $FAILED_TESTS failed, $TOTAL_TESTS total"
report "Duration:    ${ELAPSED}s"
report ""
report "--- Suite Results ---"
echo -e "$SUITE_RESULTS" | while IFS= read -r line; do
    report "$line"
done
report ""

if [[ $FAILED_SUITES -gt 0 ]]; then
    report "❌ FAILED — $FAILED_SUITES suite(s) had failures"
    report ""
    report "To debug failures, re-run individual suites:"
    echo -e "$SUITE_RESULTS" | grep '^❌' | while IFS= read -r line; do
        suite=$(echo "$line" | sed 's/❌ \([^ ]*\).*/\1/')
        report "  cargo test --features nlopt-compare -- <filter> --nocapture"
    done
    exit 1
else
    report "✅ ALL PASSED — Rust implementation matches NLOPT C"
    exit 0
fi
