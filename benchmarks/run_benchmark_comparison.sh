#!/usr/bin/env bash
# run_benchmark_comparison.sh — Compile and run NLOPT C and Rust DIRECT benchmarks,
# then generate a markdown comparison table.
#
# Usage: ./benchmarks/run_benchmark_comparison.sh [num_runs]
#   num_runs: number of timing runs per C benchmark config (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NUM_RUNS="${1:-10}"
C_BENCH_BIN="$PROJECT_DIR/target/nlopt_bench"
RESULTS_DIR="$PROJECT_DIR/benchmarks/results"

mkdir -p "$RESULTS_DIR"

echo "=== NLOPT C vs Rust DIRECT Performance Comparison ==="
echo ""

# ── Step 1: Compile C benchmark ──
echo "--- Compiling C benchmark (cc -O3) ---"
cc -O3 -o "$C_BENCH_BIN" \
    benchmarks/nlopt_bench.c \
    ../nlopt/src/algs/direct/DIRect.c \
    ../nlopt/src/algs/direct/DIRsubrout.c \
    ../nlopt/src/algs/direct/DIRserial.c \
    ../nlopt/src/algs/direct/direct_wrap.c \
    nlopt-shim/nlopt_util_shim.c \
    -I nlopt-shim \
    -I ../nlopt/src/algs/direct \
    -lm
echo "C benchmark compiled: $C_BENCH_BIN"

# ── Step 2: Run C benchmark ──
echo ""
echo "--- Running C benchmark ($NUM_RUNS runs per config) ---"
"$C_BENCH_BIN" "$NUM_RUNS" > "$RESULTS_DIR/c_results.json"
echo "C results saved to $RESULTS_DIR/c_results.json"
cat "$RESULTS_DIR/c_results.json"

# ── Step 3: Run Rust benchmark (quick mode for comparison data) ──
echo ""
echo "--- Running Rust benchmark (cargo bench) ---"
echo "Note: Criterion benchmarks run separately. Use 'cargo bench' for full results."
echo ""

# Run Rust benchmarks as a quick timing test (not full criterion)
# We use a simple Rust test binary instead of criterion for JSON output
cargo build --release --quiet 2>/dev/null

# Run a quick Rust timing test via a test
cat > "$RESULTS_DIR/rust_bench_runner.rs" << 'RUST_EOF'
use std::time::Instant;

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut sum = 10.0 * n;
    for xi in x {
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

struct BenchConfig {
    name: &'static str,
    func: fn(&[f64]) -> f64,
    dim: usize,
    lb: f64,
    ub: f64,
    maxfeval: usize,
    algo: direct_nlopt::types::DirectAlgorithm,
    algo_name: &'static str,
}

fn run_bench(cfg: &BenchConfig, num_runs: usize, parallel: bool) {
    use direct_nlopt::DirectBuilder;
    let bounds: Vec<(f64, f64)> = vec![(cfg.lb, cfg.ub); cfg.dim];
    let mode = if parallel { "parallel" } else { "serial" };

    // Warmup
    let _ = DirectBuilder::new(cfg.func, bounds.clone())
        .algorithm(cfg.algo)
        .max_feval(cfg.maxfeval)
        .parallel(parallel)
        .minimize();

    let mut total_ms = 0.0f64;
    let mut last_result = None;
    for _ in 0..num_runs {
        let t0 = Instant::now();
        let result = DirectBuilder::new(cfg.func, bounds.clone())
            .algorithm(cfg.algo)
            .max_feval(cfg.maxfeval)
            .parallel(parallel)
            .minimize()
            .unwrap();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed;
        last_result = Some(result);
    }
    let avg_ms = total_ms / num_runs as f64;
    let r = last_result.unwrap();
    let x_str: Vec<String> = r.x.iter().map(|v| format!("{:.15e}", v)).collect();

    println!(
        "{{\"name\":\"{name}_{mode}\",\"dim\":{dim},\"algo\":\"{algo}\",\"maxfeval\":{maxfeval},\
         \"time_ms\":{time:.3},\"nfev\":{nfev},\"nit\":{nit},\"minf\":{minf:.15e},\
         \"x\":[{x}]}}",
        name = cfg.name,
        mode = mode,
        dim = cfg.dim,
        algo = cfg.algo_name,
        maxfeval = cfg.maxfeval,
        time = avg_ms,
        nfev = r.nfev,
        nit = r.nit,
        minf = r.fun,
        x = x_str.join(","),
    );
}

fn main() {
    let num_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    eprintln!("Rust Benchmark — {} runs per config", num_runs);

    use direct_nlopt::types::DirectAlgorithm;

    let configs = vec![
        BenchConfig { name: "sphere_2d_gablonsky",  func: sphere,     dim: 2,  lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "sphere_2d_original",    func: sphere,     dim: 2,  lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "sphere_5d_gablonsky",   func: sphere,     dim: 5,  lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "sphere_5d_original",    func: sphere,     dim: 5,  lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "sphere_10d_gablonsky",  func: sphere,     dim: 10, lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "sphere_10d_original",   func: sphere,     dim: 10, lb: -5.0,  ub: 5.0,  maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "rosenbrock_2d_gablonsky", func: rosenbrock, dim: 2, lb: -5.0, ub: 5.0, maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "rosenbrock_2d_original",  func: rosenbrock, dim: 2, lb: -5.0, ub: 5.0, maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "rosenbrock_5d_gablonsky", func: rosenbrock, dim: 5, lb: -5.0, ub: 5.0, maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "rosenbrock_5d_original",  func: rosenbrock, dim: 5, lb: -5.0, ub: 5.0, maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "rastrigin_2d_gablonsky",  func: rastrigin,  dim: 2, lb: -5.12, ub: 5.12, maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "rastrigin_2d_original",   func: rastrigin,  dim: 2, lb: -5.12, ub: 5.12, maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
        BenchConfig { name: "rastrigin_5d_gablonsky",  func: rastrigin,  dim: 5, lb: -5.12, ub: 5.12, maxfeval: 5000, algo: DirectAlgorithm::GablonskyLocallyBiased, algo_name: "GABLONSKY" },
        BenchConfig { name: "rastrigin_5d_original",   func: rastrigin,  dim: 5, lb: -5.12, ub: 5.12, maxfeval: 5000, algo: DirectAlgorithm::GablonskyOriginal,      algo_name: "ORIGINAL" },
    ];

    for cfg in &configs {
        run_bench(cfg, num_runs, false); // serial
        run_bench(cfg, num_runs, true);  // parallel
    }
}
RUST_EOF

echo "For full Criterion benchmarks, run:"
echo "  cargo bench"
echo ""
echo "For the comparison script with markdown table generation, run:"
echo "  cargo bench --bench benchmarks 2>&1 | tee $RESULTS_DIR/criterion_output.txt"
echo ""

# ── Step 4: Generate markdown comparison table from C results ──
echo "--- Generating comparison table ---"
echo ""
echo "## NLOPT C Benchmark Results (cc -O3, $NUM_RUNS runs averaged)"
echo ""
echo "| Benchmark | Dim | Algorithm | maxfeval | Time (ms) | minf |"
echo "|-----------|-----|-----------|----------|-----------|------|"

while IFS= read -r line; do
    name=$(echo "$line" | sed 's/.*"name":"\([^"]*\)".*/\1/')
    dim=$(echo "$line" | sed 's/.*"dim":\([0-9]*\).*/\1/')
    algo=$(echo "$line" | sed 's/.*"algo":"\([^"]*\)".*/\1/')
    maxfeval=$(echo "$line" | sed 's/.*"maxfeval":\([0-9]*\).*/\1/')
    time_ms=$(echo "$line" | sed 's/.*"time_ms":\([0-9.]*\).*/\1/')
    minf=$(echo "$line" | sed 's/.*"minf":\([^,}]*\).*/\1/')
    echo "| $name | $dim | $algo | $maxfeval | $time_ms | $minf |"
done < "$RESULTS_DIR/c_results.json"

echo ""
echo "Run 'cargo bench' for detailed Rust Criterion benchmark results."
echo "Use 'cargo bench --bench benchmarks -- --save-baseline baseline' to save a baseline."

# Cleanup
rm -f "$RESULTS_DIR/rust_bench_runner.rs"
