//! Parallel stress test for DIRECT-NLOPT algorithm
//!
//! Uses expensive objective functions (10,000 sin/cos iterations per evaluation)
//! to demonstrate parallelization benefits at different dimensionalities.
//!
//! Run with: cargo run --example parallel_stress_test --release
//!
//! Limit workers: RAYON_NUM_THREADS=4 cargo run --example parallel_stress_test --release

use direct_nlopt::{DirectBuilder, types::DirectAlgorithm};
use std::time::Instant;

/// Expensive Rosenbrock: adds 10,000 sin/cos iterations per evaluation
/// to simulate a costly objective (e.g., physics simulation, FEA).
fn expensive_rosenbrock(x: &[f64]) -> f64 {
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }

    let mut result = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        result += 100.0 * t1 * t1 + t2 * t2;
    }

    result + extra_work * 1e-20
}

/// Expensive Rastrigin: adds 10,000 sin/cos iterations per evaluation.
fn expensive_rastrigin(x: &[f64]) -> f64 {
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }

    let a = 10.0;
    let n = x.len() as f64;
    let result = a * n
        + x.iter()
            .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();

    result + extra_work * 1e-20
}

struct BenchResult {
    fun: f64,
    nfev: usize,
    elapsed: std::time::Duration,
}

fn run_bench(
    name: &str,
    func: fn(&[f64]) -> f64,
    bounds: Vec<(f64, f64)>,
    max_feval: usize,
    parallel: bool,
) -> BenchResult {
    let start = Instant::now();
    let result = DirectBuilder::new(func, bounds)
        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
        .max_feval(max_feval)
        .parallel(parallel)
        .minimize()
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    BenchResult {
        fun: result.fun,
        nfev: result.nfev,
        elapsed: start.elapsed(),
    }
}

fn run_comparison(
    label: &str,
    func: fn(&[f64]) -> f64,
    bounds: Vec<(f64, f64)>,
    max_feval: usize,
) -> f64 {
    let dim = bounds.len();
    println!("--- {label} ({dim}D, maxfeval={max_feval}) ---\n");

    // Serial
    print!("  Serial   ... ");
    let serial = run_bench(label, func, bounds.clone(), max_feval, false);
    println!(
        "f={:.6e}  nfev={:<6}  time={:.3}s",
        serial.fun,
        serial.nfev,
        serial.elapsed.as_secs_f64()
    );

    // Parallel
    print!("  Parallel ... ");
    let par = run_bench(label, func, bounds, max_feval, true);
    println!(
        "f={:.6e}  nfev={:<6}  time={:.3}s",
        par.fun,
        par.nfev,
        par.elapsed.as_secs_f64()
    );

    let speedup = serial.elapsed.as_secs_f64() / par.elapsed.as_secs_f64();
    println!("  Speedup: {speedup:.2}x\n");
    speedup
}

fn main() {
    println!("=== DIRECT-NLOPT Parallel Stress Test ===");
    println!(
        "Expensive objective functions with 10,000 sin/cos iterations per evaluation\n"
    );

    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    println!("Available CPU threads: {num_cpus}\n");

    let max_feval = 500;

    // 2D Rosenbrock
    let s_2d = run_comparison(
        "Rosenbrock",
        expensive_rosenbrock,
        vec![(-5.0, 10.0); 2],
        max_feval,
    );

    // 3D Rastrigin
    let s_3d = run_comparison(
        "Rastrigin",
        expensive_rastrigin,
        vec![(-5.12, 5.12); 3],
        max_feval,
    );

    // 5D Rastrigin
    let s_5d = run_comparison(
        "Rastrigin",
        expensive_rastrigin,
        vec![(-5.12, 5.12); 5],
        max_feval,
    );

    // 8D Rastrigin
    let s_8d = run_comparison(
        "Rastrigin",
        expensive_rastrigin,
        vec![(-5.12, 5.12); 8],
        max_feval,
    );

    println!("=== Summary ===");
    println!("CPU threads: {num_cpus}");
    println!("╔══════════════════════╦══════════╗");
    println!("║ Configuration        ║ Speedup  ║");
    println!("╠══════════════════════╬══════════╣");
    println!("║ 2D Rosenbrock        ║ {:>6.2}x  ║", s_2d);
    println!("║ 3D Rastrigin         ║ {:>6.2}x  ║", s_3d);
    println!("║ 5D Rastrigin         ║ {:>6.2}x  ║", s_5d);
    println!("║ 8D Rastrigin         ║ {:>6.2}x  ║", s_8d);
    println!("╚══════════════════════╩══════════╝");
    println!();
    println!("=== Parallelization Analysis ===");
    println!();
    println!("The DIRECT algorithm parallelizes function evaluations during rectangle");
    println!("division. When a rectangle is divided, 2*d new points are evaluated");
    println!("independently. Benefits scale with:");
    println!("  - Dimension (more points per division: 2*d)");
    println!("  - Evaluation cost (parallelism overhead amortized)");
    println!();
    println!("Sequential operations (not parallelizable):");
    println!("  - Convex hull selection of potentially optimal rectangles");
    println!("  - Rectangle storage linked-list updates");
    println!("  - Main iteration loop control flow");
}
