//! Performance benchmark comparing Rust DIRECT vs NLOPT C DIRECT.
//!
//! Feature-gated behind "nlopt-compare" to avoid requiring a C compiler for
//! normal builds. Run with:
//!   cargo test --features nlopt-compare --release test_benchmark_comparison -- --nocapture
//!
//! This test times both NLOPT C and Rust implementations on identical problems
//! and prints a markdown comparison table.

#![cfg(feature = "nlopt-compare")]

#[path = "nlopt_ffi.rs"]
mod nlopt_ffi;

use nlopt_ffi::{
    DirectAlgorithmC, DirectObjectiveFuncC, direct_optimize,
};
use std::os::raw::c_int;
use std::time::Instant;

use direct_nlopt::{DirectBuilder, types::DirectAlgorithm};

// ── Test functions (Rust versions) ──

fn sphere_rs(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn rosenbrock_rs(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

fn rastrigin_rs(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut sum = 10.0 * n;
    for xi in x {
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

struct BenchResult {
    time_ms: f64,
    minf: f64,
    nfev: Option<usize>,
    nit: Option<usize>,
}

fn time_nlopt_c(
    func: DirectObjectiveFuncC,
    dim: usize,
    lb: f64,
    ub: f64,
    maxfeval: i32,
    algo: DirectAlgorithmC,
    num_runs: usize,
) -> BenchResult {
    let lower: Vec<f64> = vec![lb; dim];
    let upper: Vec<f64> = vec![ub; dim];
    let mut x = vec![0.0f64; dim];
    let mut minf: f64 = 0.0;
    let mut force_stop: c_int = 0;

    // Warmup
    unsafe {
        direct_optimize(
            func,
            std::ptr::null_mut(),
            dim as c_int,
            lower.as_ptr(),
            upper.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            maxfeval,
            -1,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            &mut force_stop,
            f64::NEG_INFINITY,
            0.0,
            std::ptr::null_mut(),
            algo,
        );
    }

    let mut total_ms = 0.0;
    for _ in 0..num_runs {
        x.fill(0.0);
        force_stop = 0;
        let t0 = Instant::now();
        unsafe {
            direct_optimize(
                func,
                std::ptr::null_mut(),
                dim as c_int,
                lower.as_ptr(),
                upper.as_ptr(),
                x.as_mut_ptr(),
                &mut minf,
                maxfeval,
                -1,
                0.0,
                0.0,
                1e-4,
                0.0,
                0.0,
                -1.0,
                &mut force_stop,
                f64::NEG_INFINITY,
                0.0,
                std::ptr::null_mut(),
                algo,
            );
        }
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    BenchResult {
        time_ms: total_ms / num_runs as f64,
        minf,
        nfev: None,
        nit: None,
    }
}

fn time_rust(
    func: fn(&[f64]) -> f64,
    dim: usize,
    lb: f64,
    ub: f64,
    maxfeval: usize,
    algo: DirectAlgorithm,
    parallel: bool,
    num_runs: usize,
) -> BenchResult {
    let bounds: Vec<(f64, f64)> = vec![(lb, ub); dim];

    // Warmup
    let _ = DirectBuilder::new(func, bounds.clone())
        .algorithm(algo)
        .max_feval(maxfeval)
        .parallel(parallel)
        .minimize();

    let mut total_ms = 0.0;
    let mut last_nfev = 0;
    let mut last_nit = 0;
    let mut last_minf = 0.0;

    for _ in 0..num_runs {
        let t0 = Instant::now();
        let result = DirectBuilder::new(func, bounds.clone())
            .algorithm(algo)
            .max_feval(maxfeval)
            .parallel(parallel)
            .minimize()
            .unwrap();
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
        last_nfev = result.nfev;
        last_nit = result.nit;
        last_minf = result.fun;
    }

    BenchResult {
        time_ms: total_ms / num_runs as f64,
        minf: last_minf,
        nfev: Some(last_nfev),
        nit: Some(last_nit),
    }
}

struct BenchConfig {
    name: &'static str,
    func_c: DirectObjectiveFuncC,
    func_rs: fn(&[f64]) -> f64,
    dim: usize,
    lb: f64,
    ub: f64,
    maxfeval: usize,
    algo_c: DirectAlgorithmC,
    algo_rs: DirectAlgorithm,
    algo_name: &'static str,
}

#[test]
fn test_benchmark_comparison() {
    let num_runs = 10;

    use nlopt_ffi::{sphere_c, rosenbrock_c, rastrigin_c};

    let configs = vec![
        // Sphere
        BenchConfig {
            name: "sphere_2d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 2,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "sphere_2d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 2,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        BenchConfig {
            name: "sphere_5d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 5,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "sphere_5d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 5,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        BenchConfig {
            name: "sphere_10d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 10,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "sphere_10d",
            func_c: sphere_c,
            func_rs: sphere_rs,
            dim: 10,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        // Rosenbrock
        BenchConfig {
            name: "rosenbrock_2d",
            func_c: rosenbrock_c,
            func_rs: rosenbrock_rs,
            dim: 2,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "rosenbrock_2d",
            func_c: rosenbrock_c,
            func_rs: rosenbrock_rs,
            dim: 2,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        BenchConfig {
            name: "rosenbrock_5d",
            func_c: rosenbrock_c,
            func_rs: rosenbrock_rs,
            dim: 5,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "rosenbrock_5d",
            func_c: rosenbrock_c,
            func_rs: rosenbrock_rs,
            dim: 5,
            lb: -5.0,
            ub: 5.0,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        // Rastrigin
        BenchConfig {
            name: "rastrigin_2d",
            func_c: rastrigin_c,
            func_rs: rastrigin_rs,
            dim: 2,
            lb: -5.12,
            ub: 5.12,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "rastrigin_2d",
            func_c: rastrigin_c,
            func_rs: rastrigin_rs,
            dim: 2,
            lb: -5.12,
            ub: 5.12,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
        BenchConfig {
            name: "rastrigin_5d",
            func_c: rastrigin_c,
            func_rs: rastrigin_rs,
            dim: 5,
            lb: -5.12,
            ub: 5.12,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_GABLONSKY,
            algo_rs: DirectAlgorithm::GablonskyLocallyBiased,
            algo_name: "GABLONSKY",
        },
        BenchConfig {
            name: "rastrigin_5d",
            func_c: rastrigin_c,
            func_rs: rastrigin_rs,
            dim: 5,
            lb: -5.12,
            ub: 5.12,
            maxfeval: 5000,
            algo_c: DirectAlgorithmC::DIRECT_ORIGINAL,
            algo_rs: DirectAlgorithm::GablonskyOriginal,
            algo_name: "ORIGINAL",
        },
    ];

    println!();
    println!("## NLOPT C vs Rust DIRECT Performance Comparison");
    println!("({} runs averaged, maxfeval=5000)", num_runs);
    println!();
    println!(
        "| {:<28} | {:>9} | {:>12} | {:>12} | {:>12} | {:>8} | {:>5} | {:>5} |",
        "Benchmark", "Algorithm", "C time (ms)", "Rust (ms)", "Rust‖ (ms)", "Speedup", "nfev", "nit"
    );
    println!(
        "|{:-<30}|{:-<11}|{:-<14}|{:-<14}|{:-<14}|{:-<10}|{:-<7}|{:-<7}|",
        "", "", "", "", "", "", "", ""
    );

    for cfg in &configs {
        // Time NLOPT C
        let c_res = time_nlopt_c(
            cfg.func_c,
            cfg.dim,
            cfg.lb,
            cfg.ub,
            cfg.maxfeval as i32,
            cfg.algo_c,
            num_runs,
        );

        // Time Rust serial (apples-to-apples vs C)
        let rs_serial = time_rust(
            cfg.func_rs,
            cfg.dim,
            cfg.lb,
            cfg.ub,
            cfg.maxfeval,
            cfg.algo_rs,
            false,
            num_runs,
        );

        // Time Rust parallel
        let rs_parallel = time_rust(
            cfg.func_rs,
            cfg.dim,
            cfg.lb,
            cfg.ub,
            cfg.maxfeval,
            cfg.algo_rs,
            true,
            num_runs,
        );

        let speedup = c_res.time_ms / rs_serial.time_ms;

        println!(
            "| {:<28} | {:>9} | {:>10.3} | {:>10.3} | {:>10.3} | {:>7.2}x | {:>5} | {:>5} |",
            format!("{}_{}", cfg.name, cfg.algo_name.to_lowercase()),
            cfg.algo_name,
            c_res.time_ms,
            rs_serial.time_ms,
            rs_parallel.time_ms,
            speedup,
            rs_serial.nfev.unwrap_or(0),
            rs_serial.nit.unwrap_or(0),
        );

        // Verify results are close (exact correctness is verified by other tests)
        let tol = if cfg.dim <= 2 { 1e-10 } else { 1e-2 };
        assert!(
            (c_res.minf - rs_serial.minf).abs() < tol
                || (c_res.minf - rs_serial.minf).abs() / c_res.minf.abs().max(1e-15) < tol,
            "{} {}: C minf={:.15e} != Rust minf={:.15e}",
            cfg.name,
            cfg.algo_name,
            c_res.minf,
            rs_serial.minf
        );
    }

    println!();
    println!("Speedup = C_time / Rust_serial_time (>1.0 means Rust is faster)");
    println!("Rust‖ = Rust with parallel=true");
}
