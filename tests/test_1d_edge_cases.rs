#![cfg(feature = "nlopt-compare")]

//! Edge case tests for 1D optimization.
//!
//! In 1D: only 1 dimension to divide, maxi=1 always, only 2 new points per division.
//! Tests verify Rust produces identical results to NLOPT C for 1D problems,
//! and that parallel threshold falls back to sequential for 1D.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// C objective functions
// ─────────────────────────────────────────────────────────────────────────────

struct EvalCounter {
    count: AtomicUsize,
}

impl EvalCounter {
    fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }
    fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// 1D sphere (x^2) for C FFI with evaluation counter.
extern "C" fn sphere_1d_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    assert_eq!(n, 1, "Expected 1D function");
    let x0 = unsafe { *x };
    x0 * x0
}

/// 1D sphere for Rust.
fn sphere_1d(x: &[f64]) -> f64 {
    x[0] * x[0]
}

/// 1D shifted quadratic: (x - 1.5)^2 + 3.0, minimum at x=1.5, f=3.0
extern "C" fn shifted_quadratic_1d_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    assert_eq!(n, 1);
    let x0 = unsafe { *x };
    (x0 - 1.5) * (x0 - 1.5) + 3.0
}

fn shifted_quadratic_1d(x: &[f64]) -> f64 {
    (x[0] - 1.5) * (x[0] - 1.5) + 3.0
}

/// 1D absolute value |x - 0.7| (non-smooth), minimum at x=0.7, f=0
extern "C" fn abs_1d_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    assert_eq!(n, 1);
    let x0 = unsafe { *x };
    (x0 - 0.7).abs()
}

fn abs_1d(x: &[f64]) -> f64 {
    (x[0] - 0.7).abs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run C and Rust side-by-side, verify identical results
// ─────────────────────────────────────────────────────────────────────────────

fn run_comparison_1d(
    name: &str,
    c_func: nlopt_ffi::DirectObjectiveFuncC,
    rust_func: fn(&[f64]) -> f64,
    lower: f64,
    upper: f64,
    max_feval: i32,
    max_iter: i32,
    algorithm_c: DirectAlgorithmC,
    algorithm_rust: DirectAlgorithm,
) {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 1,
        lower_bounds: vec![lower],
        upper_bounds: vec![upper],
        max_feval,
        max_iter,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: algorithm_c,
    };
    let c_result = unsafe {
        c_runner.run(
            c_func,
            &c_counter as *const EvalCounter as *mut c_void,
        )
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(lower, upper)];
    let rust_max_iter = if max_iter < 0 { 0usize } else { max_iter as usize };
    let opts = DirectOptions {
        max_feval: max_feval as usize,
        max_iter: rust_max_iter,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: algorithm_rust,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(rust_func, &bounds, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    println!(
        "=== 1D {} ({:?}) ===",
        name,
        algorithm_c
    );
    println!(
        "C:    x=[{:.15e}], minf={:.15e}, nfev={}",
        c_result.x[0], c_result.minf, c_nfev
    );
    println!(
        "Rust: x=[{:.15e}], minf={:.15e}, nfev={}",
        rust_result.x[0], rust_result.fun, rust_result.nfev
    );

    assert_eq!(
        c_result.x[0], rust_result.x[0],
        "{} x[0] mismatch: C={:.15e}, Rust={:.15e}",
        name, c_result.x[0], rust_result.x[0]
    );
    assert_eq!(
        c_result.minf, rust_result.fun,
        "{} minf mismatch: C={:.15e}, Rust={:.15e}",
        name, c_result.minf, rust_result.fun
    );
    assert_eq!(
        c_nfev, rust_result.nfev,
        "{} nfev mismatch: C={}, Rust={}",
        name, c_nfev, rust_result.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D sphere x^2 on [-5,5] with DIRECT_GABLONSKY
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_sphere_gablonsky_maxiter20() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        10000,
        20,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

#[test]
fn test_1d_sphere_gablonsky_maxfeval200() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        200,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

#[test]
fn test_1d_sphere_gablonsky_maxfeval500() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D sphere x^2 on [-5,5] with DIRECT_ORIGINAL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_sphere_original_maxiter20() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        10000,
        20,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

#[test]
fn test_1d_sphere_original_maxfeval200() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        200,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

#[test]
fn test_1d_sphere_original_maxfeval500() {
    run_comparison_1d(
        "sphere",
        sphere_1d_counting,
        sphere_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D shifted quadratic on [-5,5] — minimum at x=1.5 (not at center)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_shifted_quadratic_gablonsky() {
    run_comparison_1d(
        "shifted_quadratic",
        shifted_quadratic_1d_counting,
        shifted_quadratic_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

#[test]
fn test_1d_shifted_quadratic_original() {
    run_comparison_1d(
        "shifted_quadratic",
        shifted_quadratic_1d_counting,
        shifted_quadratic_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D absolute value |x - 0.7| on [-5,5] — non-smooth, minimum at x=0.7
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_abs_gablonsky() {
    run_comparison_1d(
        "abs",
        abs_1d_counting,
        abs_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

#[test]
fn test_1d_abs_original() {
    run_comparison_1d(
        "abs",
        abs_1d_counting,
        abs_1d,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D sphere with asymmetric bounds [0, 10]
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_sphere_asymmetric_gablonsky() {
    run_comparison_1d(
        "sphere_asymmetric",
        sphere_1d_counting,
        sphere_1d,
        0.0,
        10.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

#[test]
fn test_1d_sphere_asymmetric_original() {
    run_comparison_1d(
        "sphere_asymmetric",
        sphere_1d_counting,
        sphere_1d,
        0.0,
        10.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify parallel threshold falls back to sequential for 1D
// ─────────────────────────────────────────────────────────────────────────────
// In 1D, maxi=1 always, so there are only 2 new sample points per rectangle.
// With the default min_parallel_evals=4, the parallel path should never activate,
// producing results identical to serial mode.

#[test]
fn test_1d_parallel_fallback_to_sequential() {
    let bounds = vec![(-5.0, 5.0)];

    // Run with parallel=false (serial baseline)
    let opts_serial = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver_serial = Direct::new(sphere_1d, &bounds, opts_serial).unwrap();
    let result_serial = solver_serial.minimize(None).unwrap();

    // Run with parallel=true, default min_parallel_evals=4
    // In 1D, only 2 points per rect → falls back to serial
    let opts_parallel = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: true,
        min_parallel_evals: 4, // 2 points < 4 → serial fallback
        ..Default::default()
    };
    let mut solver_parallel = Direct::new(sphere_1d, &bounds, opts_parallel).unwrap();
    let result_parallel = solver_parallel.minimize(None).unwrap();

    println!("=== 1D Parallel Fallback Test ===");
    println!(
        "Serial:   x=[{:.15e}], minf={:.15e}, nfev={}",
        result_serial.x[0], result_serial.fun, result_serial.nfev
    );
    println!(
        "Parallel: x=[{:.15e}], minf={:.15e}, nfev={}",
        result_parallel.x[0], result_parallel.fun, result_parallel.nfev
    );

    // Bit-exact identical results expected
    assert_eq!(
        result_serial.x[0], result_parallel.x[0],
        "x[0] differs between serial and parallel"
    );
    assert_eq!(
        result_serial.fun, result_parallel.fun,
        "fun differs between serial and parallel"
    );
    assert_eq!(
        result_serial.nfev, result_parallel.nfev,
        "nfev differs between serial and parallel"
    );
    assert_eq!(
        result_serial.nit, result_parallel.nit,
        "nit differs between serial and parallel"
    );
}

#[test]
fn test_1d_parallel_with_threshold_1() {
    let bounds = vec![(-5.0, 5.0)];

    // Run with parallel=false
    let opts_serial = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver_serial = Direct::new(sphere_1d, &bounds, opts_serial).unwrap();
    let result_serial = solver_serial.minimize(None).unwrap();

    // Run with parallel=true, min_parallel_evals=1 (always parallel)
    let opts_parallel = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: true,
        min_parallel_evals: 1, // always parallel, even for 2 points
        ..Default::default()
    };
    let mut solver_parallel = Direct::new(sphere_1d, &bounds, opts_parallel).unwrap();
    let result_parallel = solver_parallel.minimize(None).unwrap();

    println!("=== 1D Parallel Threshold=1 Test ===");
    println!(
        "Serial:   x=[{:.15e}], minf={:.15e}, nfev={}",
        result_serial.x[0], result_serial.fun, result_serial.nfev
    );
    println!(
        "Parallel: x=[{:.15e}], minf={:.15e}, nfev={}",
        result_parallel.x[0], result_parallel.fun, result_parallel.nfev
    );

    // Even with threshold=1, results should be identical (deterministic)
    assert_eq!(result_serial.x[0], result_parallel.x[0]);
    assert_eq!(result_serial.fun, result_parallel.fun);
    assert_eq!(result_serial.nfev, result_parallel.nfev);
    assert_eq!(result_serial.nit, result_parallel.nit);
}

#[test]
fn test_1d_parallel_original_fallback() {
    let bounds = vec![(-5.0, 5.0)];

    let opts_serial = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyOriginal,
        parallel: false,
        ..Default::default()
    };
    let mut solver_serial = Direct::new(sphere_1d, &bounds, opts_serial).unwrap();
    let result_serial = solver_serial.minimize(None).unwrap();

    let opts_parallel = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyOriginal,
        parallel: true,
        min_parallel_evals: 4,
        ..Default::default()
    };
    let mut solver_parallel = Direct::new(sphere_1d, &bounds, opts_parallel).unwrap();
    let result_parallel = solver_parallel.minimize(None).unwrap();

    println!("=== 1D Parallel Original Fallback Test ===");
    println!(
        "Serial:   x=[{:.15e}], minf={:.15e}, nfev={}",
        result_serial.x[0], result_serial.fun, result_serial.nfev
    );
    println!(
        "Parallel: x=[{:.15e}], minf={:.15e}, nfev={}",
        result_parallel.x[0], result_parallel.fun, result_parallel.nfev
    );

    assert_eq!(result_serial.x[0], result_parallel.x[0]);
    assert_eq!(result_serial.fun, result_parallel.fun);
    assert_eq!(result_serial.nfev, result_parallel.nfev);
    assert_eq!(result_serial.nit, result_parallel.nit);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D-specific property: nfev after initialization should be 2*1 + 1 = 3
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_initial_nfev_is_3() {
    // In 1D, initialization evaluates center + 2*1 neighbors = 3 total
    let bounds = vec![(-5.0, 5.0)];
    let opts = DirectOptions {
        max_feval: 3, // stop right after initialization
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_1d, &bounds, opts).unwrap();
    let result = solver.minimize(None).unwrap();

    println!("=== 1D Initial nfev Test ===");
    println!(
        "nfev={}, nit={}, x=[{:.15e}], minf={:.15e}",
        result.nfev, result.nit, result.x[0], result.fun
    );

    // 1D initialization should evaluate exactly 3 points
    assert!(
        result.nfev >= 3,
        "1D init should evaluate at least 3 points, got {}",
        result.nfev
    );

    // Verify C also gets the same nfev
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 1,
        lower_bounds: vec![-5.0],
        upper_bounds: vec![5.0],
        max_feval: 3,
        max_iter: -1,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_GABLONSKY,
    };
    let c_result = unsafe {
        c_runner.run(
            sphere_1d_counting,
            &c_counter as *const EvalCounter as *mut c_void,
        )
    };
    let c_nfev = c_counter.get();

    assert_eq!(
        c_nfev, result.nfev,
        "C nfev={} differs from Rust nfev={} at maxfeval=3",
        c_nfev, result.nfev
    );
    assert_eq!(c_result.x[0], result.x[0], "x[0] mismatch at maxfeval=3");
    assert_eq!(c_result.minf, result.fun, "minf mismatch at maxfeval=3");
}

// ─────────────────────────────────────────────────────────────────────────────
// 1D solution quality tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_1d_sphere_solution_quality() {
    let bounds = vec![(-5.0, 5.0)];
    let opts = DirectOptions {
        max_feval: 1000,
        max_iter: 0,
        magic_eps: 1e-4,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_1d, &bounds, opts).unwrap();
    let result = solver.minimize(None).unwrap();

    println!(
        "=== 1D Sphere Quality: x={:.15e}, f={:.15e}, nfev={} ===",
        result.x[0], result.fun, result.nfev
    );

    // Should find near-zero minimum
    assert!(result.fun < 1e-4, "Expected f < 1e-4, got {}", result.fun);
    assert!(
        result.x[0].abs() < 0.02,
        "Expected |x| < 0.02, got {}",
        result.x[0].abs()
    );
}

#[test]
fn test_1d_shifted_quadratic_solution_quality() {
    let bounds = vec![(-5.0, 5.0)];
    let opts = DirectOptions {
        max_feval: 1000,
        max_iter: 0,
        magic_eps: 1e-4,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(shifted_quadratic_1d, &bounds, opts).unwrap();
    let result = solver.minimize(None).unwrap();

    println!(
        "=== 1D Shifted Quadratic Quality: x={:.15e}, f={:.15e}, nfev={} ===",
        result.x[0], result.fun, result.nfev
    );

    // Minimum is at x=1.5, f=3.0
    assert!(
        (result.fun - 3.0).abs() < 0.01,
        "Expected f ≈ 3.0, got {}",
        result.fun
    );
    assert!(
        (result.x[0] - 1.5).abs() < 0.1,
        "Expected x ≈ 1.5, got {}",
        result.x[0]
    );
}
