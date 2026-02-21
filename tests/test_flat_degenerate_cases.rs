#![cfg(feature = "nlopt-compare")]

//! Edge case tests for flat objective functions and degenerate configurations.
//!
//! Tests:
//! 1. f(x) = constant (42.0) — verify graceful termination
//! 2. f(x) = +Inf for some regions — verify infeasible handling
//! 3. maxfeval=1 — verify minimal initialization still works
//! 4. maxiter=0 (unlimited) with small maxfeval — verify graceful termination
//!
//! All tests compare NLOPT C vs Rust for identical behavior.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation counter
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

// ─────────────────────────────────────────────────────────────────────────────
// C objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Constant function f(x) = 42.0 (flat objective)
extern "C" fn constant_42_c(
    _n: c_int,
    _x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    if !data.is_null() {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
    }
    42.0
}

/// Constant function f(x) = 0.0 (flat at zero)
extern "C" fn constant_zero_c(
    _n: c_int,
    _x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    if !data.is_null() {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
    }
    0.0
}

/// Sphere function with infeasible region: sets undefined_flag=1 outside radius 3.0
/// NLOPT C uses the undefined_flag mechanism (not return value) for infeasibility.
extern "C" fn sphere_with_inf_region_c(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    if !data.is_null() {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
    }
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    // Signal infeasibility via undefined_flag for points outside radius 3.0
    if sum > 9.0 {
        unsafe { *undefined_flag = 1; }
        f64::MAX
    } else {
        sum
    }
}

/// Sphere function for C FFI
extern "C" fn sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    if !data.is_null() {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
    }
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust objective functions
// ─────────────────────────────────────────────────────────────────────────────

fn constant_42_rust(x: &[f64]) -> f64 {
    let _ = x;
    42.0
}

fn constant_zero_rust(x: &[f64]) -> f64 {
    let _ = x;
    0.0
}

fn sphere_with_inf_region_rust(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().map(|&xi| xi * xi).sum();
    if sum > 9.0 {
        f64::NAN // Rust uses NaN to signal infeasibility
    } else {
        sum
    }
}

fn sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run C and Rust, compare results
// ─────────────────────────────────────────────────────────────────────────────

fn run_comparison(
    name: &str,
    dim: usize,
    c_func: nlopt_ffi::DirectObjectiveFuncC,
    rust_func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    lower: f64,
    upper: f64,
    max_feval: i32,
    max_iter: i32,
    algorithm_c: DirectAlgorithmC,
    algorithm_rust: DirectAlgorithm,
) -> (nlopt_ffi::NloptDirectResult, direct_nlopt::types::DirectResult, usize) {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: vec![lower; dim],
        upper_bounds: vec![upper; dim],
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

    let bounds = vec![(lower, upper); dim];
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
        "=== {} ({:?}, {}D) ===",
        name, algorithm_c, dim
    );
    println!(
        "C:    x={:?}, minf={:.15e}, nfev={}",
        c_result.x, c_result.minf, c_nfev
    );
    println!(
        "Rust: x={:?}, minf={:.15e}, nfev={}",
        rust_result.x, rust_result.fun, rust_result.nfev
    );

    (c_result, rust_result, c_nfev)
}

fn assert_exact_match(
    name: &str,
    c_result: &nlopt_ffi::NloptDirectResult,
    rust_result: &direct_nlopt::types::DirectResult,
    c_nfev: usize,
) {
    for (i, (&cx, &rx)) in c_result.x.iter().zip(rust_result.x.iter()).enumerate() {
        assert_eq!(
            cx, rx,
            "{} x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            name, i, cx, rx
        );
    }
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

// ═════════════════════════════════════════════════════════════════════════════
// 1. Constant objective: f(x) = 42.0
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn test_constant_42_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_42",
        2,
        constant_42_c,
        constant_42_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("constant_42_2d_gablonsky", &c_result, &rust_result, c_nfev);
    // All evaluations should return 42.0
    assert_eq!(rust_result.fun, 42.0, "Expected minf=42.0 for constant function");
}

#[test]
fn test_constant_42_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_42",
        2,
        constant_42_c,
        constant_42_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("constant_42_2d_original", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}

#[test]
fn test_constant_42_3d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_42",
        3,
        constant_42_c,
        constant_42_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("constant_42_3d_gablonsky", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}

#[test]
fn test_constant_zero_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_zero",
        2,
        constant_zero_c,
        constant_zero_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("constant_zero_2d_gablonsky", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 0.0, "Expected minf=0.0 for constant-zero function");
}

#[test]
fn test_constant_zero_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_zero",
        2,
        constant_zero_c,
        constant_zero_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("constant_zero_2d_original", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. f(x) = +Inf (HUGE_VAL) for some regions — infeasible handling
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sphere_inf_region_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "sphere_inf_region",
        2,
        sphere_with_inf_region_c,
        sphere_with_inf_region_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("sphere_inf_2d_gablonsky", &c_result, &rust_result, c_nfev);
    // Should still find minimum near 0 within feasible region
    assert!(
        rust_result.fun < 1.0,
        "Expected feasible minimum < 1.0, got {}",
        rust_result.fun
    );
}

#[test]
fn test_sphere_inf_region_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "sphere_inf_region",
        2,
        sphere_with_inf_region_c,
        sphere_with_inf_region_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("sphere_inf_2d_original", &c_result, &rust_result, c_nfev);
    assert!(
        rust_result.fun < 1.0,
        "Expected feasible minimum < 1.0, got {}",
        rust_result.fun
    );
}

#[test]
fn test_sphere_inf_region_3d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "sphere_inf_region",
        3,
        sphere_with_inf_region_c,
        sphere_with_inf_region_rust,
        -5.0,
        5.0,
        1000,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("sphere_inf_3d_gablonsky", &c_result, &rust_result, c_nfev);
    assert!(
        rust_result.fun < 1.0,
        "Expected feasible minimum < 1.0, got {}",
        rust_result.fun
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. maxfeval=1 — minimal initialization
// ═════════════════════════════════════════════════════════════════════════════
// DIRECT initialization evaluates 2*n+1 points, so maxfeval=1 will trigger
// immediate termination after init. Both C and Rust should handle this.

#[test]
fn test_maxfeval_1_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_1",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        1, // maxfeval = 1
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_1_2d_gablonsky", &c_result, &rust_result, c_nfev);
    // With maxfeval=1, initialization still runs (2*2+1=5 evals)
    // and then terminates with MAXFEVAL_EXCEEDED
    println!("maxfeval=1: nfev={}", rust_result.nfev);
}

#[test]
fn test_maxfeval_1_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_1",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        1,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("maxfeval_1_2d_original", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxfeval_1_3d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_1",
        3,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        1,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_1_3d_gablonsky", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxfeval_1_1d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_1",
        1,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        1,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_1_1d_gablonsky", &c_result, &rust_result, c_nfev);
}

// Also test maxfeval=2*n+1 (exactly enough for initialization, no iterations)
#[test]
fn test_maxfeval_exact_init_2d_gablonsky() {
    // 2D: init needs 2*2+1=5 evals
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_exact_init",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        5, // exactly 2*2+1
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_exact_init_2d_gablonsky", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxfeval_exact_init_3d_gablonsky() {
    // 3D: init needs 2*3+1=7 evals
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_exact_init",
        3,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        7, // exactly 2*3+1
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_exact_init_3d_gablonsky", &c_result, &rust_result, c_nfev);
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. maxiter=0 — unlimited iterations (terminates via maxfeval)
// ═════════════════════════════════════════════════════════════════════════════
// In both NLOPT C and Rust, maxiter=0 (or ≤0 in C) means "no iteration limit".
// The algorithm runs until another stopping criterion (e.g., maxfeval) is reached.

#[test]
fn test_maxiter_0_terminates_via_maxfeval_2d_gablonsky() {
    // maxiter=0 (unlimited) + maxfeval=200 → should terminate via maxfeval
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_0_sphere",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        200,
        0, // maxiter=0 → unlimited in both C and Rust
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxiter_0_2d_gablonsky", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxiter_0_terminates_via_maxfeval_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_0_sphere",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        200,
        0,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("maxiter_0_2d_original", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxiter_0_constant_2d_gablonsky() {
    // Constant function with maxiter=0 (unlimited) + maxfeval=500
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_0_constant",
        2,
        constant_42_c,
        constant_42_rust,
        -5.0,
        5.0,
        500,
        0,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxiter_0_constant_2d_gablonsky", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}

#[test]
fn test_maxiter_0_3d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_0_sphere",
        3,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        300,
        0,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxiter_0_3d_gablonsky", &c_result, &rust_result, c_nfev);
}

// ═════════════════════════════════════════════════════════════════════════════
// Additional degenerate cases
// ═════════════════════════════════════════════════════════════════════════════

/// Test with maxfeval=2 (less than init needs)
#[test]
fn test_maxfeval_2_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxfeval_2",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        2,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxfeval_2_2d_gablonsky", &c_result, &rust_result, c_nfev);
}

/// Test constant function with maxfeval=1
#[test]
fn test_constant_maxfeval_1_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_maxfeval_1",
        2,
        constant_42_c,
        constant_42_rust,
        -5.0,
        5.0,
        1,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("constant_maxfeval_1_2d_gablonsky", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}

/// Test with maxiter=1 (single iteration after init)
#[test]
fn test_maxiter_1_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_1",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        10000,
        1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("maxiter_1_2d_gablonsky", &c_result, &rust_result, c_nfev);
}

#[test]
fn test_maxiter_1_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "maxiter_1",
        2,
        sphere_c,
        sphere_rust,
        -5.0,
        5.0,
        10000,
        1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("maxiter_1_2d_original", &c_result, &rust_result, c_nfev);
}

/// Test infeasible region + constant function in feasible region
extern "C" fn constant_with_inf_c(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    if !data.is_null() {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
    }
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    if sum > 9.0 {
        unsafe { *undefined_flag = 1; }
        f64::MAX
    } else {
        42.0 // flat within feasible region
    }
}

fn constant_with_inf_rust(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().map(|&xi| xi * xi).sum();
    if sum > 9.0 {
        f64::NAN // Rust uses NaN to signal infeasibility
    } else {
        42.0
    }
}

#[test]
fn test_constant_with_inf_region_2d_gablonsky() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_with_inf",
        2,
        constant_with_inf_c,
        constant_with_inf_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    assert_exact_match("constant_with_inf_2d_gablonsky", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}

#[test]
fn test_constant_with_inf_region_2d_original() {
    let (c_result, rust_result, c_nfev) = run_comparison(
        "constant_with_inf",
        2,
        constant_with_inf_c,
        constant_with_inf_rust,
        -5.0,
        5.0,
        500,
        -1,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
    assert_exact_match("constant_with_inf_2d_original", &c_result, &rust_result, c_nfev);
    assert_eq!(rust_result.fun, 42.0);
}
