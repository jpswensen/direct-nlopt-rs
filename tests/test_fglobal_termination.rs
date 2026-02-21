#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: fglobal termination.
//!
//! Tests that NLOPT C and Rust terminate at the SAME iteration with
//! GLOBAL_FOUND return code when fglobal and fglobal_reltol are set.
//! Also tests the divfactor handling (fglobal=0 vs fglobal≠0).

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{DirectAlgorithmC, DirectReturnCodeC, NloptDirectRunner};

use direct_nlopt::direct::Direct;
use direct_nlopt::error::DirectReturnCode;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Objective functions
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

/// Sphere function for C FFI: f(x) = sum(x_i^2), global minimum = 0 at origin.
extern "C" fn sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

/// Shifted sphere for C FFI: f(x) = sum((x_i - 1)^2) + 5.0
/// Global minimum = 5.0 at x = (1, 1, ..., 1).
/// This tests divfactor = |fglobal| = 5.0 (fglobal ≠ 0 case).
extern "C" fn shifted_sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    let n = n as usize;
    let mut sum = 5.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += (xi - 1.0) * (xi - 1.0);
    }
    sum
}

/// Rust sphere function.
fn sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Rust shifted sphere: f(x) = sum((x_i - 1)^2) + 5.0
fn shifted_sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| (xi - 1.0).powi(2)).sum::<f64>() + 5.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run fglobal comparison
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_fglobal_comparison(
    c_func: nlopt_ffi::DirectObjectiveFuncC,
    rust_func: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    fglobal: f64,
    fglobal_reltol: f64,
    max_feval: usize,
    c_alg: DirectAlgorithmC,
    rust_alg: DirectAlgorithm,
    label: &str,
) {
    let dim = bounds.len();

    // ── Run NLOPT C ──
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: bounds.iter().map(|b| b.0).collect(),
        upper_bounds: bounds.iter().map(|b| b.1).collect(),
        max_feval: max_feval as i32,
        max_iter: -1, // unlimited iterations
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal,
        fglobal_reltol,
        algorithm: c_alg,
    };
    let c_result = unsafe {
        c_runner.run(
            c_func,
            &c_counter as *const EvalCounter as *mut c_void,
        )
    };
    let c_nfev = c_counter.get();

    // ── Run Rust ──
    let opts = DirectOptions {
        max_feval,
        max_iter: 0, // 0 = unlimited in Rust
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal,
        fglobal_reltol,
        algorithm: rust_alg,
        parallel: false,
        ..Default::default()
    };
    let bounds_vec: Vec<(f64, f64)> = bounds.to_vec();
    let mut solver = Direct::new(rust_func, &bounds_vec, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    // ── Print comparison ──
    println!("=== {} ===", label);
    println!(
        "  fglobal={:.15e}, fglobal_reltol={:.15e}",
        fglobal, fglobal_reltol
    );
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    // ── Verify return code ──
    assert_eq!(
        c_result.return_code,
        DirectReturnCodeC::DIRECT_GLOBAL_FOUND,
        "[{}] C should return GLOBAL_FOUND, got {:?}",
        label,
        c_result.return_code
    );
    assert_eq!(
        rust_result.return_code,
        DirectReturnCode::GlobalFound,
        "[{}] Rust should return GlobalFound, got {:?}",
        label,
        rust_result.return_code
    );

    // ── Verify IDENTICAL results ──
    assert_eq!(
        c_nfev, rust_result.nfev,
        "[{}] nfev mismatch: C={}, Rust={}",
        label, c_nfev, rust_result.nfev
    );
    assert_eq!(
        c_result.minf, rust_result.fun,
        "[{}] minf mismatch: C={:.15e}, Rust={:.15e}",
        label, c_result.minf, rust_result.fun
    );
    for i in 0..dim {
        assert_eq!(
            c_result.x[i], rust_result.x[i],
            "[{}] x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            label, i, c_result.x[i], rust_result.x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: sphere function (fglobal=0.0, divfactor=1.0)
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere with fglobal=0.0, fglobal_reltol=1e-4 using DIRECT_GABLONSKY.
/// NLOPT C formula: (minf - 0.0) * 100 / 1.0 <= 1e-4 * 100 = 0.01
/// i.e. minf <= 0.0001
#[test]
fn test_fglobal_sphere_gablonsky() {
    run_fglobal_comparison(
        sphere_c,
        sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,     // fglobal
        1e-4,    // fglobal_reltol
        10000,   // generous maxfeval budget
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere fglobal=0.0 Gablonsky",
    );
}

/// Sphere with fglobal=0.0, fglobal_reltol=1e-4 using DIRECT_ORIGINAL.
#[test]
fn test_fglobal_sphere_original() {
    run_fglobal_comparison(
        sphere_c,
        sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,     // fglobal
        1e-4,    // fglobal_reltol
        10000,   // generous maxfeval budget
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere fglobal=0.0 Original",
    );
}

/// Sphere with a tighter tolerance (fglobal_reltol=1e-2) — should terminate earlier.
#[test]
fn test_fglobal_sphere_loose_tolerance_gablonsky() {
    run_fglobal_comparison(
        sphere_c,
        sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,     // fglobal
        1e-2,    // fglobal_reltol (loose)
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere fglobal=0.0 reltol=1e-2 Gablonsky",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: shifted sphere (fglobal=5.0, divfactor=5.0)
// Tests the fglobal≠0 case where divfactor = |fglobal|
// ─────────────────────────────────────────────────────────────────────────────

/// Shifted sphere: f(x) = sum((x_i - 1)^2) + 5.0, fglobal=5.0.
/// NLOPT C formula: (minf - 5.0) * 100 / 5.0 <= 1e-4 * 100 = 0.01
/// i.e. (minf - 5.0) / 5.0 <= 0.0001
/// i.e. minf <= 5.0005
#[test]
fn test_fglobal_shifted_sphere_gablonsky() {
    run_fglobal_comparison(
        shifted_sphere_c,
        shifted_sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        5.0,     // fglobal (non-zero)
        1e-4,    // fglobal_reltol
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Shifted sphere fglobal=5.0 Gablonsky",
    );
}

/// Shifted sphere with DIRECT_ORIGINAL.
#[test]
fn test_fglobal_shifted_sphere_original() {
    run_fglobal_comparison(
        shifted_sphere_c,
        shifted_sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        5.0,     // fglobal (non-zero)
        1e-4,    // fglobal_reltol
        10000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Shifted sphere fglobal=5.0 Original",
    );
}

/// Shifted sphere with larger tolerance — tests divfactor scaling.
#[test]
fn test_fglobal_shifted_sphere_loose_tolerance_gablonsky() {
    run_fglobal_comparison(
        shifted_sphere_c,
        shifted_sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0)],
        5.0,     // fglobal (non-zero)
        1e-2,    // fglobal_reltol (loose)
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Shifted sphere fglobal=5.0 reltol=1e-2 Gablonsky",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: verify fglobal termination happens BEFORE maxfeval
// ─────────────────────────────────────────────────────────────────────────────

/// With fglobal set, the optimizer should terminate well before maxfeval.
#[test]
fn test_fglobal_terminates_before_maxfeval() {
    // Run without fglobal (uses maxfeval as termination)
    let bounds = vec![(-5.0, 5.0); 2];
    let opts_no_fglobal = DirectOptions {
        max_feval: 10000,
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
    let mut solver = Direct::new(sphere_rust, &bounds, opts_no_fglobal).unwrap();
    let result_no_fglobal = solver.minimize(None).unwrap();

    // Run with fglobal=0.0 — should stop much earlier
    let opts_with_fglobal = DirectOptions {
        max_feval: 10000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: 0.0,
        fglobal_reltol: 1e-4,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts_with_fglobal).unwrap();
    let result_with_fglobal = solver.minimize(None).unwrap();

    println!(
        "Without fglobal: nfev={}, minf={:.15e}, code={:?}",
        result_no_fglobal.nfev, result_no_fglobal.fun, result_no_fglobal.return_code
    );
    println!(
        "With fglobal:    nfev={}, minf={:.15e}, code={:?}",
        result_with_fglobal.nfev, result_with_fglobal.fun, result_with_fglobal.return_code
    );

    assert_eq!(
        result_with_fglobal.return_code,
        DirectReturnCode::GlobalFound,
        "With fglobal set, should return GlobalFound"
    );
    assert!(
        result_with_fglobal.nfev < result_no_fglobal.nfev,
        "fglobal termination should use fewer evals: with={}, without={}",
        result_with_fglobal.nfev,
        result_no_fglobal.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: 3D sphere to verify divfactor=0 handling in higher dimensions
// ─────────────────────────────────────────────────────────────────────────────

/// 3D sphere with fglobal=0.0 — tests divfactor=1 in higher dimensions.
extern "C" fn sphere_3d_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

#[test]
fn test_fglobal_sphere_3d_gablonsky() {
    run_fglobal_comparison(
        sphere_3d_c,
        sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        0.0,     // fglobal
        1e-4,    // fglobal_reltol
        20000,   // larger budget for 3D
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 3D fglobal=0.0 Gablonsky",
    );
}

#[test]
fn test_fglobal_sphere_3d_original() {
    run_fglobal_comparison(
        sphere_3d_c,
        sphere_rust,
        &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        0.0,     // fglobal
        1e-4,    // fglobal_reltol
        20000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere 3D fglobal=0.0 Original",
    );
}
