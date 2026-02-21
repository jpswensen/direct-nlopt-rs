#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: volume_reltol and sigma_reltol termination.
//!
//! Tests that NLOPT C and Rust stop at the SAME iteration with the same
//! return code (VOLTOL or SIGMATOL) when volume_reltol or sigma_reltol
//! are set on the sphere function.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, DirectReturnCodeC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL,
    DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

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

/// Sphere function for C FFI: f(x) = sum(x_i^2).
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

/// Rust sphere function.
fn sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run volume_reltol or sigma_reltol comparison
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_tol_comparison(
    bounds: &[(f64, f64)],
    volume_reltol: f64,
    sigma_reltol: f64,
    max_feval: usize,
    c_alg: DirectAlgorithmC,
    rust_alg: DirectAlgorithm,
    expected_c_code: DirectReturnCodeC,
    expected_rust_code: DirectReturnCode,
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
        volume_reltol,
        sigma_reltol,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: c_alg,
    };
    let c_result = unsafe {
        c_runner.run(
            sphere_c,
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
        volume_reltol,
        sigma_reltol,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: rust_alg,
        parallel: false,
        ..Default::default()
    };
    let bounds_vec: Vec<(f64, f64)> = bounds.to_vec();
    let mut solver = Direct::new(sphere_rust, &bounds_vec, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    // ── Print comparison ──
    println!("=== {} ===", label);
    println!(
        "  volume_reltol={:.15e}, sigma_reltol={:.15e}",
        volume_reltol, sigma_reltol
    );
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    // ── Verify return codes ──
    assert_eq!(
        c_result.return_code, expected_c_code,
        "[{}] C should return {:?}, got {:?}",
        label, expected_c_code, c_result.return_code
    );
    assert_eq!(
        rust_result.return_code, expected_rust_code,
        "[{}] Rust should return {:?}, got {:?}",
        label, expected_rust_code, rust_result.return_code
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
// Tests: volume_reltol termination
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere [-5,5]^2 with volume_reltol=1e-8, DIRECT_GABLONSKY.
/// The algorithm should stop when the smallest hyperrectangle volume
/// (as a fraction of the original) drops below 1e-8.
#[test]
fn test_voltol_sphere_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        1e-8,  // volume_reltol
        -1.0,  // sigma_reltol disabled
        100000, // generous maxfeval
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_VOLTOL,
        DirectReturnCode::VolTol,
        "Sphere voltol=1e-8 Gablonsky",
    );
}

/// Sphere [-5,5]^2 with volume_reltol=1e-8, DIRECT_ORIGINAL.
#[test]
fn test_voltol_sphere_original() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        1e-8,  // volume_reltol
        -1.0,  // sigma_reltol disabled
        100000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        DirectReturnCodeC::DIRECT_VOLTOL,
        DirectReturnCode::VolTol,
        "Sphere voltol=1e-8 Original",
    );
}

/// Sphere [-5,5]^2 with a looser volume_reltol=1e-4.
/// Should terminate earlier than 1e-8.
#[test]
fn test_voltol_sphere_loose_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        1e-4,  // volume_reltol (looser)
        -1.0,  // sigma_reltol disabled
        100000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_VOLTOL,
        DirectReturnCode::VolTol,
        "Sphere voltol=1e-4 Gablonsky",
    );
}

/// 3D sphere with volume_reltol=1e-8, DIRECT_GABLONSKY.
#[test]
fn test_voltol_sphere_3d_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        1e-8,  // volume_reltol
        -1.0,  // sigma_reltol disabled
        200000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_VOLTOL,
        DirectReturnCode::VolTol,
        "Sphere 3D voltol=1e-8 Gablonsky",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: sigma_reltol termination
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere [-5,5]^2 with sigma_reltol=1e-3, DIRECT_GABLONSKY.
/// The algorithm should stop when the measure (sigma) of the best
/// rectangle drops below sigma_reltol.
#[test]
fn test_sigmatol_sphere_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,   // volume_reltol disabled
        1e-3,  // sigma_reltol
        100000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_SIGMATOL,
        DirectReturnCode::SigmaTol,
        "Sphere sigmatol=1e-3 Gablonsky",
    );
}

/// Sphere [-5,5]^2 with sigma_reltol=1e-3, DIRECT_ORIGINAL.
#[test]
fn test_sigmatol_sphere_original() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,   // volume_reltol disabled
        1e-3,  // sigma_reltol
        100000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        DirectReturnCodeC::DIRECT_SIGMATOL,
        DirectReturnCode::SigmaTol,
        "Sphere sigmatol=1e-3 Original",
    );
}

/// Sphere [-5,5]^2 with a tighter sigma_reltol=1e-5.
/// Should terminate later than 1e-3.
#[test]
fn test_sigmatol_sphere_tight_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0)],
        0.0,   // volume_reltol disabled
        1e-5,  // sigma_reltol (tighter)
        100000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_SIGMATOL,
        DirectReturnCode::SigmaTol,
        "Sphere sigmatol=1e-5 Gablonsky",
    );
}

/// 3D sphere with sigma_reltol=1e-3, DIRECT_GABLONSKY.
#[test]
fn test_sigmatol_sphere_3d_gablonsky() {
    run_tol_comparison(
        &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        0.0,   // volume_reltol disabled
        1e-3,  // sigma_reltol
        200000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        DirectReturnCodeC::DIRECT_SIGMATOL,
        DirectReturnCode::SigmaTol,
        "Sphere 3D sigmatol=1e-3 Gablonsky",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: voltol terminates before maxfeval
// ─────────────────────────────────────────────────────────────────────────────

/// With volume_reltol set, the optimizer should terminate before maxfeval.
#[test]
fn test_voltol_terminates_before_maxfeval() {
    let bounds = vec![(-5.0, 5.0); 2];

    // Run without any tolerance → maxfeval termination
    let opts_none = DirectOptions {
        max_feval: 100000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: -1.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts_none).unwrap();
    let result_none = solver.minimize(None).unwrap();

    // Run with volume_reltol → should stop earlier
    let opts_vol = DirectOptions {
        max_feval: 100000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 1e-8,
        sigma_reltol: -1.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts_vol).unwrap();
    let result_vol = solver.minimize(None).unwrap();

    println!(
        "No tolerance: nfev={}, minf={:.15e}, code={:?}",
        result_none.nfev, result_none.fun, result_none.return_code
    );
    println!(
        "voltol=1e-8:  nfev={}, minf={:.15e}, code={:?}",
        result_vol.nfev, result_vol.fun, result_vol.return_code
    );

    assert_eq!(
        result_vol.return_code,
        DirectReturnCode::VolTol,
        "Should return VolTol"
    );
    assert!(
        result_vol.nfev < result_none.nfev,
        "voltol should use fewer evals: with={}, without={}",
        result_vol.nfev,
        result_none.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: sigmatol terminates before maxfeval
// ─────────────────────────────────────────────────────────────────────────────

/// With sigma_reltol set, the optimizer should terminate before maxfeval.
#[test]
fn test_sigmatol_terminates_before_maxfeval() {
    let bounds = vec![(-5.0, 5.0); 2];

    // Run without any tolerance → maxfeval termination
    let opts_none = DirectOptions {
        max_feval: 100000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: -1.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts_none).unwrap();
    let result_none = solver.minimize(None).unwrap();

    // Run with sigma_reltol → should stop earlier
    let opts_sigma = DirectOptions {
        max_feval: 100000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 1e-3,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts_sigma).unwrap();
    let result_sigma = solver.minimize(None).unwrap();

    println!(
        "No tolerance:   nfev={}, minf={:.15e}, code={:?}",
        result_none.nfev, result_none.fun, result_none.return_code
    );
    println!(
        "sigmatol=1e-3:  nfev={}, minf={:.15e}, code={:?}",
        result_sigma.nfev, result_sigma.fun, result_sigma.return_code
    );

    assert_eq!(
        result_sigma.return_code,
        DirectReturnCode::SigmaTol,
        "Should return SigmaTol"
    );
    assert!(
        result_sigma.nfev < result_none.nfev,
        "sigmatol should use fewer evals: with={}, without={}",
        result_sigma.nfev,
        result_none.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: looser voltol terminates earlier than tighter voltol
// ─────────────────────────────────────────────────────────────────────────────

/// Looser volume_reltol should terminate with fewer function evaluations.
#[test]
fn test_voltol_ordering() {
    let bounds = vec![(-5.0, 5.0); 2];

    let run = |vol_tol: f64| -> (usize, f64) {
        let opts = DirectOptions {
            max_feval: 100000,
            max_iter: 0,
            magic_eps: 1e-4,
            magic_eps_abs: 0.0,
            volume_reltol: vol_tol,
            sigma_reltol: -1.0,
            fglobal: f64::NEG_INFINITY,
            fglobal_reltol: 0.0,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            parallel: false,
            ..Default::default()
        };
        let mut solver = Direct::new(sphere_rust, &bounds, opts).unwrap();
        let result = solver.minimize(None).unwrap();
        assert_eq!(result.return_code, DirectReturnCode::VolTol);
        (result.nfev, result.fun)
    };

    let (nfev_loose, _) = run(1e-4);
    let (nfev_tight, _) = run(1e-8);

    println!("voltol=1e-4: nfev={}", nfev_loose);
    println!("voltol=1e-8: nfev={}", nfev_tight);

    assert!(
        nfev_loose < nfev_tight,
        "Looser voltol should terminate earlier: loose={}, tight={}",
        nfev_loose,
        nfev_tight
    );
}
