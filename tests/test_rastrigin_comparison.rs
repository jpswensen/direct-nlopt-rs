#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: Rastrigin function (multimodal) with both algorithm variants.
//!
//! Compares NLOPT C direct_optimize() with Rust Direct implementation for
//! the Rastrigin function on [-5.12,5.12]^n using both DIRECT_ORIGINAL and
//! DIRECT_GABLONSKY algorithms with maxfeval=5000.
//! Results should be IDENTICAL (not just close).
//!
//! Rastrigin f(x) = 10*n + sum_{i=0}^{n-1} [x_i^2 - 10*cos(2*pi*x_i)]
//! Global minimum at x = (0, 0, ..., 0), f(x*) = 0.
//! Tests global exploration with many local minima.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL};

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
        Self { count: AtomicUsize::new(0) }
    }
    fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Rastrigin function for C FFI that counts evaluations.
extern "C" fn rastrigin_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);

    let n_usize = n as usize;
    let mut sum = 10.0 * n as f64;
    for i in 0..n_usize {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

/// Rust Rastrigin function.
fn rastrigin_rust(x: &[f64]) -> f64 {
    let n = x.len();
    let mut sum = 10.0 * n as f64;
    for i in 0..n {
        sum += x[i] * x[i] - 10.0 * (2.0 * std::f64::consts::PI * x[i]).cos();
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper to run both C and Rust, compare results
// ─────────────────────────────────────────────────────────────────────────────

fn run_comparison(
    label: &str,
    dim: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    max_feval: i32,
    max_iter_c: i32,
    max_iter_rust: usize,
    magic_eps: f64,
    c_algo: DirectAlgorithmC,
    rust_algo: DirectAlgorithm,
) {
    // ── Run NLOPT C ──
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: lower.clone(),
        upper_bounds: upper.clone(),
        max_feval,
        max_iter: max_iter_c,
        magic_eps,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: c_algo,
    };
    let c_result = unsafe {
        c_runner.run(rastrigin_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    // ── Run Rust ──
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let opts = DirectOptions {
        max_feval: max_feval as usize,
        max_iter: max_iter_rust,
        magic_eps,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: rust_algo,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(rastrigin_rust, &bounds, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    // ── Print comparison ──
    println!("=== {} ===", label);
    println!("C:    x = {:?}", c_result.x);
    println!("Rust: x = {:?}", rust_result.x);
    println!("C:    minf = {:.15e}", c_result.minf);
    println!("Rust: minf = {:.15e}", rust_result.fun);
    println!("C:    nfev = {}", c_nfev);
    println!("Rust: nfev = {}", rust_result.nfev);
    println!("Rust: nit  = {}", rust_result.nit);
    println!("C:    code = {:?}", c_result.return_code);
    println!("Rust: code = {:?}", rust_result.return_code);

    // ── Verify IDENTICAL results ──
    for i in 0..dim {
        assert_eq!(
            c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, c_result.x[i], rust_result.x[i]
        );
    }
    assert_eq!(
        c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        c_result.minf, rust_result.fun
    );
    assert_eq!(
        c_nfev, rust_result.nfev,
        "nfev mismatch: C={}, Rust={}",
        c_nfev, rust_result.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DIRECT_GABLONSKY (DIRECT-L) tests — 2D
// ─────────────────────────────────────────────────────────────────────────────

/// Rastrigin 2D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_rastrigin_gablonsky_2d_maxfeval5000() {
    run_comparison(
        "Rastrigin 2D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        5000,   // max_feval
        -1,     // max_iter C (unlimited)
        0,      // max_iter Rust (0 = unlimited)
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Rastrigin 2D with DIRECT_GABLONSKY, shorter run (maxfeval=1000).
#[test]
fn test_rastrigin_gablonsky_2d_maxfeval1000() {
    run_comparison(
        "Rastrigin 2D DIRECT_GABLONSKY (maxfeval=1000, eps=1e-4)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        1000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DIRECT_GABLONSKY (DIRECT-L) tests — 3D
// ─────────────────────────────────────────────────────────────────────────────

/// Rastrigin 3D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_rastrigin_gablonsky_3d_maxfeval5000() {
    run_comparison(
        "Rastrigin 3D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Rastrigin 3D with DIRECT_GABLONSKY, shorter run (maxfeval=1000).
#[test]
fn test_rastrigin_gablonsky_3d_maxfeval1000() {
    run_comparison(
        "Rastrigin 3D DIRECT_GABLONSKY (maxfeval=1000, eps=1e-4)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        1000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DIRECT_ORIGINAL (Jones) tests — 2D
// ─────────────────────────────────────────────────────────────────────────────

/// Rastrigin 2D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_rastrigin_original_2d_maxfeval5000() {
    run_comparison(
        "Rastrigin 2D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Rastrigin 2D with DIRECT_ORIGINAL, shorter run (maxfeval=1000).
#[test]
fn test_rastrigin_original_2d_maxfeval1000() {
    run_comparison(
        "Rastrigin 2D DIRECT_ORIGINAL (maxfeval=1000, eps=1e-4)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        1000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DIRECT_ORIGINAL (Jones) tests — 3D
// ─────────────────────────────────────────────────────────────────────────────

/// Rastrigin 3D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_rastrigin_original_3d_maxfeval5000() {
    run_comparison(
        "Rastrigin 3D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Rastrigin 3D with DIRECT_ORIGINAL, shorter run (maxfeval=1000).
#[test]
fn test_rastrigin_original_3d_maxfeval1000() {
    run_comparison(
        "Rastrigin 3D DIRECT_ORIGINAL (maxfeval=1000, eps=1e-4)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        1000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-variant consistency tests
// ─────────────────────────────────────────────────────────────────────────────

/// Both variants on Rastrigin 2D — verify each matches its own C counterpart.
#[test]
fn test_rastrigin_both_variants_2d() {
    run_comparison(
        "Rastrigin 2D GABLONSKY (both-variants test)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    run_comparison(
        "Rastrigin 2D ORIGINAL (both-variants test)",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Both variants on Rastrigin 3D — verify each matches its own C counterpart.
#[test]
fn test_rastrigin_both_variants_3d() {
    run_comparison(
        "Rastrigin 3D GABLONSKY (both-variants test)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    run_comparison(
        "Rastrigin 3D ORIGINAL (both-variants test)",
        3,
        vec![-5.12, -5.12, -5.12],
        vec![5.12, 5.12, 5.12],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Rastrigin with different epsilon values (eps=1e-6).
#[test]
fn test_rastrigin_gablonsky_eps_1e6() {
    run_comparison(
        "Rastrigin 2D GABLONSKY eps=1e-6",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        5000,
        -1,
        0,
        1e-6,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Rastrigin ORIGINAL with different epsilon values (eps=1e-6).
#[test]
fn test_rastrigin_original_eps_1e6() {
    run_comparison(
        "Rastrigin 2D ORIGINAL eps=1e-6",
        2,
        vec![-5.12, -5.12],
        vec![5.12, 5.12],
        5000,
        -1,
        0,
        1e-6,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}
