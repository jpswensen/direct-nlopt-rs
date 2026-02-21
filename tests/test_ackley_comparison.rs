#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: Ackley function (flat regions) with both algorithm variants.
//!
//! Compares NLOPT C direct_optimize() with Rust Direct implementation for
//! the Ackley function on [-5,5]^2 using both DIRECT_ORIGINAL and
//! DIRECT_GABLONSKY algorithms.
//! Results should be IDENTICAL (not just close).
//!
//! Ackley f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e
//! Global minimum at x = (0, 0, ..., 0), f(x*) = 0.
//! Tests behavior on nearly-flat multiscale landscape.

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

/// Ackley function for C FFI that counts evaluations.
extern "C" fn ackley_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);

    let n_usize = n as usize;
    let mut sum_sq = 0.0;
    let mut sum_cos = 0.0;
    for i in 0..n_usize {
        let xi = unsafe { *x.add(i) };
        sum_sq += xi * xi;
        sum_cos += (2.0 * std::f64::consts::PI * xi).cos();
    }
    let nd = n as f64;
    -20.0 * (-0.2 * (sum_sq / nd).sqrt()).exp()
        - (sum_cos / nd).exp()
        + 20.0
        + std::f64::consts::E
}

/// Rust Ackley function.
fn ackley_rust(x: &[f64]) -> f64 {
    let n = x.len();
    let mut sum_sq = 0.0;
    let mut sum_cos = 0.0;
    for i in 0..n {
        sum_sq += x[i] * x[i];
        sum_cos += (2.0 * std::f64::consts::PI * x[i]).cos();
    }
    let nd = n as f64;
    -20.0 * (-0.2 * (sum_sq / nd).sqrt()).exp()
        - (sum_cos / nd).exp()
        + 20.0
        + std::f64::consts::E
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
        c_runner.run(ackley_counting, &c_counter as *const EvalCounter as *mut c_void)
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
    let mut solver = Direct::new(ackley_rust, &bounds, opts).unwrap();
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

/// Ackley 2D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_ackley_gablonsky_2d_maxfeval5000() {
    run_comparison(
        "Ackley 2D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        5000,   // max_feval
        -1,     // max_iter C (unlimited)
        0,      // max_iter Rust (0 = unlimited)
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Ackley 2D with DIRECT_GABLONSKY, shorter run (maxfeval=1000).
#[test]
fn test_ackley_gablonsky_2d_maxfeval1000() {
    run_comparison(
        "Ackley 2D DIRECT_GABLONSKY (maxfeval=1000, eps=1e-4)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
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

/// Ackley 2D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_ackley_original_2d_maxfeval5000() {
    run_comparison(
        "Ackley 2D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Ackley 2D with DIRECT_ORIGINAL, shorter run (maxfeval=1000).
#[test]
fn test_ackley_original_2d_maxfeval1000() {
    run_comparison(
        "Ackley 2D DIRECT_ORIGINAL (maxfeval=1000, eps=1e-4)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
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

/// Both variants on Ackley 2D — verify each matches its own C counterpart.
#[test]
fn test_ackley_both_variants_2d() {
    run_comparison(
        "Ackley 2D GABLONSKY (both-variants test)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    run_comparison(
        "Ackley 2D ORIGINAL (both-variants test)",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Ackley with different epsilon values (eps=1e-6).
#[test]
fn test_ackley_gablonsky_eps_1e6() {
    run_comparison(
        "Ackley 2D GABLONSKY eps=1e-6",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        5000,
        -1,
        0,
        1e-6,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Ackley ORIGINAL with different epsilon values (eps=1e-6).
#[test]
fn test_ackley_original_eps_1e6() {
    run_comparison(
        "Ackley 2D ORIGINAL eps=1e-6",
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        5000,
        -1,
        0,
        1e-6,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}
