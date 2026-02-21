#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: Styblinski-Tang function (multiple minima) with both algorithm variants.
//!
//! Compares NLOPT C direct_optimize() with Rust Direct implementation for
//! the Styblinski-Tang function on [-5,5]^n (n=2,3,5) using both DIRECT_ORIGINAL
//! and DIRECT_GABLONSKY algorithms.
//! Results should be IDENTICAL (not just close).
//!
//! Styblinski-Tang f(x) = 0.5 * sum_{i=0}^{n-1} (x_i^4 - 16*x_i^2 + 5*x_i)
//! Global minimum at x_i ≈ -2.903534, f(x*) ≈ -39.16617 * n.

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

/// Styblinski-Tang function for C FFI that counts evaluations.
extern "C" fn styblinski_tang_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);

    let n_usize = n as usize;
    let mut sum = 0.0;
    for i in 0..n_usize {
        let xi = unsafe { *x.add(i) };
        let xi2 = xi * xi;
        sum += xi2 * xi2 - 16.0 * xi2 + 5.0 * xi;
    }
    0.5 * sum
}

/// Rust Styblinski-Tang function.
fn styblinski_tang_rust(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        let xi = x[i];
        let xi2 = xi * xi;
        sum += xi2 * xi2 - 16.0 * xi2 + 5.0 * xi;
    }
    0.5 * sum
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
        c_runner.run(styblinski_tang_counting, &c_counter as *const EvalCounter as *mut c_void)
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
    let mut solver = Direct::new(styblinski_tang_rust, &bounds, opts).unwrap();
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
// DIRECT_GABLONSKY (DIRECT-L) tests — 2D, 3D, 5D
// ─────────────────────────────────────────────────────────────────────────────

/// Styblinski-Tang 2D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_styblinski_tang_gablonsky_2d() {
    run_comparison(
        "Styblinski-Tang 2D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.0; 2],
        vec![5.0; 2],
        5000,   // max_feval
        -1,     // max_iter C (unlimited)
        0,      // max_iter Rust (0 = unlimited)
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Styblinski-Tang 3D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_styblinski_tang_gablonsky_3d() {
    run_comparison(
        "Styblinski-Tang 3D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        3,
        vec![-5.0; 3],
        vec![5.0; 3],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

/// Styblinski-Tang 5D with DIRECT_GABLONSKY, maxfeval=5000.
#[test]
fn test_styblinski_tang_gablonsky_5d() {
    run_comparison(
        "Styblinski-Tang 5D DIRECT_GABLONSKY (maxfeval=5000, eps=1e-4)",
        5,
        vec![-5.0; 5],
        vec![5.0; 5],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DIRECT_ORIGINAL (Jones) tests — 2D, 3D, 5D
// ─────────────────────────────────────────────────────────────────────────────

/// Styblinski-Tang 2D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_styblinski_tang_original_2d() {
    run_comparison(
        "Styblinski-Tang 2D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        2,
        vec![-5.0; 2],
        vec![5.0; 2],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Styblinski-Tang 3D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_styblinski_tang_original_3d() {
    run_comparison(
        "Styblinski-Tang 3D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        3,
        vec![-5.0; 3],
        vec![5.0; 3],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Styblinski-Tang 5D with DIRECT_ORIGINAL, maxfeval=5000.
#[test]
fn test_styblinski_tang_original_5d() {
    run_comparison(
        "Styblinski-Tang 5D DIRECT_ORIGINAL (maxfeval=5000, eps=1e-4)",
        5,
        vec![-5.0; 5],
        vec![5.0; 5],
        5000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-variant and cross-dimension consistency tests
// ─────────────────────────────────────────────────────────────────────────────

/// Both variants on Styblinski-Tang 2D — verify each matches its own C counterpart.
#[test]
fn test_styblinski_tang_both_variants_2d() {
    run_comparison(
        "Styblinski-Tang 2D GABLONSKY (both-variants test)",
        2,
        vec![-5.0; 2],
        vec![5.0; 2],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    run_comparison(
        "Styblinski-Tang 2D ORIGINAL (both-variants test)",
        2,
        vec![-5.0; 2],
        vec![5.0; 2],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}

/// Both variants on Styblinski-Tang 3D — verify each matches its own C counterpart.
#[test]
fn test_styblinski_tang_both_variants_3d() {
    run_comparison(
        "Styblinski-Tang 3D GABLONSKY (both-variants test)",
        3,
        vec![-5.0; 3],
        vec![5.0; 3],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    run_comparison(
        "Styblinski-Tang 3D ORIGINAL (both-variants test)",
        3,
        vec![-5.0; 3],
        vec![5.0; 3],
        3000,
        -1,
        0,
        1e-4,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );
}
