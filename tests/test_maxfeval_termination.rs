#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: maxfeval termination.
//!
//! Tests that NLOPT C and Rust produce IDENTICAL results when termination
//! is driven by maxfeval limits (50, 100, 200) on the sphere function.
//! Verifies nfev and best solution match exactly at each maxfeval level.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::direct::Direct;
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

/// Sphere function for C FFI that counts evaluations.
extern "C" fn sphere_counting(
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
// Helper: run comparison at a given maxfeval for a given algorithm
// ─────────────────────────────────────────────────────────────────────────────

fn run_maxfeval_comparison(
    maxfeval: usize,
    c_alg: DirectAlgorithmC,
    rust_alg: DirectAlgorithm,
    label: &str,
) {
    // ── Run NLOPT C ──
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: maxfeval as i32,
        max_iter: -1, // unlimited iterations
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: c_alg,
    };
    let c_result = unsafe {
        c_runner.run(
            sphere_counting,
            &c_counter as *const EvalCounter as *mut c_void,
        )
    };
    let c_nfev = c_counter.get();

    // ── Run Rust ──
    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: maxfeval,
        max_iter: 0, // 0 = unlimited in Rust
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: rust_alg,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    // ── Print comparison ──
    println!(
        "=== {} maxfeval={} ===",
        label, maxfeval
    );
    println!(
        "C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
        c_result.x[0], c_result.x[1], c_result.minf, c_nfev
    );
    println!(
        "Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
        rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev
    );
    println!(
        "C code: {:?}, Rust code: {:?}",
        c_result.return_code, rust_result.return_code
    );

    // ── Verify IDENTICAL results ──
    assert_eq!(
        c_nfev, rust_result.nfev,
        "[{}] nfev mismatch at maxfeval={}: C={}, Rust={}",
        label, maxfeval, c_nfev, rust_result.nfev
    );
    assert_eq!(
        c_result.x[0], rust_result.x[0],
        "[{}] x[0] mismatch at maxfeval={}: C={:.15e}, Rust={:.15e}",
        label, maxfeval, c_result.x[0], rust_result.x[0]
    );
    assert_eq!(
        c_result.x[1], rust_result.x[1],
        "[{}] x[1] mismatch at maxfeval={}: C={:.15e}, Rust={:.15e}",
        label, maxfeval, c_result.x[1], rust_result.x[1]
    );
    assert_eq!(
        c_result.minf, rust_result.fun,
        "[{}] minf mismatch at maxfeval={}: C={:.15e}, Rust={:.15e}",
        label, maxfeval, c_result.minf, rust_result.fun
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: DIRECT_GABLONSKY (DIRECT-L) with maxfeval=50, 100, 200
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_maxfeval_50_gablonsky() {
    run_maxfeval_comparison(
        50,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Gablonsky",
    );
}

#[test]
fn test_maxfeval_100_gablonsky() {
    run_maxfeval_comparison(
        100,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Gablonsky",
    );
}

#[test]
fn test_maxfeval_200_gablonsky() {
    run_maxfeval_comparison(
        200,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Gablonsky",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: DIRECT_ORIGINAL with maxfeval=50, 100, 200
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_maxfeval_50_original() {
    run_maxfeval_comparison(
        50,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Original",
    );
}

#[test]
fn test_maxfeval_100_original() {
    run_maxfeval_comparison(
        100,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Original",
    );
}

#[test]
fn test_maxfeval_200_original() {
    run_maxfeval_comparison(
        200,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Original",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: verify nfev is monotonically increasing across maxfeval levels
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_maxfeval_monotonic_nfev_gablonsky() {
    let mut prev_nfev = 0usize;
    let mut prev_minf = f64::INFINITY;

    for &maxfeval in &[50usize, 100, 200, 500] {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: maxfeval,
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
        let mut solver = Direct::new(sphere_rust, &bounds, opts).unwrap();
        let result = solver.minimize(None).unwrap();

        println!(
            "maxfeval={}: nfev={}, minf={:.15e}",
            maxfeval, result.nfev, result.fun
        );

        // nfev should increase (or stay same if terminated early) with maxfeval
        assert!(
            result.nfev >= prev_nfev,
            "nfev should be monotonically non-decreasing: prev={}, cur={} at maxfeval={}",
            prev_nfev,
            result.nfev,
            maxfeval
        );
        // minf should decrease (or stay same) with more evaluations
        assert!(
            result.fun <= prev_minf + 1e-15,
            "minf should decrease with more evals: prev={:.15e}, cur={:.15e} at maxfeval={}",
            prev_minf,
            result.fun,
            maxfeval
        );

        prev_nfev = result.nfev;
        prev_minf = result.fun;
    }
}

/// Same monotonicity test for DIRECT_ORIGINAL.
#[test]
fn test_maxfeval_monotonic_nfev_original() {
    let mut prev_nfev = 0usize;
    let mut prev_minf = f64::INFINITY;

    for &maxfeval in &[50usize, 100, 200, 500] {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: maxfeval,
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
        let mut solver = Direct::new(sphere_rust, &bounds, opts).unwrap();
        let result = solver.minimize(None).unwrap();

        println!(
            "maxfeval={}: nfev={}, minf={:.15e}",
            maxfeval, result.nfev, result.fun
        );

        assert!(
            result.nfev >= prev_nfev,
            "nfev should be monotonically non-decreasing: prev={}, cur={} at maxfeval={}",
            prev_nfev,
            result.nfev,
            maxfeval
        );
        assert!(
            result.fun <= prev_minf + 1e-15,
            "minf should decrease with more evals: prev={:.15e}, cur={:.15e} at maxfeval={}",
            prev_minf,
            result.fun,
            maxfeval
        );

        prev_nfev = result.nfev;
        prev_minf = result.fun;
    }
}
