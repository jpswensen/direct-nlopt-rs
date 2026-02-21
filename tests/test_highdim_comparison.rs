#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: higher-dimensional problems (5D, 10D, 20D).
//!
//! Tests sphere function in 5D, 10D, and 20D with both DIRECT_GABLONSKY and
//! DIRECT_ORIGINAL using maxfeval=5000. Compares nfev, fun, x between
//! NLOPT C and Rust implementations — results should be IDENTICAL.
//!
//! These stress-test memory allocation, level management, and performance
//! at higher dimensions where the number of rectangles grows rapidly.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// C objective function with evaluation counter
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

/// N-dimensional sphere function for C FFI with evaluation counter.
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
// Helper: run C and Rust, compare results
// ─────────────────────────────────────────────────────────────────────────────

fn run_comparison(
    dim: usize,
    max_feval: i32,
    c_algo: DirectAlgorithmC,
    rust_algo: DirectAlgorithm,
    label: &str,
) {
    let lb = vec![-5.0; dim];
    let ub = vec![5.0; dim];

    // Run NLOPT C
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: lb,
        upper_bounds: ub,
        max_feval,
        max_iter: -1, // unlimited iterations
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: c_algo,
    };
    let c_result = unsafe {
        c_runner.run(
            sphere_counting,
            &c_counter as *const EvalCounter as *mut c_void,
        )
    };
    let c_nfev = c_counter.get();

    // Run Rust
    let bounds = vec![(-5.0, 5.0); dim];
    let opts = DirectOptions {
        max_feval: max_feval as usize,
        max_iter: 0, // 0 = unlimited in Rust
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: rust_algo,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(sphere_rust, &bounds, opts).unwrap();
    let rust_result = solver.minimize(None).unwrap();

    // Print comparison
    println!("=== {} ===", label);
    println!(
        "  C:    minf={:.15e}, nfev={}, code={:?}",
        c_result.minf, c_nfev, c_result.return_code
    );
    println!(
        "  Rust: minf={:.15e}, nfev={}, nit={}, code={:?}",
        rust_result.fun, rust_result.nfev, rust_result.nit, rust_result.return_code
    );
    println!("  C:    x={:?}", &c_result.x[..dim.min(5)]);
    println!("  Rust: x={:?}", &rust_result.x[..dim.min(5)]);

    // Verify IDENTICAL results
    assert_eq!(
        c_nfev, rust_result.nfev,
        "{}: nfev mismatch: C={}, Rust={}",
        label, c_nfev, rust_result.nfev
    );
    assert_eq!(
        c_result.minf, rust_result.fun,
        "{}: minf mismatch: C={:.15e}, Rust={:.15e}",
        label, c_result.minf, rust_result.fun
    );
    for i in 0..dim {
        assert_eq!(
            c_result.x[i], rust_result.x[i],
            "{}: x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            label, i, c_result.x[i], rust_result.x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5D tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sphere_5d_gablonsky() {
    run_comparison(
        5,
        5000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 5D GABLONSKY maxfeval=5000",
    );
}

#[test]
fn test_sphere_5d_original() {
    run_comparison(
        5,
        5000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere 5D ORIGINAL maxfeval=5000",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 10D tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sphere_10d_gablonsky() {
    run_comparison(
        10,
        5000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 10D GABLONSKY maxfeval=5000",
    );
}

#[test]
fn test_sphere_10d_original() {
    run_comparison(
        10,
        5000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere 10D ORIGINAL maxfeval=5000",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 20D tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sphere_20d_gablonsky() {
    run_comparison(
        20,
        5000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 20D GABLONSKY maxfeval=5000",
    );
}

#[test]
fn test_sphere_20d_original() {
    run_comparison(
        20,
        5000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere 20D ORIGINAL maxfeval=5000",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stress tests with higher maxfeval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sphere_5d_gablonsky_10k() {
    run_comparison(
        5,
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 5D GABLONSKY maxfeval=10000",
    );
}

#[test]
fn test_sphere_10d_gablonsky_10k() {
    run_comparison(
        10,
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
        "Sphere 10D GABLONSKY maxfeval=10000",
    );
}

#[test]
fn test_sphere_5d_original_10k() {
    run_comparison(
        5,
        10000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
        "Sphere 5D ORIGINAL maxfeval=10000",
    );
}
