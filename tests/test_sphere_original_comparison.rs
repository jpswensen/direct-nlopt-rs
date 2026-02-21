#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: sphere function with DIRECT_ORIGINAL (Jones 1993).
//!
//! Compares NLOPT C direct_optimize() with Rust Direct implementation for
//! the sphere function on [-5,5]^2 using the Original (Jones) algorithm.
//! Results should be IDENTICAL (not just close).
//!
//! This tests the dirdoubleinsert_ path and Original level computation
//! (algmethod=0), which differs from DIRECT_GABLONSKY (algmethod=1) in:
//! - Level computation: min_side_length_index (vs k*N+p for Gablonsky)
//! - Double insert: includes equal-valued rectangles at same level
//! - Division: divides ALL longest dimensions (vs locally biased)

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL};

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
        Self { count: AtomicUsize::new(0) }
    }
    fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// Sphere function for C FFI that counts evaluations via data pointer.
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
// Main comparison tests
// ─────────────────────────────────────────────────────────────────────────────

/// Run NLOPT C and Rust on sphere [-5,5]^2 with DIRECT_ORIGINAL, maxiter=20, eps=1e-4.
/// Results should be IDENTICAL.
#[test]
fn test_sphere_original_comparison() {
    // ── Run NLOPT C ──
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 10000,
        max_iter: 20,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    // ── Run Rust ──
    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 10000,
        max_iter: 20,
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
    let rust_result = solver.minimize(None).unwrap();

    // ── Print comparison ──
    println!("=== Sphere 2D DIRECT_ORIGINAL Comparison (maxiter=20, eps=1e-4) ===");
    println!("C:    x = [{:.15e}, {:.15e}]", c_result.x[0], c_result.x[1]);
    println!("Rust: x = [{:.15e}, {:.15e}]", rust_result.x[0], rust_result.x[1]);
    println!("C:    minf = {:.15e}", c_result.minf);
    println!("Rust: minf = {:.15e}", rust_result.fun);
    println!("C:    nfev = {}", c_nfev);
    println!("Rust: nfev = {}", rust_result.nfev);
    println!("Rust: nit  = {}", rust_result.nit);
    println!("C:    code = {:?}", c_result.return_code);
    println!("Rust: code = {:?}", rust_result.return_code);

    // ── Verify IDENTICAL results ──
    assert_eq!(
        c_result.x[0], rust_result.x[0],
        "x[0] mismatch: C={:.15e}, Rust={:.15e}",
        c_result.x[0], rust_result.x[0]
    );
    assert_eq!(
        c_result.x[1], rust_result.x[1],
        "x[1] mismatch: C={:.15e}, Rust={:.15e}",
        c_result.x[1], rust_result.x[1]
    );
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

/// Verify with larger iteration count (maxiter=50) to stress-test the
/// dirdoubleinsert_ path and Original level computation.
#[test]
fn test_sphere_original_comparison_maxiter50() {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 10000,
        max_iter: 50,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 10000,
        max_iter: 50,
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
    let rust_result = solver.minimize(None).unwrap();

    println!("=== Sphere 2D DIRECT_ORIGINAL (maxiter=50) ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", c_result.x[0], c_result.x[1], c_result.minf, c_nfev);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev);

    assert_eq!(c_result.x[0], rust_result.x[0], "x[0] mismatch at maxiter=50");
    assert_eq!(c_result.x[1], rust_result.x[1], "x[1] mismatch at maxiter=50");
    assert_eq!(c_result.minf, rust_result.fun, "minf mismatch at maxiter=50");
    assert_eq!(c_nfev, rust_result.nfev, "nfev mismatch at maxiter=50");
}

/// Test with maxfeval-limited run instead of maxiter-limited.
#[test]
fn test_sphere_original_comparison_maxfeval500() {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        max_iter: -1, // unlimited iterations
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 500,
        max_iter: 0, // 0 = unlimited in Rust
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
    let rust_result = solver.minimize(None).unwrap();

    println!("=== Sphere 2D DIRECT_ORIGINAL (maxfeval=500) ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", c_result.x[0], c_result.x[1], c_result.minf, c_nfev);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev);

    assert_eq!(c_result.x[0], rust_result.x[0], "x[0] mismatch at maxfeval=500");
    assert_eq!(c_result.x[1], rust_result.x[1], "x[1] mismatch at maxfeval=500");
    assert_eq!(c_result.minf, rust_result.fun, "minf mismatch at maxfeval=500");
    assert_eq!(c_nfev, rust_result.nfev, "nfev mismatch at maxfeval=500");
}

/// Test 3D sphere for higher-dimensional verification of Original algorithm.
#[test]
fn test_sphere_original_comparison_3d() {
    extern "C" fn sphere_3d_counting(
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

    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 3,
        lower_bounds: vec![-5.0, -5.0, -5.0],
        upper_bounds: vec![5.0, 5.0, 5.0],
        max_feval: 10000,
        max_iter: 20,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_3d_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(-5.0, 5.0); 3];
    let opts = DirectOptions {
        max_feval: 10000,
        max_iter: 20,
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
    let rust_result = solver.minimize(None).unwrap();

    println!("=== Sphere 3D DIRECT_ORIGINAL (maxiter=20) ===");
    println!("C:    x={:?}, minf={:.15e}, nfev={}", c_result.x, c_result.minf, c_nfev);
    println!("Rust: x={:?}, minf={:.15e}, nfev={}", rust_result.x, rust_result.fun, rust_result.nfev);

    for i in 0..3 {
        assert_eq!(
            c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, c_result.x[i], rust_result.x[i]
        );
    }
    assert_eq!(c_result.minf, rust_result.fun, "minf mismatch for 3D");
    assert_eq!(c_nfev, rust_result.nfev, "nfev mismatch for 3D");
}

/// Test with parallel=false explicitly to verify exact equivalence.
/// This is the primary faithfulness verification for DIRECT_ORIGINAL.
#[test]
fn test_sphere_original_parallel_false_exact() {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 1000,
        max_iter: 30,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 1000,
        max_iter: 30,
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
    let rust_result = solver.minimize(None).unwrap();

    println!("=== Sphere 2D DIRECT_ORIGINAL parallel=false Exact Comparison ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             c_result.x[0], c_result.x[1], c_result.minf, c_nfev, c_result.return_code);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, nit={}, code={:?}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev,
             rust_result.nit, rust_result.return_code);

    // Exact equality — not approximate
    assert_eq!(c_result.x[0], rust_result.x[0], "x[0] not bit-identical");
    assert_eq!(c_result.x[1], rust_result.x[1], "x[1] not bit-identical");
    assert_eq!(c_result.minf, rust_result.fun, "minf not bit-identical");
    assert_eq!(c_nfev, rust_result.nfev, "nfev not identical");
}

/// Test with asymmetric bounds to verify scaling faithfulness with Original algorithm.
#[test]
fn test_sphere_original_asymmetric_bounds() {
    let c_counter = EvalCounter::new();
    let c_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-2.0, -8.0],
        upper_bounds: vec![4.0, 3.0],
        max_feval: 10000,
        max_iter: 20,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let c_result = unsafe {
        c_runner.run(sphere_counting, &c_counter as *const EvalCounter as *mut c_void)
    };
    let c_nfev = c_counter.get();

    let bounds = vec![(-2.0, 4.0), (-8.0, 3.0)];
    let opts = DirectOptions {
        max_feval: 10000,
        max_iter: 20,
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
    let rust_result = solver.minimize(None).unwrap();

    println!("=== Sphere 2D Asymmetric Bounds DIRECT_ORIGINAL (maxiter=20) ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", c_result.x[0], c_result.x[1], c_result.minf, c_nfev);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}", rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev);

    assert_eq!(c_result.x[0], rust_result.x[0], "x[0] mismatch (asymmetric)");
    assert_eq!(c_result.x[1], rust_result.x[1], "x[1] mismatch (asymmetric)");
    assert_eq!(c_result.minf, rust_result.fun, "minf mismatch (asymmetric)");
    assert_eq!(c_nfev, rust_result.nfev, "nfev mismatch (asymmetric)");
}
