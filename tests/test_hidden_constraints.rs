#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: hidden constraints (infeasible regions).
//!
//! Tests that both NLOPT C and Rust handle infeasible points identically
//! when the objective function returns NaN/Inf for points outside a
//! feasible region (a hidden constraint).
//!
//! The test function is a sphere function with a circular hidden constraint:
//! - f(x) = sum(x_i^2) when ||x|| <= R
//! - f(x) = NaN (infeasible) when ||x|| > R
//!
//! This exercises the dirreplaceinf_() / replace_infeasible() logic.

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
// Hidden constraint: sphere inside circle of radius R
// ─────────────────────────────────────────────────────────────────────────────

const RADIUS: f64 = 3.0;
const RADIUS_SQ: f64 = RADIUS * RADIUS;

/// C objective: sphere with circular hidden constraint.
/// Returns HUGE_VAL and sets *undefined_flag = 1 for ||x|| > R.
extern "C" fn constrained_sphere_c(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);

    let n = n as usize;
    let mut r_sq = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        r_sq += xi * xi;
    }
    if r_sq > RADIUS_SQ {
        unsafe { *undefined_flag = 1; }
        return f64::MAX;
    }
    r_sq
}

/// Rust objective: sphere with circular hidden constraint.
/// Returns NaN for ||x|| > R.
fn constrained_sphere_rust(x: &[f64]) -> f64 {
    let r_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    if r_sq > RADIUS_SQ {
        return f64::NAN;
    }
    r_sq
}

// ─────────────────────────────────────────────────────────────────────────────
// Hidden constraint: sphere with annular infeasible region
// The sphere minimum is at the origin, but we restrict to a ring R_inner < ||x|| < R_outer
// ─────────────────────────────────────────────────────────────────────────────

const R_INNER: f64 = 0.5;
const R_INNER_SQ: f64 = R_INNER * R_INNER;
const R_OUTER: f64 = 4.0;
const R_OUTER_SQ: f64 = R_OUTER * R_OUTER;

/// C objective: sphere with annular hidden constraint.
extern "C" fn annular_sphere_c(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const EvalCounter) };
    counter.count.fetch_add(1, Ordering::Relaxed);

    let n = n as usize;
    let mut r_sq = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        r_sq += xi * xi;
    }
    if r_sq < R_INNER_SQ || r_sq > R_OUTER_SQ {
        unsafe { *undefined_flag = 1; }
        return f64::MAX;
    }
    r_sq
}

/// Rust objective: sphere with annular hidden constraint.
fn annular_sphere_rust(x: &[f64]) -> f64 {
    let r_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    if r_sq < R_INNER_SQ || r_sq > R_OUTER_SQ {
        return f64::NAN;
    }
    r_sq
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

fn run_c_direct(
    f: nlopt_ffi::DirectObjectiveFuncC,
    dim: usize,
    lb: Vec<f64>,
    ub: Vec<f64>,
    max_feval: i32,
    algorithm: DirectAlgorithmC,
) -> (nlopt_ffi::NloptDirectResult, usize) {
    let counter = EvalCounter::new();
    let runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: lb,
        upper_bounds: ub,
        max_feval,
        max_iter: -1,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm,
    };
    let result = unsafe {
        runner.run(f, &counter as *const EvalCounter as *mut c_void)
    };
    let nfev = counter.get();
    (result, nfev)
}

fn run_rust_direct(
    f: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    bounds: Vec<(f64, f64)>,
    max_feval: usize,
    algorithm: DirectAlgorithm,
) -> direct_nlopt::types::DirectResult {
    let opts = DirectOptions {
        max_feval,
        max_iter: 0, // unlimited
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm,
        parallel: false,
        ..Default::default()
    };
    let mut solver = Direct::new(f, &bounds, opts).unwrap();
    solver.minimize(None).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: circular hidden constraint
// ─────────────────────────────────────────────────────────────────────────────

/// Test hidden circular constraint with DIRECT_GABLONSKY on [-5,5]^2, maxfeval=1000.
/// The feasible region is ||x|| <= 3, so many sample points will be infeasible.
#[test]
fn test_hidden_constraint_circle_gablonsky() {
    let (c_result, c_nfev) = run_c_direct(
        constrained_sphere_c,
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        1000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_direct(
        constrained_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Hidden Constraint Circle 2D GABLONSKY (maxfeval=1000, R={}) ===", RADIUS);
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             c_result.x[0], c_result.x[1], c_result.minf, c_nfev, c_result.return_code);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev,
             rust_result.return_code);

    // Verify IDENTICAL results
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

/// Test hidden circular constraint with DIRECT_ORIGINAL on [-5,5]^2, maxfeval=1000.
#[test]
fn test_hidden_constraint_circle_original() {
    let (c_result, c_nfev) = run_c_direct(
        constrained_sphere_c,
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        1000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
    );

    let rust_result = run_rust_direct(
        constrained_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Hidden Constraint Circle 2D ORIGINAL (maxfeval=1000, R={}) ===", RADIUS);
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             c_result.x[0], c_result.x[1], c_result.minf, c_nfev, c_result.return_code);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev,
             rust_result.return_code);

    assert_eq!(c_result.x[0], rust_result.x[0],
        "x[0] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[0], rust_result.x[0]);
    assert_eq!(c_result.x[1], rust_result.x[1],
        "x[1] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[1], rust_result.x[1]);
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    assert_eq!(c_nfev, rust_result.nfev,
        "nfev mismatch: C={}, Rust={}", c_nfev, rust_result.nfev);
}

/// Test hidden circular constraint in 3D with GABLONSKY.
#[test]
fn test_hidden_constraint_circle_3d_gablonsky() {
    extern "C" fn constrained_sphere_3d_c(
        n: c_int,
        x: *const c_double,
        undefined_flag: *mut c_int,
        data: *mut c_void,
    ) -> c_double {
        let counter = unsafe { &*(data as *const EvalCounter) };
        counter.count.fetch_add(1, Ordering::Relaxed);
        let n = n as usize;
        let mut r_sq = 0.0;
        for i in 0..n {
            let xi = unsafe { *x.add(i) };
            r_sq += xi * xi;
        }
        if r_sq > RADIUS_SQ {
            unsafe { *undefined_flag = 1; }
            return f64::MAX;
        }
        r_sq
    }

    let (c_result, c_nfev) = run_c_direct(
        constrained_sphere_3d_c,
        3,
        vec![-5.0; 3],
        vec![5.0; 3],
        1000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_direct(
        constrained_sphere_rust,
        vec![(-5.0, 5.0); 3],
        1000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Hidden Constraint Circle 3D GABLONSKY (maxfeval=1000, R={}) ===", RADIUS);
    println!("C:    x={:?}, minf={:.15e}, nfev={}", c_result.x, c_result.minf, c_nfev);
    println!("Rust: x={:?}, minf={:.15e}, nfev={}", rust_result.x, rust_result.fun, rust_result.nfev);

    for i in 0..3 {
        assert_eq!(c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}", i, c_result.x[i], rust_result.x[i]);
    }
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    assert_eq!(c_nfev, rust_result.nfev,
        "nfev mismatch: C={}, Rust={}", c_nfev, rust_result.nfev);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: annular hidden constraint (both inner and outer infeasible boundaries)
// ─────────────────────────────────────────────────────────────────────────────

/// Test annular hidden constraint (R_inner=0.5, R_outer=4) with GABLONSKY.
/// This ensures infeasible regions near the optimum are handled correctly
/// since the true minimum (origin) is infeasible.
#[test]
fn test_hidden_constraint_annular_gablonsky() {
    let (c_result, c_nfev) = run_c_direct(
        annular_sphere_c,
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        1000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_direct(
        annular_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Hidden Constraint Annular 2D GABLONSKY (R_in={}, R_out={}, maxfeval=1000) ===",
             R_INNER, R_OUTER);
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             c_result.x[0], c_result.x[1], c_result.minf, c_nfev, c_result.return_code);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev,
             rust_result.return_code);

    assert_eq!(c_result.x[0], rust_result.x[0],
        "x[0] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[0], rust_result.x[0]);
    assert_eq!(c_result.x[1], rust_result.x[1],
        "x[1] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[1], rust_result.x[1]);
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    assert_eq!(c_nfev, rust_result.nfev,
        "nfev mismatch: C={}, Rust={}", c_nfev, rust_result.nfev);
}

/// Test annular hidden constraint with DIRECT_ORIGINAL.
#[test]
fn test_hidden_constraint_annular_original() {
    let (c_result, c_nfev) = run_c_direct(
        annular_sphere_c,
        2,
        vec![-5.0, -5.0],
        vec![5.0, 5.0],
        1000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
    );

    let rust_result = run_rust_direct(
        annular_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Hidden Constraint Annular 2D ORIGINAL (R_in={}, R_out={}, maxfeval=1000) ===",
             R_INNER, R_OUTER);
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             c_result.x[0], c_result.x[1], c_result.minf, c_nfev, c_result.return_code);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}, code={:?}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev,
             rust_result.return_code);

    assert_eq!(c_result.x[0], rust_result.x[0],
        "x[0] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[0], rust_result.x[0]);
    assert_eq!(c_result.x[1], rust_result.x[1],
        "x[1] mismatch: C={:.15e}, Rust={:.15e}", c_result.x[1], rust_result.x[1]);
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    assert_eq!(c_nfev, rust_result.nfev,
        "nfev mismatch: C={}, Rust={}", c_nfev, rust_result.nfev);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: sanity check that the optimum is reasonable
// ─────────────────────────────────────────────────────────────────────────────

/// Verify the circular constraint finds the true optimum near the origin,
/// and the annular constraint finds the optimum on the inner boundary.
#[test]
fn test_hidden_constraint_solution_quality() {
    // Circle constraint: optimum should be near origin
    let circle_result = run_rust_direct(
        constrained_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    println!("Circle constraint: x={:?}, minf={:.6e}", circle_result.x, circle_result.fun);
    assert!(circle_result.fun < 0.1,
        "Circle constraint should find near-zero minimum, got {}", circle_result.fun);

    // Annular constraint: optimum should be on inner boundary (||x|| ~ R_INNER)
    let annular_result = run_rust_direct(
        annular_sphere_rust,
        vec![(-5.0, 5.0); 2],
        1000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );
    println!("Annular constraint: x={:?}, minf={:.6e}", annular_result.x, annular_result.fun);
    // The optimum should be >= R_INNER^2 since ||x|| >= R_INNER
    assert!(annular_result.fun >= R_INNER_SQ - 0.01,
        "Annular min should be >= R_inner^2={}, got {}", R_INNER_SQ, annular_result.fun);
}
