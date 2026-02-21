#![cfg(feature = "nlopt-compare")]

//! Edge case tests for asymmetric and extreme bounds.
//!
//! Tests verify that both NLOPT C and Rust handle:
//! - Heavy asymmetry: [(-100, 1), (-1, 100)]
//! - Very narrow bounds: [(0, 1e-10)]
//! - Very wide bounds: [(-1e10, 1e10)]
//! - No numerical overflow/underflow in scaling
//!
//! Results are compared between NLOPT C and Rust for both DIRECT_GABLONSKY
//! and DIRECT_ORIGINAL algorithm variants.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};

use nlopt_ffi::{
    DirectAlgorithmC, NloptDirectRunner, DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// C objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere function for C FFI: f(x) = sum(x_i^2)
extern "C" fn sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

/// 1D sphere for C FFI
extern "C" fn sphere_1d_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    assert_eq!(n, 1);
    let x0 = unsafe { *x };
    x0 * x0
}

/// Shifted sphere for C FFI: f(x) = sum((x_i - shift_i)^2)
/// shift = (-50, 50) — minimum lies inside the asymmetric domain
extern "C" fn shifted_sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let shifts = [-50.0, 50.0];
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        let diff = xi - shifts[i];
        sum += diff * diff;
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust objective functions
// ─────────────────────────────────────────────────────────────────────────────

fn sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn sphere_1d_rust(x: &[f64]) -> f64 {
    x[0] * x[0]
}

fn shifted_sphere_rust(x: &[f64]) -> f64 {
    let shifts = [-50.0, 50.0];
    x.iter()
        .zip(shifts.iter())
        .map(|(&xi, &si)| (xi - si).powi(2))
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run NLOPT C and Rust comparison
// ─────────────────────────────────────────────────────────────────────────────

struct ComparisonResult {
    c_x: Vec<f64>,
    c_minf: f64,
    rust_x: Vec<f64>,
    rust_minf: f64,
    rust_nfev: usize,
}

fn run_comparison(
    c_func: nlopt_ffi::DirectObjectiveFuncC,
    rust_func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    lower: &[f64],
    upper: &[f64],
    max_feval: i32,
    algorithm_c: DirectAlgorithmC,
    algorithm_rust: DirectAlgorithm,
) -> ComparisonResult {
    let dim = lower.len();

    // Run NLOPT C
    let c_runner = NloptDirectRunner {
        dimension: dim,
        lower_bounds: lower.to_vec(),
        upper_bounds: upper.to_vec(),
        max_feval,
        max_iter: -1,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: algorithm_c,
    };
    let c_result = unsafe { c_runner.run(c_func, std::ptr::null_mut()) };

    // Run Rust
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let opts = DirectOptions {
        max_feval: max_feval as usize,
        max_iter: 0,
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

    ComparisonResult {
        c_x: c_result.x,
        c_minf: c_result.minf,
        rust_x: rust_result.x,
        rust_minf: rust_result.fun,
        rust_nfev: rust_result.nfev,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Heavy asymmetry [(-100, 1), (-1, 100)]
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_asymmetric_bounds_sphere_gablonsky() {
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[-100.0, -1.0],
        &[1.0, 100.0],
        500,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Asymmetric [-100,1]×[-1,100] sphere GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    // Results should be identical
    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // Sphere minimum is at (0,0). With these bounds, the minimum is at (0,0)
    // only if 0 is in both intervals: [-100,1] contains 0, [-1,100] contains 0.
    assert!(res.rust_minf < 1.0, "Should find near-minimum, got {}", res.rust_minf);
}

#[test]
fn test_asymmetric_bounds_sphere_original() {
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[-100.0, -1.0],
        &[1.0, 100.0],
        500,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Asymmetric [-100,1]×[-1,100] sphere ORIGINAL ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );
}

/// Test with shifted sphere where minimum is at (-50, 50), inside the asymmetric domain.
#[test]
fn test_asymmetric_bounds_shifted_sphere_gablonsky() {
    let res = run_comparison(
        shifted_sphere_c,
        shifted_sphere_rust,
        &[-100.0, -1.0],
        &[1.0, 100.0],
        1000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Asymmetric shifted sphere GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // The true minimum at (-50, 50) is not in the domain.
    // Constrained optimum: x[0] is clamped to [-100, 1], x[1] to [-1, 100].
    // Closest feasible point to (-50,50) is (-50, 50) since both are in range.
    assert!(res.rust_minf < 100.0, "Should find reasonable minimum, got {}", res.rust_minf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Very narrow bounds [(0, 1e-10)]
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_narrow_bounds_1d_gablonsky() {
    let res = run_comparison(
        sphere_1d_c,
        sphere_1d_rust,
        &[0.0],
        &[1e-10],
        200,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Narrow [0, 1e-10] sphere 1D GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);
    println!("Rust: nfev = {}", res.rust_nfev);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // f should be very small (x is in [0, 1e-10])
    assert!(
        res.rust_minf < 1e-18,
        "f(x) should be tiny for x in [0, 1e-10], got {}",
        res.rust_minf
    );
    // x should be in bounds
    assert!(res.rust_x[0] >= 0.0 && res.rust_x[0] <= 1e-10);
}

#[test]
fn test_narrow_bounds_1d_original() {
    let res = run_comparison(
        sphere_1d_c,
        sphere_1d_rust,
        &[0.0],
        &[1e-10],
        200,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Narrow [0, 1e-10] sphere 1D ORIGINAL ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );
}

/// Test narrow bounds in 2D with different scales per dimension
#[test]
fn test_narrow_bounds_2d_mixed_gablonsky() {
    // One very narrow dimension, one normal
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[0.0, -5.0],
        &[1e-10, 5.0],
        500,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Narrow/Normal [0,1e-10]×[-5,5] sphere 2D GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Very wide bounds [(-1e10, 1e10)]
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_wide_bounds_1d_gablonsky() {
    let res = run_comparison(
        sphere_1d_c,
        sphere_1d_rust,
        &[-1e10],
        &[1e10],
        200,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Wide [-1e10, 1e10] sphere 1D GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);
    println!("Rust: nfev = {}", res.rust_nfev);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // Verify no overflow: minf should be finite
    assert!(res.rust_minf.is_finite(), "minf should be finite, got {}", res.rust_minf);
    assert!(res.rust_x[0].is_finite(), "x should be finite, got {}", res.rust_x[0]);
}

#[test]
fn test_wide_bounds_1d_original() {
    let res = run_comparison(
        sphere_1d_c,
        sphere_1d_rust,
        &[-1e10],
        &[1e10],
        200,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Wide [-1e10, 1e10] sphere 1D ORIGINAL ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    assert!(res.rust_minf.is_finite(), "minf should be finite");
}

#[test]
fn test_wide_bounds_2d_gablonsky() {
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[-1e10, -1e10],
        &[1e10, 1e10],
        500,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Wide [-1e10,1e10]^2 sphere 2D GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // All results should be finite (no overflow)
    assert!(res.rust_minf.is_finite(), "minf should be finite");
    for xi in &res.rust_x {
        assert!(xi.is_finite(), "x should be finite, got {}", xi);
    }
}

#[test]
fn test_wide_bounds_2d_original() {
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[-1e10, -1e10],
        &[1e10, 1e10],
        500,
        DirectAlgorithmC::DIRECT_ORIGINAL,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Wide [-1e10,1e10]^2 sphere 2D ORIGINAL ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Verify no numerical overflow/underflow in scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Verify scaling formula produces correct results for asymmetric bounds.
/// NLOPT uses: xs1 = u - l, xs2 = l / (u - l)
/// Unscaling: x_actual = (x_norm + xs2) * xs1
#[test]
fn test_scaling_asymmetric_values() {
    // [-100, 1]: xs1 = 101, xs2 = -100/101
    let l = -100.0_f64;
    let u = 1.0_f64;
    let xs1 = u - l;
    let xs2 = l / (u - l);

    assert_eq!(xs1, 101.0);
    assert!((xs2 - (-100.0 / 101.0)).abs() < 1e-15);

    // Verify roundtrip at center: x_norm=0.5 -> x_actual
    let center_actual = (0.5 + xs2) * xs1;
    let expected_center = (l + u) / 2.0;
    assert!(
        (center_actual - expected_center).abs() < 1e-12,
        "center: actual={}, expected={}",
        center_actual,
        expected_center
    );
}

/// Verify scaling formula for very narrow bounds doesn't lose precision.
#[test]
fn test_scaling_narrow_values() {
    let l = 0.0_f64;
    let u = 1e-10_f64;
    let xs1 = u - l;
    let xs2 = l / (u - l);

    assert_eq!(xs1, 1e-10);
    assert_eq!(xs2, 0.0);

    let center_actual = (0.5 + xs2) * xs1;
    assert_eq!(center_actual, 5e-11);
}

/// Verify scaling formula for very wide bounds doesn't overflow.
#[test]
fn test_scaling_wide_values() {
    let l = -1e10_f64;
    let u = 1e10_f64;
    let xs1 = u - l;
    let xs2 = l / (u - l);

    assert_eq!(xs1, 2e10);
    assert_eq!(xs2, -0.5);

    let center_actual = (0.5 + xs2) * xs1;
    assert_eq!(center_actual, 0.0);

    // Verify that thirds don't underflow
    let third = 1.0 / 3.0_f64;
    let offset_actual = (0.5 + third + xs2) * xs1;
    assert!(offset_actual.is_finite(), "offset should be finite");
    // x_actual = (0.5 + 1/3 + xs2) * xs1 = (5u + l) / 6
    let expected_offset = (5.0 * u + l) / 6.0;
    assert!(
        (offset_actual - expected_offset).abs() / expected_offset.abs() < 1e-14,
        "offset: actual={}, expected={}",
        offset_actual,
        expected_offset
    );
}

/// Verify scaling for bounds with extreme ratio (one tiny, one large dimension).
#[test]
fn test_scaling_extreme_ratio() {
    let bounds: [(f64, f64); 2] = [(1e-15, 1e-14), (-1e8, 1e8)];
    for &(l, u) in &bounds {
        let xs1: f64 = u - l;
        let xs2: f64 = l / (u - l);
        assert!(xs1.is_finite(), "xs1 should be finite for [{}, {}]", l, u);
        assert!(xs2.is_finite(), "xs2 should be finite for [{}, {}]", l, u);

        // Roundtrip test
        let center_actual: f64 = (0.5 + xs2) * xs1;
        let expected: f64 = (l + u) / 2.0;
        let rel_err: f64 = if expected.abs() > 0.0 {
            (center_actual - expected).abs() / expected.abs()
        } else {
            (center_actual - expected).abs()
        };
        assert!(
            rel_err < 1e-12,
            "center roundtrip failed for [{}, {}]: actual={}, expected={}",
            l,
            u,
            center_actual,
            expected
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Additional asymmetric configurations
// ─────────────────────────────────────────────────────────────────────────────

/// Test with purely positive asymmetric bounds: [(2, 10), (0.1, 0.5)]
#[test]
fn test_positive_asymmetric_bounds_gablonsky() {
    extern "C" fn shifted_c(
        n: c_int,
        x: *const c_double,
        _undefined_flag: *mut c_int,
        _data: *mut c_void,
    ) -> c_double {
        let n = n as usize;
        let targets = [5.0, 0.3];
        let mut sum = 0.0;
        for i in 0..n {
            let xi = unsafe { *x.add(i) };
            let diff = xi - targets[i];
            sum += diff * diff;
        }
        sum
    }

    fn shifted_rust(x: &[f64]) -> f64 {
        let targets = [5.0, 0.3];
        x.iter()
            .zip(targets.iter())
            .map(|(&xi, &ti)| (xi - ti).powi(2))
            .sum()
    }

    let res = run_comparison(
        shifted_c,
        shifted_rust,
        &[2.0, 0.1],
        &[10.0, 0.5],
        500,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Positive asymmetric [2,10]×[0.1,0.5] shifted sphere GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );
}

/// Test with heavily skewed width ratio between dimensions.
#[test]
fn test_skewed_width_ratio_gablonsky() {
    let res = run_comparison(
        sphere_c,
        sphere_rust,
        &[-1e-6, -1e6],
        &[1e-6, 1e6],
        500,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Skewed [-1e-6,1e-6]×[-1e6,1e6] sphere GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // All values should be finite
    assert!(res.rust_minf.is_finite());
    for xi in &res.rust_x {
        assert!(xi.is_finite());
    }
}

/// Test with bounds where lower bound is very close to zero.
#[test]
fn test_near_zero_lower_bound_gablonsky() {
    let res = run_comparison(
        sphere_1d_c,
        sphere_1d_rust,
        &[1e-15],
        &[1.0],
        200,
        DirectAlgorithmC::DIRECT_GABLONSKY,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Near-zero lower [1e-15, 1] sphere 1D GABLONSKY ===");
    println!("C:    x = {:?}, minf = {:.15e}", res.c_x, res.c_minf);
    println!("Rust: x = {:?}, minf = {:.15e}", res.rust_x, res.rust_minf);

    for i in 0..res.c_x.len() {
        assert_eq!(
            res.c_x[i], res.rust_x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, res.c_x[i], res.rust_x[i]
        );
    }
    assert_eq!(
        res.c_minf, res.rust_minf,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        res.c_minf, res.rust_minf
    );

    // x should be in bounds and finite
    assert!(res.rust_x[0] >= 1e-15 && res.rust_x[0] <= 1.0);
    assert!(res.rust_minf.is_finite());
}
