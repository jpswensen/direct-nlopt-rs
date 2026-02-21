//! Verify parallel mode produces correct results across all test functions.
//!
//! Runs sphere, rosenbrock, rastrigin, ackley, and styblinski-tang with both
//! `parallel=false` and `parallel=true` (and `parallel_batch=true`), comparing
//! final results. For deterministic functions, parallel mode evaluates in a
//! different order but produces identical final results because the algorithm
//! logic (rectangle selection, division, insertion) is identical — only the
//! function evaluation is parallelized.

use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Test functions
// ─────────────────────────────────────────────────────────────────────────────

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

fn styblinski_tang(x: &[f64]) -> f64 {
    0.5 * x.iter().map(|&xi| xi.powi(4) - 16.0 * xi * xi + 5.0 * xi).sum::<f64>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run optimization with given parallel settings and return result
// ─────────────────────────────────────────────────────────────────────────────

fn run_gablonsky(
    func: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    algorithm: DirectAlgorithm,
    max_feval: usize,
    max_iter: usize,
    parallel: bool,
    parallel_batch: bool,
    min_parallel_evals: usize,
) -> direct_nlopt::types::DirectResult {
    let opts = DirectOptions {
        max_feval,
        max_iter,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm,
        parallel,
        parallel_batch,
        min_parallel_evals,
        ..Default::default()
    };
    let bounds_vec: Vec<(f64, f64)> = bounds.to_vec();
    let mut solver = Direct::new(func, &bounds_vec, opts).unwrap();
    solver.minimize(None).unwrap()
}

/// Compare two results for identical output (x, fun, nfev).
/// For deterministic functions with the same algorithm flow, parallel mode
/// should produce bit-identical results.
fn assert_results_identical(
    serial: &direct_nlopt::types::DirectResult,
    parallel: &direct_nlopt::types::DirectResult,
    label: &str,
) {
    assert_eq!(
        serial.nfev, parallel.nfev,
        "{}: nfev mismatch: serial={}, parallel={}",
        label, serial.nfev, parallel.nfev
    );
    assert_eq!(
        serial.nit, parallel.nit,
        "{}: nit mismatch: serial={}, parallel={}",
        label, serial.nit, parallel.nit
    );
    assert_eq!(
        serial.fun, parallel.fun,
        "{}: fun mismatch: serial={:.15e}, parallel={:.15e}",
        label, serial.fun, parallel.fun
    );
    for (i, (&s, &p)) in serial.x.iter().zip(parallel.x.iter()).enumerate() {
        assert_eq!(
            s, p,
            "{}: x[{}] mismatch: serial={:.15e}, parallel={:.15e}",
            label, i, s, p
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_sphere_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, false, false, 4);
    let parallel = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "sphere_2d_gablonsky_parallel");
}

#[test]
fn test_parallel_batch_sphere_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, false, false, 4);
    let batch = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "sphere_2d_gablonsky_batch");
}

#[test]
fn test_parallel_sphere_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "sphere_3d_gablonsky_parallel");
}

#[test]
fn test_parallel_sphere_5d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 5];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let parallel = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "sphere_5d_gablonsky_parallel");
}

#[test]
fn test_parallel_sphere_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyOriginal, 500, 0, false, false, 4);
    let parallel = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyOriginal, 500, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "sphere_2d_original_parallel");
}

#[test]
fn test_parallel_batch_sphere_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyOriginal, 500, 0, false, false, 4);
    let batch = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyOriginal, 500, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "sphere_2d_original_batch");
}

// ─────────────────────────────────────────────────────────────────────────────
// Rosenbrock tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_rosenbrock_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rosenbrock_2d_gablonsky_parallel");
}

#[test]
fn test_parallel_batch_rosenbrock_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let batch = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "rosenbrock_2d_gablonsky_batch");
}

#[test]
fn test_parallel_rosenbrock_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rosenbrock_2d_original_parallel");
}

#[test]
fn test_parallel_rosenbrock_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let parallel = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rosenbrock_3d_gablonsky_parallel");
}

// ─────────────────────────────────────────────────────────────────────────────
// Rastrigin tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_rastrigin_2d_gablonsky() {
    let bounds = vec![(-5.12, 5.12); 2];
    let serial = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rastrigin_2d_gablonsky_parallel");
}

#[test]
fn test_parallel_batch_rastrigin_2d_gablonsky() {
    let bounds = vec![(-5.12, 5.12); 2];
    let serial = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let batch = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "rastrigin_2d_gablonsky_batch");
}

#[test]
fn test_parallel_rastrigin_3d_gablonsky() {
    let bounds = vec![(-5.12, 5.12); 3];
    let serial = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let parallel = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rastrigin_3d_gablonsky_parallel");
}

#[test]
fn test_parallel_rastrigin_2d_original() {
    let bounds = vec![(-5.12, 5.12); 2];
    let serial = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "rastrigin_2d_original_parallel");
}

// ─────────────────────────────────────────────────────────────────────────────
// Ackley tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_ackley_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "ackley_2d_gablonsky_parallel");
}

#[test]
fn test_parallel_batch_ackley_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let batch = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "ackley_2d_gablonsky_batch");
}

#[test]
fn test_parallel_ackley_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "ackley_2d_original_parallel");
}

#[test]
fn test_parallel_ackley_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let parallel = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "ackley_3d_gablonsky_parallel");
}

// ─────────────────────────────────────────────────────────────────────────────
// Styblinski-Tang tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_styblinski_tang_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "styblinski_tang_2d_gablonsky_parallel");
}

#[test]
fn test_parallel_batch_styblinski_tang_2d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, false, false, 4);
    let batch = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "styblinski_tang_2d_gablonsky_batch");
}

#[test]
fn test_parallel_styblinski_tang_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let parallel = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "styblinski_tang_3d_gablonsky_parallel");
}

#[test]
fn test_parallel_styblinski_tang_5d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 5];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 3000, 0, false, false, 4);
    let parallel = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 3000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "styblinski_tang_5d_gablonsky_parallel");
}

#[test]
fn test_parallel_styblinski_tang_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, false, false, 4);
    let parallel = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyOriginal, 1000, 0, true, false, 1);
    assert_results_identical(&serial, &parallel, "styblinski_tang_2d_original_parallel");
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-cutting: parallel threshold tests
// ─────────────────────────────────────────────────────────────────────────────

/// Verify min_parallel_evals threshold: when threshold is very high, parallel
/// mode should fall back to serial and produce bit-identical results.
#[test]
fn test_parallel_threshold_fallback_to_serial() {
    let bounds = vec![(-5.0, 5.0); 2];
    let serial = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, false, false, 4);
    // parallel=true but threshold=1000 means serial path is always used
    let high_threshold = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 500, 0, true, false, 1000);
    assert_results_identical(&serial, &high_threshold, "sphere_2d_high_threshold_fallback");
}

/// Verify that different min_parallel_evals values all produce the same result.
#[test]
fn test_parallel_threshold_consistency() {
    let bounds = vec![(-5.0, 5.0); 3];
    let t1 = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 1);
    let t2 = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 4);
    let t8 = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 8);
    let t16 = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 1000, 0, true, false, 16);
    assert_results_identical(&t1, &t2, "threshold_1_vs_4");
    assert_results_identical(&t1, &t8, "threshold_1_vs_8");
    assert_results_identical(&t1, &t16, "threshold_1_vs_16");
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch parallelization: comprehensive coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_batch_rosenbrock_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let batch = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "rosenbrock_3d_gablonsky_batch");
}

#[test]
fn test_parallel_batch_rastrigin_3d_gablonsky() {
    let bounds = vec![(-5.12, 5.12); 3];
    let serial = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let batch = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "rastrigin_3d_gablonsky_batch");
}

#[test]
fn test_parallel_batch_ackley_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let batch = run_gablonsky(ackley, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "ackley_3d_gablonsky_batch");
}

#[test]
fn test_parallel_batch_styblinski_tang_3d_gablonsky() {
    let bounds = vec![(-5.0, 5.0); 3];
    let serial = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, false, false, 4);
    let batch = run_gablonsky(styblinski_tang, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert_results_identical(&serial, &batch, "styblinski_tang_3d_gablonsky_batch");
}

// ─────────────────────────────────────────────────────────────────────────────
// Solution quality: parallel mode should find good minima
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_solution_quality_sphere() {
    let bounds = vec![(-5.0, 5.0); 3];
    let result = run_gablonsky(sphere, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert!(result.fun < 1e-4, "sphere parallel should find near-zero minimum, got {}", result.fun);
}

#[test]
fn test_parallel_solution_quality_rosenbrock() {
    let bounds = vec![(-5.0, 5.0); 2];
    let result = run_gablonsky(rosenbrock, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert!(result.fun < 1.0, "rosenbrock parallel should find good minimum, got {}", result.fun);
}

#[test]
fn test_parallel_solution_quality_rastrigin() {
    let bounds = vec![(-5.12, 5.12); 2];
    let result = run_gablonsky(rastrigin, &bounds, DirectAlgorithm::GablonskyLocallyBiased, 2000, 0, true, true, 1);
    assert!(result.fun < 5.0, "rastrigin parallel should find good minimum, got {}", result.fun);
}
