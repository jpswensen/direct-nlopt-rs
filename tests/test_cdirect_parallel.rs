//! Tests for CDirect parallel evaluation.
//!
//! Verifies that the parallel CDirect path produces results of equivalent
//! quality to the serial path across all CDirect algorithm variants
//! (Original, LocallyBiased, Randomized, and their unscaled counterparts).
//!
//! Note: The parallel path may produce slightly different results from
//! serial due to different age/id assignment ordering (affecting tiebreaking),
//! but should achieve equivalent or better objective values.

use direct_nlopt::cdirect::CDirect;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions, DirectResult};
use direct_nlopt::{direct_optimize, DirectBuilder};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Test objective functions
// ─────────────────────────────────────────────────────────────────────────────

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
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
    let mut sum = 10.0 * n;
    for &xi in x {
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

fn styblinski_tang(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for &xi in x {
        sum += xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi;
    }
    sum / 2.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper to run serial and parallel and compare
// ─────────────────────────────────────────────────────────────────────────────

fn run_serial_and_parallel(
    func: impl Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
    bounds: &[(f64, f64)],
    algorithm: DirectAlgorithm,
    max_feval: usize,
) -> (DirectResult, DirectResult) {
    let serial_opts = DirectOptions {
        max_feval,
        algorithm,
        magic_eps: 1e-4,
        parallel: false,
        ..Default::default()
    };
    let func_clone = func.clone();
    let serial_result = CDirect::new(func_clone, bounds.to_vec(), serial_opts)
        .minimize()
        .unwrap();

    let parallel_opts = DirectOptions {
        max_feval,
        algorithm,
        magic_eps: 1e-4,
        parallel: true,
        min_parallel_evals: 1, // Always use parallel path
        ..Default::default()
    };
    let parallel_result = CDirect::new(func, bounds.to_vec(), parallel_opts)
        .minimize()
        .unwrap();

    (serial_result, parallel_result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic correctness: parallel finds a good minimum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_sphere_2d_original() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(sphere, &bounds, DirectAlgorithm::Original, 500);

    println!("Serial:   fun={:.6e}, nfev={}, x={:?}", serial.fun, serial.nfev, serial.x);
    println!("Parallel: fun={:.6e}, nfev={}, x={:?}", parallel.fun, parallel.nfev, parallel.x);

    // Both should find a good minimum for sphere
    assert!(serial.fun < 1e-2, "Serial should find good minimum: {}", serial.fun);
    assert!(parallel.fun < 1e-2, "Parallel should find good minimum: {}", parallel.fun);
}

#[test]
fn test_parallel_sphere_2d_locally_biased() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(sphere, &bounds, DirectAlgorithm::LocallyBiased, 500);

    println!("Serial:   fun={:.6e}, nfev={}", serial.fun, serial.nfev);
    println!("Parallel: fun={:.6e}, nfev={}", parallel.fun, parallel.nfev);

    assert!(serial.fun < 1e-2);
    assert!(parallel.fun < 1e-2);
}

#[test]
fn test_parallel_sphere_2d_randomized() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(sphere, &bounds, DirectAlgorithm::Randomized, 500);

    println!("Serial:   fun={:.6e}, nfev={}", serial.fun, serial.nfev);
    println!("Parallel: fun={:.6e}, nfev={}", parallel.fun, parallel.nfev);

    assert!(serial.fun < 1e-2);
    assert!(parallel.fun < 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unscaled variants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_sphere_2d_original_unscaled() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        sphere, &bounds, DirectAlgorithm::OriginalUnscaled, 500,
    );

    assert!(serial.fun < 1e-2);
    assert!(parallel.fun < 1e-2);
}

#[test]
fn test_parallel_sphere_2d_locally_biased_unscaled() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        sphere, &bounds, DirectAlgorithm::LocallyBiasedUnscaled, 500,
    );

    assert!(serial.fun < 1e-2);
    assert!(parallel.fun < 1e-2);
}

#[test]
fn test_parallel_sphere_2d_randomized_unscaled() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        sphere, &bounds, DirectAlgorithm::LocallyBiasedRandomizedUnscaled, 500,
    );

    assert!(serial.fun < 1e-2);
    assert!(parallel.fun < 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Higher dimensions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_sphere_5d_locally_biased() {
    let bounds = vec![(-5.0, 5.0); 5];
    let (serial, parallel) = run_serial_and_parallel(
        sphere, &bounds, DirectAlgorithm::LocallyBiased, 2000,
    );

    println!("5D Serial:   fun={:.6e}, nfev={}", serial.fun, serial.nfev);
    println!("5D Parallel: fun={:.6e}, nfev={}", parallel.fun, parallel.nfev);

    assert!(serial.fun < 1.0);
    assert!(parallel.fun < 1.0);
}

#[test]
fn test_parallel_rosenbrock_2d_locally_biased() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        rosenbrock, &bounds, DirectAlgorithm::LocallyBiased, 1000,
    );

    println!("Rosenbrock Serial:   fun={:.6e}, nfev={}", serial.fun, serial.nfev);
    println!("Rosenbrock Parallel: fun={:.6e}, nfev={}", parallel.fun, parallel.nfev);

    // Rosenbrock with 1000 evals should get a reasonable minimum
    assert!(serial.fun < 10.0);
    assert!(parallel.fun < 10.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel through the high-level API (DirectBuilder / direct_optimize)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_via_direct_optimize() {
    let result = direct_optimize(
        sphere,
        &vec![(-5.0, 5.0); 2],
        DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiased,
            parallel: true,
            min_parallel_evals: 1,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(result.fun < 1e-2);
}

#[test]
fn test_parallel_via_builder() {
    let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0); 2])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(500)
        .parallel(true)
        .min_parallel_evals(1)
        .minimize()
        .unwrap();

    assert!(result.fun < 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify parallel actually evaluates in parallel (thread count check)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_uses_multiple_threads() {
    use std::collections::HashSet;
    use std::sync::Mutex;

    let thread_ids = Arc::new(Mutex::new(HashSet::new()));
    let thread_ids_clone = Arc::clone(&thread_ids);

    let func = move |x: &[f64]| -> f64 {
        let id = std::thread::current().id();
        thread_ids_clone.lock().unwrap().insert(format!("{:?}", id));
        // Simulate some work to encourage parallel scheduling
        std::thread::sleep(std::time::Duration::from_micros(10));
        x.iter().map(|xi| xi * xi).sum()
    };

    let result = DirectBuilder::new(func, vec![(-5.0, 5.0); 5])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(2000)
        .parallel(true)
        .min_parallel_evals(1)
        .minimize()
        .unwrap();

    let num_threads = thread_ids.lock().unwrap().len();
    println!("Parallel used {} distinct threads", num_threads);
    println!("Result: fun={:.6e}, nfev={}", result.fun, result.nfev);

    // On multi-core machines, parallel should use more than 1 thread
    // (but don't fail on single-core CI machines)
    if std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1) > 1 {
        assert!(
            num_threads > 1,
            "Expected parallel to use multiple threads, got {}",
            num_threads
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify function evaluation count is reasonable
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_nfev_reasonable() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        sphere, &bounds, DirectAlgorithm::LocallyBiased, 500,
    );

    // Parallel may slightly overshoot max_feval but should be in the ballpark
    // (within 2x is a generous bound — in practice it's much closer)
    assert!(
        parallel.nfev <= serial.nfev * 2 + 50,
        "Parallel nfev ({}) should be within 2x of serial ({})",
        parallel.nfev,
        serial.nfev,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Callback works with parallel CDirect
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_callback_force_stop() {
    let eval_count = Arc::new(AtomicUsize::new(0));
    let eval_count_clone = Arc::clone(&eval_count);

    let func = move |x: &[f64]| -> f64 {
        eval_count_clone.fetch_add(1, Ordering::Relaxed);
        x.iter().map(|xi| xi * xi).sum()
    };

    let result = DirectBuilder::new(func, vec![(-5.0, 5.0); 2])
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(5000) // Large budget
        .parallel(true)
        .min_parallel_evals(1)
        .with_callback(|_x, _f, nfev, _nit| {
            // Force stop after 100 evaluations
            nfev >= 100
        })
        .minimize()
        .unwrap();

    println!("Callback stopped at nfev={}", result.nfev);
    // Should have stopped reasonably early (callback checked each iteration)
    assert!(result.nfev < 1000, "Callback should have stopped: nfev={}", result.nfev);
}

// ─────────────────────────────────────────────────────────────────────────────
// All 6 CDirect variants work with parallel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_all_cdirect_variants() {
    let algorithms = [
        DirectAlgorithm::Original,
        DirectAlgorithm::LocallyBiased,
        DirectAlgorithm::Randomized,
        DirectAlgorithm::OriginalUnscaled,
        DirectAlgorithm::LocallyBiasedUnscaled,
        DirectAlgorithm::LocallyBiasedRandomizedUnscaled,
    ];

    let bounds = vec![(-5.0, 5.0); 3];

    for alg in &algorithms {
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: *alg,
            magic_eps: 1e-4,
            parallel: true,
            min_parallel_evals: 1,
            ..Default::default()
        };

        let result = CDirect::new(sphere, bounds.clone(), opts).minimize().unwrap();

        println!(
            "{:?}: fun={:.6e}, nfev={}, code={:?}",
            alg, result.fun, result.nfev, result.return_code
        );
        assert!(
            result.fun < 1.0,
            "{:?} parallel should find reasonable minimum, got {}",
            alg,
            result.fun,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel CDirect on more challenging functions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parallel_rastrigin_2d() {
    let bounds = vec![(-5.12, 5.12); 2];
    let (serial, parallel) = run_serial_and_parallel(
        rastrigin, &bounds, DirectAlgorithm::LocallyBiased, 1000,
    );

    println!("Rastrigin Serial:   fun={:.6e}", serial.fun);
    println!("Rastrigin Parallel: fun={:.6e}", parallel.fun);

    // Both should find something reasonable (rastrigin global min = 0)
    assert!(serial.fun < 20.0);
    assert!(parallel.fun < 20.0);
}

#[test]
fn test_parallel_styblinski_tang_2d() {
    let bounds = vec![(-5.0, 5.0); 2];
    let (serial, parallel) = run_serial_and_parallel(
        styblinski_tang, &bounds, DirectAlgorithm::LocallyBiased, 1000,
    );

    println!("Styblinski-Tang Serial:   fun={:.6e}", serial.fun);
    println!("Styblinski-Tang Parallel: fun={:.6e}", parallel.fun);

    // Global min ≈ -78.33 for 2D
    assert!(serial.fun < -50.0);
    assert!(parallel.fun < -50.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// min_parallel_evals threshold fallback to serial
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_min_parallel_evals_threshold() {
    // With a very high threshold, parallel path should behave like serial
    let opts = DirectOptions {
        max_feval: 200,
        algorithm: DirectAlgorithm::LocallyBiased,
        magic_eps: 1e-4,
        parallel: true,
        min_parallel_evals: 10000, // Very high threshold → always serial fallback
        ..Default::default()
    };

    let result = CDirect::new(sphere, vec![(-5.0, 5.0); 2], opts)
        .minimize()
        .unwrap();

    assert!(result.fun < 1e-1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression: objective function that itself uses rayon internally
// ─────────────────────────────────────────────────────────────────────────────

/// Regression test for <https://github.com/...> — rayon work-stealing re-entrancy.
///
/// When the user's objective function internally uses rayon (e.g. parallel
/// simulations), rayon's work-stealing can cause the same thread to re-enter
/// the scaled_func closure while a RefCell borrow was held, panicking with
/// "RefCell already borrowed".
///
/// This test verifies that a rayon-using objective runs to completion without
/// panic under both serial and parallel CDirect.
#[test]
fn test_objective_using_rayon_internally_no_panic() {
    use rayon::prelude::*;

    // Objective that internally uses rayon to compute a parallel sum-of-squares
    let rayon_objective = |x: &[f64]| -> f64 {
        // Force rayon work inside the objective
        let partial_sums: Vec<f64> = (0..100)
            .into_par_iter()
            .map(|i| {
                let scale = (i as f64) * 0.01;
                x.iter().map(|&xi| (xi * scale).powi(2)).sum::<f64>()
            })
            .collect();
        partial_sums.iter().sum::<f64>() / 100.0
    };

    let bounds = vec![(-5.0, 5.0); 3];

    // Serial CDirect with rayon-using objective
    let serial_opts = DirectOptions {
        max_feval: 300,
        algorithm: DirectAlgorithm::Original,
        parallel: false,
        ..Default::default()
    };
    let serial_result = CDirect::new(rayon_objective, bounds.clone(), serial_opts)
        .minimize()
        .unwrap();
    assert!(serial_result.fun.is_finite());

    // Parallel CDirect with rayon-using objective — this was the crash scenario
    let parallel_opts = DirectOptions {
        max_feval: 300,
        algorithm: DirectAlgorithm::Original,
        parallel: true,
        min_parallel_evals: 2,
        ..Default::default()
    };
    let parallel_result = CDirect::new(rayon_objective, bounds, parallel_opts)
        .minimize()
        .unwrap();
    assert!(parallel_result.fun.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify CDirect parallel actually uses multiple threads
// ─────────────────────────────────────────────────────────────────────────────

/// Verifies that CDirect LocallyBiased parallel mode actually evaluates
/// objective function calls on multiple threads.
///
/// Regression: the old threshold check `qualifying_keys.len() * 2` underestimated
/// the actual batch size for Path A (which produces 2 * n_dims points per rect).
/// With LocallyBiased (DIRECT-L), often only 1 rect qualifies per iteration,
/// causing the estimate to be 2 < min_parallel_evals (4), which fell back to a
/// completely serial code path. The fix: check all_points.len() AFTER building
/// the candidate list (matching the Gablonsky parallel pattern).
#[test]
fn test_cdirect_locally_biased_parallel_uses_multiple_threads() {
    use std::collections::HashSet;
    use std::sync::Mutex;

    let thread_ids: Arc<Mutex<HashSet<std::thread::ThreadId>>> =
        Arc::new(Mutex::new(HashSet::new()));
    let thread_ids_clone = Arc::clone(&thread_ids);

    // Objective with a small sleep to ensure rayon distributes work across threads
    let slow_sphere = move |x: &[f64]| -> f64 {
        thread_ids_clone
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        // Small sleep so rayon has time to distribute across threads
        std::thread::sleep(std::time::Duration::from_millis(1));
        x.iter().map(|xi| xi * xi).sum()
    };

    // 5D problem: Path A produces 2*5=10 candidate points per rect — well above threshold
    let bounds = vec![(-5.0, 5.0); 5];

    let opts = DirectOptions {
        max_feval: 500,
        algorithm: DirectAlgorithm::LocallyBiased,
        parallel: true,
        min_parallel_evals: 4,
        ..Default::default()
    };

    let result = CDirect::new(slow_sphere, bounds, opts)
        .minimize()
        .unwrap();

    assert!(result.fun.is_finite());
    assert!(result.nfev > 50, "Should have done many evaluations, got {}", result.nfev);

    let unique_threads = thread_ids.lock().unwrap().len();
    assert!(
        unique_threads >= 2,
        "CDirect parallel should use multiple threads, but only used {}",
        unique_threads
    );
}
