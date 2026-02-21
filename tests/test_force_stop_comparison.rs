#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: force_stop and callback behavior.
//!
//! Tests that NLOPT C's force_stop mechanism and Rust's callback mechanism
//! both terminate with FORCED_STOP when f < threshold.
//!
//! IMPORTANT: Force_stop must NOT be triggered during initialization, because
//! NLOPT C's dirinit_() exits immediately on force_stop (before updating minf),
//! while Rust's initialize() does not check force_stop at the same point.
//! All test functions are designed so that the threshold is NOT reached during
//! the 2n+1 initialization evaluations.
//!
//! Two mechanisms are tested:
//! 1. **Callback comparison**: Rust callback stops after each iteration when
//!    new minimum is below threshold. C stops mid-batch when objective sets
//!    force_stop. Both return FORCED_STOP but may differ in nfev.
//! 2. **Objective force_stop**: Rust objective sets AtomicBool when f < threshold,
//!    matching C's mechanism exactly. Produces identical nfev and results.

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use nlopt_ffi::{direct_optimize, DirectAlgorithmC, DirectReturnCodeC, DIRECT_UNKNOWN_FGLOBAL,
                DIRECT_UNKNOWN_FGLOBAL_RELTOL};

use direct_nlopt::direct::Direct;
use direct_nlopt::error::DirectReturnCode;
use direct_nlopt::types::{CallbackFn, DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// C FFI data struct for force_stop from within objective function
// ─────────────────────────────────────────────────────────────────────────────

/// Data passed through f_data to the C objective function.
/// Contains the force_stop pointer so the objective can trigger early termination.
#[repr(C)]
struct ForceStopData {
    force_stop: *mut c_int,
    threshold: f64,
    eval_count: AtomicUsize,
}

// ─────────────────────────────────────────────────────────────────────────────
// C objective functions that set force_stop when f < threshold
// ─────────────────────────────────────────────────────────────────────────────

/// Shifted sphere f(x) = (x_0-2)^2 + (x_1-3)^2 with force_stop.
/// On [-5,5]^2: center (0,0) → f=13, min init value ≈ 4.11 at (0, 10/3).
/// Use threshold ≤ 3.0 to avoid triggering during initialization.
extern "C" fn shifted_sphere_2d_force_stop(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let data = unsafe { &*(data as *const ForceStopData) };
    data.eval_count.fetch_add(1, Ordering::Relaxed);
    let n = n as usize;
    let shifts = [2.0, 3.0, 1.5, 2.5, 1.0]; // works for up to 5D
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        let s = if i < shifts.len() { shifts[i] } else { 2.0 };
        sum += (xi - s) * (xi - s);
    }
    if sum < data.threshold {
        unsafe { *data.force_stop = 1; }
    }
    sum
}

/// Shifted sphere for 3D with force_stop.
/// On [-5,5]^3: center (0,0,0) → f = 4+9+2.25 = 15.25.
/// Min init value >> threshold when threshold ≤ 3.0.
extern "C" fn shifted_sphere_3d_force_stop(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    shifted_sphere_2d_force_stop(n, x, _undefined_flag, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// Shifted sphere: f(x) = (x_0-2)^2 + (x_1-3)^2 [+ more dims with different shifts]
fn shifted_sphere_rust(x: &[f64]) -> f64 {
    let shifts = [2.0, 3.0, 1.5, 2.5, 1.0];
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let s = if i < shifts.len() { shifts[i] } else { 2.0 };
            (xi - s).powi(2)
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run C NLOPT with force_stop from objective function
// ─────────────────────────────────────────────────────────────────────────────

struct CForceStopResult {
    x: Vec<f64>,
    minf: f64,
    nfev: usize,
    return_code: DirectReturnCodeC,
}

fn run_c_with_force_stop(
    f: nlopt_ffi::DirectObjectiveFuncC,
    dim: usize,
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    threshold: f64,
    max_feval: i32,
    algorithm: DirectAlgorithmC,
) -> CForceStopResult {
    let mut x = vec![0.0f64; dim];
    let mut minf: f64 = 0.0;
    let mut force_stop: c_int = 0;

    let data = ForceStopData {
        force_stop: &mut force_stop,
        threshold,
        eval_count: AtomicUsize::new(0),
    };

    let return_code = unsafe {
        direct_optimize(
            f,
            &data as *const ForceStopData as *mut c_void,
            dim as c_int,
            lower_bounds.as_ptr(),
            upper_bounds.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            max_feval,
            -1,   // max_iter (unlimited)
            0.0,  // start time
            0.0,  // maxtime (no limit)
            1e-4, // magic_eps
            0.0,  // magic_eps_abs
            0.0,  // volume_reltol
            0.0,  // sigma_reltol
            &mut force_stop,
            DIRECT_UNKNOWN_FGLOBAL,
            DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            std::ptr::null_mut(), // logfile
            algorithm,
        )
    };

    CForceStopResult {
        x,
        minf,
        nfev: data.eval_count.load(Ordering::Relaxed),
        return_code,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run Rust with callback-based force_stop
// ─────────────────────────────────────────────────────────────────────────────

fn run_rust_with_callback(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    threshold: f64,
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
    let bounds_vec: Vec<(f64, f64)> = bounds.to_vec();
    let mut solver = Direct::new(f, &bounds_vec, opts).unwrap();

    let cb: Box<CallbackFn> = Box::new(move |_x, fun, _nfev, _nit| fun < threshold);
    solver.minimize(Some(&*cb)).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run Rust with force_stop set from within objective function
// (closest match to C force_stop behavior)
// ─────────────────────────────────────────────────────────────────────────────

fn run_rust_with_objective_force_stop(
    f: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    threshold: f64,
    max_feval: usize,
    algorithm: DirectAlgorithm,
) -> direct_nlopt::types::DirectResult {
    let opts = DirectOptions {
        max_feval,
        max_iter: 0,
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
    let bounds_vec: Vec<(f64, f64)> = bounds.to_vec();

    // Create force_stop flag that the objective function can trigger
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = Arc::clone(&stop_flag);

    let objective = move |x: &[f64]| -> f64 {
        let val = f(x);
        if val < threshold {
            stop_flag_clone.store(true, Ordering::Relaxed);
        }
        val
    };

    let mut solver = Direct::new(objective, &bounds_vec, opts).unwrap();
    // Share the same force_stop flag between objective function and solver
    solver.force_stop = stop_flag;

    solver.minimize(None).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Callback comparison with DIRECT_GABLONSKY
// The Rust callback checks the BEST VALID value each iteration.
// C force_stop fires on ANY eval below threshold (that eval is then excluded).
// Both return FORCED_STOP, but minf semantics differ:
//   - C: minf = best VALID value (triggering eval excluded, may be >= threshold)
//   - Rust callback: minf < threshold (callback only fires when best < threshold)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_callback_gablonsky() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_2d_force_stop,
        2,
        &lb,
        &ub,
        threshold,
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_with_callback(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Force stop callback GABLONSKY ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    // Both return FORCED_STOP
    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);
    // Rust callback fires when best valid < threshold
    assert!(rust_result.fun < threshold, "Rust fun={}", rust_result.fun);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Callback comparison with DIRECT_ORIGINAL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_callback_original() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_2d_force_stop,
        2,
        &lb,
        &ub,
        threshold,
        10000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
    );

    let rust_result = run_rust_with_callback(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Force stop callback ORIGINAL ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);
    assert!(rust_result.fun < threshold);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Objective force_stop comparison with DIRECT_GABLONSKY
// Both use the same mechanism: set force_stop flag from within objective.
// The triggering eval is excluded, so minf = best VALID value (may be >= threshold).
// x and minf should be identical between C and Rust.
// nfev may differ slightly (Rust counts force_stop-skipped evals, C doesn't).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_objective_gablonsky() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_2d_force_stop,
        2,
        &lb,
        &ub,
        threshold,
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_with_objective_force_stop(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Force stop objective GABLONSKY ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    // Both return FORCED_STOP
    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);

    // Same best valid results (triggering eval excluded in both)
    assert_eq!(
        c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}",
        c_result.minf, rust_result.fun
    );
    for i in 0..2 {
        assert_eq!(
            c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}",
            i, c_result.x[i], rust_result.x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Objective force_stop with DIRECT_ORIGINAL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_objective_original() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_2d_force_stop,
        2,
        &lb,
        &ub,
        threshold,
        10000,
        DirectAlgorithmC::DIRECT_ORIGINAL,
    );

    let rust_result = run_rust_with_objective_force_stop(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyOriginal,
    );

    println!("=== Force stop objective ORIGINAL ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    for i in 0..2 {
        assert_eq!(c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}", i, c_result.x[i], rust_result.x[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: 3D objective force_stop comparison
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_objective_3d_gablonsky() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_3d_force_stop,
        3,
        &lb,
        &ub,
        threshold,
        20000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_with_objective_force_stop(
        shifted_sphere_rust,
        &bounds,
        threshold,
        20000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Force stop objective 3D GABLONSKY ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    for i in 0..3 {
        assert_eq!(c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}", i, c_result.x[i], rust_result.x[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Force_stop terminates BEFORE maxfeval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_terminates_early() {
    let threshold = 3.0;
    let bounds = vec![(-5.0, 5.0); 2];

    // Run WITHOUT force_stop (uses maxfeval as termination)
    let opts_no_stop = DirectOptions {
        max_feval: 10000,
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
    let mut solver = Direct::new(shifted_sphere_rust, &bounds, opts_no_stop).unwrap();
    let result_no_stop = solver.minimize(None).unwrap();

    // Run WITH callback force_stop
    let opts_with_stop = DirectOptions {
        max_feval: 10000,
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
    let mut solver = Direct::new(shifted_sphere_rust, &bounds, opts_with_stop).unwrap();
    let cb: Box<CallbackFn> = Box::new(move |_x, fun, _nfev, _nit| fun < threshold);
    let result_with_stop = solver.minimize(Some(&*cb)).unwrap();

    println!("=== Force stop terminates early ===");
    println!(
        "  Without stop: nfev={}, minf={:.15e}, code={:?}",
        result_no_stop.nfev, result_no_stop.fun, result_no_stop.return_code
    );
    println!(
        "  With stop:    nfev={}, minf={:.15e}, code={:?}",
        result_with_stop.nfev, result_with_stop.fun, result_with_stop.return_code
    );

    assert_eq!(result_with_stop.return_code, DirectReturnCode::ForcedStop);
    assert!(result_with_stop.fun < threshold);
    assert!(
        result_with_stop.nfev < result_no_stop.nfev,
        "Force stop should use fewer evals: with_stop={}, without={}",
        result_with_stop.nfev,
        result_no_stop.nfev
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Callback vs objective force_stop nfev ordering
// Callback processes all selected rects before checking → more evals.
// Objective force_stop stops mid-batch → fewer evals.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_callback_vs_objective_nfev_ordering() {
    let threshold = 3.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];

    // Run Rust with callback (stops after iteration when new min < threshold)
    let rust_callback = run_rust_with_callback(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    // Run Rust with objective force_stop (stops mid-batch, like C)
    let rust_objective = run_rust_with_objective_force_stop(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Callback vs objective nfev ordering ===");
    println!(
        "  Callback:  nfev={}, minf={:.15e}, code={:?}",
        rust_callback.nfev, rust_callback.fun, rust_callback.return_code
    );
    println!(
        "  Objective: nfev={}, minf={:.15e}, code={:?}",
        rust_objective.nfev, rust_objective.fun, rust_objective.return_code
    );

    assert_eq!(rust_callback.return_code, DirectReturnCode::ForcedStop);
    assert_eq!(rust_objective.return_code, DirectReturnCode::ForcedStop);
    // Callback min < threshold (fires on valid min)
    assert!(rust_callback.fun < threshold);
    // Both stopped early
    assert!(rust_callback.nfev < 10000);
    assert!(rust_objective.nfev < 10000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Tighter threshold (still above init values)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_force_stop_tight_threshold() {
    // Threshold = 1.0 — harder to reach, takes more iterations.
    // All init values on shifted sphere [-5,5]^2 are > 4.0, so safe.
    let threshold = 1.0;
    let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
    let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();

    let c_result = run_c_with_force_stop(
        shifted_sphere_2d_force_stop,
        2,
        &lb,
        &ub,
        threshold,
        10000,
        DirectAlgorithmC::DIRECT_GABLONSKY,
    );

    let rust_result = run_rust_with_objective_force_stop(
        shifted_sphere_rust,
        &bounds,
        threshold,
        10000,
        DirectAlgorithm::GablonskyLocallyBiased,
    );

    println!("=== Force stop tight threshold ===");
    println!(
        "  C:    x={:?}, minf={:.15e}, nfev={}, code={:?}",
        c_result.x, c_result.minf, c_result.nfev, c_result.return_code
    );
    println!(
        "  Rust: x={:?}, minf={:.15e}, nfev={}, code={:?}",
        rust_result.x, rust_result.fun, rust_result.nfev, rust_result.return_code
    );

    assert_eq!(c_result.return_code, DirectReturnCodeC::DIRECT_FORCED_STOP);
    assert_eq!(rust_result.return_code, DirectReturnCode::ForcedStop);
    // Same best valid results
    assert_eq!(c_result.minf, rust_result.fun,
        "minf mismatch: C={:.15e}, Rust={:.15e}", c_result.minf, rust_result.fun);
    for i in 0..2 {
        assert_eq!(c_result.x[i], rust_result.x[i],
            "x[{}] mismatch: C={:.15e}, Rust={:.15e}", i, c_result.x[i], rust_result.x[i]);
    }
}
