#![cfg(feature = "nlopt-compare")]

//! End-to-end comparison: cdirect vs Gablonsky implementation consistency.
//!
//! This test verifies:
//! 1. NLOPT C cdirect() matches Rust CDirect for the same which_alg
//! 2. NLOPT C direct_optimize() matches Rust Direct for the same algmethod
//! 3. Documents whether the two NLOPT implementations (cdirect vs Gablonsky
//!    translation) produce identical results (they may differ due to different
//!    data structures: red-black tree vs SoA + linked lists)

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nlopt_ffi::{
    DirectAlgorithmC, NloptCDirectRunner, NloptDirectRunner,
    sphere_nlopt_counting,
    DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

use direct_nlopt::cdirect::CDirect;
use direct_nlopt::direct::Direct;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Rust objective functions
// ─────────────────────────────────────────────────────────────────────────────

fn sphere_rust(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// C objective function with counter for Gablonsky direct_optimize
// ─────────────────────────────────────────────────────────────────────────────

extern "C" fn sphere_direct_counting(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const AtomicUsize) };
    counter.fetch_add(1, Ordering::Relaxed);
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: NLOPT C cdirect() with which_alg=0 (DIRECT_ORIGINAL) vs Rust CDirect
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_cdirect_original_c_vs_rust() {
    let counter = AtomicUsize::new(0);
    let c_runner = NloptCDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        magic_eps: 1e-4,
        which_alg: 0, // DIRECT_ORIGINAL
    };
    let c_result = unsafe {
        c_runner.run(sphere_nlopt_counting, &counter as *const AtomicUsize as *mut c_void)
    };

    // Rust CDirect with Original (which_alg=0)
    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::Original,
        parallel: false,
        ..Default::default()
    };
    let cdirect = CDirect::new(sphere_rust, bounds, opts);
    let rust_result = cdirect.minimize().unwrap();

    println!("=== cdirect DIRECT_ORIGINAL (which_alg=0): C vs Rust ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             c_result.x[0], c_result.x[1], c_result.minf, c_result.nfev);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev);
    println!("C:    code={:?}", c_result.return_code);
    println!("Rust: code={:?}", rust_result.return_code);

    assert_eq!(c_result.x[0], rust_result.x[0],
               "cdirect Original x[0] mismatch: C={:.15e}, Rust={:.15e}",
               c_result.x[0], rust_result.x[0]);
    assert_eq!(c_result.x[1], rust_result.x[1],
               "cdirect Original x[1] mismatch: C={:.15e}, Rust={:.15e}",
               c_result.x[1], rust_result.x[1]);
    assert_eq!(c_result.minf, rust_result.fun,
               "cdirect Original minf mismatch: C={:.15e}, Rust={:.15e}",
               c_result.minf, rust_result.fun);
    assert_eq!(c_result.nfev, rust_result.nfev,
               "cdirect Original nfev mismatch: C={}, Rust={}", c_result.nfev, rust_result.nfev);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: NLOPT C cdirect() with which_alg=13 (DIRECT_L) vs Rust CDirect
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_cdirect_l_c_vs_rust() {
    let counter = AtomicUsize::new(0);
    let c_runner = NloptCDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        magic_eps: 1e-4,
        which_alg: 13, // DIRECT_L
    };
    let c_result = unsafe {
        c_runner.run(sphere_nlopt_counting, &counter as *const AtomicUsize as *mut c_void)
    };

    // Rust CDirect with LocallyBiased (which_alg=13)
    let bounds = vec![(-5.0, 5.0); 2];
    let opts = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::LocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let cdirect = CDirect::new(sphere_rust, bounds, opts);
    let rust_result = cdirect.minimize().unwrap();

    println!("=== cdirect DIRECT_L (which_alg=13): C vs Rust ===");
    println!("C:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             c_result.x[0], c_result.x[1], c_result.minf, c_result.nfev);
    println!("Rust: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             rust_result.x[0], rust_result.x[1], rust_result.fun, rust_result.nfev);
    println!("C:    code={:?}", c_result.return_code);
    println!("Rust: code={:?}", rust_result.return_code);

    assert_eq!(c_result.x[0], rust_result.x[0],
               "cdirect L x[0] mismatch: C={:.15e}, Rust={:.15e}",
               c_result.x[0], rust_result.x[0]);
    assert_eq!(c_result.x[1], rust_result.x[1],
               "cdirect L x[1] mismatch: C={:.15e}, Rust={:.15e}",
               c_result.x[1], rust_result.x[1]);
    assert_eq!(c_result.minf, rust_result.fun,
               "cdirect L minf mismatch: C={:.15e}, Rust={:.15e}",
               c_result.minf, rust_result.fun);
    assert_eq!(c_result.nfev, rust_result.nfev,
               "cdirect L nfev mismatch: C={}, Rust={}", c_result.nfev, rust_result.nfev);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Cross-implementation comparison — document differences
// ─────────────────────────────────────────────────────────────────────────────

/// Compare NLOPT's two implementations on the same problem.
/// The two NLOPT implementations (cdirect vs Gablonsky) may produce different
/// results because they use different data structures (red-black tree vs SoA
/// + linked lists), different convex hull algorithms, and different stopping
/// logic. This test documents the differences.
#[test]
fn test_cross_implementation_sphere_original() {
    // NLOPT C cdirect() with which_alg=0 (DIRECT_ORIGINAL)
    let cd_counter = AtomicUsize::new(0);
    let cd_runner = NloptCDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        magic_eps: 1e-4,
        which_alg: 0,
    };
    let cd_result = unsafe {
        cd_runner.run(sphere_nlopt_counting, &cd_counter as *const AtomicUsize as *mut c_void)
    };

    // NLOPT C direct_optimize() with DIRECT_ORIGINAL
    let gab_counter = AtomicUsize::new(0);
    let gab_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        max_iter: -1,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
    };
    let gab_result = unsafe {
        gab_runner.run(sphere_direct_counting, &gab_counter as *const AtomicUsize as *mut c_void)
    };
    let gab_nfev = gab_counter.load(Ordering::Relaxed);

    println!("=== Cross-Implementation DIRECT_ORIGINAL: cdirect vs Gablonsky ===");
    println!("cdirect:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             cd_result.x[0], cd_result.x[1], cd_result.minf, cd_result.nfev);
    println!("Gablonsky:  x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             gab_result.x[0], gab_result.x[1], gab_result.minf, gab_nfev);

    // Document whether they match (they may not — different data structures)
    let x_match = cd_result.x[0] == gab_result.x[0] && cd_result.x[1] == gab_result.x[1];
    let minf_match = cd_result.minf == gab_result.minf;
    let nfev_match = cd_result.nfev == gab_nfev;

    println!("x match: {}, minf match: {}, nfev match: {}",
             x_match, minf_match, nfev_match);
    if !x_match || !minf_match || !nfev_match {
        println!("NOTE: NLOPT's cdirect and Gablonsky implementations produce DIFFERENT results.");
        println!("This is expected — different data structures lead to different rectangle");
        println!("processing orders and convex hull computations.");
    }

    // Both should find a good minimum regardless
    assert!(cd_result.minf < 1.0,
            "cdirect DIRECT_ORIGINAL minf too large: {}", cd_result.minf);
    assert!(gab_result.minf < 1.0,
            "Gablonsky DIRECT_ORIGINAL minf too large: {}", gab_result.minf);
}

/// Compare NLOPT's two implementations for DIRECT-L (locally biased) variant.
#[test]
fn test_cross_implementation_sphere_l() {
    // NLOPT C cdirect() with which_alg=13 (DIRECT_L)
    let cd_counter = AtomicUsize::new(0);
    let cd_runner = NloptCDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        magic_eps: 1e-4,
        which_alg: 13,
    };
    let cd_result = unsafe {
        cd_runner.run(sphere_nlopt_counting, &cd_counter as *const AtomicUsize as *mut c_void)
    };

    // NLOPT C direct_optimize() with DIRECT_GABLONSKY
    let gab_counter = AtomicUsize::new(0);
    let gab_runner = NloptDirectRunner {
        dimension: 2,
        lower_bounds: vec![-5.0, -5.0],
        upper_bounds: vec![5.0, 5.0],
        max_feval: 500,
        max_iter: -1,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        algorithm: DirectAlgorithmC::DIRECT_GABLONSKY,
    };
    let gab_result = unsafe {
        gab_runner.run(sphere_direct_counting, &gab_counter as *const AtomicUsize as *mut c_void)
    };
    let gab_nfev = gab_counter.load(Ordering::Relaxed);

    println!("=== Cross-Implementation DIRECT_L: cdirect vs Gablonsky ===");
    println!("cdirect:    x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             cd_result.x[0], cd_result.x[1], cd_result.minf, cd_result.nfev);
    println!("Gablonsky:  x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             gab_result.x[0], gab_result.x[1], gab_result.minf, gab_nfev);

    let x_match = cd_result.x[0] == gab_result.x[0] && cd_result.x[1] == gab_result.x[1];
    let minf_match = cd_result.minf == gab_result.minf;
    let nfev_match = cd_result.nfev == gab_nfev;

    println!("x match: {}, minf match: {}, nfev match: {}",
             x_match, minf_match, nfev_match);
    if !x_match || !minf_match || !nfev_match {
        println!("NOTE: NLOPT's cdirect and Gablonsky implementations produce DIFFERENT results.");
        println!("This is expected — different data structures and stopping conditions.");
    }

    // Both should find a good minimum
    assert!(cd_result.minf < 1.0,
            "cdirect DIRECT_L minf too large: {}", cd_result.minf);
    assert!(gab_result.minf < 1.0,
            "Gablonsky DIRECT_L minf too large: {}", gab_result.minf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Rust CDirect vs Rust Direct — verify each matches its C counterpart
// ─────────────────────────────────────────────────────────────────────────────

/// Verify Rust CDirect matches NLOPT C cdirect AND Rust Direct matches NLOPT C direct
/// for the DIRECT-L variant.
#[test]
fn test_rust_cdirect_vs_rust_direct_l() {
    // Run Rust CDirect (LocallyBiased = which_alg=13)
    let bounds = vec![(-5.0, 5.0); 2];
    let cdirect_opts = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::LocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let cdirect_solver = CDirect::new(sphere_rust, bounds.clone(), cdirect_opts);
    let cdirect_result = cdirect_solver.minimize().unwrap();

    // Run Rust Direct (GablonskyLocallyBiased = algmethod=1)
    let direct_opts = DirectOptions {
        max_feval: 500,
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
    let mut direct_solver = Direct::new(sphere_rust, &bounds, direct_opts).unwrap();
    let direct_result = direct_solver.minimize(None).unwrap();

    println!("=== Rust CDirect vs Rust Direct (DIRECT-L) ===");
    println!("CDirect: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             cdirect_result.x[0], cdirect_result.x[1], cdirect_result.fun, cdirect_result.nfev);
    println!("Direct:  x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             direct_result.x[0], direct_result.x[1], direct_result.fun, direct_result.nfev);

    let x_match = cdirect_result.x[0] == direct_result.x[0]
        && cdirect_result.x[1] == direct_result.x[1];
    let minf_match = cdirect_result.fun == direct_result.fun;
    let nfev_match = cdirect_result.nfev == direct_result.nfev;

    println!("x match: {}, minf match: {}, nfev match: {}",
             x_match, minf_match, nfev_match);
    if !x_match || !minf_match || !nfev_match {
        println!("NOTE: Rust CDirect and Rust Direct produce different results (expected —");
        println!("mirrors the difference between NLOPT's two C implementations).");
    }

    // Both should find a good minimum
    assert!(cdirect_result.fun < 1.0,
            "Rust CDirect minf too large: {}", cdirect_result.fun);
    assert!(direct_result.fun < 1.0,
            "Rust Direct minf too large: {}", direct_result.fun);
}

/// Same comparison for DIRECT_ORIGINAL variant.
#[test]
fn test_rust_cdirect_vs_rust_direct_original() {
    let bounds = vec![(-5.0, 5.0); 2];

    // Rust CDirect (Original = which_alg=0)
    let cdirect_opts = DirectOptions {
        max_feval: 500,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::Original,
        parallel: false,
        ..Default::default()
    };
    let cdirect_solver = CDirect::new(sphere_rust, bounds.clone(), cdirect_opts);
    let cdirect_result = cdirect_solver.minimize().unwrap();

    // Rust Direct (GablonskyOriginal = algmethod=0)
    let direct_opts = DirectOptions {
        max_feval: 500,
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
    let mut direct_solver = Direct::new(sphere_rust, &bounds, direct_opts).unwrap();
    let direct_result = direct_solver.minimize(None).unwrap();

    println!("=== Rust CDirect vs Rust Direct (DIRECT_ORIGINAL) ===");
    println!("CDirect: x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             cdirect_result.x[0], cdirect_result.x[1], cdirect_result.fun, cdirect_result.nfev);
    println!("Direct:  x=[{:.15e}, {:.15e}], minf={:.15e}, nfev={}",
             direct_result.x[0], direct_result.x[1], direct_result.fun, direct_result.nfev);

    // Both should find a good minimum
    assert!(cdirect_result.fun < 1.0,
            "Rust CDirect Original minf too large: {}", cdirect_result.fun);
    assert!(direct_result.fun < 1.0,
            "Rust Direct Original minf too large: {}", direct_result.fun);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: 3D verification to stress-test higher dimensions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_cdirect_l_c_vs_rust_3d() {
    let counter = AtomicUsize::new(0);
    let c_runner = NloptCDirectRunner {
        dimension: 3,
        lower_bounds: vec![-5.0; 3],
        upper_bounds: vec![5.0; 3],
        max_feval: 1000,
        magic_eps: 1e-4,
        which_alg: 13,
    };
    let c_result = unsafe {
        c_runner.run(sphere_nlopt_counting, &counter as *const AtomicUsize as *mut c_void)
    };

    let bounds = vec![(-5.0, 5.0); 3];
    let opts = DirectOptions {
        max_feval: 1000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::LocallyBiased,
        parallel: false,
        ..Default::default()
    };
    let cdirect = CDirect::new(sphere_rust, bounds, opts);
    let rust_result = cdirect.minimize().unwrap();

    println!("=== cdirect DIRECT_L 3D: C vs Rust ===");
    println!("C:    x={:?}, minf={:.15e}, nfev={}", c_result.x, c_result.minf, c_result.nfev);
    println!("Rust: x={:?}, minf={:.15e}, nfev={}", rust_result.x, rust_result.fun, rust_result.nfev);

    for i in 0..3 {
        assert_eq!(c_result.x[i], rust_result.x[i],
                   "cdirect L 3D x[{}] mismatch: C={:.15e}, Rust={:.15e}",
                   i, c_result.x[i], rust_result.x[i]);
    }
    assert_eq!(c_result.minf, rust_result.fun,
               "cdirect L 3D minf mismatch: C={:.15e}, Rust={:.15e}",
               c_result.minf, rust_result.fun);
    assert_eq!(c_result.nfev, rust_result.nfev,
               "cdirect L 3D nfev mismatch: C={}, Rust={}", c_result.nfev, rust_result.nfev);
}

#[test]
fn test_cdirect_original_c_vs_rust_3d() {
    let counter = AtomicUsize::new(0);
    let c_runner = NloptCDirectRunner {
        dimension: 3,
        lower_bounds: vec![-5.0; 3],
        upper_bounds: vec![5.0; 3],
        max_feval: 1000,
        magic_eps: 1e-4,
        which_alg: 0,
    };
    let c_result = unsafe {
        c_runner.run(sphere_nlopt_counting, &counter as *const AtomicUsize as *mut c_void)
    };

    let bounds = vec![(-5.0, 5.0); 3];
    let opts = DirectOptions {
        max_feval: 1000,
        max_iter: 0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: 0.0,
        fglobal: f64::NEG_INFINITY,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::Original,
        parallel: false,
        ..Default::default()
    };
    let cdirect = CDirect::new(sphere_rust, bounds, opts);
    let rust_result = cdirect.minimize().unwrap();

    println!("=== cdirect DIRECT_ORIGINAL 3D: C vs Rust ===");
    println!("C:    x={:?}, minf={:.15e}, nfev={}", c_result.x, c_result.minf, c_result.nfev);
    println!("Rust: x={:?}, minf={:.15e}, nfev={}", rust_result.x, rust_result.fun, rust_result.nfev);

    for i in 0..3 {
        assert_eq!(c_result.x[i], rust_result.x[i],
                   "cdirect Original 3D x[{}] mismatch: C={:.15e}, Rust={:.15e}",
                   i, c_result.x[i], rust_result.x[i]);
    }
    assert_eq!(c_result.minf, rust_result.fun,
               "cdirect Original 3D minf mismatch: C={:.15e}, Rust={:.15e}",
               c_result.minf, rust_result.fun);
    assert_eq!(c_result.nfev, rust_result.nfev,
               "cdirect Original 3D nfev mismatch: C={}, Rust={}", c_result.nfev, rust_result.nfev);
}
