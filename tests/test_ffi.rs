//! Integration tests for the C FFI interface.
//!
//! Tests that the Rust FFI layer produces correct results when called via
//! C-compatible function signatures, and verifies consistency with NLOPT C
//! when compiled with the `nlopt-compare` feature.

use std::os::raw::{c_double, c_int, c_void};
use std::ptr;

use direct_nlopt::ffi::*;
use direct_nlopt::DIRECT_UNKNOWN_FGLOBAL;

// ──────────────────────────────────────────────────────────────────────────────
// Test objective functions with C-compatible signatures
// ──────────────────────────────────────────────────────────────────────────────

unsafe extern "C" fn sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let x = std::slice::from_raw_parts(x, n as usize);
    x.iter().map(|xi| xi * xi).sum()
}

unsafe extern "C" fn rosenbrock_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let x = std::slice::from_raw_parts(x, n as usize);
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
}

unsafe extern "C" fn counting_sphere_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double {
    let counter = &mut *(data as *mut i32);
    *counter += 1;
    let x = std::slice::from_raw_parts(x, n as usize);
    x.iter().map(|xi| xi * xi).sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// FFI tests — calling Rust via C-compatible interface
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ffi_sphere_gablonsky_2d() {
    unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;

        let ret = direct_nlopt_optimize(
            sphere_c,
            ptr::null_mut(),
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            500,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Gablonsky,
        );

        assert!(ret > 0, "expected success, got ret={}", ret);
        assert!(minf < 1e-4, "minf={}", minf);
        assert!(x[0].abs() < 0.1, "x[0]={}", x[0]);
        assert!(x[1].abs() < 0.1, "x[1]={}", x[1]);
    }
}

#[test]
fn test_ffi_sphere_original_2d() {
    unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;

        let ret = direct_nlopt_optimize(
            sphere_c,
            ptr::null_mut(),
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            500,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Original,
        );

        assert!(ret > 0, "expected success, got ret={}", ret);
        assert!(minf < 1e-2, "minf={}", minf);
    }
}

#[test]
fn test_ffi_rosenbrock_2d() {
    unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;

        let ret = direct_nlopt_optimize(
            rosenbrock_c,
            ptr::null_mut(),
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            2000,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Gablonsky,
        );

        assert!(ret > 0, "expected success, got ret={}", ret);
        assert!(minf < 5.0, "rosenbrock minf={}", minf);
    }
}

#[test]
fn test_ffi_optimize_full_returns_stats() {
    unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;

        let result = direct_nlopt_optimize_full(
            sphere_c,
            ptr::null_mut(),
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            500,
            0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            DirectAlgorithmC::Gablonsky,
        );

        assert!(result.return_code > 0, "return_code={}", result.return_code);
        assert!(result.nfev > 0, "nfev={}", result.nfev);
        assert!(result.nit > 0, "nit={}", result.nit);
        assert!(minf < 1e-4, "minf={}", minf);
    }
}

#[test]
fn test_ffi_user_data_passthrough() {
    unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;
        let mut counter: i32 = 0;

        let ret = direct_nlopt_optimize(
            counting_sphere_c,
            &mut counter as *mut i32 as *mut c_void,
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            200,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Gablonsky,
        );

        assert!(ret > 0, "expected success, got ret={}", ret);
        assert!(counter > 0, "counter should be incremented");
        // With maxfeval=200, we should get at least 5 (init) and roughly 200 evaluations
        // (may slightly overshoot due to batch processing in the Gablonsky translation)
        assert!(
            counter >= 5 && counter <= 250,
            "counter={} should be in [5, 250]",
            counter
        );
    }
}

#[test]
fn test_ffi_5d_sphere() {
    unsafe {
        let lb = [-5.0, -5.0, -5.0, -5.0, -5.0];
        let ub = [5.0, 5.0, 5.0, 5.0, 5.0];
        let mut x = [0.0; 5];
        let mut minf = f64::MAX;

        let ret = direct_nlopt_optimize(
            sphere_c,
            ptr::null_mut(),
            5,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            5000,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Gablonsky,
        );

        assert!(ret > 0, "expected success, got ret={}", ret);
        assert!(minf < 1.0, "5D sphere minf={}", minf);
    }
}

#[test]
fn test_ffi_invalid_dim_zero() {
    unsafe {
        let lb = [-5.0];
        let ub = [5.0];
        let mut x = [0.0];
        let mut minf = f64::MAX;

        let ret = direct_nlopt_optimize(
            sphere_c,
            ptr::null_mut(),
            0,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            500,
            0,
            0.0,
            0.0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            ptr::null_mut(),
            DirectAlgorithmC::Gablonsky,
        );

        assert!(ret < 0, "expected error for dim=0, got ret={}", ret);
    }
}

#[test]
fn test_ffi_version_string() {
    let ver = direct_nlopt_version();
    assert!(!ver.is_null());
    let c_str = unsafe { std::ffi::CStr::from_ptr(ver) };
    let version = c_str.to_str().unwrap();
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
}

// ──────────────────────────────────────────────────────────────────────────────
// Comparison tests: FFI result consistency with Rust API
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ffi_matches_rust_api_sphere() {
    // Run via FFI
    let (ffi_x, ffi_minf, ffi_nfev) = unsafe {
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let mut x = [0.0; 2];
        let mut minf = f64::MAX;

        let result = direct_nlopt_optimize_full(
            sphere_c,
            ptr::null_mut(),
            2,
            lb.as_ptr(),
            ub.as_ptr(),
            x.as_mut_ptr(),
            &mut minf,
            500,
            0,
            1e-4,
            0.0,
            0.0,
            -1.0,
            ptr::null(),
            DIRECT_UNKNOWN_FGLOBAL,
            0.0,
            DirectAlgorithmC::Gablonsky,
        );

        assert!(result.return_code > 0);
        (x.to_vec(), minf, result.nfev)
    };

    // Run via Rust API with identical parameters
    use direct_nlopt::{direct_optimize, DirectAlgorithm, DirectOptions};
    let rust_result = direct_optimize(
        |x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
        &vec![(-5.0, 5.0), (-5.0, 5.0)],
        DirectOptions {
            max_feval: 500,
            magic_eps: 1e-4,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        },
    )
    .unwrap();

    // FFI and Rust API should produce identical results
    assert_eq!(ffi_x[0], rust_result.x[0], "x[0] mismatch");
    assert_eq!(ffi_x[1], rust_result.x[1], "x[1] mismatch");
    assert_eq!(ffi_minf, rust_result.fun, "minf mismatch");
    assert_eq!(
        ffi_nfev as usize, rust_result.nfev,
        "nfev mismatch: ffi={} rust={}",
        ffi_nfev, rust_result.nfev
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Comparison with NLOPT C (when nlopt-compare feature is enabled)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "nlopt-compare")]
mod nlopt_compare {
    use super::*;

    extern "C" {
        fn direct_optimize(
            f: unsafe extern "C" fn(c_int, *const c_double, *mut c_int, *mut c_void) -> c_double,
            f_data: *mut c_void,
            dimension: c_int,
            lower_bounds: *const c_double,
            upper_bounds: *const c_double,
            x: *mut c_double,
            minf: *mut c_double,
            max_feval: c_int,
            max_iter: c_int,
            start: c_double,
            maxtime: c_double,
            magic_eps: c_double,
            magic_eps_abs: c_double,
            volume_reltol: c_double,
            sigma_reltol: c_double,
            force_stop: *mut c_int,
            fglobal: c_double,
            fglobal_reltol: c_double,
            logfile: *mut c_void,
            algorithm: c_int,
        ) -> c_int;
    }

    #[test]
    fn test_ffi_matches_nlopt_c_sphere_gablonsky() {
        unsafe {
            // Run NLOPT C
            let lb = [-5.0, -5.0];
            let ub = [5.0, 5.0];
            let mut c_x = [0.0; 2];
            let mut c_minf = f64::MAX;

            let c_ret = direct_optimize(
                sphere_c,
                ptr::null_mut(),
                2,
                lb.as_ptr(),
                ub.as_ptr(),
                c_x.as_mut_ptr(),
                &mut c_minf,
                500,
                0,
                0.0,
                0.0,
                1e-4,
                0.0,
                0.0,
                -1.0,
                ptr::null_mut(),
                -f64::MAX,
                0.0,
                ptr::null_mut(),
                1, // DIRECT_GABLONSKY
            );

            // Run Rust FFI
            let mut rs_x = [0.0; 2];
            let mut rs_minf = f64::MAX;

            let rs_ret = direct_nlopt_optimize(
                sphere_c,
                ptr::null_mut(),
                2,
                lb.as_ptr(),
                ub.as_ptr(),
                rs_x.as_mut_ptr(),
                &mut rs_minf,
                500,
                0,
                0.0,
                0.0,
                1e-4,
                0.0,
                0.0,
                -1.0,
                ptr::null(),
                DIRECT_UNKNOWN_FGLOBAL,
                0.0,
                ptr::null_mut(),
                DirectAlgorithmC::Gablonsky,
            );

            assert_eq!(c_ret, rs_ret, "return codes differ: C={} Rust={}", c_ret, rs_ret);
            assert_eq!(c_minf, rs_minf, "minf differs: C={} Rust={}", c_minf, rs_minf);
            assert_eq!(c_x[0], rs_x[0], "x[0] differs");
            assert_eq!(c_x[1], rs_x[1], "x[1] differs");
        }
    }

    #[test]
    fn test_ffi_matches_nlopt_c_sphere_original() {
        unsafe {
            let lb = [-5.0, -5.0];
            let ub = [5.0, 5.0];
            let mut c_x = [0.0; 2];
            let mut c_minf = f64::MAX;

            let c_ret = direct_optimize(
                sphere_c,
                ptr::null_mut(),
                2,
                lb.as_ptr(),
                ub.as_ptr(),
                c_x.as_mut_ptr(),
                &mut c_minf,
                500,
                0,
                0.0,
                0.0,
                1e-4,
                0.0,
                0.0,
                -1.0,
                ptr::null_mut(),
                -f64::MAX,
                0.0,
                ptr::null_mut(),
                0, // DIRECT_ORIGINAL
            );

            let mut rs_x = [0.0; 2];
            let mut rs_minf = f64::MAX;

            let rs_ret = direct_nlopt_optimize(
                sphere_c,
                ptr::null_mut(),
                2,
                lb.as_ptr(),
                ub.as_ptr(),
                rs_x.as_mut_ptr(),
                &mut rs_minf,
                500,
                0,
                0.0,
                0.0,
                1e-4,
                0.0,
                0.0,
                -1.0,
                ptr::null(),
                DIRECT_UNKNOWN_FGLOBAL,
                0.0,
                ptr::null_mut(),
                DirectAlgorithmC::Original,
            );

            assert_eq!(c_ret, rs_ret, "return codes differ: C={} Rust={}", c_ret, rs_ret);
            assert_eq!(c_minf, rs_minf, "minf differs: C={} Rust={}", c_minf, rs_minf);
            assert_eq!(c_x[0], rs_x[0], "x[0] differs");
            assert_eq!(c_x[1], rs_x[1], "x[1] differs");
        }
    }
}
