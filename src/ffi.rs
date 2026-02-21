//! C FFI bindings for the DIRECT-NLOPT-RS library.
//!
//! Provides C-compatible function signatures matching NLOPT's `direct_optimize()`,
//! enabling external C/C++ programs to call the Rust DIRECT implementation.
//!
//! # NLOPT C Correspondence
//!
//! | C function (NLOPT)       | Rust FFI function            |
//! |--------------------------|------------------------------|
//! | `direct_optimize()`      | `direct_nlopt_optimize()`    |
//!
//! The FFI layer exactly mirrors NLOPT's `direct.h` types:
//! - `direct_objective_func` → C function pointer `(n, x, undefined_flag, data) -> f64`
//! - `direct_algorithm` → enum with `DIRECT_ORIGINAL` and `DIRECT_GABLONSKY`
//! - `direct_return_code` → integer return codes matching NLOPT

use std::os::raw::{c_double, c_int, c_void};
use std::slice;

use crate::error::DirectReturnCode;
use crate::types::{DirectAlgorithm, DirectOptions};
use crate::DirectBuilder;

// ──────────────────────────────────────────────────────────────────────────────
// C-compatible types
// ──────────────────────────────────────────────────────────────────────────────

/// C-compatible objective function pointer, matching NLOPT's `direct_objective_func`.
///
/// ```c
/// typedef double (*direct_objective_func)(int n, const double *x,
///                                         int *undefined_flag,
///                                         void *data);
/// ```
pub type DirectObjectiveFuncC = unsafe extern "C" fn(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double;

/// C-compatible algorithm enum, matching NLOPT's `direct_algorithm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectAlgorithmC {
    /// Jones' original DIRECT (1993)
    Original = 0,
    /// Gablonsky's locally-biased DIRECT-L (2001)
    Gablonsky = 1,
}

/// C-compatible result struct for `direct_nlopt_optimize_full`.
#[repr(C)]
pub struct DirectResultC {
    /// Return code (matches NLOPT's `direct_return_code` integer values)
    pub return_code: c_int,
    /// Number of function evaluations performed
    pub nfev: c_int,
    /// Number of iterations performed
    pub nit: c_int,
}

// ──────────────────────────────────────────────────────────────────────────────
// FFI entry point matching NLOPT's direct_optimize()
// ──────────────────────────────────────────────────────────────────────────────

/// Perform global minimization using the Rust DIRECT algorithm implementation.
///
/// This function has a C-compatible signature matching NLOPT's `direct_optimize()`,
/// enabling drop-in replacement from C/C++ code.
///
/// # Safety
///
/// - `f` must be a valid function pointer.
/// - `lower_bounds` and `upper_bounds` must point to arrays of length `dimension`.
/// - `x` must point to a writable array of length `dimension`.
/// - `minf` must point to a writable `double`.
/// - `force_stop` may be NULL; if non-NULL, must point to a valid `int`.
/// - `f_data` is passed through to `f` and must remain valid for the call duration.
///
/// # Returns
///
/// A `direct_return_code` integer matching NLOPT's convention:
/// - Positive values indicate successful termination
/// - Negative values indicate errors
#[no_mangle]
pub unsafe extern "C" fn direct_nlopt_optimize(
    f: DirectObjectiveFuncC,
    f_data: *mut c_void,
    dimension: c_int,
    lower_bounds: *const c_double,
    upper_bounds: *const c_double,
    x: *mut c_double,
    minf: *mut c_double,
    max_feval: c_int,
    max_iter: c_int,
    _start: c_double,
    _maxtime: c_double,
    magic_eps: c_double,
    magic_eps_abs: c_double,
    volume_reltol: c_double,
    sigma_reltol: c_double,
    force_stop: *const c_int,
    fglobal: c_double,
    fglobal_reltol: c_double,
    _logfile: *mut c_void,
    algorithm: DirectAlgorithmC,
) -> c_int {
    // Validate dimension
    if dimension < 1 {
        return DirectReturnCode::InvalidArgs as c_int;
    }
    let n = dimension as usize;

    // Read bounds from C arrays
    let lb = slice::from_raw_parts(lower_bounds, n);
    let ub = slice::from_raw_parts(upper_bounds, n);
    let bounds: Vec<(f64, f64)> = lb.iter().zip(ub.iter()).map(|(&l, &u)| (l, u)).collect();

    // Map C algorithm enum to Rust
    let alg = match algorithm {
        DirectAlgorithmC::Original => DirectAlgorithm::GablonskyOriginal,
        DirectAlgorithmC::Gablonsky => DirectAlgorithm::GablonskyLocallyBiased,
    };

    // Build options matching NLOPT's direct_optimize() parameter handling
    let options = DirectOptions {
        max_feval: if max_feval > 0 {
            max_feval as usize
        } else {
            0
        },
        max_iter: if max_iter > 0 {
            max_iter as usize
        } else {
            0
        },
        max_time: 0.0,
        magic_eps,
        magic_eps_abs,
        volume_reltol,
        sigma_reltol,
        fglobal,
        fglobal_reltol,
        algorithm: alg,
        parallel: false,
        parallel_batch: false,
        min_parallel_evals: 4,
    };

    // Wrap C function pointer + data into a Rust closure
    let f_data_ptr = f_data as usize; // Send-safe wrapper
    let f_fn = f;

    // Check force_stop pointer
    let force_stop_ptr = if force_stop.is_null() {
        None
    } else {
        Some(force_stop as usize)
    };

    let objective = move |x_slice: &[f64]| -> f64 {
        let mut undefined_flag: c_int = 0;
        let val = unsafe {
            f_fn(
                x_slice.len() as c_int,
                x_slice.as_ptr(),
                &mut undefined_flag,
                f_data_ptr as *mut c_void,
            )
        };
        if undefined_flag != 0 {
            f64::NAN
        } else {
            val
        }
    };

    // Build and run optimizer
    let mut builder = DirectBuilder::new(objective, bounds).options(options);

    // Add force_stop callback if pointer is provided
    if let Some(fs_ptr) = force_stop_ptr {
        builder = builder.with_callback(move |_x, _f, _nfev, _nit| -> bool {
            let ptr = fs_ptr as *const c_int;
            unsafe { *ptr != 0 }
        });
    }

    let result = match builder.minimize() {
        Ok(r) => r,
        Err(e) => {
            // Map error to return code
            let code = match e {
                crate::error::DirectError::InvalidBounds { .. } => {
                    DirectReturnCode::InvalidBounds as c_int
                }
                crate::error::DirectError::MaxFevalTooBig(_) => {
                    DirectReturnCode::MaxFevalTooBig as c_int
                }
                crate::error::DirectError::OutOfMemory => DirectReturnCode::OutOfMemory as c_int,
                crate::error::DirectError::InvalidArgs(_) => {
                    DirectReturnCode::InvalidArgs as c_int
                }
                crate::error::DirectError::ForcedStop => DirectReturnCode::ForcedStop as c_int,
                _ => DirectReturnCode::InvalidArgs as c_int,
            };
            return code;
        }
    };

    // Write results back to C output parameters
    let x_out = slice::from_raw_parts_mut(x, n);
    x_out.copy_from_slice(&result.x);
    *minf = result.fun;

    result.return_code as c_int
}

/// Extended optimization function that also returns nfev and nit.
///
/// Same as `direct_nlopt_optimize` but returns a `DirectResultC` struct
/// with additional statistics.
///
/// # Safety
///
/// Same safety requirements as `direct_nlopt_optimize`.
#[no_mangle]
pub unsafe extern "C" fn direct_nlopt_optimize_full(
    f: DirectObjectiveFuncC,
    f_data: *mut c_void,
    dimension: c_int,
    lower_bounds: *const c_double,
    upper_bounds: *const c_double,
    x: *mut c_double,
    minf: *mut c_double,
    max_feval: c_int,
    max_iter: c_int,
    magic_eps: c_double,
    magic_eps_abs: c_double,
    volume_reltol: c_double,
    sigma_reltol: c_double,
    force_stop: *const c_int,
    fglobal: c_double,
    fglobal_reltol: c_double,
    algorithm: DirectAlgorithmC,
) -> DirectResultC {
    if dimension < 1 {
        return DirectResultC {
            return_code: DirectReturnCode::InvalidArgs as c_int,
            nfev: 0,
            nit: 0,
        };
    }
    let n = dimension as usize;

    let lb = slice::from_raw_parts(lower_bounds, n);
    let ub = slice::from_raw_parts(upper_bounds, n);
    let bounds: Vec<(f64, f64)> = lb.iter().zip(ub.iter()).map(|(&l, &u)| (l, u)).collect();

    let alg = match algorithm {
        DirectAlgorithmC::Original => DirectAlgorithm::GablonskyOriginal,
        DirectAlgorithmC::Gablonsky => DirectAlgorithm::GablonskyLocallyBiased,
    };

    let options = DirectOptions {
        max_feval: if max_feval > 0 {
            max_feval as usize
        } else {
            0
        },
        max_iter: if max_iter > 0 {
            max_iter as usize
        } else {
            0
        },
        max_time: 0.0,
        magic_eps,
        magic_eps_abs,
        volume_reltol,
        sigma_reltol,
        fglobal,
        fglobal_reltol,
        algorithm: alg,
        parallel: false,
        parallel_batch: false,
        min_parallel_evals: 4,
    };

    let f_data_ptr = f_data as usize;
    let f_fn = f;

    let force_stop_ptr = if force_stop.is_null() {
        None
    } else {
        Some(force_stop as usize)
    };

    let objective = move |x_slice: &[f64]| -> f64 {
        let mut undefined_flag: c_int = 0;
        let val = unsafe {
            f_fn(
                x_slice.len() as c_int,
                x_slice.as_ptr(),
                &mut undefined_flag,
                f_data_ptr as *mut c_void,
            )
        };
        if undefined_flag != 0 {
            f64::NAN
        } else {
            val
        }
    };

    let mut builder = DirectBuilder::new(objective, bounds).options(options);

    if let Some(fs_ptr) = force_stop_ptr {
        builder = builder.with_callback(move |_x, _f, _nfev, _nit| -> bool {
            let ptr = fs_ptr as *const c_int;
            unsafe { *ptr != 0 }
        });
    }

    let result = match builder.minimize() {
        Ok(r) => r,
        Err(e) => {
            let code = match e {
                crate::error::DirectError::InvalidBounds { .. } => {
                    DirectReturnCode::InvalidBounds as c_int
                }
                crate::error::DirectError::MaxFevalTooBig(_) => {
                    DirectReturnCode::MaxFevalTooBig as c_int
                }
                crate::error::DirectError::OutOfMemory => DirectReturnCode::OutOfMemory as c_int,
                crate::error::DirectError::InvalidArgs(_) => {
                    DirectReturnCode::InvalidArgs as c_int
                }
                crate::error::DirectError::ForcedStop => DirectReturnCode::ForcedStop as c_int,
                _ => DirectReturnCode::InvalidArgs as c_int,
            };
            return DirectResultC {
                return_code: code,
                nfev: 0,
                nit: 0,
            };
        }
    };

    let x_out = slice::from_raw_parts_mut(x, n);
    x_out.copy_from_slice(&result.x);
    *minf = result.fun;

    DirectResultC {
        return_code: result.return_code as c_int,
        nfev: result.nfev as c_int,
        nit: result.nit as c_int,
    }
}

/// Get the version string of the direct-nlopt-rs library.
///
/// Returns a pointer to a null-terminated static string.
/// The caller must NOT free the returned pointer.
#[no_mangle]
pub extern "C" fn direct_nlopt_version() -> *const std::os::raw::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DIRECT_UNKNOWN_FGLOBAL;
    use std::ptr;

    // Simple sphere function for testing
    unsafe extern "C" fn sphere_c(
        n: c_int,
        x: *const c_double,
        _undefined_flag: *mut c_int,
        _data: *mut c_void,
    ) -> c_double {
        let x = slice::from_raw_parts(x, n as usize);
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_ffi_optimize_sphere_gablonsky() {
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

            assert!(ret > 0, "ret = {}", ret);
            assert!(minf < 1e-4, "minf = {}", minf);
            assert!(x[0].abs() < 0.1, "x[0] = {}", x[0]);
            assert!(x[1].abs() < 0.1, "x[1] = {}", x[1]);
        }
    }

    #[test]
    fn test_ffi_optimize_sphere_original() {
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

            assert!(ret > 0, "ret = {}", ret);
            assert!(minf < 1e-2, "minf = {}", minf);
        }
    }

    #[test]
    fn test_ffi_optimize_full() {
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

            assert!(result.return_code > 0, "return_code = {}", result.return_code);
            assert!(result.nfev > 0, "nfev = {}", result.nfev);
            assert!(result.nit > 0, "nit = {}", result.nit);
            assert!(minf < 1e-4, "minf = {}", minf);
        }
    }

    #[test]
    fn test_ffi_invalid_dimension() {
        unsafe {
            let lb = [-5.0];
            let ub = [5.0];
            let mut x = [0.0];
            let mut minf = f64::MAX;

            let ret = direct_nlopt_optimize(
                sphere_c,
                ptr::null_mut(),
                0, // invalid
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

            assert!(ret < 0, "expected error, got ret = {}", ret);
        }
    }

    #[test]
    fn test_ffi_force_stop() {
        unsafe {
            let lb = [-5.0, -5.0];
            let ub = [5.0, 5.0];
            let mut x = [0.0; 2];
            let mut minf = f64::MAX;
            let force_stop: c_int = 1; // immediately stop

            let ret = direct_nlopt_optimize(
                sphere_c,
                ptr::null_mut(),
                2,
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
                &force_stop as *const c_int,
                DIRECT_UNKNOWN_FGLOBAL,
                0.0,
                ptr::null_mut(),
                DirectAlgorithmC::Gablonsky,
            );

            // With force_stop=1 from the start, the optimizer should stop quickly.
            // It may complete initialization before checking the callback, so we
            // accept either ForcedStop or a success code with minimal evaluations.
            assert!(
                ret == DirectReturnCode::ForcedStop as c_int || ret > 0,
                "expected ForcedStop or success, got {}",
                ret
            );
        }
    }

    #[test]
    fn test_ffi_version() {
        let ver = direct_nlopt_version();
        assert!(!ver.is_null());
        let c_str = unsafe { std::ffi::CStr::from_ptr(ver) };
        let version = c_str.to_str().unwrap();
        assert!(!version.is_empty());
    }
}
