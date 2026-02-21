#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types, dead_code)]

//! FFI bindings for the NLOPT DIRECT C implementation.
//! Feature-gated behind "nlopt-compare" for comparison testing.

use std::os::raw::{c_double, c_int, c_void};

/// NLOPT's direct_algorithm enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectAlgorithmC {
    DIRECT_ORIGINAL = 0,
    DIRECT_GABLONSKY = 1,
}

/// NLOPT's direct_return_code enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectReturnCodeC {
    DIRECT_INVALID_BOUNDS = -1,
    DIRECT_MAXFEVAL_TOOBIG = -2,
    DIRECT_INIT_FAILED = -3,
    DIRECT_SAMPLEPOINTS_FAILED = -4,
    DIRECT_SAMPLE_FAILED = -5,
    DIRECT_MAXFEVAL_EXCEEDED = 1,
    DIRECT_MAXITER_EXCEEDED = 2,
    DIRECT_GLOBAL_FOUND = 3,
    DIRECT_VOLTOL = 4,
    DIRECT_SIGMATOL = 5,
    DIRECT_MAXTIME_EXCEEDED = 6,
    DIRECT_OUT_OF_MEMORY = -100,
    DIRECT_INVALID_ARGS = -101,
    DIRECT_FORCED_STOP = -102,
}

/// NLOPT's direct_objective_func type
pub type DirectObjectiveFuncC = extern "C" fn(
    n: c_int,
    x: *const c_double,
    undefined_flag: *mut c_int,
    data: *mut c_void,
) -> c_double;

/// NLOPT's nlopt_func type (used by cdirect)
pub type NloptFuncC = extern "C" fn(
    n: c_uint,
    x: *const c_double,
    gradient: *mut c_double,
    func_data: *mut c_void,
) -> c_double;

/// NLOPT's nlopt_result enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NloptResultC {
    NLOPT_FAILURE = -1,
    NLOPT_INVALID_ARGS = -2,
    NLOPT_OUT_OF_MEMORY = -3,
    NLOPT_ROUNDOFF_LIMITED = -4,
    NLOPT_FORCED_STOP = -5,
    NLOPT_SUCCESS = 1,
    NLOPT_STOPVAL_REACHED = 2,
    NLOPT_FTOL_REACHED = 3,
    NLOPT_XTOL_REACHED = 4,
    NLOPT_MAXEVAL_REACHED = 5,
    NLOPT_MAXTIME_REACHED = 6,
}

/// NLOPT's nlopt_stopping struct (used by cdirect)
#[repr(C)]
pub struct NloptStopping {
    pub n: c_uint,
    pub minf_max: c_double,
    pub ftol_rel: c_double,
    pub ftol_abs: c_double,
    pub xtol_rel: c_double,
    pub xtol_abs: *const c_double,
    pub x_weights: *const c_double,
    pub nevals_p: *mut c_int,
    pub maxeval: c_int,
    pub maxtime: c_double,
    pub start: c_double,
    pub force_stop: *mut c_int,
    pub stop_msg: *mut *mut std::os::raw::c_char,
}

use std::os::raw::c_uint;

extern "C" {
    pub fn direct_optimize(
        f: DirectObjectiveFuncC,
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
        logfile: *mut c_void, // FILE*, pass null
        algorithm: DirectAlgorithmC,
    ) -> DirectReturnCodeC;

    /// NLOPT's cdirect() — SGJ re-implementation using red-black trees.
    /// Rescales bounds to [0,1]^n before calling cdirect_unscaled.
    pub fn cdirect(
        n: c_int,
        f: NloptFuncC,
        f_data: *mut c_void,
        lb: *const c_double,
        ub: *const c_double,
        x: *mut c_double,
        minf: *mut c_double,
        stop: *mut NloptStopping,
        magic_eps: c_double,
        which_alg: c_int,
    ) -> NloptResultC;

    /// NLOPT's cdirect_unscaled() — unscaled variant.
    pub fn cdirect_unscaled(
        n: c_int,
        f: NloptFuncC,
        f_data: *mut c_void,
        lb: *const c_double,
        ub: *const c_double,
        x: *mut c_double,
        minf: *mut c_double,
        stop: *mut NloptStopping,
        magic_eps: c_double,
        which_alg: c_int,
    ) -> NloptResultC;

    pub fn nlopt_seconds() -> c_double;

    /// Tracing wrapper for NLOPT DIRECT — runs the algorithm with trace output.
    pub fn nlopt_trace_direct(
        fcn: DirectObjectiveFuncC,
        fcn_data: *mut c_void,
        n: c_int,
        eps_in: c_double,
        epsabs: c_double,
        maxf: c_int,
        maxt_in: c_int,
        minf_out: *mut c_double,
        x_out: *mut c_double,
        lower: *const c_double,
        upper: *const c_double,
        algmethod: c_int,
        fglobal: c_double,
        fglper_in: c_double,
        volper_in: c_double,
        sigmaper_in: c_double,
        trace_buf: *mut std::os::raw::c_char,
        trace_buf_size: c_int,
    ) -> c_int;
}

/// Safe wrapper around NLOPT's direct_optimize()
pub struct NloptDirectRunner {
    pub dimension: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub max_feval: i32,
    pub max_iter: i32,
    pub magic_eps: f64,
    pub magic_eps_abs: f64,
    pub volume_reltol: f64,
    pub sigma_reltol: f64,
    pub fglobal: f64,
    pub fglobal_reltol: f64,
    pub algorithm: DirectAlgorithmC,
}

/// DIRECT_UNKNOWN_FGLOBAL = -HUGE_VAL
pub const DIRECT_UNKNOWN_FGLOBAL: f64 = f64::NEG_INFINITY;
pub const DIRECT_UNKNOWN_FGLOBAL_RELTOL: f64 = 0.0;

/// Result from running NLOPT DIRECT
#[derive(Debug, Clone)]
pub struct NloptDirectResult {
    pub x: Vec<f64>,
    pub minf: f64,
    pub return_code: DirectReturnCodeC,
}

impl NloptDirectRunner {
    pub fn new(dimension: usize, lower_bounds: Vec<f64>, upper_bounds: Vec<f64>) -> Self {
        Self {
            dimension,
            lower_bounds,
            upper_bounds,
            max_feval: 10000,
            max_iter: -1,
            magic_eps: 1e-4,
            magic_eps_abs: 0.0,
            volume_reltol: 0.0,
            sigma_reltol: 0.0,
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            algorithm: DirectAlgorithmC::DIRECT_GABLONSKY,
        }
    }

    /// Run the NLOPT DIRECT optimizer with the given objective function.
    ///
    /// # Safety
    /// The callback `f` must be a valid C function pointer that follows
    /// NLOPT's direct_objective_func convention.
    pub unsafe fn run(&self, f: DirectObjectiveFuncC, f_data: *mut c_void) -> NloptDirectResult {
        let mut x = vec![0.0f64; self.dimension];
        let mut minf: f64 = 0.0;
        let mut force_stop: c_int = 0;

        let return_code = unsafe {
            direct_optimize(
                f,
                f_data,
                self.dimension as c_int,
                self.lower_bounds.as_ptr(),
                self.upper_bounds.as_ptr(),
                x.as_mut_ptr(),
                &mut minf,
                self.max_feval,
                self.max_iter,
                0.0, // start time
                0.0, // maxtime (no limit)
                self.magic_eps,
                self.magic_eps_abs,
                self.volume_reltol,
                self.sigma_reltol,
                &mut force_stop,
                self.fglobal,
                self.fglobal_reltol,
                std::ptr::null_mut(), // logfile
                self.algorithm,
            )
        };

        NloptDirectResult {
            x,
            minf,
            return_code,
        }
    }
}

// ============================================================
// Test objective functions as extern "C" callbacks
// ============================================================

/// Sphere function: f(x) = sum(x_i^2)
pub extern "C" fn sphere_c(
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

/// Rosenbrock function: f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2]
pub extern "C" fn rosenbrock_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n - 1 {
        let xi = unsafe { *x.add(i) };
        let xi1 = unsafe { *x.add(i + 1) };
        sum += 100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2);
    }
    sum
}

/// Rastrigin function: f(x) = 10*n + sum_{i=0}^{n-1} [x_i^2 - 10*cos(2*pi*x_i)]
pub extern "C" fn rastrigin_c(
    n: c_int,
    x: *const c_double,
    _undefined_flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let n_usize = n as usize;
    let mut sum = 10.0 * n as f64;
    for i in 0..n_usize {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

// ============================================================
// nlopt_func-compatible objective functions (for cdirect)
// ============================================================

/// Sphere function with nlopt_func signature: f(n, x, grad, data) -> double
pub extern "C" fn sphere_nlopt(
    n: c_uint,
    x: *const c_double,
    _gradient: *mut c_double,
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

/// Sphere function with nlopt_func signature that counts evaluations.
pub extern "C" fn sphere_nlopt_counting(
    n: c_uint,
    x: *const c_double,
    _gradient: *mut c_double,
    data: *mut c_void,
) -> c_double {
    let counter = unsafe { &*(data as *const std::sync::atomic::AtomicUsize) };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n = n as usize;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = unsafe { *x.add(i) };
        sum += xi * xi;
    }
    sum
}

// ============================================================
// Safe wrapper for NLOPT cdirect()
// ============================================================

/// Result from running NLOPT cdirect
#[derive(Debug, Clone)]
pub struct NloptCDirectResult {
    pub x: Vec<f64>,
    pub minf: f64,
    pub nfev: usize,
    pub return_code: NloptResultC,
}

/// Safe wrapper around NLOPT's cdirect() function.
pub struct NloptCDirectRunner {
    pub dimension: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub max_feval: i32,
    pub magic_eps: f64,
    pub which_alg: i32,
}

impl NloptCDirectRunner {
    pub fn new(dimension: usize, lower_bounds: Vec<f64>, upper_bounds: Vec<f64>) -> Self {
        Self {
            dimension,
            lower_bounds,
            upper_bounds,
            max_feval: 10000,
            magic_eps: 1e-4,
            which_alg: 0,
        }
    }

    /// Run NLOPT cdirect() with the given objective function.
    pub unsafe fn run(&self, f: NloptFuncC, f_data: *mut c_void) -> NloptCDirectResult {
        let mut x = vec![0.0f64; self.dimension];
        let mut minf: f64 = f64::INFINITY;
        let mut nevals: c_int = 0;
        let mut force_stop: c_int = 0;
        let mut stop_msg: *mut std::os::raw::c_char = std::ptr::null_mut();

        let start = unsafe { nlopt_seconds() };

        let mut stop = NloptStopping {
            n: self.dimension as c_uint,
            minf_max: f64::NEG_INFINITY, // no stopval
            ftol_rel: 0.0,
            ftol_abs: 0.0,
            xtol_rel: 0.0,
            xtol_abs: std::ptr::null(),
            x_weights: std::ptr::null(),
            nevals_p: &mut nevals,
            maxeval: self.max_feval,
            maxtime: 0.0, // no time limit
            start,
            force_stop: &mut force_stop,
            stop_msg: &mut stop_msg,
        };

        let return_code = unsafe {
            cdirect(
                self.dimension as c_int,
                f,
                f_data,
                self.lower_bounds.as_ptr(),
                self.upper_bounds.as_ptr(),
                x.as_mut_ptr(),
                &mut minf,
                &mut stop,
                self.magic_eps,
                self.which_alg,
            )
        };

        // Free stop_msg if allocated
        if !stop_msg.is_null() {
            unsafe { libc::free(stop_msg as *mut c_void) };
        }

        NloptCDirectResult {
            x,
            minf,
            nfev: nevals as usize,
            return_code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlopt_sphere_2d_gablonsky() {
        let runner = NloptDirectRunner {
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

        let result = unsafe { runner.run(sphere_c, std::ptr::null_mut()) };

        println!("NLOPT C sphere 2D GABLONSKY:");
        println!("  x = {:?}", result.x);
        println!("  minf = {}", result.minf);
        println!("  return_code = {:?}", result.return_code);

        assert!(
            result.return_code == DirectReturnCodeC::DIRECT_MAXFEVAL_EXCEEDED
                || result.return_code == DirectReturnCodeC::DIRECT_GLOBAL_FOUND
                || result.return_code == DirectReturnCodeC::DIRECT_VOLTOL
                || result.return_code == DirectReturnCodeC::DIRECT_SIGMATOL,
            "Unexpected return code: {:?}",
            result.return_code
        );
        assert!(result.minf < 0.01, "Sphere minimum should be near 0, got {}", result.minf);
        for xi in &result.x {
            assert!(xi.abs() < 0.5, "Sphere optimum should be near 0, got {}", xi);
        }
    }

    #[test]
    fn test_nlopt_sphere_2d_original() {
        let runner = NloptDirectRunner {
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

        let result = unsafe { runner.run(sphere_c, std::ptr::null_mut()) };

        println!("NLOPT C sphere 2D ORIGINAL:");
        println!("  x = {:?}", result.x);
        println!("  minf = {}", result.minf);
        println!("  return_code = {:?}", result.return_code);

        assert!(result.minf < 0.01, "Sphere minimum should be near 0, got {}", result.minf);
    }

    #[test]
    fn test_nlopt_rosenbrock_2d_gablonsky() {
        let runner = NloptDirectRunner {
            dimension: 2,
            lower_bounds: vec![-5.0, -5.0],
            upper_bounds: vec![5.0, 5.0],
            max_feval: 2000,
            max_iter: -1,
            magic_eps: 1e-4,
            magic_eps_abs: 0.0,
            volume_reltol: 0.0,
            sigma_reltol: 0.0,
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            algorithm: DirectAlgorithmC::DIRECT_GABLONSKY,
        };

        let result = unsafe { runner.run(rosenbrock_c, std::ptr::null_mut()) };

        println!("NLOPT C rosenbrock 2D GABLONSKY:");
        println!("  x = {:?}", result.x);
        println!("  minf = {}", result.minf);
        println!("  return_code = {:?}", result.return_code);

        // Rosenbrock is harder; just verify it found a reasonable result
        assert!(result.minf < 10.0, "Rosenbrock minimum should be reasonable, got {}", result.minf);
    }

    #[test]
    fn test_nlopt_rosenbrock_2d_original() {
        let runner = NloptDirectRunner {
            dimension: 2,
            lower_bounds: vec![-5.0, -5.0],
            upper_bounds: vec![5.0, 5.0],
            max_feval: 2000,
            max_iter: -1,
            magic_eps: 1e-4,
            magic_eps_abs: 0.0,
            volume_reltol: 0.0,
            sigma_reltol: 0.0,
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            algorithm: DirectAlgorithmC::DIRECT_ORIGINAL,
        };

        let result = unsafe { runner.run(rosenbrock_c, std::ptr::null_mut()) };

        println!("NLOPT C rosenbrock 2D ORIGINAL:");
        println!("  x = {:?}", result.x);
        println!("  minf = {}", result.minf);
        println!("  return_code = {:?}", result.return_code);

        assert!(result.minf < 10.0, "Rosenbrock minimum should be reasonable, got {}", result.minf);
    }

    /// Compare NLOPT C and Rust implementations for sphere function
    #[test]
    fn test_compare_sphere_2d_gablonsky() {
        // Run NLOPT C
        let c_runner = NloptDirectRunner {
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
        let c_result = unsafe { c_runner.run(sphere_c, std::ptr::null_mut()) };

        // Run Rust implementation
        use direct_nlopt::types::{DirectAlgorithm, DirectOptions};
        use direct_nlopt::direct::Direct;

        let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
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
        let mut solver = Direct::new(sphere, &bounds, opts).unwrap();
        let rust_result = solver.minimize(None).unwrap();

        println!("COMPARISON sphere 2D GABLONSKY:");
        println!("  C:    x={:?}, minf={}, code={:?}", c_result.x, c_result.minf, c_result.return_code);
        println!("  Rust: x={:?}, minf={}", rust_result.x, rust_result.fun);

        // Both should find near-zero minimum
        assert!(c_result.minf < 0.01, "C minf={}", c_result.minf);
        assert!(rust_result.fun < 0.01, "Rust minf={}", rust_result.fun);
    }

    /// Compare NLOPT C and Rust implementations for Rosenbrock function
    #[test]
    fn test_compare_rosenbrock_2d_gablonsky() {
        // Run NLOPT C
        let c_runner = NloptDirectRunner {
            dimension: 2,
            lower_bounds: vec![-5.0, -5.0],
            upper_bounds: vec![5.0, 5.0],
            max_feval: 2000,
            max_iter: -1,
            magic_eps: 1e-4,
            magic_eps_abs: 0.0,
            volume_reltol: 0.0,
            sigma_reltol: 0.0,
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            algorithm: DirectAlgorithmC::DIRECT_GABLONSKY,
        };
        let c_result = unsafe { c_runner.run(rosenbrock_c, std::ptr::null_mut()) };

        // Run Rust
        use direct_nlopt::types::{DirectAlgorithm, DirectOptions};
        use direct_nlopt::direct::Direct;

        let rosenbrock = |x: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..x.len() - 1 {
                sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
            }
            sum
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 2000,
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
        let mut solver = Direct::new(rosenbrock, &bounds, opts).unwrap();
        let rust_result = solver.minimize(None).unwrap();

        println!("COMPARISON rosenbrock 2D GABLONSKY:");
        println!("  C:    x={:?}, minf={}, code={:?}", c_result.x, c_result.minf, c_result.return_code);
        println!("  Rust: x={:?}, minf={}", rust_result.x, rust_result.fun);

        // Both should find a reasonable minimum
        assert!(c_result.minf < 10.0, "C minf={}", c_result.minf);
        assert!(rust_result.fun < 10.0, "Rust minf={}", rust_result.fun);
    }
}
