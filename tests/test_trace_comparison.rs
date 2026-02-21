#![cfg(all(feature = "nlopt-compare", feature = "trace"))]

//! Step-by-step trace comparison between NLOPT C and Rust DIRECT implementations.
//!
//! Runs both implementations on sphere [-5,5]^2 with maxiter=5 and compares
//! trace output line-by-line. This is the ultimate faithfulness verification tool.
//!
//! Run with: cargo test --features "nlopt-compare,trace" test_trace_comparison

mod nlopt_ffi;

use std::os::raw::{c_double, c_int, c_void};
use std::sync::{Arc, Mutex};

use direct_nlopt::direct::Direct;
use direct_nlopt::trace::TraceWriter;
use direct_nlopt::types::{DirectAlgorithm, DirectOptions, DIRECT_UNKNOWN_FGLOBAL};

// Serialize all C FFI calls — the NLOPT C code is not thread-safe.
static C_FFI_MUTEX: Mutex<()> = Mutex::new(());

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

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

/// Run the C tracing implementation and return trace output as a String.
fn run_c_trace(
    n: usize,
    lower: &[f64],
    upper: &[f64],
    max_feval: i32,
    max_iter: i32,
    magic_eps: f64,
    algmethod: i32,
) -> (String, Vec<f64>, f64) {
    let _lock = C_FFI_MUTEX.lock().unwrap();
    let mut x_out = vec![0.0f64; n];
    let mut minf_out: f64 = 0.0;
    let buf_size = 1024 * 1024; // 1MB buffer
    let mut trace_buf: Vec<u8> = vec![0u8; buf_size];

    let _ierror = unsafe {
        nlopt_ffi::nlopt_trace_direct(
            sphere_c,
            std::ptr::null_mut(),
            n as c_int,
            magic_eps,
            0.0,   // epsabs
            max_feval,
            max_iter,
            &mut minf_out,
            x_out.as_mut_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            algmethod,
            f64::NEG_INFINITY, // fglobal = unknown
            0.0,               // fglper
            0.0,               // volper
            0.0,               // sigmaper
            trace_buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf_size as c_int,
        )
    };

    let trace_str = unsafe {
        let c_str = std::ffi::CStr::from_ptr(trace_buf.as_ptr() as *const std::os::raw::c_char);
        c_str.to_string_lossy().into_owned()
    };

    (trace_str, x_out, minf_out)
}

/// Run the Rust tracing implementation and return trace output as a String.
fn run_rust_trace(
    bounds: &Vec<(f64, f64)>,
    max_feval: usize,
    max_iter: usize,
    magic_eps: f64,
    algorithm: DirectAlgorithm,
) -> (String, Vec<f64>, f64) {
    let opts = DirectOptions {
        max_feval,
        max_iter,
        magic_eps,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: -1.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: 0.0,
        parallel: false,
        parallel_batch: false,
        algorithm,
        ..Default::default()
    };

    let mut solver = Direct::new(sphere, &bounds, opts).unwrap();
    let tracer = Arc::new(TraceWriter::new());
    solver.set_tracer(tracer.clone());

    let result = solver.minimize(None).unwrap();

    let trace_str = tracer.get_output();
    (trace_str, result.x, result.fun)
}

/// Parse trace lines, filtering only TRACE lines.
fn parse_trace_lines(trace: &str) -> Vec<String> {
    trace
        .lines()
        .filter(|line| line.starts_with("TRACE "))
        .map(|s| s.to_string())
        .collect()
}

/// Normalize exponent format: C uses `e+01`/`e-01`, Rust uses `e1`/`e-1`.
/// Convert both to a canonical form by stripping leading zeros and unnecessary `+`.
fn normalize_exponent(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'e' || bytes[i] == b'E' {
            result.push(bytes[i] as char);
            i += 1;
            // Handle sign
            if i < bytes.len() && bytes[i] == b'+' {
                i += 1; // skip unnecessary '+'
            } else if i < bytes.len() && bytes[i] == b'-' {
                result.push('-');
                i += 1;
            }
            // Skip leading zeros in exponent
            while i < bytes.len() && bytes[i] == b'0' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                i += 1;
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

/// Compare trace outputs line-by-line, with detailed error reporting.
fn compare_traces(c_trace: &str, rust_trace: &str) -> Vec<String> {
    let c_lines = parse_trace_lines(c_trace);
    let r_lines = parse_trace_lines(rust_trace);
    let mut mismatches = Vec::new();

    let max_lines = c_lines.len().max(r_lines.len());
    for i in 0..max_lines {
        let c_line = c_lines.get(i).map(|s| s.as_str()).unwrap_or("<missing>");
        let r_line = r_lines.get(i).map(|s| s.as_str()).unwrap_or("<missing>");

        let c_norm = normalize_exponent(c_line);
        let r_norm = normalize_exponent(r_line);

        if c_norm != r_norm {
            mismatches.push(format!(
                "Line {}: \n  C:    {}\n  Rust: {}",
                i + 1,
                c_line,
                r_line
            ));
        }
    }

    mismatches
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trace_sphere_2d_gablonsky_5iter() {
    let n = 2;
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let max_feval = 10000;
    let max_iter = 5;
    let magic_eps = 1e-4;

    let (c_trace, c_x, c_minf) = run_c_trace(n, &lower, &upper, max_feval as i32, max_iter as i32, magic_eps, 1);
    let (rust_trace, rust_x, rust_minf) = run_rust_trace(&bounds, max_feval, max_iter, magic_eps, DirectAlgorithm::GablonskyLocallyBiased);

    // Print both traces for debugging
    eprintln!("=== C TRACE ===");
    eprintln!("{}", c_trace);
    eprintln!("=== RUST TRACE ===");
    eprintln!("{}", rust_trace);

    // Compare results
    assert_eq!(c_minf, rust_minf, "minf mismatch: C={}, Rust={}", c_minf, rust_minf);
    assert_eq!(c_x, rust_x, "x mismatch: C={:?}, Rust={:?}", c_x, rust_x);

    // Compare traces
    let mismatches = compare_traces(&c_trace, &rust_trace);
    if !mismatches.is_empty() {
        eprintln!("=== TRACE MISMATCHES ===");
        for m in &mismatches {
            eprintln!("{}", m);
        }
        panic!("{} trace line mismatches found (see above)", mismatches.len());
    }
}

#[test]
fn test_trace_sphere_2d_original_5iter() {
    let n = 2;
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let max_feval = 10000;
    let max_iter = 5;
    let magic_eps = 1e-4;

    let (c_trace, c_x, c_minf) = run_c_trace(n, &lower, &upper, max_feval as i32, max_iter as i32, magic_eps, 0);
    let (rust_trace, rust_x, rust_minf) = run_rust_trace(&bounds, max_feval, max_iter, magic_eps, DirectAlgorithm::GablonskyOriginal);

    eprintln!("=== C TRACE (Original) ===");
    eprintln!("{}", c_trace);
    eprintln!("=== RUST TRACE (Original) ===");
    eprintln!("{}", rust_trace);

    assert_eq!(c_minf, rust_minf, "minf mismatch: C={}, Rust={}", c_minf, rust_minf);
    assert_eq!(c_x, rust_x, "x mismatch: C={:?}, Rust={:?}", c_x, rust_x);

    let mismatches = compare_traces(&c_trace, &rust_trace);
    if !mismatches.is_empty() {
        eprintln!("=== TRACE MISMATCHES ===");
        for m in &mismatches {
            eprintln!("{}", m);
        }
        panic!("{} trace line mismatches found (see above)", mismatches.len());
    }
}

#[test]
fn test_trace_sphere_3d_gablonsky_5iter() {
    let n = 3;
    let lower = vec![-5.0; n];
    let upper = vec![5.0; n];
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let max_feval = 10000;
    let max_iter = 5;
    let magic_eps = 1e-4;

    let (c_trace, c_x, c_minf) = run_c_trace(n, &lower, &upper, max_feval as i32, max_iter as i32, magic_eps, 1);
    let (rust_trace, rust_x, rust_minf) = run_rust_trace(&bounds, max_feval, max_iter, magic_eps, DirectAlgorithm::GablonskyLocallyBiased);

    assert_eq!(c_minf, rust_minf, "minf mismatch: C={}, Rust={}", c_minf, rust_minf);
    assert_eq!(c_x, rust_x, "x mismatch: C={:?}, Rust={:?}", c_x, rust_x);

    let mismatches = compare_traces(&c_trace, &rust_trace);
    if !mismatches.is_empty() {
        eprintln!("=== C TRACE (3D Gablonsky) ===");
        eprintln!("{}", c_trace);
        eprintln!("=== RUST TRACE (3D Gablonsky) ===");
        eprintln!("{}", rust_trace);
        eprintln!("=== TRACE MISMATCHES ===");
        for m in &mismatches {
            eprintln!("{}", m);
        }
        panic!("{} trace line mismatches found (see above)", mismatches.len());
    }
}

#[test]
fn test_trace_sphere_2d_gablonsky_20iter() {
    let n = 2;
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
    let max_feval = 10000;
    let max_iter = 20;
    let magic_eps = 1e-4;

    let (c_trace, c_x, c_minf) = run_c_trace(n, &lower, &upper, max_feval as i32, max_iter as i32, magic_eps, 1);
    let (rust_trace, rust_x, rust_minf) = run_rust_trace(&bounds, max_feval, max_iter, magic_eps, DirectAlgorithm::GablonskyLocallyBiased);

    assert_eq!(c_minf, rust_minf, "minf mismatch: C={}, Rust={}", c_minf, rust_minf);
    assert_eq!(c_x, rust_x, "x mismatch: C={:?}, Rust={:?}", c_x, rust_x);

    let mismatches = compare_traces(&c_trace, &rust_trace);
    if !mismatches.is_empty() {
        eprintln!("{} trace mismatches in 20-iter test", mismatches.len());
        for m in mismatches.iter().take(5) {
            eprintln!("{}", m);
        }
        panic!("{} trace line mismatches found", mismatches.len());
    }
}
