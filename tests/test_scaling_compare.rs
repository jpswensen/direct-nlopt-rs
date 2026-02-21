#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for scaling/unscaling (dirpreprc_ + dirinfcn_).
//! Verifies that the Rust implementation produces identical results to NLOPT C.

use std::os::raw::{c_double, c_int, c_void};

// FFI declarations for NLOPT's internal scaling functions.
// These are compiled from DIRsubrout.c via build.rs when "nlopt-compare" is enabled.
extern "C" {
    fn direct_dirpreprc_(
        u: *const c_double,
        l: *const c_double,
        n: *const c_int,
        xs1: *mut c_double,
        xs2: *mut c_double,
        oops: *mut c_int,
    );

    fn direct_dirinfcn_(
        fcn: extern "C" fn(c_int, *const c_double, *mut c_int, *mut c_void) -> c_double,
        x: *mut c_double,
        c1: *const c_double,
        c2: *const c_double,
        n: *const c_int,
        f: *mut c_double,
        flag: *mut c_int,
        fcn_data: *mut c_void,
    );
}

/// Identity function for testing unscaling: returns sum(x_i) so we can verify
/// that the C code passes the correctly unscaled coordinates.
extern "C" fn sum_fn(
    n: c_int,
    x: *const c_double,
    _flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let mut s = 0.0;
    for i in 0..n as usize {
        s += unsafe { *x.add(i) };
    }
    s
}

/// Sphere function for infcn_ testing
extern "C" fn sphere_c(
    n: c_int,
    x: *const c_double,
    _flag: *mut c_int,
    _data: *mut c_void,
) -> c_double {
    let mut s = 0.0;
    for i in 0..n as usize {
        let xi = unsafe { *x.add(i) };
        s += xi * xi;
    }
    s
}

/// Call NLOPT's direct_dirpreprc_ and return (xs1, xs2, oops).
/// The C function uses 1-based Fortran indexing via pointer decrements.
fn call_c_dirpreprc(lower: &[f64], upper: &[f64]) -> (Vec<f64>, Vec<f64>, i32) {
    let n = lower.len() as c_int;
    let mut xs1 = vec![0.0f64; n as usize];
    let mut xs2 = vec![0.0f64; n as usize];
    let mut oops: c_int = 0;

    unsafe {
        direct_dirpreprc_(
            upper.as_ptr(),
            lower.as_ptr(),
            &n,
            xs1.as_mut_ptr(),
            xs2.as_mut_ptr(),
            &mut oops,
        );
    }

    (xs1, xs2, oops)
}

/// Call NLOPT's direct_dirinfcn_ to unscale x, evaluate, and rescale.
/// Returns (f_value, x_after_rescale).
/// The C function expects c1=xs1, c2=xs2 with Fortran 1-based indexing.
fn call_c_dirinfcn(
    fcn: extern "C" fn(c_int, *const c_double, *mut c_int, *mut c_void) -> c_double,
    x_norm: &[f64],
    xs1: &[f64],
    xs2: &[f64],
) -> (f64, Vec<f64>) {
    let n = x_norm.len() as c_int;
    let mut x = x_norm.to_vec();
    let mut f: f64 = 0.0;
    let mut flag: c_int = 0;

    unsafe {
        direct_dirinfcn_(
            fcn,
            x.as_mut_ptr(),
            xs1.as_ptr(),
            xs2.as_ptr(),
            &n,
            &mut f,
            &mut flag,
            std::ptr::null_mut(),
        );
    }

    (f, x)
}

// ====================================================================
// Tests for direct_dirpreprc_: xs1 and xs2 computation
// ====================================================================

#[test]
fn test_dirpreprc_symmetric_bounds() {
    // [-5, 5]: expected xs1=10, xs2=-0.5
    let lower = vec![-5.0];
    let upper = vec![5.0];
    let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);

    assert_eq!(oops, 0);
    assert_eq!(xs1[0], 10.0);
    assert_eq!(xs2[0], -0.5);

    // Compare with Rust
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(-5.0, 5.0)];
    let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

    assert_eq!(d.xs1[0], xs1[0], "xs1 mismatch: Rust={} C={}", d.xs1[0], xs1[0]);
    assert_eq!(d.xs2[0], xs2[0], "xs2 mismatch: Rust={} C={}", d.xs2[0], xs2[0]);
}

#[test]
fn test_dirpreprc_asymmetric_bounds() {
    // [2, 10]: expected xs1=8, xs2=0.25
    let lower = vec![2.0];
    let upper = vec![10.0];
    let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);

    assert_eq!(oops, 0);
    assert_eq!(xs1[0], 8.0);
    assert_eq!(xs2[0], 0.25);

    // Compare with Rust
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(2.0, 10.0)];
    let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

    assert_eq!(d.xs1[0], xs1[0], "xs1 mismatch: Rust={} C={}", d.xs1[0], xs1[0]);
    assert_eq!(d.xs2[0], xs2[0], "xs2 mismatch: Rust={} C={}", d.xs2[0], xs2[0]);
}

#[test]
fn test_dirpreprc_near_zero_bounds() {
    // [0, 1]: expected xs1=1, xs2=0
    let lower = vec![0.0];
    let upper = vec![1.0];
    let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);

    assert_eq!(oops, 0);
    assert_eq!(xs1[0], 1.0);
    assert_eq!(xs2[0], 0.0);

    // Compare with Rust
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(0.0, 1.0)];
    let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

    assert_eq!(d.xs1[0], xs1[0], "xs1 mismatch: Rust={} C={}", d.xs1[0], xs1[0]);
    assert_eq!(d.xs2[0], xs2[0], "xs2 mismatch: Rust={} C={}", d.xs2[0], xs2[0]);
}

#[test]
fn test_dirpreprc_multidim() {
    // 3D: [-5,5], [2,10], [0,1]
    let lower = vec![-5.0, 2.0, 0.0];
    let upper = vec![5.0, 10.0, 1.0];
    let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);

    assert_eq!(oops, 0);
    assert_eq!(xs1, vec![10.0, 8.0, 1.0]);
    assert_eq!(xs2, vec![-0.5, 0.25, 0.0]);

    // Compare with Rust
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(-5.0, 5.0), (2.0, 10.0), (0.0, 1.0)];
    let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

    for i in 0..3 {
        assert_eq!(d.xs1[i], xs1[i], "xs1[{}] mismatch: Rust={} C={}", i, d.xs1[i], xs1[i]);
        assert_eq!(d.xs2[i], xs2[i], "xs2[{}] mismatch: Rust={} C={}", i, d.xs2[i], xs2[i]);
    }
}

#[test]
fn test_dirpreprc_invalid_bounds() {
    // u <= l should set oops=1
    let lower = vec![5.0];
    let upper = vec![-5.0];
    let (_xs1, _xs2, oops) = call_c_dirpreprc(&lower, &upper);
    assert_eq!(oops, 1, "C dirpreprc_ should detect invalid bounds");

    // Rust should also reject invalid bounds
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(5.0, -5.0)];
    assert!(Direct::new(sphere, &bounds, DirectOptions::default()).is_err());
}

#[test]
fn test_dirpreprc_extreme_bounds() {
    // Very wide bounds: [-1e10, 1e10]
    let lower = vec![-1e10];
    let upper = vec![1e10];
    let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);

    assert_eq!(oops, 0);
    assert_eq!(xs1[0], 2e10);
    assert_eq!(xs2[0], -0.5);

    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let bounds = vec![(-1e10, 1e10)];
    let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

    assert_eq!(d.xs1[0], xs1[0]);
    assert_eq!(d.xs2[0], xs2[0]);

    // Very narrow bounds: [0, 1e-10]
    let lower2 = vec![0.0];
    let upper2 = vec![1e-10];
    let (xs1_2, xs2_2, oops2) = call_c_dirpreprc(&lower2, &upper2);

    assert_eq!(oops2, 0);
    assert_eq!(xs1_2[0], 1e-10);
    assert_eq!(xs2_2[0], 0.0);

    let bounds2 = vec![(0.0, 1e-10)];
    let d2 = Direct::new(sphere, &bounds2, DirectOptions::default()).unwrap();
    assert_eq!(d2.xs1[0], xs1_2[0]);
    assert_eq!(d2.xs2[0], xs2_2[0]);
}

// ====================================================================
// Tests for direct_dirinfcn_: unscaling + evaluation + rescaling
// ====================================================================

#[test]
fn test_dirinfcn_symmetric_unscaling() {
    // [-5, 5]: xs1=10, xs2=-0.5
    // Normalized center 0.5 → actual 0.0 → sphere = 0.0
    let lower = vec![-5.0];
    let upper = vec![5.0];
    let (xs1, xs2, _) = call_c_dirpreprc(&lower, &upper);

    // Test center point: x_norm=0.5 → x_actual = (0.5 + (-0.5)) * 10 = 0.0
    let (f_c, x_after) = call_c_dirinfcn(sphere_c, &[0.5], &xs1, &xs2);
    assert!((f_c - 0.0).abs() < 1e-15, "C sphere at center should be 0, got {}", f_c);
    // After rescaling, x should be back to 0.5
    assert!((x_after[0] - 0.5).abs() < 1e-15, "C rescale should restore x=0.5, got {}", x_after[0]);

    // Compare with Rust to_actual + evaluate
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let d = Direct::new(sphere, &vec![(-5.0, 5.0)], DirectOptions::default()).unwrap();
    let (f_rust, _feasible) = d.evaluate(&[0.5]);

    assert_eq!(f_rust, f_c, "Rust f={} != C f={} at center", f_rust, f_c);
}

#[test]
fn test_dirinfcn_asymmetric_unscaling() {
    // [2, 10]: xs1=8, xs2=0.25
    let lower = vec![2.0];
    let upper = vec![10.0];
    let (xs1, xs2, _) = call_c_dirpreprc(&lower, &upper);

    // x_norm=0.0 → x_actual = (0.0 + 0.25) * 8 = 2.0 → sphere = 4.0
    let (f_c_0, x_after_0) = call_c_dirinfcn(sphere_c, &[0.0], &xs1, &xs2);
    assert!((f_c_0 - 4.0).abs() < 1e-12, "C sphere at x_norm=0.0 should be 4.0, got {}", f_c_0);
    assert!((x_after_0[0] - 0.0).abs() < 1e-15);

    // x_norm=1.0 → x_actual = (1.0 + 0.25) * 8 = 10.0 → sphere = 100.0
    let (f_c_1, x_after_1) = call_c_dirinfcn(sphere_c, &[1.0], &xs1, &xs2);
    assert!((f_c_1 - 100.0).abs() < 1e-10, "C sphere at x_norm=1.0 should be 100.0, got {}", f_c_1);
    assert!((x_after_1[0] - 1.0).abs() < 1e-15);

    // x_norm=0.5 → x_actual = (0.5 + 0.25) * 8 = 6.0 → sphere = 36.0
    let (f_c_half, _) = call_c_dirinfcn(sphere_c, &[0.5], &xs1, &xs2);
    assert!((f_c_half - 36.0).abs() < 1e-10);

    // Compare all with Rust
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let d = Direct::new(sphere, &vec![(2.0, 10.0)], DirectOptions::default()).unwrap();

    let (f_r_0, _) = d.evaluate(&[0.0]);
    let (f_r_1, _) = d.evaluate(&[1.0]);
    let (f_r_half, _) = d.evaluate(&[0.5]);

    assert_eq!(f_r_0, f_c_0, "f at x_norm=0.0: Rust={} C={}", f_r_0, f_c_0);
    assert_eq!(f_r_1, f_c_1, "f at x_norm=1.0: Rust={} C={}", f_r_1, f_c_1);
    assert_eq!(f_r_half, f_c_half, "f at x_norm=0.5: Rust={} C={}", f_r_half, f_c_half);
}

#[test]
fn test_dirinfcn_near_zero_unscaling() {
    // [0, 1]: xs1=1, xs2=0
    let lower = vec![0.0];
    let upper = vec![1.0];
    let (xs1, xs2, _) = call_c_dirpreprc(&lower, &upper);

    // x_norm=0.5 → x_actual = (0.5 + 0.0) * 1.0 = 0.5 → sphere = 0.25
    let (f_c, _) = call_c_dirinfcn(sphere_c, &[0.5], &xs1, &xs2);
    assert!((f_c - 0.25).abs() < 1e-15);

    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let d = Direct::new(sphere, &vec![(0.0, 1.0)], DirectOptions::default()).unwrap();
    let (f_r, _) = d.evaluate(&[0.5]);

    assert_eq!(f_r, f_c, "f at x_norm=0.5: Rust={} C={}", f_r, f_c);
}

#[test]
fn test_dirinfcn_multidim_unscaling() {
    // 2D: [-5,5] x [2,10]
    let lower = vec![-5.0, 2.0];
    let upper = vec![5.0, 10.0];
    let (xs1, xs2, _) = call_c_dirpreprc(&lower, &upper);

    // Use sum_fn to verify the actual coordinates passed to the objective
    // x_norm = [0.5, 0.0]
    // → x_actual = [(0.5-0.5)*10, (0.0+0.25)*8] = [0.0, 2.0]
    // → sum = 2.0
    let (f_c, x_after) = call_c_dirinfcn(sum_fn, &[0.5, 0.0], &xs1, &xs2);
    assert!((f_c - 2.0).abs() < 1e-12, "sum([0.0, 2.0]) should be 2.0, got {}", f_c);
    assert!((x_after[0] - 0.5).abs() < 1e-15);
    assert!((x_after[1] - 0.0).abs() < 1e-15);

    // Sphere at center: x_norm = [0.5, 0.5]
    // → x_actual = [0.0, 6.0] → sphere = 36.0
    let (f_c2, _) = call_c_dirinfcn(sphere_c, &[0.5, 0.5], &xs1, &xs2);
    assert!((f_c2 - 36.0).abs() < 1e-10);

    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::DirectOptions;
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
    let d = Direct::new(sphere, &vec![(-5.0, 5.0), (2.0, 10.0)], DirectOptions::default()).unwrap();
    let (f_r, _) = d.evaluate(&[0.5, 0.5]);
    assert_eq!(f_r, f_c2, "2D sphere at center: Rust={} C={}", f_r, f_c2);
}

// ====================================================================
// Roundtrip tests: unscale → rescale should restore original x
// ====================================================================

#[test]
fn test_dirinfcn_roundtrip_rescaling() {
    // Verify that dirinfcn_ correctly rescales x back to normalized form
    // after evaluation, matching Rust to_normalized(to_actual(x)) == x.
    let test_cases: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = vec![
        (vec![-5.0], vec![5.0], vec![0.0]),
        (vec![-5.0], vec![5.0], vec![0.5]),
        (vec![-5.0], vec![5.0], vec![1.0]),
        (vec![-5.0], vec![5.0], vec![0.3]),
        (vec![2.0], vec![10.0], vec![0.0]),
        (vec![2.0], vec![10.0], vec![0.75]),
        (vec![0.0], vec![1.0], vec![0.5]),
        (vec![-100.0], vec![1.0], vec![0.5]),
        (vec![-1e10], vec![1e10], vec![0.5]),
        (vec![0.0], vec![1e-10], vec![0.5]),
        // Multi-dimensional
        (vec![-5.0, 2.0], vec![5.0, 10.0], vec![0.3, 0.7]),
        (vec![-5.0, 2.0, 0.0], vec![5.0, 10.0, 1.0], vec![0.1, 0.5, 0.9]),
    ];

    for (lower, upper, x_norm_orig) in &test_cases {
        let (xs1, xs2, oops) = call_c_dirpreprc(lower, upper);
        assert_eq!(oops, 0);

        let (_f, x_after) = call_c_dirinfcn(sphere_c, x_norm_orig, &xs1, &xs2);

        for i in 0..x_norm_orig.len() {
            let err = (x_after[i] - x_norm_orig[i]).abs();
            assert!(
                err < 1e-12,
                "C roundtrip failed: bounds=[{},{}], x_norm={}, got {} (err={})",
                lower[i], upper[i], x_norm_orig[i], x_after[i], err
            );
        }

        // Verify Rust roundtrip matches
        use direct_nlopt::direct::Direct;
        use direct_nlopt::types::DirectOptions;
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
        let bounds: Vec<(f64, f64)> = lower.iter().zip(upper.iter()).map(|(&l, &u)| (l, u)).collect();
        let d = Direct::new(sphere, &bounds, DirectOptions::default()).unwrap();

        let dim = x_norm_orig.len();
        let mut x_actual = vec![0.0; dim];
        let mut x_norm_back = vec![0.0; dim];

        d.to_actual(x_norm_orig, &mut x_actual);
        d.to_normalized(&x_actual, &mut x_norm_back);

        for i in 0..dim {
            let err = (x_norm_back[i] - x_norm_orig[i]).abs();
            assert!(
                err < 1e-12,
                "Rust roundtrip failed: bounds=[{},{}], x_norm={}, got {} (err={})",
                lower[i], upper[i], x_norm_orig[i], x_norm_back[i], err
            );
        }
    }
}

// ====================================================================
// Algebraic equivalence: (x + xs2) * xs1 == x * xs1 + l
// ====================================================================

#[test]
fn test_dirinfcn_algebraic_equivalence_vs_c() {
    // Verify: the C formula (x+c2)*c1 produces the same result as x*c1 + l
    // for all bounds configurations, using the actual C-computed xs1/xs2.
    let bounds_list: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![-5.0], vec![5.0]),
        (vec![2.0], vec![10.0]),
        (vec![0.0], vec![1.0]),
        (vec![-100.0], vec![1.0]),
        (vec![-1e10], vec![1e10]),
        (vec![0.0], vec![1e-10]),
    ];
    let x_norms = [0.0, 0.25, 0.5, 0.75, 1.0];

    for (lower, upper) in &bounds_list {
        let (xs1, xs2, oops) = call_c_dirpreprc(lower, upper);
        assert_eq!(oops, 0);

        for &x in &x_norms {
            let formula_c = (x + xs2[0]) * xs1[0]; // C's unscaling formula
            let formula_direct = x * xs1[0] + lower[0]; // algebraic equivalent
            let err = (formula_c - formula_direct).abs();
            assert!(
                err < 1e-6,
                "Algebraic equiv failed for bounds=[{},{}], x={}: C_formula={} direct={}",
                lower[0], upper[0], x, formula_c, formula_direct
            );
        }
    }
}

// ====================================================================
// Comprehensive comparison: C and Rust produce bit-identical f-values
// ====================================================================

#[test]
fn test_dirinfcn_bit_identical_fvalues() {
    // For a grid of normalized points, verify that the C dirinfcn_ and
    // Rust evaluate() produce identical f-values (bit-exact for simple cases).
    let bounds_configs: Vec<Vec<(f64, f64)>> = vec![
        vec![(-5.0, 5.0)],
        vec![(2.0, 10.0)],
        vec![(0.0, 1.0)],
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        vec![(-5.0, 5.0), (2.0, 10.0)],
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    ];

    let x_norms_1d = vec![0.0, 0.1, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 0.9, 1.0];

    for bounds in &bounds_configs {
        let dim = bounds.len();
        let lower: Vec<f64> = bounds.iter().map(|&(l, _)| l).collect();
        let upper: Vec<f64> = bounds.iter().map(|&(_, u)| u).collect();
        let (xs1, xs2, oops) = call_c_dirpreprc(&lower, &upper);
        assert_eq!(oops, 0);

        use direct_nlopt::direct::Direct;
        use direct_nlopt::types::DirectOptions;
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
        let d = Direct::new(sphere, bounds, DirectOptions::default()).unwrap();

        // Test the center point and a few others
        let test_points: Vec<Vec<f64>> = if dim == 1 {
            x_norms_1d.iter().map(|&x| vec![x]).collect()
        } else if dim == 2 {
            let mut pts = Vec::new();
            for &x0 in &[0.0, 0.5, 1.0, 1.0 / 3.0] {
                for &x1 in &[0.0, 0.5, 1.0, 1.0 / 3.0] {
                    pts.push(vec![x0, x1]);
                }
            }
            pts
        } else {
            // 3D: just test center and a few corners
            vec![
                vec![0.5; dim],
                vec![0.0; dim],
                vec![1.0; dim],
                vec![1.0 / 3.0; dim],
            ]
        };

        for x_norm in &test_points {
            let (f_c, _) = call_c_dirinfcn(sphere_c, x_norm, &xs1, &xs2);
            let (f_r, _) = d.evaluate(x_norm);

            let err = (f_c - f_r).abs();
            let tol = if f_c.abs() > 1e-10 {
                f_c.abs() * 1e-14
            } else {
                1e-15
            };

            assert!(
                err <= tol,
                "f-value mismatch for bounds={:?}, x_norm={:?}: C={} Rust={} err={}",
                bounds, x_norm, f_c, f_r, err
            );
        }
    }
}
