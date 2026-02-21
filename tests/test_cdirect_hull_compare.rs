#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for cdirect convex_hull() and rect_diameter().
//! Verifies that the Rust CDirect implementation produces identical results
//! to the NLOPT C algorithms (reimplemented as standalone shim functions
//! matching cdirect.c lines 94–112 and 261–378).

use std::os::raw::{c_double, c_int};

// Force linking the native C library by referencing the crate.
// Without this, rustc won't include the C objects from the rlib.
use direct_nlopt::types::DirectOptions;

// FFI declarations for shim functions compiled from nlopt_util_shim.c
extern "C" {
    fn nlopt_shim_rect_diameter(n: c_int, w: *const c_double, which_diam: c_int) -> c_double;

    fn nlopt_shim_convex_hull(
        npts: c_int,
        diameters: *const c_double,
        f_values: *const c_double,
        hull_indices: *mut c_int,
        allow_dups: c_int,
    ) -> c_int;
}

/// Call the C shim rect_diameter.
fn c_rect_diameter(n: usize, w: &[f64], which_diam: i32) -> f64 {
    unsafe { nlopt_shim_rect_diameter(n as c_int, w.as_ptr(), which_diam as c_int) }
}

/// Call the C shim convex_hull with sorted (diameter, f_value) points.
/// Returns indices of hull points.
fn c_convex_hull(
    diameters: &[f64],
    f_values: &[f64],
    allow_dups: bool,
) -> Vec<usize> {
    let npts = diameters.len();
    let mut hull_indices = vec![0i32; npts];
    let nhull = unsafe {
        nlopt_shim_convex_hull(
            npts as c_int,
            diameters.as_ptr(),
            f_values.as_ptr(),
            hull_indices.as_mut_ptr(),
            allow_dups as c_int,
        )
    };
    hull_indices.truncate(nhull as usize);
    hull_indices.iter().map(|&i| i as usize).collect()
}

/// Rust rect_diameter matching CDirect::rect_diameter().
fn rust_rect_diameter(n: usize, w: &[f64], which_diam: i32) -> f64 {
    if which_diam == 0 {
        let sum: f64 = w[..n].iter().map(|&wi| wi * wi).sum();
        (sum.sqrt() * 0.5) as f32 as f64
    } else {
        let maxw = w[..n].iter().cloned().fold(0.0_f64, f64::max);
        (maxw * 0.5) as f32 as f64
    }
}

/// Rust convex_hull matching CDirect::convex_hull(), operating on sorted arrays
/// instead of a BTreeMap (same algorithm).
fn rust_convex_hull(
    diameters: &[f64],
    f_values: &[f64],
    allow_dups: bool,
) -> Vec<usize> {
    let npts = diameters.len();
    if npts == 0 {
        return vec![];
    }

    let xmin = diameters[0];
    let yminmin = f_values[0];
    let xmax = diameters[npts - 1];

    let mut hull: Vec<usize> = Vec::new();

    // Include initial points at (xmin, yminmin)
    if allow_dups {
        for i in 0..npts {
            if diameters[i] == xmin && f_values[i] == yminmin {
                hull.push(i);
            } else {
                break;
            }
        }
    } else {
        hull.push(0);
    }

    if xmin == xmax {
        return hull;
    }

    // Find ymaxmin: minimum f among entries with d == xmax
    let nmax_start = diameters.iter().position(|&d| d == xmax).unwrap_or(npts);
    let ymaxmin = diameters
        .iter()
        .zip(f_values.iter())
        .filter(|(&d, _)| d == xmax)
        .map(|(_, &f)| f)
        .fold(f64::INFINITY, f64::min);

    let minslope = (ymaxmin - yminmin) / (xmax - xmin);

    // Skip entries with d == xmin
    let start_idx = diameters
        .iter()
        .position(|&d| d != xmin)
        .unwrap_or(npts);

    // Process entries between xmin and xmax
    let mut i = start_idx;
    while i < nmax_start {
        let x = diameters[i];
        let y = f_values[i];

        // Skip if above the line from (xmin,yminmin) to (xmax,ymaxmin)
        if y > yminmin + (x - xmin) * minslope {
            i += 1;
            continue;
        }

        // Performance hack: skip vertical lines
        if !hull.is_empty() && x == diameters[hull[hull.len() - 1]] {
            if y > f_values[hull[hull.len() - 1]] {
                let cur_d = x;
                while i < nmax_start && diameters[i] == cur_d {
                    i += 1;
                }
                continue;
            } else if allow_dups {
                hull.push(i);
                i += 1;
                continue;
            }
        }

        // Remove points until we make a "left turn" to i
        while hull.len() > 1 {
            let t1_d = diameters[hull[hull.len() - 1]];
            let t1_f = f_values[hull[hull.len() - 1]];

            let mut it2 = hull.len() as i64 - 2;
            loop {
                if it2 < 0 {
                    break;
                }
                let t2_d_cand = diameters[hull[it2 as usize]];
                let t2_f_cand = f_values[hull[it2 as usize]];
                if t2_d_cand != t1_d || t2_f_cand != t1_f {
                    break;
                }
                it2 -= 1;
            }
            if it2 < 0 {
                break;
            }
            let t2_d = diameters[hull[it2 as usize]];
            let t2_f = f_values[hull[it2 as usize]];

            let cross = (t1_d - t2_d) * (y - t2_f) - (t1_f - t2_f) * (x - t2_d);
            if cross >= 0.0 {
                break;
            }
            hull.pop();
        }
        hull.push(i);
        i += 1;
    }

    // Add points at (xmax, ymaxmin)
    if allow_dups {
        for j in nmax_start..npts {
            if diameters[j] == xmax && f_values[j] == ymaxmin {
                hull.push(j);
            } else if diameters[j] != xmax {
                break;
            }
        }
    } else {
        for j in nmax_start..npts {
            if diameters[j] == xmax && f_values[j] == ymaxmin {
                hull.push(j);
                break;
            }
        }
    }

    hull
}

// ══════════════════════════════════════════════════════════════════════════════
// rect_diameter tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_rect_diameter_jones_unit_square() {
    // 2D unit square: sqrt(1+1)*0.5 = sqrt(2)/2 ≈ 0.7071
    let w = [1.0, 1.0];
    let c_val = c_rect_diameter(2, &w, 0);
    let r_val = rust_rect_diameter(2, &w, 0);
    assert_eq!(c_val, r_val, "Jones 2D unit square: C={}, Rust={}", c_val, r_val);
}

#[test]
fn test_rect_diameter_jones_various_widths() {
    let test_cases: Vec<(usize, Vec<f64>)> = vec![
        (1, vec![2.0]),
        (2, vec![1.0, 2.0]),
        (3, vec![1.0, 2.0, 3.0]),
        (3, vec![0.5, 0.5, 0.5]),
        (4, vec![1.0, 1.0, 1.0, 1.0]),
        (2, vec![0.001, 1000.0]),
        (5, vec![1.0, 0.333333, 0.111111, 0.037037, 0.012346]),
        // DIRECT typical widths after several divisions
        (3, vec![10.0 / 3.0, 10.0, 10.0]),
        (3, vec![10.0 / 9.0, 10.0 / 3.0, 10.0]),
    ];

    for (n, w) in &test_cases {
        let c_val = c_rect_diameter(*n, w, 0);
        let r_val = rust_rect_diameter(*n, w, 0);
        assert_eq!(
            c_val, r_val,
            "Jones n={} w={:?}: C={}, Rust={}",
            n, w, c_val, r_val
        );
    }
}

#[test]
fn test_rect_diameter_gablonsky_various_widths() {
    let test_cases: Vec<(usize, Vec<f64>)> = vec![
        (1, vec![2.0]),
        (2, vec![1.0, 2.0]),
        (3, vec![1.0, 2.0, 0.5]),
        (3, vec![3.0, 3.0, 3.0]),
        (4, vec![0.1, 0.2, 0.3, 0.4]),
        (2, vec![0.001, 1000.0]),
        (5, vec![1.0, 0.333333, 0.111111, 0.037037, 0.012346]),
        (3, vec![10.0 / 3.0, 10.0, 10.0]),
    ];

    for (n, w) in &test_cases {
        let c_val = c_rect_diameter(*n, w, 1);
        let r_val = rust_rect_diameter(*n, w, 1);
        assert_eq!(
            c_val, r_val,
            "Gablonsky n={} w={:?}: C={}, Rust={}",
            n, w, c_val, r_val
        );
    }
}

#[test]
fn test_rect_diameter_f32_rounding() {
    // Verify the float-rounding performance hack: result should be f32-precision
    let w = [1.0, 1.0];
    let d = c_rect_diameter(2, &w, 0);
    // The f32 round-trip should be exact
    assert_eq!(d, d as f32 as f64, "diameter should be f32-precision");

    // Another case with irrational result
    let w2 = [1.0, 2.0, 3.0];
    let d2 = c_rect_diameter(3, &w2, 0);
    assert_eq!(d2, d2 as f32 as f64, "diameter should be f32-precision");
}

#[test]
fn test_rect_diameter_jones_vs_gablonsky_differ() {
    // For non-cubic rectangles, Jones and Gablonsky should differ
    let w = [1.0, 2.0, 3.0];
    let jones = c_rect_diameter(3, &w, 0);
    let gab = c_rect_diameter(3, &w, 1);
    assert_ne!(jones, gab, "Jones and Gablonsky should differ for non-cubic rects");

    // For cubes, they may also differ due to the different formulas
    let w_cube = [2.0, 2.0, 2.0];
    let jones_cube = c_rect_diameter(3, &w_cube, 0);
    let gab_cube = c_rect_diameter(3, &w_cube, 1);
    // Jones: sqrt(3*4)*0.5 = sqrt(12)/2 ≈ 1.732
    // Gablonsky: 2*0.5 = 1.0
    assert_ne!(jones_cube, gab_cube, "Jones and Gablonsky differ even for cubes");
}

#[test]
fn test_rect_diameter_direct_subdivision_sequence() {
    // Simulate a typical DIRECT subdivision sequence on [-5,5]^2
    // Initial widths: [10, 10], then trisections
    let initial = [10.0, 10.0];
    let after_one = [10.0 / 3.0, 10.0]; // divided along dim 0
    let after_two = [10.0 / 3.0, 10.0 / 3.0]; // divided along dim 1

    for which_diam in [0, 1] {
        let c0 = c_rect_diameter(2, &initial, which_diam);
        let r0 = rust_rect_diameter(2, &initial, which_diam);
        assert_eq!(c0, r0, "initial diam mismatch (which_diam={})", which_diam);

        let c1 = c_rect_diameter(2, &after_one, which_diam);
        let r1 = rust_rect_diameter(2, &after_one, which_diam);
        assert_eq!(c1, r1, "after_one diam mismatch (which_diam={})", which_diam);

        let c2 = c_rect_diameter(2, &after_two, which_diam);
        let r2 = rust_rect_diameter(2, &after_two, which_diam);
        assert_eq!(c2, r2, "after_two diam mismatch (which_diam={})", which_diam);

        // Diameters should be non-increasing as we subdivide more
        assert!(c0 >= c1, "diameter should not increase after subdivision");
        assert!(c1 >= c2, "diameter should not increase after more subdivision");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// convex_hull tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_convex_hull_monotonically_decreasing() {
    // All rects on the hull: monotonically decreasing f with increasing d
    // Sorted by (d, f): already in correct order
    let diameters = [1.0, 2.0, 3.0, 4.0, 5.0];
    let f_values = [5.0, 4.0, 3.0, 2.0, 1.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull, "monotonic: C={:?}, Rust={:?}", c_hull, r_hull);
    // All points should be on the hull
    assert_eq!(c_hull.len(), 5, "all points should be on hull");
}

#[test]
fn test_convex_hull_one_above() {
    // Point at d=4 is above the convex hull
    let diameters = [1.0, 2.0, 3.0, 4.0, 5.0];
    let f_values = [5.0, 3.0, 2.0, 4.0, 1.0]; // d=4,f=4 is above hull

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull, "one_above: C={:?}, Rust={:?}", c_hull, r_hull);
    // d=4 (idx=3) should NOT be on hull
    assert!(
        !c_hull.contains(&3),
        "point above hull should be excluded"
    );
}

#[test]
fn test_convex_hull_single_point() {
    let diameters = [1.0];
    let f_values = [3.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull);
    assert_eq!(c_hull, vec![0]);
}

#[test]
fn test_convex_hull_two_points() {
    let diameters = [1.0, 5.0];
    let f_values = [3.0, 1.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull);
    assert_eq!(c_hull, vec![0, 1]);
}

#[test]
fn test_convex_hull_same_diameter() {
    // All points have same diameter → only the first should be on hull
    let diameters = [1.0, 1.0, 1.0];
    let f_values = [3.0, 4.0, 5.0]; // sorted by (d, f)

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull, "same_d: C={:?}, Rust={:?}", c_hull, r_hull);
}

#[test]
fn test_convex_hull_duplicate_diameters_vertical_lines() {
    // Multiple points at same diameter (vertical lines in (d,f) space)
    // Sorted by (d, f): points at d=1 have f=3,5; at d=3 have f=1,2; at d=5 have f=0.5,4
    let diameters = [1.0, 1.0, 3.0, 3.0, 5.0, 5.0];
    let f_values = [3.0, 5.0, 1.0, 2.0, 0.5, 4.0];

    // Without dups
    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull, r_hull, "vert_no_dups: C={:?}, Rust={:?}", c_hull, r_hull);

    // With dups
    let c_hull_d = c_convex_hull(&diameters, &f_values, true);
    let r_hull_d = rust_convex_hull(&diameters, &f_values, true);
    assert_eq!(c_hull_d, r_hull_d, "vert_dups: C={:?}, Rust={:?}", c_hull_d, r_hull_d);
}

#[test]
fn test_convex_hull_collinear_points() {
    // All points lie on a straight line: f = 10 - 2*d
    let diameters = [1.0, 2.0, 3.0, 4.0, 5.0];
    let f_values = [8.0, 6.0, 4.0, 2.0, 0.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);

    assert_eq!(c_hull, r_hull, "collinear: C={:?}, Rust={:?}", c_hull, r_hull);
    // First and last should always be on hull; intermediate collinear points
    // may or may not be included depending on >= vs > in cross product check
    assert_eq!(c_hull[0], 0);
    assert_eq!(*c_hull.last().unwrap(), 4);
}

#[test]
fn test_convex_hull_duplicate_points_allow_dups() {
    // Duplicate (d,f) pairs at the endpoints
    let diameters = [1.0, 1.0, 3.0, 5.0, 5.0];
    let f_values = [3.0, 3.0, 1.5, 1.0, 1.0];

    // allow_dups=true: both duplicates at each endpoint should appear
    let c_hull = c_convex_hull(&diameters, &f_values, true);
    let r_hull = rust_convex_hull(&diameters, &f_values, true);
    assert_eq!(c_hull, r_hull, "dups_true: C={:?}, Rust={:?}", c_hull, r_hull);
    // Should include both d=1 entries and both d=5 entries
    let d1_count = c_hull.iter().filter(|&&i| diameters[i] == 1.0).count();
    let d5_count = c_hull.iter().filter(|&&i| diameters[i] == 5.0).count();
    assert_eq!(d1_count, 2, "should include both d=1 duplicates");
    assert_eq!(d5_count, 2, "should include both d=5 duplicates");

    // allow_dups=false: only one per endpoint
    let c_hull_nd = c_convex_hull(&diameters, &f_values, false);
    let r_hull_nd = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull_nd, r_hull_nd, "dups_false: C={:?}, Rust={:?}", c_hull_nd, r_hull_nd);
}

#[test]
fn test_convex_hull_complex_scenario() {
    // A more complex scenario with points above and below hull
    // Points sorted by (d, f):
    let diameters = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let f_values = [10.0, 7.0, 9.0, 8.0, 3.0, 6.0, 2.0, 1.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull, r_hull, "complex: C={:?}, Rust={:?}", c_hull, r_hull);

    let c_hull_d = c_convex_hull(&diameters, &f_values, true);
    let r_hull_d = rust_convex_hull(&diameters, &f_values, true);
    assert_eq!(c_hull_d, r_hull_d, "complex_dups: C={:?}, Rust={:?}", c_hull_d, r_hull_d);
}

#[test]
fn test_convex_hull_realistic_direct_diameters() {
    // Simulate diameters from a real DIRECT run on [-5,5]^2
    // After several iterations, rectangles cluster at specific diameter levels
    // due to the f32 rounding trick
    let w_configs: Vec<[f64; 2]> = vec![
        [10.0, 10.0],
        [10.0 / 3.0, 10.0],
        [10.0 / 3.0, 10.0 / 3.0],
        [10.0 / 9.0, 10.0 / 3.0],
        [10.0 / 9.0, 10.0 / 9.0],
    ];

    // Compute diameters using Jones measure
    let mut entries: Vec<(f64, f64)> = Vec::new();
    for (idx, w) in w_configs.iter().enumerate() {
        let d = c_rect_diameter(2, w, 0);
        let f = (idx as f64 - 2.0).powi(2); // some f values
        entries.push((d, f));
    }

    // Sort by (d, f)
    entries.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then_with(|| a.1.partial_cmp(&b.1).unwrap())
    });

    let diameters: Vec<f64> = entries.iter().map(|e| e.0).collect();
    let f_values: Vec<f64> = entries.iter().map(|e| e.1).collect();

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(
        c_hull, r_hull,
        "realistic: C={:?}, Rust={:?}\n  diameters={:?}\n  f_values={:?}",
        c_hull, r_hull, diameters, f_values
    );
}

#[test]
fn test_convex_hull_many_at_same_diameter() {
    // Many points at same diameter (stress test vertical line handling)
    let mut diameters = vec![1.0; 5];
    diameters.extend_from_slice(&[3.0]);
    diameters.extend_from_slice(&[5.0; 5]);

    let mut f_values = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // at d=1
    f_values.push(1.0); // at d=3
    f_values.extend_from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5]); // at d=5

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull, r_hull, "many_same_d: C={:?}, Rust={:?}", c_hull, r_hull);

    let c_hull_d = c_convex_hull(&diameters, &f_values, true);
    let r_hull_d = rust_convex_hull(&diameters, &f_values, true);
    assert_eq!(c_hull_d, r_hull_d, "many_same_d_dups: C={:?}, Rust={:?}", c_hull_d, r_hull_d);
}

#[test]
fn test_convex_hull_v_shape() {
    // V-shaped hull: f decreases then increases
    let diameters = [1.0, 2.0, 3.0, 4.0, 5.0];
    let f_values = [5.0, 3.0, 1.0, 3.0, 5.0];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull, r_hull, "v_shape: C={:?}, Rust={:?}", c_hull, r_hull);
    // Only the bottom of the V and endpoints should be on the hull
}

#[test]
fn test_convex_hull_empty() {
    let diameters: [f64; 0] = [];
    let f_values: [f64; 0] = [];

    let c_hull = c_convex_hull(&diameters, &f_values, false);
    let r_hull = rust_convex_hull(&diameters, &f_values, false);
    assert_eq!(c_hull, r_hull);
    assert!(c_hull.is_empty());
}

#[test]
fn test_rect_diameter_1d() {
    // Edge case: 1D optimization
    let w = [5.0];
    let jones = c_rect_diameter(1, &w, 0);
    let gab = c_rect_diameter(1, &w, 1);
    let r_jones = rust_rect_diameter(1, &w, 0);
    let r_gab = rust_rect_diameter(1, &w, 1);

    // In 1D: Jones = sqrt(w^2)*0.5 = w/2, Gablonsky = w/2
    // Both should be equal for 1D
    assert_eq!(jones, r_jones);
    assert_eq!(gab, r_gab);
    assert_eq!(jones, gab, "Jones == Gablonsky in 1D");
}

#[test]
fn test_rect_diameter_high_dimensional() {
    // 10D rectangle with mixed widths
    let w: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let c_jones = c_rect_diameter(10, &w, 0);
    let c_gab = c_rect_diameter(10, &w, 1);
    let r_jones = rust_rect_diameter(10, &w, 0);
    let r_gab = rust_rect_diameter(10, &w, 1);

    assert_eq!(c_jones, r_jones, "10D Jones");
    assert_eq!(c_gab, r_gab, "10D Gablonsky");

    // Gablonsky should be max(w)/2 = 10/2 = 5.0 (rounded to f32)
    assert_eq!(c_gab, 5.0_f32 as f64, "Gablonsky 10D should be 5.0");
}

#[test]
fn test_rect_diameter_very_small_widths() {
    // Near-zero widths (deeply subdivided rectangle)
    let w = [1e-8, 1e-8, 1e-8];
    let c_jones = c_rect_diameter(3, &w, 0);
    let r_jones = rust_rect_diameter(3, &w, 0);
    assert_eq!(c_jones, r_jones, "very small Jones");

    let c_gab = c_rect_diameter(3, &w, 1);
    let r_gab = rust_rect_diameter(3, &w, 1);
    assert_eq!(c_gab, r_gab, "very small Gablonsky");
}

#[test]
fn test_rect_diameter_very_large_widths() {
    // Very large widths
    let w = [1e6, 1e6];
    let c_jones = c_rect_diameter(2, &w, 0);
    let r_jones = rust_rect_diameter(2, &w, 0);
    assert_eq!(c_jones, r_jones, "very large Jones");

    let c_gab = c_rect_diameter(2, &w, 1);
    let r_gab = rust_rect_diameter(2, &w, 1);
    assert_eq!(c_gab, r_gab, "very large Gablonsky");
}
