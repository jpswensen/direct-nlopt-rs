#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for divide_rectangle (direct_dirdivide_) and
//! dimension sorting.
//!
//! Verifies that the Rust Direct::divide_rectangle() produces identical
//! results to NLOPT C's direct_dirdivide_() for all test scenarios.
//!
//! NLOPT C function: direct_dirdivide_() in DIRsubrout.c lines 944-1009
//! Rust function: Direct::divide_rectangle() in direct.rs

use std::os::raw::{c_double, c_int};

// FFI declaration for NLOPT's direct_dirdivide_().
// Compiled from DIRsubrout.c via build.rs when "nlopt-compare" is enabled.
//
// C signature (direct-internal.h:71-75):
//   extern void direct_dirdivide_(
//       integer *new__, integer *currentlength,
//       integer *length, integer *point, integer *arrayi, integer *sample,
//       integer *list2, doublereal *w, integer *maxi, doublereal *f,
//       integer *maxfunc, const integer *maxdeep, integer *n);
extern "C" {
    fn direct_dirdivide_(
        new__: *mut c_int,
        currentlength: *mut c_int,
        length: *mut c_int,
        point: *mut c_int,
        arrayi: *mut c_int,
        sample: *mut c_int,
        list2: *mut c_int,
        w: *mut c_double,
        maxi: *mut c_int,
        f: *mut c_double,
        maxfunc: *mut c_int,
        maxdeep: *const c_int,
        n: *mut c_int,
    );
}

/// Describes one divide_rectangle test scenario.
struct DivideTestCase {
    name: &'static str,
    dim: usize,
    /// 1-based dimension indices to divide (matching NLOPT's arrayi)
    arrayi: Vec<i32>,
    /// (f_pos, f_neg) for each dimension in arrayi
    f_pairs: Vec<(f64, f64)>,
    /// Starting depth/length index
    current_length: i32,
    /// Initial lengths for the parent rectangle (0-based, per dimension)
    parent_lengths: Vec<i32>,
}

/// Call NLOPT C's direct_dirdivide_() with a test scenario and return the
/// resulting length arrays for the parent and all children.
///
/// Returns: (parent_lengths, children_lengths) where children_lengths
/// is a Vec of (pos_child_lengths, neg_child_lengths) tuples.
fn call_c_dirdivide(tc: &DivideTestCase) -> (Vec<i32>, Vec<(Vec<i32>, Vec<i32>)>) {
    let n = tc.dim as c_int;
    let maxi_val = tc.arrayi.len() as c_int;
    let maxfunc: c_int = 2 + 2 * maxi_val + 2; // parent + children + extra
    let maxdeep: c_int = 100;
    let sample: c_int = 1; // parent at position 1 (1-based)

    // Allocate arrays with NLOPT's expected layout.
    // length: n × maxfunc, accessed as length[dim + pos * n] after adjustment
    // The C code does: length -= (1 + n), so length[i + pos*n] maps to
    // our array at index (i + pos*n - 1 - n) = (i - 1) + (pos - 1)*n
    // for 1-based i=1..n and 1-based pos.
    //
    // In 0-based: arr[(pos-1)*n + (dim-1)]
    let len_size = (maxfunc as usize) * (n as usize);
    let mut length = vec![0i32; len_size];

    // Set parent lengths at pos=1 (0-based: row 0)
    for j in 0..tc.dim {
        length[0 * tc.dim + j] = tc.parent_lengths[j];
    }

    // point: maxfunc elements, 1-based via --point adjustment
    // C does: --point, so point[idx] maps to our array at (idx - 1)
    // We need 0-based: point_arr[idx-1] = next
    let mut point = vec![0i32; maxfunc as usize];

    // Set up the child chain: start at pos=2 (1-based)
    // Chain: 2 → 3 → 4 → 5 → ... → 2+2*maxi-1 → 0
    let start = 2i32;
    for k in 0..(2 * maxi_val) {
        let idx = (start + k) as usize;
        if k < 2 * maxi_val - 1 {
            point[idx - 1] = start + k + 1;
        } else {
            point[idx - 1] = 0;
        }
    }

    // Copy parent lengths to all children
    for k in 0..(2 * maxi_val) {
        let idx = (start + k) as usize;
        for j in 0..tc.dim {
            length[(idx - 1) * tc.dim + j] = tc.parent_lengths[j];
        }
    }

    // f: 2 × maxfunc, accessed as f[(pos << 1) + 1] after f -= 3 adjustment
    // C does: f -= 3, so f[(pos << 1) + 1] maps to our array at
    // (pos*2 + 1 - 3) = pos*2 - 2
    // The f-value for pos is at f_arr[pos*2 - 2]
    let f_size = (maxfunc as usize) * 2 + 4; // extra padding
    let mut f = vec![0.0f64; f_size];

    // Set f-values for children
    for (k, &(f_pos, f_neg)) in tc.f_pairs.iter().enumerate() {
        let pos_idx = (start as usize) + 2 * k;
        let neg_idx = (start as usize) + 2 * k + 1;
        // f-value at pos_idx: f_arr[pos_idx*2 - 2]
        f[pos_idx * 2 - 2] = f_pos;
        // f-value at neg_idx: f_arr[neg_idx*2 - 2]
        f[neg_idx * 2 - 2] = f_neg;
    }

    // arrayi: 1-based dimension indices
    // C does: --arrayi, so arrayi[i] (1-based) maps to our array at (i-1)
    let mut arrayi = vec![0i32; tc.dim + 1];
    for (k, &dim_idx) in tc.arrayi.iter().enumerate() {
        arrayi[k] = dim_idx;
    }

    // Scratch arrays
    let mut w = vec![0.0f64; tc.dim + 1];
    let list2_size = (n as usize) * 2 + 2;
    let mut list2 = vec![0i32; list2_size * 2]; // extra space

    let mut new_val = start;
    let mut current_length = tc.current_length;
    let mut sample_val = sample;
    let mut maxi_mut = maxi_val;
    let mut maxfunc_mut = maxfunc;
    let mut n_mut = n;

    unsafe {
        direct_dirdivide_(
            &mut new_val,
            &mut current_length,
            length.as_mut_ptr(),
            point.as_mut_ptr(),
            arrayi.as_mut_ptr(),
            &mut sample_val,
            list2.as_mut_ptr(),
            w.as_mut_ptr(),
            &mut maxi_mut,
            f.as_mut_ptr(),
            &mut maxfunc_mut,
            &maxdeep,
            &mut n_mut,
        );
    }

    // Extract resulting lengths
    let parent_lengths: Vec<i32> = (0..tc.dim)
        .map(|j| length[0 * tc.dim + j])
        .collect();

    let mut children_lengths = Vec::new();
    for k in 0..(maxi_val as usize) {
        let pos_idx = (start as usize) + 2 * k;
        let neg_idx = (start as usize) + 2 * k + 1;
        let pos_lens: Vec<i32> = (0..tc.dim)
            .map(|j| length[(pos_idx - 1) * tc.dim + j])
            .collect();
        let neg_lens: Vec<i32> = (0..tc.dim)
            .map(|j| length[(neg_idx - 1) * tc.dim + j])
            .collect();
        children_lengths.push((pos_lens, neg_lens));
    }

    (parent_lengths, children_lengths)
}

/// Call Rust's divide_rectangle with the same scenario and return
/// the resulting length arrays.
fn call_rust_divide(tc: &DivideTestCase) -> (Vec<i32>, Vec<(Vec<i32>, Vec<i32>)>) {
    use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

    let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); tc.dim];
    let opts = DirectOptions {
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        max_feval: 10000,
        ..Default::default()
    };
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
    let mut d = direct_nlopt::direct::Direct::new(sphere, &bounds, opts).unwrap();

    let sample = 1usize;
    let maxi = tc.arrayi.len();

    // Set parent rect at index 1
    for j in 0..tc.dim {
        d.storage.set_center(sample, j, 0.5);
        d.storage.set_length(sample, j, tc.parent_lengths[j]);
    }
    d.storage.point[sample] = 0;
    d.storage.free = 2;

    // Allocate 2*maxi children
    let start = 2usize;
    for k in 0..(2 * maxi) {
        let idx = start + k;
        d.storage.copy_center(idx, sample);
        d.storage.copy_lengths(idx, sample);
        if k < 2 * maxi - 1 {
            d.storage.point[idx] = (idx + 1) as i32;
        } else {
            d.storage.point[idx] = 0;
        }
    }

    // Set f-values for each pos/neg pair
    for (k, &(f_pos, f_neg)) in tc.f_pairs.iter().enumerate() {
        let pos_idx = start + 2 * k;
        let neg_idx = start + 2 * k + 1;
        d.storage.set_f(pos_idx, f_pos, 0.0);
        d.storage.set_f(neg_idx, f_neg, 0.0);
    }

    // Convert arrayi to usize for Rust API (already 1-based)
    let arrayi_usize: Vec<usize> = tc.arrayi.iter().map(|&x| x as usize).collect();

    // Call divide_rectangle
    d.divide_rectangle(start, tc.current_length, sample, &arrayi_usize, maxi);

    // Extract resulting lengths
    let parent_lengths: Vec<i32> = (0..tc.dim)
        .map(|j| d.storage.length(sample, j))
        .collect();

    let mut children_lengths = Vec::new();
    for k in 0..maxi {
        let pos_idx = start + 2 * k;
        let neg_idx = start + 2 * k + 1;
        let pos_lens: Vec<i32> = (0..tc.dim)
            .map(|j| d.storage.length(pos_idx, j))
            .collect();
        let neg_lens: Vec<i32> = (0..tc.dim)
            .map(|j| d.storage.length(neg_idx, j))
            .collect();
        children_lengths.push((pos_lens, neg_lens));
    }

    (parent_lengths, children_lengths)
}

/// Compare C and Rust outputs for a single test case.
fn compare_divide(tc: &DivideTestCase) {
    let (c_parent, c_children) = call_c_dirdivide(tc);
    let (r_parent, r_children) = call_rust_divide(tc);

    assert_eq!(
        c_parent, r_parent,
        "[{}] Parent lengths mismatch: C={:?} Rust={:?}",
        tc.name, c_parent, r_parent
    );

    assert_eq!(
        c_children.len(),
        r_children.len(),
        "[{}] Children count mismatch",
        tc.name
    );

    for (k, ((c_pos, c_neg), (r_pos, r_neg))) in
        c_children.iter().zip(r_children.iter()).enumerate()
    {
        assert_eq!(
            c_pos, r_pos,
            "[{}] Dim {} pos child lengths mismatch: C={:?} Rust={:?}",
            tc.name, k, c_pos, r_pos
        );
        assert_eq!(
            c_neg, r_neg,
            "[{}] Dim {} neg child lengths mismatch: C={:?} Rust={:?}",
            tc.name, k, c_neg, r_neg
        );
    }
}

// ====================================================================
// Test 1: 3D rectangle division along all dims with unequal w-values
//         Verifies dimension sort by min(f+, f-)
// ====================================================================

#[test]
fn test_divide_3d_unequal_w_sort() {
    // 3D, all dims longest: arrayi = [1, 2, 3]
    // f-values: dim1: (10.0, 12.0), dim2: (5.0, 8.0), dim3: (15.0, 20.0)
    // w = [min(10,12)=10, min(5,8)=5, min(15,20)=15]
    // Sort by w ascending: dim2(5) < dim1(10) < dim3(15)
    compare_divide(&DivideTestCase {
        name: "3d_unequal_w",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![(10.0, 12.0), (5.0, 8.0), (15.0, 20.0)],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    });
}

// ====================================================================
// Test 2: 2D with equal w-values — stable sort preserves order
// ====================================================================

#[test]
fn test_divide_2d_equal_w_stable_sort() {
    // 2D sphere: all w values equal → stable sort preserves original order
    compare_divide(&DivideTestCase {
        name: "2d_equal_w",
        dim: 2,
        arrayi: vec![1, 2],
        f_pairs: vec![(10.0, 10.0), (10.0, 10.0)],
        current_length: 0,
        parent_lengths: vec![0, 0],
    });
}

// ====================================================================
// Test 3: 1D — single dimension only
// ====================================================================

#[test]
fn test_divide_1d_single_dim() {
    compare_divide(&DivideTestCase {
        name: "1d_single",
        dim: 1,
        arrayi: vec![1],
        f_pairs: vec![(7.0, 3.0)],
        current_length: 0,
        parent_lengths: vec![0],
    });
}

// ====================================================================
// Test 4: Length increment with nonzero current_length
// ====================================================================

#[test]
fn test_divide_nonzero_current_length() {
    compare_divide(&DivideTestCase {
        name: "nonzero_current_length",
        dim: 2,
        arrayi: vec![1, 2],
        f_pairs: vec![(5.0, 8.0), (10.0, 12.0)],
        current_length: 3,
        parent_lengths: vec![3, 3],
    });
}

// ====================================================================
// Test 5: Parent center should be unchanged (lengths only test)
//         and children get correct length updates
// ====================================================================

#[test]
fn test_divide_parent_lengths_all_set() {
    // After divide, parent should have ALL divided dims set to new_len
    let tc = DivideTestCase {
        name: "parent_all_set",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    };

    let (c_parent, _) = call_c_dirdivide(&tc);
    let (r_parent, _) = call_rust_divide(&tc);

    // All dims should be set to current_length + 1 = 1
    assert_eq!(c_parent, vec![1, 1, 1], "C parent should have all dims = 1");
    assert_eq!(r_parent, vec![1, 1, 1], "Rust parent should have all dims = 1");
}

// ====================================================================
// Test 6: 3D with only 2 dims longest (partial division)
// ====================================================================

#[test]
fn test_divide_3d_two_dims_longest() {
    // 3D but only dims 1 and 3 are longest (dim 2 already divided)
    compare_divide(&DivideTestCase {
        name: "3d_two_dims",
        dim: 3,
        arrayi: vec![1, 3],
        f_pairs: vec![(20.0, 15.0), (8.0, 12.0)],
        current_length: 1,
        parent_lengths: vec![1, 1, 1],
    });
}

// ====================================================================
// Test 7: w uses min(f_pos, f_neg) — large f_pos, small f_neg
// ====================================================================

#[test]
fn test_divide_w_uses_min_f() {
    // dim1: f_pos=100, f_neg=1 → w=1
    // dim2: f_pos=2, f_neg=50 → w=2
    // Sort: dim1(1) < dim2(2)
    compare_divide(&DivideTestCase {
        name: "w_uses_min",
        dim: 2,
        arrayi: vec![1, 2],
        f_pairs: vec![(100.0, 1.0), (2.0, 50.0)],
        current_length: 0,
        parent_lengths: vec![0, 0],
    });
}

// ====================================================================
// Test 8: 5D with all dims — verifies complete sort and length cascade
// ====================================================================

#[test]
fn test_divide_5d_all_dims() {
    // Sort by w: dim4(5) < dim2(10) < dim5(18) < dim1(25) < dim3(45)
    compare_divide(&DivideTestCase {
        name: "5d_all_dims",
        dim: 5,
        arrayi: vec![1, 2, 3, 4, 5],
        f_pairs: vec![
            (30.0, 25.0), // dim1: w=25
            (10.0, 15.0), // dim2: w=10
            (50.0, 45.0), // dim3: w=45
            (5.0, 8.0),   // dim4: w=5
            (20.0, 18.0), // dim5: w=18
        ],
        current_length: 0,
        parent_lengths: vec![0, 0, 0, 0, 0],
    });
}

// ====================================================================
// Test 9: Deeper division — current_length=5
// ====================================================================

#[test]
fn test_divide_deep_level() {
    compare_divide(&DivideTestCase {
        name: "deep_level_5",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![(1.0, 2.0), (0.5, 0.3), (3.0, 4.0)],
        current_length: 5,
        parent_lengths: vec![5, 5, 5],
    });
}

// ====================================================================
// Test 10: All same f-values — degenerate case
// ====================================================================

#[test]
fn test_divide_all_same_f() {
    compare_divide(&DivideTestCase {
        name: "all_same_f",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![(7.0, 7.0), (7.0, 7.0), (7.0, 7.0)],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    });
}

// ====================================================================
// Test 11: Negative f-values
// ====================================================================

#[test]
fn test_divide_negative_f_values() {
    compare_divide(&DivideTestCase {
        name: "negative_f",
        dim: 2,
        arrayi: vec![1, 2],
        f_pairs: vec![(-10.0, -5.0), (-3.0, -8.0)],
        current_length: 0,
        parent_lengths: vec![0, 0],
    });
}

// ====================================================================
// Test 12: Mixed parent lengths (not all equal)
// ====================================================================

#[test]
fn test_divide_mixed_parent_lengths() {
    // Parent has mixed lengths — only divide dims 2 and 3 (the longest)
    compare_divide(&DivideTestCase {
        name: "mixed_parent_lengths",
        dim: 3,
        arrayi: vec![2, 3],
        f_pairs: vec![(4.0, 6.0), (2.0, 9.0)],
        current_length: 1,
        parent_lengths: vec![2, 1, 1],
    });
}

// ====================================================================
// Test 13: Reversed sort order — dim indices not in ascending order
// ====================================================================

#[test]
fn test_divide_reversed_arrayi() {
    // arrayi = [3, 1] — non-sequential dimension order
    compare_divide(&DivideTestCase {
        name: "reversed_arrayi",
        dim: 3,
        arrayi: vec![3, 1],
        f_pairs: vec![(10.0, 20.0), (5.0, 3.0)],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    });
}

// ====================================================================
// Test 14: Very small f-value differences
// ====================================================================

#[test]
fn test_divide_tiny_f_differences() {
    compare_divide(&DivideTestCase {
        name: "tiny_f_diff",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![
            (1.0000001, 1.0000002),
            (1.0000000, 1.0000003),
            (1.0000004, 1.0000005),
        ],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    });
}

// ====================================================================
// Test 15: Comprehensive sweep — many test cases at once
// ====================================================================

#[test]
fn test_divide_comprehensive_sweep() {
    let test_cases = vec![
        // 1D cases
        DivideTestCase {
            name: "sweep_1d_a",
            dim: 1,
            arrayi: vec![1],
            f_pairs: vec![(1.0, 2.0)],
            current_length: 0,
            parent_lengths: vec![0],
        },
        DivideTestCase {
            name: "sweep_1d_b",
            dim: 1,
            arrayi: vec![1],
            f_pairs: vec![(100.0, 0.01)],
            current_length: 10,
            parent_lengths: vec![10],
        },
        // 2D cases
        DivideTestCase {
            name: "sweep_2d_a",
            dim: 2,
            arrayi: vec![1, 2],
            f_pairs: vec![(3.0, 1.0), (2.0, 4.0)],
            current_length: 0,
            parent_lengths: vec![0, 0],
        },
        DivideTestCase {
            name: "sweep_2d_b",
            dim: 2,
            arrayi: vec![2],
            f_pairs: vec![(5.0, 5.0)],
            current_length: 2,
            parent_lengths: vec![3, 2],
        },
        // 4D case
        DivideTestCase {
            name: "sweep_4d",
            dim: 4,
            arrayi: vec![1, 2, 3, 4],
            f_pairs: vec![
                (40.0, 35.0), // w=35
                (20.0, 25.0), // w=20
                (10.0, 15.0), // w=10
                (30.0, 28.0), // w=28
            ],
            current_length: 0,
            parent_lengths: vec![0, 0, 0, 0],
        },
        // 3D with subset of dims
        DivideTestCase {
            name: "sweep_3d_subset",
            dim: 3,
            arrayi: vec![2],
            f_pairs: vec![(12.0, 8.0)],
            current_length: 4,
            parent_lengths: vec![5, 4, 5],
        },
    ];

    for tc in &test_cases {
        compare_divide(tc);
    }
}

// ====================================================================
// Test 16: Verify resulting lengths have correct cascade pattern
//          Dims divided first → their lengths set in ALL subsequent children
// ====================================================================

#[test]
fn test_divide_length_cascade_pattern() {
    let tc = DivideTestCase {
        name: "cascade_verify",
        dim: 3,
        arrayi: vec![1, 2, 3],
        f_pairs: vec![
            (30.0, 25.0), // dim1: w=25
            (10.0, 15.0), // dim2: w=10
            (50.0, 45.0), // dim3: w=45
        ],
        current_length: 0,
        parent_lengths: vec![0, 0, 0],
    };

    let (c_parent, c_children) = call_c_dirdivide(&tc);
    let (r_parent, r_children) = call_rust_divide(&tc);

    // Both should match
    assert_eq!(c_parent, r_parent);
    assert_eq!(c_children, r_children);

    // Verify the cascade pattern explicitly:
    // Sort by w: dim2(10) < dim1(25) < dim3(45)
    //
    // dim2 children (sorted first) — only get their own dim set
    // dim1 children (sorted second) — get dim2's dim AND their own
    // dim3 children (sorted last) — get all dims set

    let new_len = 1; // current_length(0) + 1

    // Children are in arrayi order: dim1 pair = children[0], dim2 = children[1], dim3 = children[2]
    let (_dim1_pos, _dim1_neg) = &c_children[0]; // arrayi[0]=1 → dim1 pair
    let (dim2_pos, dim2_neg) = &c_children[1]; // arrayi[1]=2 → dim2 pair
    let (dim3_pos, dim3_neg) = &c_children[2]; // arrayi[2]=3 → dim3 pair

    // dim2 pair is sorted first (lowest w=10):
    //   Only covered by sort position 0 → only dim 1 (0-indexed) updated
    //   Wait: sorted i=0 is dim2 (dim_k=1), so only dim_k=1 is set in ALL children at i=0
    //   Then dim2 pair itself only participates in i=0, so it only gets dim_k=1 set
    assert_eq!(dim2_pos[1], new_len, "dim2 pos should have dim1 (sorted) set");
    assert_eq!(dim2_neg[1], new_len, "dim2 neg should have dim1 (sorted) set");

    // dim3 pair is sorted last (highest w=45):
    //   Covered by all sort positions → all dims set
    for j in 0..3 {
        assert_eq!(dim3_pos[j], new_len, "dim3 pos should have all dims set");
        assert_eq!(dim3_neg[j], new_len, "dim3 neg should have all dims set");
    }
}

// ====================================================================
// Test 17: Verify lengths resulting from integrated init+divide sequence
//          (end-to-end: initialize, pick rect, sample, evaluate, divide)
// ====================================================================

#[test]
fn test_divide_integrated_with_initialization() {
    use direct_nlopt::direct::Direct;
    use direct_nlopt::types::{DirectAlgorithm, DirectOptions};

    let bounds = vec![(-5.0, 5.0); 3];
    let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
    let options = DirectOptions {
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        max_feval: 10000,
        max_iter: 100,
        ..Default::default()
    };
    let mut d = Direct::new(sphere, &bounds, options).unwrap();
    d.initialize().unwrap();

    // Pick a non-cube rectangle and divide it further
    // After init in 3D, rects exist at different depth levels.
    // Find one with non-uniform lengths
    let (arrayi, maxi) = d.storage.get_longest_dims(2);
    let depth = d.storage.get_max_deep(2);
    let delta = d.storage.thirds[(depth + 1) as usize];

    let new_start = d.sample_points(2, &arrayi, delta).unwrap();
    d.evaluate_sample_points(new_start, maxi).unwrap();
    d.divide_rectangle(new_start, depth, 2, &arrayi, maxi);

    // Verify: parent had its divided dimension lengths incremented
    let new_len = depth + 1;
    for k in 0..maxi {
        let dim_j = arrayi[k] - 1; // 0-based
        assert_eq!(
            d.storage.length(2, dim_j),
            new_len,
            "Parent dim {} should be updated to {}",
            dim_j,
            new_len
        );
    }

    // Verify: children exist and have correct lengths
    let mut pos = new_start;
    for k in 0..maxi {
        let dim_j = arrayi[k] - 1;
        let pos_child = pos;
        let neg_child = d.storage.point[pos_child] as usize;
        assert_eq!(d.storage.length(pos_child, dim_j), new_len);
        assert_eq!(d.storage.length(neg_child, dim_j), new_len);
        pos = d.storage.point[neg_child] as usize;
    }
}
