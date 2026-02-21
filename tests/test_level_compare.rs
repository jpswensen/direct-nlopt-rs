#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for level computation (dirgetlevel_) for both algorithm variants.
//! Verifies that the Rust get_level() produces identical results to NLOPT C direct_dirgetlevel_().

use std::os::raw::c_int;

// FFI declaration for NLOPT's internal level computation function.
// Compiled from DIRsubrout.c via build.rs when "nlopt-compare" is enabled.
//
// C signature:
//   integer direct_dirgetlevel_(integer *pos, integer *length, integer *maxfunc,
//                                integer *n, integer jones)
//
// The C function uses Fortran-style 1-based column-major indexing:
//   length_dim1 = *n;
//   length_offset = 1 + length_dim1;
//   length -= length_offset;   // pointer decrement for 1-based access
//   Then accesses length[dim + pos * length_dim1] for dim=1..n, pos=1-based
extern "C" {
    fn direct_dirgetlevel_(
        pos: *const c_int,
        length: *const c_int,
        maxfunc: *const c_int,
        n: *const c_int,
        jones: c_int,
    ) -> c_int;
}

/// Call NLOPT C direct_dirgetlevel_() with given length indices for a rectangle at position `pos`.
///
/// The `all_lengths` array must be laid out in Fortran column-major order:
///   all_lengths[dim * maxfunc + pos] for dim=0..n-1, pos=0..maxfunc-1
/// But since the C code does pointer arithmetic to convert to 1-based indexing,
/// we need to pass the raw array and let the C code handle the offset.
///
/// For simplicity, we build the full length array as the C code expects:
///   length[dim + pos * n] where dim and pos are 1-based (after the pointer decrement).
///   The raw array (before decrement) is indexed as: [1 + n + (dim-1) + (pos-1)*n]
///   = [n + dim + (pos-1)*n] = [dim + pos*n]
///   So the raw 0-based index is: dim + pos * n  (with dim 1-based, pos 1-based)
///   = (dim_0based) + 1 + (pos_1based) * n
///   Actually, the C pointer decrement is: length -= (1 + n)
///   Then access: length[i__ + pos * n]  where i__ is 1-based dim
///   Raw index = (1 + n) + i__ + pos * n - 1 = n + i__ + pos * n
///   Wait, let me re-derive. The C code does:
///     length_dim1 = *n;
///     length_offset = 1 + length_dim1;   // = 1 + n
///     length -= length_offset;           // pointer moves back by (1+n)
///     Then accesses length[i__ + *pos * length_dim1] where i__ is 1..n, *pos is 1-based
///   So the actual array index (in the original array) is:
///     (i__ + *pos * n) - (1 + n) = i__ + pos*n - 1 - n = (i__-1) + (pos-1)*n
///   This is just 0-based (dim, pos) indexing: dim_0 + pos_0 * n
///   So the C code's length array should have: at 0-based index [dim_0 + pos_0 * n] = length value
///   for dim_0 = 0..n-1, pos_0 = 0..maxfunc-1
fn call_c_dirgetlevel(lengths_for_rect: &[i32], n: usize, jones: i32) -> i32 {
    // We need to set up the length array in the format the C code expects.
    // The C code accesses length[(dim_0) + (pos_0) * n] after pointer adjustment.
    // We'll place our rectangle at position 1 (1-based, so pos_0 = 0 in 0-based)
    // to keep it simple. Actually, the C code uses 1-based pos, and the pointer
    // decrement converts to 0-based. So if we pass pos=1, the actual access is:
    //   raw_index = (dim_1based - 1) + (pos_1based - 1) * n = dim_0 + 0*n = dim_0
    // So for pos=1, the length values are just at indices 0..n-1 in the raw array.
    //
    // But we need the array to be large enough. Let's use maxfunc=10.
    let maxfunc: c_int = 10;
    let n_c: c_int = n as c_int;
    let pos: c_int = 1; // 1-based position

    // Create length array: maxfunc * n elements
    let mut length_array = vec![0i32; (maxfunc as usize) * n];

    // Set lengths for our rectangle at pos=1 (0-based index: pos_0=0)
    // raw_index = dim_0 + pos_0 * n = dim_0 + 0 = dim_0
    for (dim, &len_val) in lengths_for_rect.iter().enumerate() {
        length_array[dim] = len_val;
    }

    unsafe { direct_dirgetlevel_(&pos, length_array.as_ptr(), &maxfunc, &n_c, jones) }
}

/// Call Rust get_level() with the given lengths for a rectangle.
fn call_rust_get_level(lengths_for_rect: &[i32], n: usize, jones: i32) -> i32 {
    use direct_nlopt::storage::RectangleStorage;

    let mut storage = RectangleStorage::new(n, 100, 50);

    // Set lengths for rectangle at position 1 (1-based, matching NLOPT convention).
    // In Rust storage: lengths[pos * dim + j] for pos, dimension j.
    let pos: usize = 1;
    for (j, &len_val) in lengths_for_rect.iter().enumerate() {
        storage.lengths[pos * n + j] = len_val;
    }

    storage.get_level(pos, jones)
}

/// Helper: test both C and Rust produce the same level for given lengths and jones flag.
fn assert_level_matches(lengths: &[i32], jones: i32, test_name: &str) {
    let n = lengths.len();
    let c_level = call_c_dirgetlevel(lengths, n, jones);
    let rust_level = call_rust_get_level(lengths, n, jones);

    assert_eq!(
        c_level, rust_level,
        "{}: C level={} != Rust level={} for lengths={:?}, jones={}, n={}",
        test_name, c_level, rust_level, lengths, jones, n
    );
}

// ====================================================================
// Tests for jones=0 (Original DIRECT)
// Level formula: if k == help then k*n + n - p else k*n + p
// where help = length[dim 0], k = min of all lengths, p = count of dims equal to help
// ====================================================================

#[test]
fn test_level_jones_original_uniform_000() {
    // lengths = [0, 0, 0]: all equal, all at min
    // help = 0, k = 0, p = 3 (all equal to help)
    // k == help → level = 0*3 + 3 - 3 = 0
    assert_level_matches(&[0, 0, 0], 0, "uniform [0,0,0] jones=0");
}

#[test]
fn test_level_jones_original_mixed_101() {
    // lengths = [1, 0, 1]: help = 1 (dim 0), k = min(1,0,1) = 0, p = count(== help=1) = 2 (dims 0,2)
    // k != help → level = 0*3 + 2 = 2
    assert_level_matches(&[1, 0, 1], 0, "mixed [1,0,1] jones=0");
}

#[test]
fn test_level_jones_original_all_equal_222() {
    // lengths = [2, 2, 2]: help = 2, k = 2, p = 3
    // k == help → level = 2*3 + 3 - 3 = 6
    assert_level_matches(&[2, 2, 2], 0, "all_equal [2,2,2] jones=0");
}

#[test]
fn test_level_jones_original_asymmetric_0123() {
    // lengths = [0, 1, 2, 3]: help = 0 (dim 0), k = min(0,1,2,3) = 0, p = count(==0) = 1
    // k == help → level = 0*4 + 4 - 1 = 3
    assert_level_matches(&[0, 1, 2, 3], 0, "asymmetric [0,1,2,3] jones=0");
}

#[test]
fn test_level_jones_original_1d() {
    // 1D: lengths = [0]: help = 0, k = 0, p = 1
    // k == help → level = 0*1 + 1 - 1 = 0
    assert_level_matches(&[0], 0, "1D [0] jones=0");

    // 1D: lengths = [3]: help = 3, k = 3, p = 1
    // k == help → level = 3*1 + 1 - 1 = 3
    assert_level_matches(&[3], 0, "1D [3] jones=0");

    // 1D: lengths = [5]: help = 5, k = 5, p = 1
    // k == help → level = 5*1 + 1 - 1 = 5
    assert_level_matches(&[5], 0, "1D [5] jones=0");
}

#[test]
fn test_level_jones_original_2d_cases() {
    // 2D: [0, 0]: help=0, k=0, p=2, k==help → 0*2 + 2 - 2 = 0
    assert_level_matches(&[0, 0], 0, "2D [0,0] jones=0");

    // 2D: [1, 0]: help=1, k=0, p=1 (dim 0 == help), k!=help → 0*2 + 1 = 1
    assert_level_matches(&[1, 0], 0, "2D [1,0] jones=0");

    // 2D: [0, 1]: help=0, k=0, p=1 (dim 0 == help), k==help → 0*2 + 2 - 1 = 1
    assert_level_matches(&[0, 1], 0, "2D [0,1] jones=0");

    // 2D: [1, 1]: help=1, k=1, p=2, k==help → 1*2 + 2 - 2 = 2
    assert_level_matches(&[1, 1], 0, "2D [1,1] jones=0");

    // 2D: [2, 1]: help=2, k=1, p=1, k!=help → 1*2 + 1 = 3
    assert_level_matches(&[2, 1], 0, "2D [2,1] jones=0");

    // 2D: [3, 0]: help=3, k=0, p=1, k!=help → 0*2 + 1 = 1
    assert_level_matches(&[3, 0], 0, "2D [3,0] jones=0");
}

#[test]
fn test_level_jones_original_partial_match() {
    // lengths = [2, 2, 1]: help=2, k=1, p=2 (dims 0,1 == help)
    // k != help → level = 1*3 + 2 = 5
    assert_level_matches(&[2, 2, 1], 0, "partial [2,2,1] jones=0");

    // lengths = [1, 2, 2]: help=1, k=1, p=1
    // k == help → level = 1*3 + 3 - 1 = 5
    assert_level_matches(&[1, 2, 2], 0, "partial [1,2,2] jones=0");

    // lengths = [3, 1, 3, 1]: help=3, k=1, p=2 (dims 0,2 == help)
    // k != help → level = 1*4 + 2 = 6
    assert_level_matches(&[3, 1, 3, 1], 0, "partial [3,1,3,1] jones=0");
}

#[test]
fn test_level_jones_original_5d() {
    // 5D: [0,0,0,0,0]: help=0, k=0, p=5, k==help → 0*5+5-5 = 0
    assert_level_matches(&[0, 0, 0, 0, 0], 0, "5D [0,0,0,0,0] jones=0");

    // 5D: [1,1,1,1,0]: help=1, k=0, p=4, k!=help → 0*5+4 = 4
    assert_level_matches(&[1, 1, 1, 1, 0], 0, "5D [1,1,1,1,0] jones=0");

    // 5D: [2,2,2,2,2]: help=2, k=2, p=5, k==help → 2*5+5-5 = 10
    assert_level_matches(&[2, 2, 2, 2, 2], 0, "5D [2,2,2,2,2] jones=0");

    // 5D: [3,0,1,2,4]: help=3, k=0, p=1, k!=help → 0*5+1 = 1
    assert_level_matches(&[3, 0, 1, 2, 4], 0, "5D [3,0,1,2,4] jones=0");
}

// ====================================================================
// Tests for jones!=0 (Gablonsky DIRECT-L)
// Level = min of all length indices (same as get_max_deep)
// ====================================================================

#[test]
fn test_level_gablonsky_uniform_000() {
    // lengths = [0, 0, 0]: min = 0
    assert_level_matches(&[0, 0, 0], 1, "uniform [0,0,0] jones=1");
}

#[test]
fn test_level_gablonsky_mixed_101() {
    // lengths = [1, 0, 1]: min = 0
    assert_level_matches(&[1, 0, 1], 1, "mixed [1,0,1] jones=1");
}

#[test]
fn test_level_gablonsky_all_equal_222() {
    // lengths = [2, 2, 2]: min = 2
    assert_level_matches(&[2, 2, 2], 1, "all_equal [2,2,2] jones=1");
}

#[test]
fn test_level_gablonsky_asymmetric_0123() {
    // lengths = [0, 1, 2, 3]: min = 0
    assert_level_matches(&[0, 1, 2, 3], 1, "asymmetric [0,1,2,3] jones=1");
}

#[test]
fn test_level_gablonsky_1d() {
    // 1D: min is always the single value
    assert_level_matches(&[0], 1, "1D [0] jones=1");
    assert_level_matches(&[3], 1, "1D [3] jones=1");
    assert_level_matches(&[5], 1, "1D [5] jones=1");
}

#[test]
fn test_level_gablonsky_2d_cases() {
    assert_level_matches(&[0, 0], 1, "2D [0,0] jones=1");
    assert_level_matches(&[1, 0], 1, "2D [1,0] jones=1");
    assert_level_matches(&[0, 1], 1, "2D [0,1] jones=1");
    assert_level_matches(&[1, 1], 1, "2D [1,1] jones=1");
    assert_level_matches(&[2, 1], 1, "2D [2,1] jones=1");
    assert_level_matches(&[3, 0], 1, "2D [3,0] jones=1");
}

#[test]
fn test_level_gablonsky_5d() {
    assert_level_matches(&[0, 0, 0, 0, 0], 1, "5D [0,0,0,0,0] jones=1");
    assert_level_matches(&[1, 1, 1, 1, 0], 1, "5D [1,1,1,1,0] jones=1");
    assert_level_matches(&[2, 2, 2, 2, 2], 1, "5D [2,2,2,2,2] jones=1");
    assert_level_matches(&[3, 0, 1, 2, 4], 1, "5D [3,0,1,2,4] jones=1");
}

// ====================================================================
// Cross-validation: both jones values on same inputs
// ====================================================================

#[test]
fn test_level_both_variants_comprehensive() {
    // Systematically test many length configurations with both jones values
    let test_cases: Vec<Vec<i32>> = vec![
        // 2D cases
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![2, 0],
        vec![0, 2],
        vec![2, 1],
        vec![1, 2],
        vec![2, 2],
        vec![3, 1],
        vec![1, 3],
        vec![5, 5],
        vec![10, 0],
        // 3D cases
        vec![0, 0, 0],
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
        vec![1, 1, 0],
        vec![1, 0, 1],
        vec![0, 1, 1],
        vec![1, 1, 1],
        vec![2, 1, 0],
        vec![0, 2, 1],
        vec![3, 3, 3],
        vec![4, 2, 3],
        // 4D cases
        vec![0, 0, 0, 0],
        vec![1, 0, 0, 0],
        vec![1, 1, 1, 0],
        vec![2, 2, 2, 2],
        vec![3, 1, 2, 0],
    ];

    for lengths in &test_cases {
        let label = format!("{:?}", lengths);
        assert_level_matches(lengths, 0, &format!("{} jones=0", label));
        assert_level_matches(lengths, 1, &format!("{} jones=1", label));
    }
}

// ====================================================================
// Verify specific known values (manually computed)
// ====================================================================

#[test]
fn test_level_known_values_jones_original() {
    // Verify our understanding of the formula by checking against hand-computed values
    let test_cases: Vec<(&[i32], i32)> = vec![
        // lengths, expected level
        // [0,0,0]: k=0, help=0, k==help, p=3, n=3 → 0*3 + 3-3 = 0
        (&[0, 0, 0], 0),
        // [1,1,1]: k=1, help=1, k==help, p=3, n=3 → 1*3 + 3-3 = 3
        (&[1, 1, 1], 3),
        // [2,2,2]: k=2, help=2, k==help, p=3, n=3 → 2*3 + 3-3 = 6
        (&[2, 2, 2], 6),
        // [1,0,1]: help=1, k=0, p=2 (dims 0,2==help=1), k!=help → 0*3 + 2 = 2
        (&[1, 0, 1], 2),
        // [0,1,0]: help=0, k=0, p=2 (dims 0,2==help=0), k==help → 0*3 + 3-2 = 1
        (&[0, 1, 0], 1),
    ];

    for (lengths, expected) in &test_cases {
        let n = lengths.len();
        let c_level = call_c_dirgetlevel(lengths, n, 0);
        let rust_level = call_rust_get_level(lengths, n, 0);

        assert_eq!(
            c_level, *expected,
            "C known value: lengths={:?}, expected={}, got={}",
            lengths, expected, c_level
        );
        assert_eq!(
            rust_level, *expected,
            "Rust known value: lengths={:?}, expected={}, got={}",
            lengths, expected, rust_level
        );
    }
}

#[test]
fn test_level_known_values_gablonsky() {
    // Gablonsky: level = min(lengths)
    let test_cases: Vec<(&[i32], i32)> = vec![
        (&[0, 0, 0], 0),
        (&[1, 1, 1], 1),
        (&[2, 2, 2], 2),
        (&[1, 0, 1], 0),
        (&[3, 1, 2], 1),
        (&[5], 5),
        (&[0, 1, 2, 3], 0),
    ];

    for (lengths, expected) in &test_cases {
        let n = lengths.len();
        let c_level = call_c_dirgetlevel(lengths, n, 1);
        let rust_level = call_rust_get_level(lengths, n, 1);

        assert_eq!(
            c_level, *expected,
            "C Gablonsky: lengths={:?}, expected={}, got={}",
            lengths, expected, c_level
        );
        assert_eq!(
            rust_level, *expected,
            "Rust Gablonsky: lengths={:?}, expected={}, got={}",
            lengths, expected, rust_level
        );
    }
}

// ====================================================================
// Test with rectangle at different positions (pos > 1)
// ====================================================================

#[test]
fn test_level_at_different_positions() {
    // Verify that both C and Rust handle rectangles at positions other than 1
    use direct_nlopt::storage::RectangleStorage;

    let n: usize = 3;
    let maxfunc: c_int = 20;
    let n_c: c_int = n as c_int;

    // Set up rectangles at positions 1, 5, and 10 with different length patterns
    let test_rects: Vec<(c_int, Vec<i32>)> = vec![
        (1, vec![0, 0, 0]),
        (5, vec![2, 1, 3]),
        (10, vec![1, 1, 0]),
    ];

    // Build C length array
    let mut c_length_array = vec![0i32; (maxfunc as usize) * n];
    for &(pos, ref lengths) in &test_rects {
        for (dim, &len_val) in lengths.iter().enumerate() {
            // C 0-based: index = dim + (pos-1) * n
            c_length_array[dim + ((pos - 1) as usize) * n] = len_val;
        }
    }

    // Build Rust storage
    let mut storage = RectangleStorage::new(n, 100, 50);
    for &(pos, ref lengths) in &test_rects {
        for (j, &len_val) in lengths.iter().enumerate() {
            storage.lengths[(pos as usize) * n + j] = len_val;
        }
    }

    for jones in [0, 1] {
        for &(pos, ref lengths) in &test_rects {
            let c_level = unsafe {
                direct_dirgetlevel_(&pos, c_length_array.as_ptr(), &maxfunc, &n_c, jones)
            };
            let rust_level = storage.get_level(pos as usize, jones);

            assert_eq!(
                c_level, rust_level,
                "pos={}, lengths={:?}, jones={}: C={} Rust={}",
                pos, lengths, jones, c_level, rust_level
            );
        }
    }
}
