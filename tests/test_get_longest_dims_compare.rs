#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for get_longest_dims (direct_dirget_i__).
//! Verifies that the Rust RectangleStorage::get_longest_dims() produces identical
//! results to NLOPT C's direct_dirget_i__() for all test scenarios.
//!
//! NLOPT C function: direct_dirget_i__() in DIRsubrout.c lines 1093-1129
//! Rust function: RectangleStorage::get_longest_dims() in storage.rs

use std::os::raw::c_int;

// FFI declaration for NLOPT's direct_dirget_i__().
// Compiled from DIRsubrout.c via build.rs when "nlopt-compare" is enabled.
//
// C signature (direct-internal.h:61):
//   void direct_dirget_i__(integer *length, integer *pos, integer *arrayi,
//                          integer *maxi, integer *n, integer *maxfunc);
//
// The C code uses Fortran-style 1-based column-major indexing:
//   length is n×maxfunc, accessed as length[dim + pos * n] (after adjustment)
//   arrayi is 1-based output
//   pos is 1-based
extern "C" {
    fn direct_dirget_i__(
        length: *mut c_int,
        pos: *const c_int,
        arrayi: *mut c_int,
        maxi: *mut c_int,
        n: *const c_int,
        maxfunc: *const c_int,
    );
}

/// Helper: Call NLOPT's direct_dirget_i__ with a given length array.
///
/// `lengths_for_pos` is the length vector for the rectangle at position `pos`,
/// with `n` dimensions. We construct a minimal length array in column-major
/// format matching NLOPT's layout and call the C function.
///
/// Returns (arrayi, maxi) where arrayi contains 1-based dimension indices.
fn call_c_dirget_i(lengths_for_pos: &[i32], n: usize) -> (Vec<i32>, i32) {
    let pos: c_int = 1; // 1-based position; we put data at column 1
    let n_c = n as c_int;
    // NLOPT uses column-major: length[dim + pos * n]
    // We need at least (pos+1)*n elements. With pos=1, we need 2*n elements.
    let maxfunc: c_int = 2;
    let mut length = vec![0i32; (maxfunc as usize) * n];

    // Fill column 1 (pos=1): length[dim + 1*n] for dim=1..n (1-based)
    // After the C parameter adjustment, the access is length[i + pos * length_dim1]
    // where length_dim1 = n, and the base pointer has been shifted by -(1 + n).
    // So the actual 0-based memory access is: (i + pos * n) - (1 + n) = i + pos*n - n - 1
    // For pos=1, dim i (1-based): index = i + n - n - 1 = i - 1
    // Wait, let me re-read the C code more carefully.

    // C code parameter adjustments:
    //   --arrayi;                          // arrayi becomes 1-based writable
    //   length_dim1 = *n;
    //   length_offset = 1 + length_dim1;   // = 1 + n
    //   length -= length_offset;           // shifts pointer back by (1+n) positions
    //
    // Then access: length[i + pos * length_dim1] where i=1..n, pos=*pos
    // In 0-based terms: (original_ptr - (1+n))[i + pos*n]
    //                 = original_ptr[i + pos*n - 1 - n]
    // For pos=1, i=1: original_ptr[1 + n - 1 - n] = original_ptr[0]
    // For pos=1, i=2: original_ptr[2 + n - 1 - n] = original_ptr[1]
    // For pos=1, i=k: original_ptr[k - 1]
    // So for pos=1, dim i (1-based) maps to 0-based index (i-1).
    // This means lengths_for_pos[j] (0-based j) goes into length[j].
    for j in 0..n {
        length[j] = lengths_for_pos[j];
    }

    let mut arrayi = vec![0i32; n + 1]; // extra space; C writes 1-based
    let mut maxi: c_int = 0;

    unsafe {
        direct_dirget_i__(
            length.as_mut_ptr(),
            &pos,
            arrayi.as_mut_ptr(),
            &mut maxi,
            &n_c,
            &maxfunc,
        );
    }

    // arrayi[1..=maxi] are the 1-based dimension indices (C wrote with --arrayi adjustment)
    // But since we passed arrayi.as_mut_ptr() and C does --arrayi, it writes to
    // arrayi[-1 + j] in C terms... wait, let me think again.
    //
    // C does: --arrayi; then arrayi[j] = i__; where j starts at 1.
    // --arrayi shifts the pointer back by 1, so arrayi[1] in C = our arrayi[0].
    // So the results are in arrayi[0..maxi] (0-based).
    let result = arrayi[0..maxi as usize].to_vec();
    (result, maxi)
}

// ====================================================================
// Test 1: All dimensions equal length → all are "longest" (minimum length)
// ====================================================================

#[test]
fn test_dirget_i_all_equal_3d() {
    // lengths [0,0,0]: min=0, all dims match → maxi=3, arrayi=[1,2,3]
    let lengths = vec![0, 0, 0];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 3);

    assert_eq!(c_maxi, 3, "C: all equal lengths should give maxi=3");
    assert_eq!(c_arrayi, vec![1, 2, 3], "C: arrayi should be [1,2,3] (1-based)");

    // Compare with Rust
    let mut storage = direct_nlopt::storage::RectangleStorage::new(3, 100, 0);
    // Set lengths for rect at position 1 (1-based)
    for j in 0..3 {
        storage.lengths[1 * 3 + j] = 0;
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi, "maxi mismatch: Rust={} C={}", rust_maxi, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(
            rust_arrayi[i] as i32, c_arrayi[i],
            "arrayi[{}] mismatch: Rust={} C={}", i, rust_arrayi[i], c_arrayi[i]
        );
    }
}

// ====================================================================
// Test 2: One dim has larger length → the two with smaller length are longest
// ====================================================================

#[test]
fn test_dirget_i_one_larger_3d() {
    // lengths [1,0,0]: min=0, dims 2,3 match → maxi=2, arrayi=[2,3]
    let lengths = vec![1, 0, 0];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 3);

    assert_eq!(c_maxi, 2, "C: [1,0,0] should give maxi=2");
    assert_eq!(c_arrayi, vec![2, 3], "C: arrayi should be [2,3] (1-based)");

    // Compare with Rust
    let mut storage = direct_nlopt::storage::RectangleStorage::new(3, 100, 0);
    storage.lengths[1 * 3 + 0] = 1;
    storage.lengths[1 * 3 + 1] = 0;
    storage.lengths[1 * 3 + 2] = 0;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi, "maxi mismatch: Rust={} C={}", rust_maxi, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(
            rust_arrayi[i] as i32, c_arrayi[i],
            "arrayi[{}] mismatch: Rust={} C={}", i, rust_arrayi[i], c_arrayi[i]
        );
    }
}

// ====================================================================
// Test 3: All different lengths → only the minimum dim is longest
// ====================================================================

#[test]
fn test_dirget_i_all_different_3d() {
    // lengths [0,1,2]: min=0, only dim 1 matches → maxi=1, arrayi=[1]
    let lengths = vec![0, 1, 2];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 3);

    assert_eq!(c_maxi, 1, "C: [0,1,2] should give maxi=1");
    assert_eq!(c_arrayi, vec![1], "C: arrayi should be [1] (1-based)");

    // Compare with Rust
    let mut storage = direct_nlopt::storage::RectangleStorage::new(3, 100, 0);
    storage.lengths[1 * 3 + 0] = 0;
    storage.lengths[1 * 3 + 1] = 1;
    storage.lengths[1 * 3 + 2] = 2;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi, "maxi mismatch: Rust={} C={}", rust_maxi, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(
            rust_arrayi[i] as i32, c_arrayi[i],
            "arrayi[{}] mismatch: Rust={} C={}", i, rust_arrayi[i], c_arrayi[i]
        );
    }
}

// ====================================================================
// Test 4: 4D with two pairs → two minimum dims
// ====================================================================

#[test]
fn test_dirget_i_4d_two_pairs() {
    // lengths [2,2,1,1]: min=1, dims 3,4 match → maxi=2, arrayi=[3,4]
    let lengths = vec![2, 2, 1, 1];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 4);

    assert_eq!(c_maxi, 2, "C: [2,2,1,1] should give maxi=2");
    assert_eq!(c_arrayi, vec![3, 4], "C: arrayi should be [3,4] (1-based)");

    // Compare with Rust
    let mut storage = direct_nlopt::storage::RectangleStorage::new(4, 100, 0);
    storage.lengths[1 * 4 + 0] = 2;
    storage.lengths[1 * 4 + 1] = 2;
    storage.lengths[1 * 4 + 2] = 1;
    storage.lengths[1 * 4 + 3] = 1;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi, "maxi mismatch: Rust={} C={}", rust_maxi, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(
            rust_arrayi[i] as i32, c_arrayi[i],
            "arrayi[{}] mismatch: Rust={} C={}", i, rust_arrayi[i], c_arrayi[i]
        );
    }
}

// ====================================================================
// Test 5: 1D edge case
// ====================================================================

#[test]
fn test_dirget_i_1d() {
    // 1D: lengths [0] → maxi=1, arrayi=[1]
    let lengths = vec![0];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 1);

    assert_eq!(c_maxi, 1, "C: 1D should give maxi=1");
    assert_eq!(c_arrayi, vec![1], "C: arrayi should be [1] (1-based)");

    let mut storage = direct_nlopt::storage::RectangleStorage::new(1, 100, 0);
    storage.lengths[1 * 1 + 0] = 0;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    assert_eq!(rust_arrayi[0] as i32, c_arrayi[0]);
}

#[test]
fn test_dirget_i_1d_nonzero_length() {
    // 1D: lengths [5] → maxi=1, arrayi=[1]
    let lengths = vec![5];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 1);

    assert_eq!(c_maxi, 1, "C: 1D with length 5 should give maxi=1");
    assert_eq!(c_arrayi, vec![1]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(1, 100, 0);
    storage.lengths[1 * 1 + 0] = 5;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    assert_eq!(rust_arrayi[0] as i32, c_arrayi[0]);
}

// ====================================================================
// Test 6: 2D cases
// ====================================================================

#[test]
fn test_dirget_i_2d_equal() {
    // [0,0] → maxi=2, arrayi=[1,2]
    let lengths = vec![0, 0];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 2);

    assert_eq!(c_maxi, 2);
    assert_eq!(c_arrayi, vec![1, 2]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(2, 100, 0);
    storage.lengths[1 * 2 + 0] = 0;
    storage.lengths[1 * 2 + 1] = 0;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(rust_arrayi[i] as i32, c_arrayi[i]);
    }
}

#[test]
fn test_dirget_i_2d_first_longer() {
    // [1,0] → min=0, only dim 2 → maxi=1, arrayi=[2]
    let lengths = vec![1, 0];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 2);

    assert_eq!(c_maxi, 1);
    assert_eq!(c_arrayi, vec![2]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(2, 100, 0);
    storage.lengths[1 * 2 + 0] = 1;
    storage.lengths[1 * 2 + 1] = 0;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    assert_eq!(rust_arrayi[0] as i32, c_arrayi[0]);
}

#[test]
fn test_dirget_i_2d_second_longer() {
    // [0,1] → min=0, only dim 1 → maxi=1, arrayi=[1]
    let lengths = vec![0, 1];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 2);

    assert_eq!(c_maxi, 1);
    assert_eq!(c_arrayi, vec![1]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(2, 100, 0);
    storage.lengths[1 * 2 + 0] = 0;
    storage.lengths[1 * 2 + 1] = 1;
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    assert_eq!(rust_arrayi[0] as i32, c_arrayi[0]);
}

// ====================================================================
// Test 7: 5D with mixed lengths
// ====================================================================

#[test]
fn test_dirget_i_5d_mixed() {
    // [3,1,3,1,2] → min=1, dims 2,4 match → maxi=2, arrayi=[2,4]
    let lengths = vec![3, 1, 3, 1, 2];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 5);

    assert_eq!(c_maxi, 2, "C: [3,1,3,1,2] should give maxi=2");
    assert_eq!(c_arrayi, vec![2, 4], "C: arrayi should be [2,4] (1-based)");

    let mut storage = direct_nlopt::storage::RectangleStorage::new(5, 100, 0);
    for j in 0..5 {
        storage.lengths[1 * 5 + j] = lengths[j];
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(
            rust_arrayi[i] as i32, c_arrayi[i],
            "arrayi[{}] mismatch: Rust={} C={}", i, rust_arrayi[i], c_arrayi[i]
        );
    }
}

#[test]
fn test_dirget_i_5d_all_same() {
    // [2,2,2,2,2] → min=2, all dims → maxi=5, arrayi=[1,2,3,4,5]
    let lengths = vec![2, 2, 2, 2, 2];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 5);

    assert_eq!(c_maxi, 5);
    assert_eq!(c_arrayi, vec![1, 2, 3, 4, 5]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(5, 100, 0);
    for j in 0..5 {
        storage.lengths[1 * 5 + j] = 2;
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(rust_arrayi[i] as i32, c_arrayi[i]);
    }
}

#[test]
fn test_dirget_i_5d_single_minimum() {
    // [1,2,3,4,5] → min=1, only dim 1 → maxi=1, arrayi=[1]
    let lengths = vec![1, 2, 3, 4, 5];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 5);

    assert_eq!(c_maxi, 1);
    assert_eq!(c_arrayi, vec![1]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(5, 100, 0);
    for j in 0..5 {
        storage.lengths[1 * 5 + j] = lengths[j];
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    assert_eq!(rust_arrayi[0] as i32, c_arrayi[0]);
}

// ====================================================================
// Test 8: Large values (matching what happens after many subdivisions)
// ====================================================================

#[test]
fn test_dirget_i_large_length_values() {
    // After many subdivisions, length indices can be quite large.
    // [10, 5, 10, 5, 5] → min=5, dims 2,4,5 → maxi=3, arrayi=[2,4,5]
    let lengths = vec![10, 5, 10, 5, 5];
    let (c_arrayi, c_maxi) = call_c_dirget_i(&lengths, 5);

    assert_eq!(c_maxi, 3);
    assert_eq!(c_arrayi, vec![2, 4, 5]);

    let mut storage = direct_nlopt::storage::RectangleStorage::new(5, 100, 0);
    for j in 0..5 {
        storage.lengths[1 * 5 + j] = lengths[j];
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

    assert_eq!(rust_maxi as i32, c_maxi);
    for i in 0..c_maxi as usize {
        assert_eq!(rust_arrayi[i] as i32, c_arrayi[i]);
    }
}

// ====================================================================
// Test 9: Position != 1 (verify different rect positions work)
// ====================================================================

#[test]
fn test_dirget_i_different_position() {
    // Test with rect at position 2 (1-based) — uses different memory offset
    let n = 3usize;
    let pos: c_int = 2;
    let maxfunc: c_int = 4;
    let lengths_at_pos = [1, 0, 1]; // min=0, only dim 2 → maxi=1, arrayi=[2]

    // Build column-major length array for C
    // For pos=2, i (1-based): 0-based index = i + pos*n - 1 - n = i + 2*3 - 1 - 3 = i + 2
    // So dim 1 → idx 3, dim 2 → idx 4, dim 3 → idx 5
    let mut length = vec![0i32; (maxfunc as usize) * n];
    length[3] = 1; // dim 1
    length[4] = 0; // dim 2
    length[5] = 1; // dim 3

    let mut arrayi = vec![0i32; n + 1];
    let mut maxi: c_int = 0;
    let n_c = n as c_int;

    unsafe {
        direct_dirget_i__(
            length.as_mut_ptr(),
            &pos,
            arrayi.as_mut_ptr(),
            &mut maxi,
            &n_c,
            &maxfunc,
        );
    }

    assert_eq!(maxi, 1);
    assert_eq!(arrayi[0], 2); // 1-based dim index

    // Compare with Rust
    let mut storage = direct_nlopt::storage::RectangleStorage::new(n, 100, 0);
    for j in 0..n {
        storage.lengths[2 * n + j] = lengths_at_pos[j];
    }
    let (rust_arrayi, rust_maxi) = storage.get_longest_dims(2);

    assert_eq!(rust_maxi as i32, maxi);
    assert_eq!(rust_arrayi[0] as i32, arrayi[0]);
}

// ====================================================================
// Test 10: Comprehensive sweep — many configurations, verify C == Rust
// ====================================================================

#[test]
fn test_dirget_i_comprehensive_sweep() {
    // Test many length configurations systematically
    let test_cases: Vec<(Vec<i32>, usize, i32, Vec<i32>)> = vec![
        // (lengths, n, expected_maxi, expected_arrayi)
        (vec![0], 1, 1, vec![1]),
        (vec![0, 0], 2, 2, vec![1, 2]),
        (vec![0, 1], 2, 1, vec![1]),
        (vec![1, 0], 2, 1, vec![2]),
        (vec![1, 1], 2, 2, vec![1, 2]),
        (vec![0, 0, 0], 3, 3, vec![1, 2, 3]),
        (vec![1, 0, 0], 3, 2, vec![2, 3]),
        (vec![0, 1, 0], 3, 2, vec![1, 3]),
        (vec![0, 0, 1], 3, 2, vec![1, 2]),
        (vec![1, 1, 0], 3, 1, vec![3]),
        (vec![1, 0, 1], 3, 1, vec![2]),
        (vec![0, 1, 1], 3, 1, vec![1]),
        (vec![0, 1, 2], 3, 1, vec![1]),
        (vec![2, 1, 0], 3, 1, vec![3]),
        (vec![2, 2, 1, 1], 4, 2, vec![3, 4]),
        (vec![3, 1, 3, 1, 2], 5, 2, vec![2, 4]),
    ];

    for (lengths, n, expected_maxi, expected_arrayi) in &test_cases {
        let (c_arrayi, c_maxi) = call_c_dirget_i(lengths, *n);

        assert_eq!(
            c_maxi, *expected_maxi,
            "C: lengths={:?} expected maxi={} got {}", lengths, expected_maxi, c_maxi
        );
        assert_eq!(
            c_arrayi, *expected_arrayi,
            "C: lengths={:?} expected arrayi={:?} got {:?}", lengths, expected_arrayi, c_arrayi
        );

        // Compare with Rust
        let mut storage = direct_nlopt::storage::RectangleStorage::new(*n, 100, 0);
        for j in 0..*n {
            storage.lengths[1 * n + j] = lengths[j];
        }
        let (rust_arrayi, rust_maxi) = storage.get_longest_dims(1);

        assert_eq!(
            rust_maxi as i32, c_maxi,
            "Rust vs C maxi mismatch for lengths={:?}: Rust={} C={}", lengths, rust_maxi, c_maxi
        );
        for i in 0..c_maxi as usize {
            assert_eq!(
                rust_arrayi[i] as i32, c_arrayi[i],
                "Rust vs C arrayi[{}] mismatch for lengths={:?}: Rust={} C={}",
                i, lengths, rust_arrayi[i], c_arrayi[i]
            );
        }
    }
}
