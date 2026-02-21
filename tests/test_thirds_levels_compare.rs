#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types)]

//! Comparative unit tests for thirds[] and levels[] precomputation.
//! Verifies that the Rust `precompute_thirds()` and `precompute_levels()` produce
//! bit-identical results to the NLOPT C computation in `direct_dirinit_()` (DIRsubrout.c).

use std::os::raw::{c_double, c_int};

// FFI declarations for the C shim helpers that replicate the exact loops from direct_dirinit_().
extern "C" {
    fn nlopt_shim_precompute_thirds(thirds: *mut c_double, maxdeep: c_int);
    fn nlopt_shim_precompute_levels(levels: *mut c_double, maxdeep: c_int, n: c_int, jones: c_int);
}

/// Compute thirds[] using the C shim (identical logic to NLOPT direct_dirinit_).
fn c_compute_thirds(maxdeep: usize) -> Vec<f64> {
    let mut thirds = vec![0.0_f64; maxdeep + 1];
    unsafe {
        nlopt_shim_precompute_thirds(thirds.as_mut_ptr(), maxdeep as c_int);
    }
    thirds
}

/// Compute levels[] using the C shim (identical logic to NLOPT direct_dirinit_).
fn c_compute_levels(maxdeep: usize, n: usize, jones: i32) -> Vec<f64> {
    let mut levels = vec![0.0_f64; maxdeep + 1];
    unsafe {
        nlopt_shim_precompute_levels(levels.as_mut_ptr(), maxdeep as c_int, n as c_int, jones);
    }
    levels
}

/// Compute thirds[] using the Rust implementation.
fn rust_compute_thirds(maxdeep: usize) -> Vec<f64> {
    use direct_nlopt::storage::RectangleStorage;
    let mut storage = RectangleStorage::new(1, 100, maxdeep);
    storage.precompute_thirds();
    storage.thirds[..=maxdeep].to_vec()
}

/// Compute levels[] using the Rust implementation.
fn rust_compute_levels(maxdeep: usize, n: usize, jones: i32) -> Vec<f64> {
    use direct_nlopt::storage::RectangleStorage;
    let mut storage = RectangleStorage::new(n, 100, maxdeep);
    storage.precompute_levels(jones);
    storage.levels[..=maxdeep].to_vec()
}

// ====================================================================
// Tests for thirds[k] = 1/3^k
// ====================================================================

#[test]
fn test_thirds_maxdeep_50() {
    let maxdeep = 50;
    let c_thirds = c_compute_thirds(maxdeep);
    let rust_thirds = rust_compute_thirds(maxdeep);

    // thirds[0] = 1.0
    assert_eq!(c_thirds[0], 1.0, "C thirds[0] should be 1.0");
    assert_eq!(rust_thirds[0], 1.0, "Rust thirds[0] should be 1.0");

    for k in 0..=maxdeep {
        assert_eq!(
            c_thirds[k].to_bits(),
            rust_thirds[k].to_bits(),
            "thirds[{}]: C={:.18e} != Rust={:.18e} (bit mismatch)",
            k, c_thirds[k], rust_thirds[k]
        );
    }
}

#[test]
fn test_thirds_match_formula() {
    let maxdeep = 50;
    let c_thirds = c_compute_thirds(maxdeep);

    // Verify C output matches 1/3^k formula (computed the same way the C code does)
    let mut expected = vec![0.0_f64; maxdeep + 1];
    expected[0] = 1.0;
    let mut help2 = 3.0_f64;
    for k in 1..=maxdeep {
        expected[k] = 1.0 / help2;
        help2 *= 3.0;
    }

    for k in 0..=maxdeep {
        assert_eq!(
            c_thirds[k].to_bits(),
            expected[k].to_bits(),
            "thirds[{}]: C={:.18e} != expected={:.18e}",
            k, c_thirds[k], expected[k]
        );
    }
}

#[test]
fn test_thirds_small_maxdeep() {
    // Test with very small maxdeep values
    for maxdeep in [1, 2, 3, 5, 10] {
        let c_thirds = c_compute_thirds(maxdeep);
        let rust_thirds = rust_compute_thirds(maxdeep);

        for k in 0..=maxdeep {
            assert_eq!(
                c_thirds[k].to_bits(),
                rust_thirds[k].to_bits(),
                "maxdeep={}, thirds[{}]: C={:.18e} != Rust={:.18e}",
                maxdeep, k, c_thirds[k], rust_thirds[k]
            );
        }
    }
}

#[test]
fn test_thirds_specific_values() {
    let c_thirds = c_compute_thirds(10);
    let rust_thirds = rust_compute_thirds(10);

    // Check specific well-known values
    assert_eq!(c_thirds[0], 1.0);
    assert_eq!(rust_thirds[0], 1.0);

    // thirds[1] = 1/3
    let expected_1 = 1.0_f64 / 3.0;
    assert_eq!(c_thirds[1].to_bits(), expected_1.to_bits());
    assert_eq!(rust_thirds[1].to_bits(), expected_1.to_bits());

    // thirds[2] = 1/9
    let expected_2 = 1.0_f64 / 9.0;
    assert_eq!(c_thirds[2].to_bits(), expected_2.to_bits());
    assert_eq!(rust_thirds[2].to_bits(), expected_2.to_bits());

    // thirds[3] = 1/27
    let expected_3 = 1.0_f64 / 27.0;
    assert_eq!(c_thirds[3].to_bits(), expected_3.to_bits());
    assert_eq!(rust_thirds[3].to_bits(), expected_3.to_bits());
}

// ====================================================================
// Tests for levels[] with jones=1 (Gablonsky DIRECT-L)
// Levels should be identical to thirds for Gablonsky.
// ====================================================================

#[test]
fn test_levels_gablonsky_maxdeep_50_n2() {
    let maxdeep = 50;
    let n = 2;
    let c_levels = c_compute_levels(maxdeep, n, 1);
    let rust_levels = rust_compute_levels(maxdeep, n, 1);

    for k in 0..=maxdeep {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_gablonsky_maxdeep_50_n3() {
    let maxdeep = 50;
    let n = 3;
    let c_levels = c_compute_levels(maxdeep, n, 1);
    let rust_levels = rust_compute_levels(maxdeep, n, 1);

    for k in 0..=maxdeep {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_gablonsky_maxdeep_50_n5() {
    let maxdeep = 50;
    let n = 5;
    let c_levels = c_compute_levels(maxdeep, n, 1);
    let rust_levels = rust_compute_levels(maxdeep, n, 1);

    for k in 0..=maxdeep {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_gablonsky_equals_thirds() {
    // For Gablonsky (jones=1), levels[] should be identical to thirds[]
    let maxdeep = 50;
    let c_thirds = c_compute_thirds(maxdeep);
    let c_levels = c_compute_levels(maxdeep, 2, 1);

    for k in 0..=maxdeep {
        assert_eq!(
            c_thirds[k].to_bits(),
            c_levels[k].to_bits(),
            "thirds[{}]={:.18e} != levels[{}]={:.18e} (should be identical for Gablonsky)",
            k, c_thirds[k], k, c_levels[k]
        );
    }
}

#[test]
fn test_levels_gablonsky_dimension_independent() {
    // For Gablonsky, levels[] should not depend on n
    let maxdeep = 50;
    let levels_n2 = c_compute_levels(maxdeep, 2, 1);
    let levels_n3 = c_compute_levels(maxdeep, 3, 1);
    let levels_n5 = c_compute_levels(maxdeep, 5, 1);

    for k in 0..=maxdeep {
        assert_eq!(
            levels_n2[k].to_bits(),
            levels_n3[k].to_bits(),
            "Gablonsky levels should not depend on n: levels_n2[{}] != levels_n3[{}]",
            k, k
        );
        assert_eq!(
            levels_n2[k].to_bits(),
            levels_n5[k].to_bits(),
            "Gablonsky levels should not depend on n: levels_n2[{}] != levels_n5[{}]",
            k, k
        );
    }
}

// ====================================================================
// Tests for levels[] with jones=0 (Jones DIRECT Original)
// levels[(i-1)*n + j] = w[j] / 3^(i-1)
// where w[j] = sqrt(n - j + j/9) * 0.5
// ====================================================================

#[test]
fn test_levels_jones_original_n2_maxdeep_50() {
    let maxdeep = 50;
    let n = 2;
    let c_levels = c_compute_levels(maxdeep, n, 0);
    let rust_levels = rust_compute_levels(maxdeep, n, 0);

    // Only indices 0..maxdeep/n * n are filled for jones=0
    let filled = (maxdeep / n) * n;
    for k in 0..filled {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, jones=0, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_jones_original_n3_maxdeep_50() {
    let maxdeep = 50;
    let n = 3;
    let c_levels = c_compute_levels(maxdeep, n, 0);
    let rust_levels = rust_compute_levels(maxdeep, n, 0);

    let filled = (maxdeep / n) * n;
    for k in 0..filled {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, jones=0, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_jones_original_n5_maxdeep_50() {
    let maxdeep = 50;
    let n = 5;
    let c_levels = c_compute_levels(maxdeep, n, 0);
    let rust_levels = rust_compute_levels(maxdeep, n, 0);

    let filled = (maxdeep / n) * n;
    for k in 0..filled {
        assert_eq!(
            c_levels[k].to_bits(),
            rust_levels[k].to_bits(),
            "n={}, jones=0, levels[{}]: C={:.18e} != Rust={:.18e}",
            n, k, c_levels[k], rust_levels[k]
        );
    }
}

#[test]
fn test_levels_jones_original_w_values() {
    // Verify the w[j] = sqrt(n - j + j/9) * 0.5 formula for n=3
    let n = 3;
    let maxdeep = 30;
    let c_levels = c_compute_levels(maxdeep, n, 0);

    // First "row" (i=1) of levels: levels[0..n-1] = w[0..n-1] / 3^0 = w[0..n-1]
    for j in 0..n {
        let expected_w = ((n - j) as f64 + j as f64 / 9.0).sqrt() * 0.5;
        assert_eq!(
            c_levels[j].to_bits(),
            expected_w.to_bits(),
            "w[{}] = levels[{}]: C={:.18e} != expected={:.18e}",
            j, j, c_levels[j], expected_w
        );
    }

    // Second "row" (i=2): levels[n..2n-1] = w[0..n-1] / 3
    for j in 0..n {
        let expected_w = ((n - j) as f64 + j as f64 / 9.0).sqrt() * 0.5;
        let expected = expected_w / 3.0;
        assert_eq!(
            c_levels[n + j].to_bits(),
            expected.to_bits(),
            "levels[{}]: C={:.18e} != expected={:.18e}",
            n + j, c_levels[n + j], expected
        );
    }
}

#[test]
fn test_levels_jones_original_depends_on_n() {
    // For Jones Original (jones=0), levels[] SHOULD depend on n
    let maxdeep = 50;
    let levels_n2 = c_compute_levels(maxdeep, 2, 0);
    let levels_n3 = c_compute_levels(maxdeep, 3, 0);

    // First values should differ because w[0] = sqrt(n)*0.5 depends on n
    assert_ne!(
        levels_n2[0].to_bits(),
        levels_n3[0].to_bits(),
        "Jones Original levels should depend on n"
    );
}

// ====================================================================
// Cross-dimension tests with various n
// ====================================================================

#[test]
fn test_thirds_various_maxdeep() {
    // Verify thirds are consistent across different maxdeep values
    let maxdeep_large = 50;
    let c_thirds_large = c_compute_thirds(maxdeep_large);
    let rust_thirds_large = rust_compute_thirds(maxdeep_large);

    for maxdeep in [10, 20, 30, 40] {
        let c_thirds = c_compute_thirds(maxdeep);
        let rust_thirds = rust_compute_thirds(maxdeep);

        // All values up to the smaller maxdeep should match
        for k in 0..=maxdeep {
            assert_eq!(
                c_thirds[k].to_bits(),
                c_thirds_large[k].to_bits(),
                "C thirds consistency: maxdeep={} vs {}, k={}",
                maxdeep, maxdeep_large, k
            );
            assert_eq!(
                rust_thirds[k].to_bits(),
                rust_thirds_large[k].to_bits(),
                "Rust thirds consistency: maxdeep={} vs {}, k={}",
                maxdeep, maxdeep_large, k
            );
        }
    }
}

#[test]
fn test_levels_both_variants_n2_n3_n5() {
    // Comprehensive test: both jones variants across n=2,3,5 with maxdeep=50
    let maxdeep = 50;

    for &n in &[2, 3, 5] {
        for jones in [0, 1] {
            let c_levels = c_compute_levels(maxdeep, n, jones);
            let rust_levels = rust_compute_levels(maxdeep, n, jones);

            let check_len = if jones == 0 {
                (maxdeep / n) * n
            } else {
                maxdeep + 1
            };

            for k in 0..check_len {
                assert_eq!(
                    c_levels[k].to_bits(),
                    rust_levels[k].to_bits(),
                    "n={}, jones={}, levels[{}]: C={:.18e} != Rust={:.18e}",
                    n, jones, k, c_levels[k], rust_levels[k]
                );
            }
        }
    }
}

// ====================================================================
// Monotonicity and ordering properties
// ====================================================================

#[test]
fn test_thirds_monotonically_decreasing() {
    let maxdeep = 50;
    let c_thirds = c_compute_thirds(maxdeep);

    for k in 1..=maxdeep {
        assert!(
            c_thirds[k] < c_thirds[k - 1],
            "thirds should be monotonically decreasing: thirds[{}]={} >= thirds[{}]={}",
            k, c_thirds[k], k - 1, c_thirds[k - 1]
        );
    }
}

#[test]
fn test_levels_gablonsky_monotonically_decreasing() {
    let maxdeep = 50;
    let c_levels = c_compute_levels(maxdeep, 3, 1);

    for k in 1..=maxdeep {
        assert!(
            c_levels[k] < c_levels[k - 1],
            "Gablonsky levels should decrease: levels[{}]={} >= levels[{}]={}",
            k, c_levels[k], k - 1, c_levels[k - 1]
        );
    }
}

#[test]
fn test_levels_jones_row_decreasing() {
    // For Jones Original, within each "column" j, levels should decrease as i increases
    let maxdeep = 50;
    let n = 3;
    let c_levels = c_compute_levels(maxdeep, n, 0);

    let num_rows = maxdeep / n;
    for j in 0..n {
        for i in 1..num_rows {
            let curr = c_levels[i * n + j];
            let prev = c_levels[(i - 1) * n + j];
            assert!(
                curr < prev,
                "Jones levels should decrease per column: levels[{}]={} >= levels[{}]={}",
                i * n + j, curr, (i - 1) * n + j, prev
            );
        }
    }
}

// ====================================================================
// Positive values test
// ====================================================================

#[test]
fn test_thirds_all_positive() {
    let maxdeep = 50;
    let c_thirds = c_compute_thirds(maxdeep);

    for k in 0..=maxdeep {
        assert!(
            c_thirds[k] > 0.0,
            "thirds[{}] = {} should be positive",
            k, c_thirds[k]
        );
    }
}

#[test]
fn test_levels_all_positive() {
    let maxdeep = 50;

    for &n in &[2, 3, 5] {
        for jones in [0, 1] {
            let c_levels = c_compute_levels(maxdeep, n, jones);
            let check_len = if jones == 0 {
                (maxdeep / n) * n
            } else {
                maxdeep + 1
            };

            for k in 0..check_len {
                assert!(
                    c_levels[k] > 0.0,
                    "n={}, jones={}, levels[{}] = {} should be positive",
                    n, jones, k, c_levels[k]
                );
            }
        }
    }
}
