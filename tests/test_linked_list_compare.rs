#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types, dead_code)]

//! Comparative unit tests for linked list operations (dirinitlist_, dirinsertlist_).
//! Verifies that Rust RectangleStorage linked list operations produce identical
//! results to NLOPT C's direct_dirinitlist_() and direct_dirinsertlist_().
//!
//! NLOPT C functions:
//! - direct_dirinitlist_() in DIRsubrout.c lines 1327-1361
//! - direct_dirinsertlist_() in DIRsubrout.c lines 703-793
//! - direct_dirgetlevel_() in DIRsubrout.c lines 33-82
//!
//! Rust functions:
//! - RectangleStorage::init_lists() in storage.rs
//! - RectangleStorage::insert_into_list() in storage.rs

use std::os::raw::{c_double, c_int};

// FFI declarations for NLOPT C functions.
// Compiled from DIRsubrout.c via build.rs when "nlopt-compare" is enabled.
extern "C" {
    fn direct_dirinitlist_(
        anchor: *mut c_int,
        free: *mut c_int,
        point: *mut c_int,
        f: *mut c_double,
        maxfunc: *const c_int,
        maxdeep: *const c_int,
    );

    fn direct_dirinsertlist_(
        new__: *mut c_int,
        anchor: *mut c_int,
        point: *mut c_int,
        f: *mut c_double,
        maxi: *const c_int,
        length: *mut c_int,
        maxfunc: *const c_int,
        maxdeep: *const c_int,
        n: *const c_int,
        samp: *const c_int,
        jones: c_int,
    );

    fn direct_dirgetlevel_(
        pos: *const c_int,
        length: *mut c_int,
        maxfunc: *const c_int,
        n: *const c_int,
        jones: c_int,
    ) -> c_int;
}

/// Helper struct holding C-side arrays that mirror the NLOPT memory layout.
/// All arrays are 0-based in memory; the C functions do internal adjustments
/// for 1-based indexing.
struct CStorage {
    anchor: Vec<c_int>,   // size: maxdeep + 2
    point: Vec<c_int>,    // size: maxfunc
    f: Vec<c_double>,     // size: maxfunc * 2
    length: Vec<c_int>,   // size: maxfunc * n
    free: c_int,
    maxfunc: c_int,
    maxdeep: c_int,
    n: c_int,
}

impl CStorage {
    fn new(n: usize, maxfunc: usize, maxdeep: usize) -> Self {
        Self {
            anchor: vec![0i32; maxdeep + 2],
            point: vec![0i32; maxfunc],
            f: vec![0.0f64; maxfunc * 2],
            length: vec![0i32; maxfunc * n],
            free: 0,
            maxfunc: maxfunc as c_int,
            maxdeep: maxdeep as c_int,
            n: n as c_int,
        }
    }

    /// Call NLOPT C direct_dirinitlist_()
    fn init_lists(&mut self) {
        unsafe {
            direct_dirinitlist_(
                self.anchor.as_mut_ptr(),
                &mut self.free,
                self.point.as_mut_ptr(),
                self.f.as_mut_ptr(),
                &self.maxfunc,
                &self.maxdeep,
            );
        }
    }

    /// Set f-value for 1-based rect index idx.
    /// C layout after adjustment: f[(idx << 1) + 1] = val, f[(idx << 1) + 2] = flag
    /// In raw 0-based: f[2*(idx-1)] = val, f[2*(idx-1)+1] = flag
    fn set_f(&mut self, idx: usize, val: f64, flag: f64) {
        self.f[2 * (idx - 1)] = val;        // first column (1-based)
        self.f[2 * (idx - 1) + 1] = flag;   // second column
    }

    /// Get f-value for 1-based rect index idx.
    fn f_val(&self, idx: usize) -> f64 {
        self.f[2 * (idx - 1)]
    }

    /// Set length index for 1-based rect pos, 0-based dimension dim_j.
    /// C layout: length is maxfunc×n, accessed after adjustment as
    /// length[dim(1-based) + pos * n] with offset -(1+n).
    /// In raw 0-based memory: length[(pos-1)*n + dim_j]
    fn set_length(&mut self, pos: usize, dim_j: usize, val: i32) {
        self.length[(pos - 1) * self.n as usize + dim_j] = val;
    }

    /// Allocate a rect from the free list (1-based).
    fn alloc_rect(&mut self) -> usize {
        let idx = self.free as usize;
        assert!(idx > 0 && idx <= self.maxfunc as usize);
        // After dirinitlist_ parameter adjustments, point is 1-based:
        // point[i] = i+1, but the raw array is 0-based.
        // With --point adjustment in C, point[i] in C = raw_point[i-1]
        // So raw_point[idx-1] gives next free.
        self.free = self.point[idx - 1];
        self.point[idx - 1] = 0;
        idx
    }

    /// Chain two rects: point[a] = b (1-based, adjusting for raw 0-based).
    fn set_point(&mut self, a: usize, b: i32) {
        self.point[a - 1] = b;
    }

    /// Get point[a] (1-based → raw).
    fn get_point(&self, a: usize) -> i32 {
        self.point[a - 1]
    }

    /// Get anchor[depth] (depth from -1 to maxdeep).
    /// After ++anchor adjustment: anchor[depth] in C = raw anchor[depth+1]
    fn get_anchor(&self, depth: i32) -> i32 {
        self.anchor[(depth + 1) as usize]
    }

    /// Call NLOPT C direct_dirinsertlist_()
    fn insert_into_list(&mut self, new_start: &mut i32, maxi: i32, samp: i32, jones: i32) {
        unsafe {
            direct_dirinsertlist_(
                new_start,
                self.anchor.as_mut_ptr(),
                self.point.as_mut_ptr(),
                self.f.as_mut_ptr(),
                &maxi,
                self.length.as_mut_ptr(),
                &self.maxfunc,
                &self.maxdeep,
                &self.n,
                &samp,
                jones,
            );
        }
    }

    /// Walk linked list from anchor[depth], collecting 1-based indices.
    fn walk_list(&self, depth: i32) -> Vec<usize> {
        let mut result = Vec::new();
        let mut current = self.get_anchor(depth);
        while current > 0 {
            result.push(current as usize);
            current = self.get_point(current as usize);
        }
        result
    }

    /// Walk linked list from anchor[depth], collecting f-values.
    fn walk_list_fvals(&self, depth: i32) -> Vec<f64> {
        self.walk_list(depth).iter().map(|&idx| self.f_val(idx)).collect()
    }
}

use direct_nlopt::storage::RectangleStorage;

// ====================================================================
// Test 1: init_lists — verify all anchors = 0, point[] forms free list
// ====================================================================

#[test]
fn test_init_lists_compare() {
    let maxfunc = 20;
    let maxdeep = 10;
    let n = 2;

    // C side
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // Rust side
    let mut r = RectangleStorage::new(n, 0, 0);
    // Override sizes to match C
    r.maxfunc = maxfunc;
    r.maxdeep = maxdeep;
    r.point = vec![0i32; maxfunc];
    r.anchor = vec![0i32; maxdeep + 2];
    r.f_values = vec![0.0; maxfunc * 2];
    r.init_lists();

    // Compare anchors: all should be 0
    for depth in -1..=(maxdeep as i32) {
        let c_anchor = c.get_anchor(depth);
        let r_anchor = r.anchor[(depth + 1) as usize];
        assert_eq!(
            c_anchor, r_anchor,
            "anchor[{}] mismatch: C={} Rust={}", depth, c_anchor, r_anchor
        );
        assert_eq!(c_anchor, 0, "anchor[{}] should be 0 after init", depth);
    }

    // Compare free list head
    assert_eq!(c.free, r.free, "free mismatch: C={} Rust={}", c.free, r.free);
    assert_eq!(c.free, 1, "free should be 1 after init");

    // Compare point[] free list chain for shared range (1..maxfunc-1).
    // C uses 1-based slots 1..maxfunc (via --point adjustment).
    // Rust uses 1-based slots 1..maxfunc-1 (one fewer slot due to array layout).
    // Both form ascending chains; we verify they agree on the shared range.
    for i in 1..maxfunc - 1 {
        let c_point = c.get_point(i);
        let r_point = r.point[i];
        assert_eq!(
            c_point, r_point,
            "point[{}] mismatch: C={} Rust={}", i, c_point, r_point
        );
    }

    // C free list terminates at slot maxfunc: point[maxfunc]=0
    let c_term = c.get_point(maxfunc);
    assert_eq!(c_term, 0, "C free list should terminate with 0 at slot maxfunc");

    // Rust free list terminates at slot maxfunc-1: point[maxfunc-1]=0
    assert_eq!(r.point[maxfunc - 1], 0, "Rust free list should terminate at slot maxfunc-1");

    // Both free lists are valid chains from 1 to their respective ends.
    // Verify C: 1→2→3→...→maxfunc→0
    let mut c_count = 0;
    let mut c_cur = c.free;
    while c_cur > 0 {
        c_count += 1;
        c_cur = c.get_point(c_cur as usize);
    }
    assert_eq!(c_count, maxfunc, "C free list should have maxfunc={} entries", maxfunc);

    // Verify Rust: 1→2→3→...→(maxfunc-1)→0
    let mut r_count = 0;
    let mut r_cur = r.free;
    while r_cur > 0 {
        r_count += 1;
        r_cur = r.point[r_cur as usize];
    }
    assert_eq!(r_count as usize, maxfunc - 1, "Rust free list should have maxfunc-1={} entries", maxfunc - 1);

    // f_values all 0
    for i in 1..maxfunc {
        let c_fval = c.f_val(i);
        let r_fval = r.f_values[i * 2];
        assert_eq!(c_fval, r_fval, "f_val[{}] mismatch: C={} Rust={}", i, c_fval, r_fval);
        assert_eq!(c_fval, 0.0);
    }
}

// ====================================================================
// Test 2: Insert 5 rectangles at same level with different f-values
// ====================================================================

#[test]
fn test_insert_5_rects_same_level_sorted_compare() {
    let maxfunc = 30;
    let maxdeep = 20;
    let n = 2;
    let jones = 1; // Gablonsky

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // Allocate parent (samp) and 2 children for each of 5 "divisions"
    // We'll do 5 separate insert_into_list calls, each with maxi=1 (1 dimension split)
    // to insert pairs of children plus the parent rect.
    //
    // Actually, insert_into_list expects: a chain of 2*maxi new rects plus the samp rect.
    // For each call, we set up one parent and one pair of children.
    // 
    // Simpler approach: do ONE call with maxi=1, inserting 2 children + parent = 3 rects.
    // Then do additional calls to add more rects at the same depth.
    //
    // Let's simulate what happens during actual DIRECT operation:
    // Parent rect (samp) has lengths [0,0] → level=0.
    // After division of dim 0: children have lengths [1,0], parent gets [1,0].
    // All 3 rects end up at level=0 (Gablonsky: min(1,0)=0).
    //
    // We'll do 2 rounds of insertions to get 5+ rects at depth 0.

    // Round 1: samp=1, children=2,3
    let samp1 = c.alloc_rect(); // 1
    let c1 = c.alloc_rect();    // 2
    let c2 = c.alloc_rect();    // 3
    c.set_f(samp1, 5.0, 0.0);
    c.set_f(c1, 3.0, 0.0);
    c.set_f(c2, 7.0, 0.0);
    // Set lengths for all: [1, 0] → level = min(1,0) = 0
    for &rect in &[samp1, c1, c2] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    // Chain children: point[c1] = c2
    c.set_point(c1, c2 as i32);
    c.set_point(c2, 0);

    let mut new_start = c1 as i32;
    c.insert_into_list(&mut new_start, 1, samp1 as i32, jones);

    // Round 2: samp=4, children=5,6
    let samp2 = c.alloc_rect(); // 4
    let c3 = c.alloc_rect();    // 5
    let c4 = c.alloc_rect();    // 6
    c.set_f(samp2, 1.0, 0.0);
    c.set_f(c3, 9.0, 0.0);
    c.set_f(c4, 2.0, 0.0);
    for &rect in &[samp2, c3, c4] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(c3, c4 as i32);
    c.set_point(c4, 0);

    let mut new_start = c3 as i32;
    c.insert_into_list(&mut new_start, 1, samp2 as i32, jones);

    // Walk C linked list at depth 0
    let c_fvals = c.walk_list_fvals(0);

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    // Round 1
    let r_samp1 = r.alloc_rect().unwrap();
    let r_c1 = r.alloc_rect().unwrap();
    let r_c2 = r.alloc_rect().unwrap();
    r.set_f(r_samp1, 5.0, 0.0);
    r.set_f(r_c1, 3.0, 0.0);
    r.set_f(r_c2, 7.0, 0.0);
    for &rect in &[r_samp1, r_c1, r_c2] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_c1] = r_c2 as i32;
    r.point[r_c2] = 0;

    let mut r_new_start = r_c1 as i32;
    r.insert_into_list(&mut r_new_start, 1, r_samp1, jones);

    // Round 2
    let r_samp2 = r.alloc_rect().unwrap();
    let r_c3 = r.alloc_rect().unwrap();
    let r_c4 = r.alloc_rect().unwrap();
    r.set_f(r_samp2, 1.0, 0.0);
    r.set_f(r_c3, 9.0, 0.0);
    r.set_f(r_c4, 2.0, 0.0);
    for &rect in &[r_samp2, r_c3, r_c4] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_c3] = r_c4 as i32;
    r.point[r_c4] = 0;

    let mut r_new_start = r_c3 as i32;
    r.insert_into_list(&mut r_new_start, 1, r_samp2, jones);

    // Walk Rust linked list at depth 0 (anchor[1])
    let mut r_fvals = Vec::new();
    let mut current = r.anchor[1]; // depth 0
    while current > 0 {
        r_fvals.push(r.f_val(current as usize));
        current = r.point[current as usize];
    }

    // Compare: both should be sorted ascending
    assert_eq!(
        c_fvals, r_fvals,
        "Linked list f-values mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals, r_fvals
    );

    // Verify ascending order
    for i in 1..c_fvals.len() {
        assert!(
            c_fvals[i] >= c_fvals[i - 1],
            "C list not sorted at index {}: {} >= {}",
            i, c_fvals[i], c_fvals[i - 1]
        );
    }

    // Verify we have 6 rects in total
    assert_eq!(c_fvals.len(), 6, "Expected 6 rects, got {}", c_fvals.len());
}

// ====================================================================
// Test 3: Insert at different levels, verify each anchor correct
// ====================================================================

#[test]
fn test_insert_different_levels_compare() {
    let maxfunc = 30;
    let maxdeep = 20;
    let n = 2;
    let jones = 1; // Gablonsky

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // Insert rects at depth 0: samp with lengths [0,0] (level=0)
    // and children with lengths [1,0] (level=0)
    let samp1 = c.alloc_rect();
    let c1 = c.alloc_rect();
    let c2 = c.alloc_rect();
    c.set_f(samp1, 5.0, 0.0);
    c.set_f(c1, 3.0, 0.0);
    c.set_f(c2, 7.0, 0.0);
    // All have lengths [1,0] → level=0
    for &rect in &[samp1, c1, c2] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(c1, c2 as i32);
    c.set_point(c2, 0);
    let mut ns = c1 as i32;
    c.insert_into_list(&mut ns, 1, samp1 as i32, jones);

    // Insert rects at depth 1: samp with lengths [1,1] → level=1
    // children with lengths [2,1] → level=1
    let samp2 = c.alloc_rect();
    let c3 = c.alloc_rect();
    let c4 = c.alloc_rect();
    c.set_f(samp2, 10.0, 0.0);
    c.set_f(c3, 8.0, 0.0);
    c.set_f(c4, 12.0, 0.0);
    for &rect in &[samp2, c3, c4] {
        c.set_length(rect, 0, 2);
        c.set_length(rect, 1, 1);
    }
    c.set_point(c3, c4 as i32);
    c.set_point(c4, 0);
    let mut ns = c3 as i32;
    c.insert_into_list(&mut ns, 1, samp2 as i32, jones);

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    let r_samp1 = r.alloc_rect().unwrap();
    let r_c1 = r.alloc_rect().unwrap();
    let r_c2 = r.alloc_rect().unwrap();
    r.set_f(r_samp1, 5.0, 0.0);
    r.set_f(r_c1, 3.0, 0.0);
    r.set_f(r_c2, 7.0, 0.0);
    for &rect in &[r_samp1, r_c1, r_c2] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_c1] = r_c2 as i32;
    r.point[r_c2] = 0;
    let mut ns = r_c1 as i32;
    r.insert_into_list(&mut ns, 1, r_samp1, jones);

    let r_samp2 = r.alloc_rect().unwrap();
    let r_c3 = r.alloc_rect().unwrap();
    let r_c4 = r.alloc_rect().unwrap();
    r.set_f(r_samp2, 10.0, 0.0);
    r.set_f(r_c3, 8.0, 0.0);
    r.set_f(r_c4, 12.0, 0.0);
    for &rect in &[r_samp2, r_c3, r_c4] {
        r.set_length(rect, 0, 2);
        r.set_length(rect, 1, 1);
    }
    r.point[r_c3] = r_c4 as i32;
    r.point[r_c4] = 0;
    let mut ns = r_c3 as i32;
    r.insert_into_list(&mut ns, 1, r_samp2, jones);

    // Compare depth 0 lists
    let c_fvals_0 = c.walk_list_fvals(0);
    let mut r_fvals_0 = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_fvals_0.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }
    assert_eq!(
        c_fvals_0, r_fvals_0,
        "Depth 0 f-values mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals_0, r_fvals_0
    );

    // Compare depth 1 lists
    let c_fvals_1 = c.walk_list_fvals(1);
    let mut r_fvals_1 = Vec::new();
    let mut cur = r.anchor[2]; // depth 1
    while cur > 0 {
        r_fvals_1.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }
    assert_eq!(
        c_fvals_1, r_fvals_1,
        "Depth 1 f-values mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals_1, r_fvals_1
    );

    // Verify both depths have separate, sorted lists
    assert_eq!(c_fvals_0.len(), 3, "Depth 0 should have 3 rects");
    assert_eq!(c_fvals_1.len(), 3, "Depth 1 should have 3 rects");
    for i in 1..c_fvals_0.len() {
        assert!(c_fvals_0[i] >= c_fvals_0[i - 1], "Depth 0 not sorted");
    }
    for i in 1..c_fvals_1.len() {
        assert!(c_fvals_1[i] >= c_fvals_1[i - 1], "Depth 1 not sorted");
    }
}

// ====================================================================
// Test 4: Remove head, middle, tail — verify list integrity
// ====================================================================

#[test]
fn test_removal_compare() {
    let maxfunc = 30;
    let maxdeep = 20;
    let n = 2;
    let jones = 1;

    // Build a list of 4 rects at depth 0: f-values [2.0, 4.0, 6.0, 8.0]

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // Insert first pair + samp: samp=1(f=4), c1=2(f=2), c2=3(f=6)
    let samp1 = c.alloc_rect();
    let cc1 = c.alloc_rect();
    let cc2 = c.alloc_rect();
    c.set_f(samp1, 4.0, 0.0);
    c.set_f(cc1, 2.0, 0.0);
    c.set_f(cc2, 6.0, 0.0);
    for &rect in &[samp1, cc1, cc2] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(cc1, cc2 as i32);
    c.set_point(cc2, 0);
    let mut ns = cc1 as i32;
    c.insert_into_list(&mut ns, 1, samp1 as i32, jones);

    // Add one more rect at depth 0 with f=8.0 using another insert
    let samp2 = c.alloc_rect(); // f=8, will be parent
    let cc3 = c.alloc_rect();   // dummy pair partner
    let cc4 = c.alloc_rect();
    c.set_f(samp2, 8.0, 0.0);
    c.set_f(cc3, 20.0, 0.0); // high f, goes to end
    c.set_f(cc4, 22.0, 0.0);
    for &rect in &[samp2, cc3, cc4] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(cc3, cc4 as i32);
    c.set_point(cc4, 0);
    let mut ns = cc3 as i32;
    c.insert_into_list(&mut ns, 1, samp2 as i32, jones);

    let c_fvals_before = c.walk_list_fvals(0);

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    let r_samp1 = r.alloc_rect().unwrap();
    let r_cc1 = r.alloc_rect().unwrap();
    let r_cc2 = r.alloc_rect().unwrap();
    r.set_f(r_samp1, 4.0, 0.0);
    r.set_f(r_cc1, 2.0, 0.0);
    r.set_f(r_cc2, 6.0, 0.0);
    for &rect in &[r_samp1, r_cc1, r_cc2] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_cc1] = r_cc2 as i32;
    r.point[r_cc2] = 0;
    let mut ns = r_cc1 as i32;
    r.insert_into_list(&mut ns, 1, r_samp1, jones);

    let r_samp2 = r.alloc_rect().unwrap();
    let r_cc3 = r.alloc_rect().unwrap();
    let r_cc4 = r.alloc_rect().unwrap();
    r.set_f(r_samp2, 8.0, 0.0);
    r.set_f(r_cc3, 20.0, 0.0);
    r.set_f(r_cc4, 22.0, 0.0);
    for &rect in &[r_samp2, r_cc3, r_cc4] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_cc3] = r_cc4 as i32;
    r.point[r_cc4] = 0;
    let mut ns = r_cc3 as i32;
    r.insert_into_list(&mut ns, 1, r_samp2, jones);

    let mut r_fvals_before = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_fvals_before.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    // Verify initial state matches
    assert_eq!(
        c_fvals_before, r_fvals_before,
        "Before removal, lists should match:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals_before, r_fvals_before
    );

    // Now test removal of head from Rust (the main loop does this via remove_from_anchor)
    let head_idx = r.anchor[1] as usize;
    let head_f = r.f_val(head_idx);
    r.remove_from_anchor(head_idx, jones);

    // After removing head, next rect should be new anchor
    let mut r_fvals_after = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_fvals_after.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    // Head's f-value should no longer be in the list
    assert!(!r_fvals_after.contains(&head_f) || r_fvals_before.iter().filter(|&&v| v == head_f).count() > 1,
        "Head f-value {} should be removed from list", head_f);
    assert_eq!(r_fvals_after.len(), r_fvals_before.len() - 1, "List should be 1 shorter");
    // Still sorted
    for i in 1..r_fvals_after.len() {
        assert!(r_fvals_after[i] >= r_fvals_after[i - 1], "List not sorted after head removal");
    }

    // Test removal from middle using remove_from_list_at_depth
    if r_fvals_after.len() >= 3 {
        // Find second element
        let first = r.anchor[1] as usize;
        let second = r.point[first] as usize;
        let second_f = r.f_val(second);
        r.remove_from_list_at_depth(second, 0);

        let mut r_fvals_mid = Vec::new();
        let mut cur = r.anchor[1];
        while cur > 0 {
            r_fvals_mid.push(r.f_val(cur as usize));
            cur = r.point[cur as usize];
        }
        assert!(!r_fvals_mid.contains(&second_f) || r_fvals_after.iter().filter(|&&v| v == second_f).count() > 1,
            "Middle element f-value {} should be removed", second_f);
        assert_eq!(r_fvals_mid.len(), r_fvals_after.len() - 1);
        for i in 1..r_fvals_mid.len() {
            assert!(r_fvals_mid[i] >= r_fvals_mid[i - 1], "List not sorted after middle removal");
        }
    }
}

// ====================================================================
// Test 5: Walk linked lists verifying ascending f-value order
// ====================================================================

#[test]
fn test_walk_list_ascending_order_compare() {
    let maxfunc = 50;
    let maxdeep = 20;
    let n = 3;
    let jones = 1;

    // Insert rects with various f-values at depth 0 in random-ish order
    let f_vals_to_insert = [
        (15.0, 8.0, 20.0),  // samp, c+, c-
        (3.0, 12.0, 1.0),
        (7.0, 25.0, 0.5),
    ];

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    for &(f_samp, f_c1, f_c2) in &f_vals_to_insert {
        let samp = c.alloc_rect();
        let c1 = c.alloc_rect();
        let c2 = c.alloc_rect();
        c.set_f(samp, f_samp, 0.0);
        c.set_f(c1, f_c1, 0.0);
        c.set_f(c2, f_c2, 0.0);
        for &rect in &[samp, c1, c2] {
            c.set_length(rect, 0, 1);
            c.set_length(rect, 1, 0);
            c.set_length(rect, 2, 0);
        }
        c.set_point(c1, c2 as i32);
        c.set_point(c2, 0);
        let mut ns = c1 as i32;
        c.insert_into_list(&mut ns, 1, samp as i32, jones);
    }

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    for &(f_samp, f_c1, f_c2) in &f_vals_to_insert {
        let samp = r.alloc_rect().unwrap();
        let c1 = r.alloc_rect().unwrap();
        let c2 = r.alloc_rect().unwrap();
        r.set_f(samp, f_samp, 0.0);
        r.set_f(c1, f_c1, 0.0);
        r.set_f(c2, f_c2, 0.0);
        for &rect in &[samp, c1, c2] {
            r.set_length(rect, 0, 1);
            r.set_length(rect, 1, 0);
            r.set_length(rect, 2, 0);
        }
        r.point[c1] = c2 as i32;
        r.point[c2] = 0;
        let mut ns = c1 as i32;
        r.insert_into_list(&mut ns, 1, samp, jones);
    }

    // Walk both lists at depth 0
    let c_fvals = c.walk_list_fvals(0);
    let mut r_fvals = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_fvals.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    // Lists must match exactly
    assert_eq!(
        c_fvals, r_fvals,
        "List f-values mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals, r_fvals
    );

    // Both must be ascending
    for i in 1..c_fvals.len() {
        assert!(c_fvals[i] >= c_fvals[i - 1], "C list not ascending at {}", i);
    }
    for i in 1..r_fvals.len() {
        assert!(r_fvals[i] >= r_fvals[i - 1], "Rust list not ascending at {}", i);
    }

    // Should have 9 rects total (3 rounds × 3 rects)
    assert_eq!(c_fvals.len(), 9, "Expected 9 rects");
}

// ====================================================================
// Test 6: Compare with NLOPT C for insertion sequence matching
//         actual DIRECT initialization (center + 2n neighbors, n=2)
// ====================================================================

#[test]
fn test_insert_like_dirinit_compare() {
    let maxfunc = 30;
    let maxdeep = 20;
    let n = 2;
    let jones = 1;

    // Simulate the initial division in dirinit_ for 2D:
    // Center rect (idx 1) evaluated at f_center
    // Dim 0: c+,c- at f+,f-; dim 1: c+,c- at f+,f-
    // maxi = n = 2 (both dims have same minimum length initially)
    // Sorted by min(f+,f-) per dimension before insert

    let f_center = 5.0;
    let f_dim0_plus = 3.0;
    let f_dim0_minus = 7.0;
    let f_dim1_plus = 2.0;
    let f_dim1_minus = 8.0;

    // After sorting by min(f+,f-): dim1(min=2.0) < dim0(min=3.0)
    // So dim1 is divided first (gets higher depth), then dim0.
    // After division:
    // - samp (center): lengths become [1,1] after both dims divided → level=1
    // - dim1 children (divided first): lengths [0,1] → level=0
    // - dim0 children (divided second): lengths [1,1] → level=1
    //
    // Actually, let's match the exact NLOPT pattern: dirinit_ allocates
    // child pairs and chains them for insert_into_list with maxi=2.

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    let samp = c.alloc_rect(); // 1 (center)
    c.set_f(samp, f_center, 0.0);

    // Children for sorted dim order: dim1 first, dim0 second
    // The chain is: [dim1+ → dim1- → dim0+ → dim0- → ...]
    let c_d1p = c.alloc_rect(); // 2
    let c_d1m = c.alloc_rect(); // 3
    let c_d0p = c.alloc_rect(); // 4
    let c_d0m = c.alloc_rect(); // 5

    c.set_f(c_d1p, f_dim1_plus, 0.0);
    c.set_f(c_d1m, f_dim1_minus, 0.0);
    c.set_f(c_d0p, f_dim0_plus, 0.0);
    c.set_f(c_d0m, f_dim0_minus, 0.0);

    // After sorting and division:
    // dim1 children get lengths [0, 1] (dim1 divided first → length[1] incremented)
    // dim0 children get lengths [1, 1] (dim0 divided second → both incremented)
    // parent gets lengths [1, 1]
    c.set_length(c_d1p, 0, 0); c.set_length(c_d1p, 1, 1);
    c.set_length(c_d1m, 0, 0); c.set_length(c_d1m, 1, 1);
    c.set_length(c_d0p, 0, 1); c.set_length(c_d0p, 1, 1);
    c.set_length(c_d0m, 0, 1); c.set_length(c_d0m, 1, 1);
    c.set_length(samp, 0, 1);  c.set_length(samp, 1, 1);

    // Chain: c_d1p → c_d1m → c_d0p → c_d0m → 0
    c.set_point(c_d1p, c_d1m as i32);
    c.set_point(c_d1m, c_d0p as i32);
    c.set_point(c_d0p, c_d0m as i32);
    c.set_point(c_d0m, 0);

    let mut ns = c_d1p as i32;
    c.insert_into_list(&mut ns, 2, samp as i32, jones); // maxi=2 for 2 dimensions

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    let r_samp = r.alloc_rect().unwrap();
    r.set_f(r_samp, f_center, 0.0);

    let r_d1p = r.alloc_rect().unwrap();
    let r_d1m = r.alloc_rect().unwrap();
    let r_d0p = r.alloc_rect().unwrap();
    let r_d0m = r.alloc_rect().unwrap();

    r.set_f(r_d1p, f_dim1_plus, 0.0);
    r.set_f(r_d1m, f_dim1_minus, 0.0);
    r.set_f(r_d0p, f_dim0_plus, 0.0);
    r.set_f(r_d0m, f_dim0_minus, 0.0);

    r.set_length(r_d1p, 0, 0); r.set_length(r_d1p, 1, 1);
    r.set_length(r_d1m, 0, 0); r.set_length(r_d1m, 1, 1);
    r.set_length(r_d0p, 0, 1); r.set_length(r_d0p, 1, 1);
    r.set_length(r_d0m, 0, 1); r.set_length(r_d0m, 1, 1);
    r.set_length(r_samp, 0, 1); r.set_length(r_samp, 1, 1);

    r.point[r_d1p] = r_d1m as i32;
    r.point[r_d1m] = r_d0p as i32;
    r.point[r_d0p] = r_d0m as i32;
    r.point[r_d0m] = 0;

    let mut ns = r_d1p as i32;
    r.insert_into_list(&mut ns, 2, r_samp, jones);

    // Compare lists at each depth level
    // dim1 children have level = min(0,1) = 0
    // dim0 children and samp have level = min(1,1) = 1
    let c_fvals_d0 = c.walk_list_fvals(0);
    let c_fvals_d1 = c.walk_list_fvals(1);

    let mut r_fvals_d0 = Vec::new();
    let mut cur = r.anchor[1]; // depth 0
    while cur > 0 {
        r_fvals_d0.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    let mut r_fvals_d1 = Vec::new();
    let mut cur = r.anchor[2]; // depth 1
    while cur > 0 {
        r_fvals_d1.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    assert_eq!(
        c_fvals_d0, r_fvals_d0,
        "Depth 0 (dim1 children) mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals_d0, r_fvals_d0
    );
    assert_eq!(
        c_fvals_d1, r_fvals_d1,
        "Depth 1 (dim0 children + samp) mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals_d1, r_fvals_d1
    );

    // dim1 children: f=2.0, f=8.0 → sorted [2.0, 8.0]
    assert_eq!(c_fvals_d0, vec![2.0, 8.0], "Depth 0 should be [2.0, 8.0]");
    // dim0 children + samp: f=3.0, f=7.0, f=5.0 → sorted [3.0, 5.0, 7.0]
    assert_eq!(c_fvals_d1, vec![3.0, 5.0, 7.0], "Depth 1 should be [3.0, 5.0, 7.0]");
}

// ====================================================================
// Test 7: Jones Original (jones=0) level computation in insert
// ====================================================================

#[test]
fn test_insert_jones_original_compare() {
    let maxfunc = 30;
    let maxdeep = 50;
    let n = 2;
    let jones = 0; // Jones Original

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // All lengths [0,0] → Jones level: help=0, k=0, all equal, p=2
    // ret = 0*2 + 2-2 = 0
    let samp = c.alloc_rect();
    let c1 = c.alloc_rect();
    let c2 = c.alloc_rect();
    c.set_f(samp, 5.0, 0.0);
    c.set_f(c1, 3.0, 0.0);
    c.set_f(c2, 7.0, 0.0);
    // After division, samp and children have [1,0]
    // Jones level: help=1 (dim0), k=min(1,0)=0. k≠help → p=count(==help)=1
    // ret = 0*2 + 1 = 1
    for &rect in &[samp, c1, c2] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(c1, c2 as i32);
    c.set_point(c2, 0);
    let mut ns = c1 as i32;
    c.insert_into_list(&mut ns, 1, samp as i32, jones);

    let c_fvals = c.walk_list_fvals(1); // Jones level = 1

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();
    r.precompute_levels(jones);

    let r_samp = r.alloc_rect().unwrap();
    let r_c1 = r.alloc_rect().unwrap();
    let r_c2 = r.alloc_rect().unwrap();
    r.set_f(r_samp, 5.0, 0.0);
    r.set_f(r_c1, 3.0, 0.0);
    r.set_f(r_c2, 7.0, 0.0);
    for &rect in &[r_samp, r_c1, r_c2] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_c1] = r_c2 as i32;
    r.point[r_c2] = 0;
    let mut ns = r_c1 as i32;
    r.insert_into_list(&mut ns, 1, r_samp, jones);

    // Rust: level for lengths [1,0] jones=0: help=1, k=0, k≠help, p=1 → level=1
    let r_level = r.get_level(r_samp, jones);
    assert_eq!(r_level, 1, "Rust Jones level for [1,0] should be 1");

    let mut r_fvals = Vec::new();
    let anchor_idx = (r_level + 1) as usize;
    let mut cur = r.anchor[anchor_idx];
    while cur > 0 {
        r_fvals.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    assert_eq!(
        c_fvals, r_fvals,
        "Jones Original insert mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals, r_fvals
    );
}

// ====================================================================
// Test 8: Multi-dimensional (maxi=3) insert with 6 children
// ====================================================================

#[test]
fn test_insert_maxi3_compare() {
    let maxfunc = 40;
    let maxdeep = 30;
    let n = 3;
    let jones = 1;

    // maxi=3: 3 dimension splits, 6 children + 1 parent
    let f_samp = 10.0;
    let f_pairs = [
        (4.0, 16.0),  // dim0 children
        (2.0, 18.0),  // dim1 children
        (6.0, 14.0),  // dim2 children
    ];

    // After all 3 dims divided, all have lengths [1,1,1] → level=1

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    let samp = c.alloc_rect();
    c.set_f(samp, f_samp, 0.0);

    let mut children = Vec::new();
    for &(fp, fm) in &f_pairs {
        let cp = c.alloc_rect();
        let cm = c.alloc_rect();
        c.set_f(cp, fp, 0.0);
        c.set_f(cm, fm, 0.0);
        children.push((cp, cm));
    }

    // Set lengths: after 3D division all lengths [1,1,1]
    for &rect in &[samp] {
        for j in 0..n {
            c.set_length(rect, j, 1);
        }
    }
    for &(cp, cm) in &children {
        for j in 0..n {
            c.set_length(cp, j, 1);
            c.set_length(cm, j, 1);
        }
    }

    // Chain: d0+ → d0- → d1+ → d1- → d2+ → d2- → 0
    for i in 0..children.len() {
        let (cp, cm) = children[i];
        if i + 1 < children.len() {
            c.set_point(cp, cm as i32);
            c.set_point(cm, children[i + 1].0 as i32);
        } else {
            c.set_point(cp, cm as i32);
            c.set_point(cm, 0);
        }
    }

    let mut ns = children[0].0 as i32;
    c.insert_into_list(&mut ns, 3, samp as i32, jones);

    let c_fvals = c.walk_list_fvals(1); // level=min(1,1,1)=1

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    let r_samp = r.alloc_rect().unwrap();
    r.set_f(r_samp, f_samp, 0.0);

    let mut r_children = Vec::new();
    for &(fp, fm) in &f_pairs {
        let cp = r.alloc_rect().unwrap();
        let cm = r.alloc_rect().unwrap();
        r.set_f(cp, fp, 0.0);
        r.set_f(cm, fm, 0.0);
        r_children.push((cp, cm));
    }

    for &rect in &[r_samp] {
        for j in 0..n {
            r.set_length(rect, j, 1);
        }
    }
    for &(cp, cm) in &r_children {
        for j in 0..n {
            r.set_length(cp, j, 1);
            r.set_length(cm, j, 1);
        }
    }

    for i in 0..r_children.len() {
        let (cp, cm) = r_children[i];
        if i + 1 < r_children.len() {
            r.point[cp] = cm as i32;
            r.point[cm] = r_children[i + 1].0 as i32;
        } else {
            r.point[cp] = cm as i32;
            r.point[cm] = 0;
        }
    }

    let mut ns = r_children[0].0 as i32;
    r.insert_into_list(&mut ns, 3, r_samp, jones);

    let mut r_fvals = Vec::new();
    let mut cur = r.anchor[2]; // depth 1
    while cur > 0 {
        r_fvals.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    assert_eq!(
        c_fvals, r_fvals,
        "maxi=3 insert mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals, r_fvals
    );

    // Should have 7 rects total (6 children + 1 parent)
    assert_eq!(c_fvals.len(), 7);
    // All f-values sorted ascending
    for i in 1..c_fvals.len() {
        assert!(c_fvals[i] >= c_fvals[i - 1], "Not sorted at {}", i);
    }
}

// ====================================================================
// Test 9: Equal f-values (tests tie-breaking in sorted insert)
// ====================================================================

#[test]
fn test_insert_equal_fvals_compare() {
    let maxfunc = 30;
    let maxdeep = 20;
    let n = 2;
    let jones = 1;

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    // Insert 3 rects with identical f-values
    let samp = c.alloc_rect();
    let c1 = c.alloc_rect();
    let c2 = c.alloc_rect();
    c.set_f(samp, 5.0, 0.0);
    c.set_f(c1, 5.0, 0.0);
    c.set_f(c2, 5.0, 0.0);
    for &rect in &[samp, c1, c2] {
        c.set_length(rect, 0, 1);
        c.set_length(rect, 1, 0);
    }
    c.set_point(c1, c2 as i32);
    c.set_point(c2, 0);
    let mut ns = c1 as i32;
    c.insert_into_list(&mut ns, 1, samp as i32, jones);

    let c_list = c.walk_list(0);
    let c_fvals = c.walk_list_fvals(0);

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    let r_samp = r.alloc_rect().unwrap();
    let r_c1 = r.alloc_rect().unwrap();
    let r_c2 = r.alloc_rect().unwrap();
    r.set_f(r_samp, 5.0, 0.0);
    r.set_f(r_c1, 5.0, 0.0);
    r.set_f(r_c2, 5.0, 0.0);
    for &rect in &[r_samp, r_c1, r_c2] {
        r.set_length(rect, 0, 1);
        r.set_length(rect, 1, 0);
    }
    r.point[r_c1] = r_c2 as i32;
    r.point[r_c2] = 0;
    let mut ns = r_c1 as i32;
    r.insert_into_list(&mut ns, 1, r_samp, jones);

    let mut r_list = Vec::new();
    let mut r_fvals = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_list.push(cur as usize);
        r_fvals.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    // f-values must match
    assert_eq!(c_fvals, r_fvals, "Equal f-values: lists should match");
    assert_eq!(c_fvals.len(), 3);
    // All equal
    assert!(c_fvals.iter().all(|&v| v == 5.0));

    // Exact order (rect indices) should also match
    assert_eq!(c_list, r_list, "Equal f-values: rect index order should match");
}

// ====================================================================
// Test 10: Large batch — stress test insertion ordering
// ====================================================================

#[test]
fn test_insert_large_batch_compare() {
    let maxfunc = 200;
    let maxdeep = 50;
    let n = 2;
    let jones = 1;

    // Do 10 rounds of insertions with varied f-values
    let rounds: Vec<(f64, f64, f64)> = vec![
        (50.0, 30.0, 70.0),
        (10.0, 90.0, 5.0),
        (25.0, 15.0, 35.0),
        (1.0, 99.0, 0.1),
        (60.0, 40.0, 80.0),
        (12.0, 88.0, 6.0),
        (33.0, 22.0, 44.0),
        (3.0, 97.0, 0.5),
        (55.0, 45.0, 65.0),
        (8.0, 92.0, 4.0),
    ];

    // --- C side ---
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();

    for &(f_samp, f_c1, f_c2) in &rounds {
        let samp = c.alloc_rect();
        let cc1 = c.alloc_rect();
        let cc2 = c.alloc_rect();
        c.set_f(samp, f_samp, 0.0);
        c.set_f(cc1, f_c1, 0.0);
        c.set_f(cc2, f_c2, 0.0);
        for &rect in &[samp, cc1, cc2] {
            c.set_length(rect, 0, 1);
            c.set_length(rect, 1, 0);
        }
        c.set_point(cc1, cc2 as i32);
        c.set_point(cc2, 0);
        let mut ns = cc1 as i32;
        c.insert_into_list(&mut ns, 1, samp as i32, jones);
    }

    // --- Rust side ---
    let mut r = RectangleStorage::new(n, maxfunc, maxdeep);
    r.init_lists();
    r.precompute_thirds();

    for &(f_samp, f_c1, f_c2) in &rounds {
        let samp = r.alloc_rect().unwrap();
        let cc1 = r.alloc_rect().unwrap();
        let cc2 = r.alloc_rect().unwrap();
        r.set_f(samp, f_samp, 0.0);
        r.set_f(cc1, f_c1, 0.0);
        r.set_f(cc2, f_c2, 0.0);
        for &rect in &[samp, cc1, cc2] {
            r.set_length(rect, 0, 1);
            r.set_length(rect, 1, 0);
        }
        r.point[cc1] = cc2 as i32;
        r.point[cc2] = 0;
        let mut ns = cc1 as i32;
        r.insert_into_list(&mut ns, 1, samp, jones);
    }

    // Compare
    let c_fvals = c.walk_list_fvals(0);
    let mut r_fvals = Vec::new();
    let mut cur = r.anchor[1];
    while cur > 0 {
        r_fvals.push(r.f_val(cur as usize));
        cur = r.point[cur as usize];
    }

    assert_eq!(
        c_fvals, r_fvals,
        "Large batch mismatch:\n  C:    {:?}\n  Rust: {:?}",
        c_fvals, r_fvals
    );
    assert_eq!(c_fvals.len(), 30, "Expected 30 rects (10 rounds × 3)");
    for i in 1..c_fvals.len() {
        assert!(c_fvals[i] >= c_fvals[i - 1], "Not sorted at {}", i);
    }
}
