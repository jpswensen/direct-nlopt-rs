#![cfg(feature = "nlopt-compare")]
#![allow(non_camel_case_types, dead_code)]

//! Comparative unit tests for PotentiallyOptimal::select() (dirchoose_) and
//! PotentiallyOptimal::double_insert() (dirdoubleinsert_).
//!
//! Verifies that the Rust convex hull selection produces identical results to
//! NLOPT C's direct_dirchoose_() and direct_dirdoubleinsert_() across multiple
//! scenarios.
//!
//! NLOPT C functions:
//! - direct_dirchoose_() in DIRsubrout.c lines 102–261
//! - direct_dirdoubleinsert_() in DIRsubrout.c lines 274–332
//! - direct_dirgetlevel_() in DIRsubrout.c lines 33–82
//!
//! Rust functions:
//! - PotentiallyOptimal::select() in storage.rs
//! - PotentiallyOptimal::double_insert() in storage.rs

use std::os::raw::{c_double, c_int};

// FFI declarations for NLOPT C functions.
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

    fn direct_dirchoose_(
        anchor: *mut c_int,
        s: *mut c_int,
        actdeep: *mut c_int,
        f: *mut c_double,
        minf: *mut c_double,
        epsrel: c_double,
        epsabs: c_double,
        thirds: *mut c_double, // actually receives "levels" from caller
        maxpos: *mut c_int,
        length: *mut c_int,
        maxfunc: *mut c_int,
        maxdeep: *const c_int,
        maxdiv: *const c_int,
        n: *mut c_int,
        logfile: *mut std::ffi::c_void,
        cheat: *mut c_int,
        kmax: *mut c_double,
        ifeasiblef: *mut c_int,
        jones: c_int,
    );

    fn direct_dirdoubleinsert_(
        anchor: *mut c_int,
        s: *mut c_int,
        maxpos: *mut c_int,
        point: *mut c_int,
        f: *mut c_double,
        maxdeep: *const c_int,
        maxfunc: *mut c_int,
        maxdiv: *const c_int,
        ierror: *mut c_int,
    );

    fn direct_dirgetlevel_(
        pos: *const c_int,
        length: *mut c_int,
        maxfunc: *const c_int,
        n: *const c_int,
        jones: c_int,
    ) -> c_int;
}

/// Mirror of NLOPT C memory layout for comparison testing.
struct CStorage {
    anchor: Vec<c_int>,   // size: maxdeep + 2
    point: Vec<c_int>,    // size: maxfunc
    f: Vec<c_double>,     // size: maxfunc * 2
    length: Vec<c_int>,   // size: maxfunc * n
    levels: Vec<c_double>, // size: maxdeep + 1
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
            levels: vec![0.0f64; maxdeep + 1],
            free: 0,
            maxfunc: maxfunc as c_int,
            maxdeep: maxdeep as c_int,
            n: n as c_int,
        }
    }

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

    /// Precompute levels[] matching NLOPT's dirinit_ logic.
    fn precompute_levels(&mut self, jones: i32) {
        let n = self.n as usize;
        if jones == 0 {
            // Jones Original: distance from midpoint to corner
            let mut w = vec![0.0f64; n];
            for j in 0..n {
                w[j] = ((n - j) as f64 + j as f64 / 9.0).sqrt() * 0.5;
            }
            let mut help2 = 1.0f64;
            let imax = self.maxdeep as usize / n;
            for i in 1..=imax {
                for j in 0..n {
                    let idx = (i - 1) * n + j;
                    if idx < self.levels.len() {
                        self.levels[idx] = w[j] / help2;
                    }
                }
                help2 *= 3.0;
            }
        } else {
            // Gablonsky: 1/3^k
            self.levels[0] = 1.0;
            let mut help2 = 3.0f64;
            for i in 1..=self.maxdeep as usize {
                if i < self.levels.len() {
                    self.levels[i] = 1.0 / help2;
                    help2 *= 3.0;
                }
            }
        }
    }

    /// Set f-value and flag for 1-based rect index.
    fn set_f(&mut self, idx: usize, val: f64, flag: f64) {
        self.f[2 * (idx - 1)] = val;
        self.f[2 * (idx - 1) + 1] = flag;
    }

    fn f_val(&self, idx: usize) -> f64 {
        self.f[2 * (idx - 1)]
    }

    /// Set length index for 1-based rect pos, 0-based dimension dim_j.
    fn set_length(&mut self, pos: usize, dim_j: usize, val: i32) {
        self.length[(pos - 1) * self.n as usize + dim_j] = val;
    }

    /// Allocate a rect from the free list (returns 1-based index).
    fn alloc_rect(&mut self) -> usize {
        let idx = self.free as usize;
        assert!(idx > 0 && idx <= self.maxfunc as usize);
        self.free = self.point[idx - 1];
        self.point[idx - 1] = 0;
        idx
    }

    /// Insert rects into anchor linked list via NLOPT C.
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

    fn get_anchor(&self, depth: i32) -> i32 {
        self.anchor[(depth + 1) as usize]
    }

    /// Call direct_dirchoose_ and return selected indices and levels.
    fn dirchoose(
        &mut self,
        act_deep: i32,
        minf: f64,
        eps_rel: f64,
        eps_abs: f64,
        ifeasible_f: i32,
        jones: i32,
        maxdiv: i32,
    ) -> (Vec<i32>, Vec<i32>, i32) {
        // Allocate s array: 2D array with dimensions maxdiv × 2 (column-major)
        // s[k + s_dim1*col] where s_dim1 = maxdiv
        let mut s = vec![0i32; maxdiv as usize * 2];
        let mut maxpos: c_int = 0;
        let mut act_deep_mut = act_deep;
        let mut minf_mut = minf;
        let mut cheat: c_int = 0;
        let mut kmax: c_double = 1e10;
        let mut ifeasiblef_mut = ifeasible_f;

        unsafe {
            direct_dirchoose_(
                self.anchor.as_mut_ptr(),
                s.as_mut_ptr(),
                &mut act_deep_mut,
                self.f.as_mut_ptr(),
                &mut minf_mut,
                eps_rel,
                eps_abs,
                self.levels.as_mut_ptr(),
                &mut maxpos,
                self.length.as_mut_ptr(),
                &mut self.maxfunc,
                &self.maxdeep,
                &maxdiv,
                &mut self.n,
                std::ptr::null_mut(), // logfile
                &mut cheat,
                &mut kmax,
                &mut ifeasiblef_mut,
                jones,
            );
        }

        // Extract results: s is 1-based, column major with s_dim1 = maxdiv
        // After parameter adjustments: s[k + s_dim1] is index, s[k + 2*s_dim1] is level
        // In raw memory: s[(k-1) + (col-1)*maxdiv]
        // col1 (indices): raw index = (k-1) + 0 = k-1
        // col2 (levels): raw index = (k-1) + maxdiv = k-1 + maxdiv
        let mut indices = Vec::new();
        let mut levels = Vec::new();
        let md = maxdiv as usize;
        for k in 1..=maxpos {
            let ku = k as usize;
            let idx = s[ku - 1];
            let lvl = s[ku - 1 + md];
            indices.push(idx);
            levels.push(lvl);
        }

        (indices, levels, maxpos)
    }

    /// Call direct_dirdoubleinsert_ on existing s/maxpos data.
    fn dirdoubleinsert(
        &mut self,
        s: &mut Vec<i32>,
        maxpos: &mut i32,
        maxdiv: i32,
    ) -> i32 {
        let mut ierror: c_int = 0;
        unsafe {
            direct_dirdoubleinsert_(
                self.anchor.as_mut_ptr(),
                s.as_mut_ptr(),
                maxpos,
                self.point.as_mut_ptr(),
                self.f.as_mut_ptr(),
                &self.maxdeep,
                &mut self.maxfunc,
                &maxdiv,
                &mut ierror,
            );
        }
        ierror
    }
}

use direct_nlopt::storage::{PotentiallyOptimal, RectangleStorage, FEASIBLE, INFEASIBLE};

/// Create a matching Rust RectangleStorage with given parameters.
fn make_rust_storage(n: usize, maxfunc: usize, maxdeep: usize) -> RectangleStorage {
    let mut r = RectangleStorage::new(n, 0, 0);
    r.dim = n;
    r.maxfunc = maxfunc;
    r.maxdeep = maxdeep;
    r.centers = vec![0.0; maxfunc * n];
    r.f_values = vec![0.0; maxfunc * 2];
    r.lengths = vec![0i32; maxfunc * n];
    r.point = vec![0i32; maxfunc];
    r.anchor = vec![0i32; maxdeep + 2];
    r.thirds = vec![0.0; maxdeep + 1];
    r.levels = vec![0.0; maxdeep + 1];
    r.free = 0;
    r.init_lists();
    r
}

/// Helper: set up C and Rust storages with identical rect configurations.
fn setup_pair(n: usize, maxfunc: usize, maxdeep: usize) -> (CStorage, RectangleStorage) {
    let mut c = CStorage::new(n, maxfunc, maxdeep);
    c.init_lists();
    let r = make_rust_storage(n, maxfunc, maxdeep);
    (c, r)
}

/// Manually insert a single rect into an anchor-linked list sorted by f-value.
/// This directly manipulates the anchor/point arrays, bypassing insert_into_list
/// which is designed for the pair-insert division pattern.
fn manual_sorted_insert_c(c: &mut CStorage, idx: usize, depth: i32) {
    let anchor_idx = (depth + 1) as usize;
    let fv = c.f_val(idx);
    let head = c.anchor[anchor_idx];

    if head == 0 {
        // Empty list
        c.anchor[anchor_idx] = idx as i32;
        c.point[idx - 1] = 0; // terminate
        return;
    }

    // Check if we should become the new head
    if fv < c.f_val(head as usize) {
        c.anchor[anchor_idx] = idx as i32;
        c.point[idx - 1] = head;
        return;
    }

    // Walk the list to find insertion point
    let mut prev = head as usize;
    let mut cur = c.point[prev - 1];
    while cur > 0 {
        if fv < c.f_val(cur as usize) {
            c.point[prev - 1] = idx as i32;
            c.point[idx - 1] = cur;
            return;
        }
        prev = cur as usize;
        cur = c.point[prev - 1];
    }
    // Insert at end
    c.point[prev - 1] = idx as i32;
    c.point[idx - 1] = 0;
}

fn manual_sorted_insert_r(r: &mut RectangleStorage, idx: usize, depth: i32) {
    let anchor_idx = (depth + 1) as usize;
    let fv = r.f_val(idx);
    let head = r.anchor[anchor_idx];

    if head == 0 {
        r.anchor[anchor_idx] = idx as i32;
        r.point[idx] = 0;
        return;
    }

    if fv < r.f_val(head as usize) {
        r.anchor[anchor_idx] = idx as i32;
        r.point[idx] = head;
        return;
    }

    let mut prev = head as usize;
    let mut cur = r.point[prev];
    while cur > 0 {
        if fv < r.f_val(cur as usize) {
            r.point[prev] = idx as i32;
            r.point[idx] = cur;
            return;
        }
        prev = cur as usize;
        cur = r.point[prev];
    }
    r.point[prev] = idx as i32;
    r.point[idx] = 0;
}

/// Insert a single rect into both C and Rust storages.
/// Allocates from free list, sets f-value, lengths, computes level, and inserts
/// into the anchor linked list sorted by f-value.
fn insert_rect(
    c: &mut CStorage,
    r: &mut RectangleStorage,
    f_val: f64,
    f_flag: f64,
    lengths: &[i32],
    jones: i32,
) -> usize {
    let n = c.n as usize;
    assert_eq!(lengths.len(), n);

    // Allocate from C free list
    let c_idx = c.alloc_rect();

    // Set f-value and lengths in C
    c.set_f(c_idx, f_val, f_flag);
    for j in 0..n {
        c.set_length(c_idx, j, lengths[j]);
    }

    // Compute level via C FFI
    let level = unsafe {
        direct_dirgetlevel_(
            &(c_idx as c_int),
            c.length.as_mut_ptr(),
            &c.maxfunc,
            &c.n,
            jones,
        )
    };

    // Insert into C anchor list (sorted by f-value)
    manual_sorted_insert_c(c, c_idx, level);

    // Allocate from Rust free list (should give same index)
    let r_idx = r.alloc_rect().expect("Rust free list exhausted");
    assert_eq!(c_idx, r_idx, "Free list mismatch");

    // Set f-value and lengths in Rust
    r.set_f(r_idx, f_val, f_flag);
    for j in 0..n {
        r.lengths[r_idx * n + j] = lengths[j];
    }

    // Compute level via Rust
    let r_level = r.get_level(r_idx, jones);
    assert_eq!(level, r_level, "Level mismatch for rect {}", c_idx);

    // Insert into Rust anchor list
    manual_sorted_insert_r(r, r_idx, r_level);

    c_idx
}

/// Compare dirchoose output between C and Rust.
fn compare_dirchoose(
    c: &mut CStorage,
    r: &RectangleStorage,
    act_deep: i32,
    minf: f64,
    eps_rel: f64,
    eps_abs: f64,
    ifeasible_f: i32,
    jones: i32,
    maxdiv: i32,
    scenario: &str,
) {
    // C side
    let (c_indices, c_levels, c_maxpos) =
        c.dirchoose(act_deep, minf, eps_rel, eps_abs, ifeasible_f, jones, maxdiv);

    // Rust side
    let mut po = PotentiallyOptimal::new(maxdiv as usize);
    po.select(r, act_deep, minf, eps_rel, eps_abs, ifeasible_f, jones);

    // Collect Rust selected (non-zero entries)
    let mut r_indices: Vec<i32> = Vec::new();
    let mut r_levels: Vec<i32> = Vec::new();
    for i in 0..po.count {
        if po.indices[i] > 0 {
            r_indices.push(po.indices[i]);
            r_levels.push(po.rect_levels[i]);
        }
    }

    // Also collect non-zero from C side (C may have zeros for eliminated entries)
    let mut c_nonzero_indices: Vec<i32> = Vec::new();
    let mut c_nonzero_levels: Vec<i32> = Vec::new();
    for i in 0..c_indices.len() {
        if c_indices[i] > 0 {
            c_nonzero_indices.push(c_indices[i]);
            c_nonzero_levels.push(c_levels[i]);
        }
    }

    println!("--- {} ---", scenario);
    println!("  C maxpos={}, selected indices={:?}, levels={:?}",
             c_maxpos, c_nonzero_indices, c_nonzero_levels);
    println!("  R count={}, selected indices={:?}, levels={:?}",
             po.count, r_indices, r_levels);

    // Sort both for comparison (selection order may differ but contents should match)
    let mut c_sorted = c_nonzero_indices.clone();
    c_sorted.sort();
    let mut r_sorted = r_indices.clone();
    r_sorted.sort();

    assert_eq!(
        c_sorted, r_sorted,
        "{}: selected indices mismatch.\n  C={:?}\n  R={:?}",
        scenario, c_nonzero_indices, r_indices
    );

    // Also verify levels match for corresponding indices
    for &idx in &c_nonzero_indices {
        let c_pos = c_nonzero_indices.iter().position(|&x| x == idx).unwrap();
        let r_pos = r_indices.iter().position(|&x| x == idx).unwrap();
        assert_eq!(
            c_nonzero_levels[c_pos], r_levels[r_pos],
            "{}: level mismatch for rect {}. C={}, R={}",
            scenario, idx, c_nonzero_levels[c_pos], r_levels[r_pos]
        );
    }
}

// ====================================================================
// Scenario 1: All rects at different levels with monotonically
// decreasing f → all should be selected (all on convex hull)
// ====================================================================
#[test]
fn test_dirchoose_scenario1_all_selected_gablonsky() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1; // Gablonsky

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Insert rects at different depth levels with decreasing f-values.
    // Deeper rects have lower f → all lie on the convex hull lower envelope.
    // Level 0: f=10.0, Level 1: f=5.0, Level 2: f=2.0, Level 3: f=0.5
    let configs = vec![
        (10.0, vec![0, 0]),  // level 0 (all lengths = 0)
        (5.0,  vec![1, 1]),  // level 1
        (2.0,  vec![2, 2]),  // level 2
        (0.5,  vec![3, 3]),  // level 3
    ];

    for (f_val, lens) in &configs {
        insert_rect(&mut c, &mut r, *f_val, FEASIBLE, lens, jones);
    }

    let act_deep = 3;
    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, act_deep, minf,
        1e-4, 0.0,  // eps_rel, eps_abs
        0,           // ifeasible_f = 0 (feasible points exist)
        jones,
        100,         // maxdiv
        "Scenario 1 (Gablonsky): all on hull, all selected",
    );
}

#[test]
fn test_dirchoose_scenario1_all_selected_original() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 0; // Original

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // For Jones Original with n=2: level = k*n + p
    // All same lengths → cube → level = k*n + n - p = k*2 + 2 - 2 = 2k
    let configs = vec![
        (10.0, vec![0, 0]),  // level = 0*2 + 2-2 = 0
        (5.0,  vec![1, 1]),  // level = 1*2 + 2-2 = 2
        (2.0,  vec![2, 2]),  // level = 2*2 + 2-2 = 4
        (0.5,  vec![3, 3]),  // level = 3*2 + 2-2 = 6
    ];

    for (f_val, lens) in &configs {
        insert_rect(&mut c, &mut r, *f_val, FEASIBLE, lens, jones);
    }

    let act_deep = 6; // max level is 6 for Original
    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, act_deep, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 1 (Original): all on hull, all selected",
    );
}

// ====================================================================
// Scenario 2: One rect above convex hull → should be excluded
// ====================================================================
#[test]
fn test_dirchoose_scenario2_one_above_hull_gablonsky() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Level 0: f=2.0, Level 1: f=8.0 (above hull!), Level 2: f=1.0
    // The line from (level0, 2.0) to (level2, 1.0) is below (level1, 8.0)
    // → rect at level 1 should be eliminated.
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 8.0, FEASIBLE, &[1, 1], jones);  // level 1 — above hull
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[2, 2], jones);  // level 2

    let minf = 1.0;

    compare_dirchoose(
        &mut c, &r, 2, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 2 (Gablonsky): one rect above convex hull",
    );
}

#[test]
fn test_dirchoose_scenario2_one_above_hull_original() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 0;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // For jones=0 with n=2: cube levels are 0, 2, 4
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 8.0, FEASIBLE, &[1, 1], jones);  // level 2 — above hull
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[2, 2], jones);  // level 4

    let minf = 1.0;

    compare_dirchoose(
        &mut c, &r, 8, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 2 (Original): one rect above convex hull",
    );
}

// ====================================================================
// Scenario 3: Multiple rects at same level → only anchor (lowest f)
// should be considered by dirchoose
// ====================================================================
#[test]
fn test_dirchoose_scenario3_same_level_gablonsky() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Three rects at level 0 with different f-values.
    // Only the anchor (lowest f) is used by dirchoose.
    insert_rect(&mut c, &mut r, 3.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[0, 0], jones); // lowest → becomes anchor
    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);

    // One rect at level 1 with lower f
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[1, 1], jones);

    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, 1, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 3 (Gablonsky): multiple rects at same level",
    );
}

#[test]
fn test_dirchoose_scenario3_same_level_original() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 0;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Three rects at same cube-level 0 (all lengths [0,0])
    insert_rect(&mut c, &mut r, 3.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);

    // One at deeper level
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[1, 1], jones);

    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, 2, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 3 (Original): multiple rects at same level",
    );
}

// ====================================================================
// Scenario 4: Epsilon test eliminates a hull-point rect
// ====================================================================
#[test]
fn test_dirchoose_scenario4_epsilon_eliminates_gablonsky() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Set up rects where one is on the convex hull but fails the epsilon test.
    // Level 0: f=5.0 (large rect, high f → on hull but may fail epsilon)
    // Level 3: f=0.1 (deep, low f → minf candidate)
    // With eps_rel=0.5 (very aggressive), the epsilon test
    // threshold = min(minf - eps*|minf|, minf - epsabs) = min(0.1 - 0.5*0.1, 0.1 - 0) = 0.05
    // For the level-0 rect: test_val = 5.0 - helplower * levels[0]
    // helplower depends on slope to the level-3 rect.
    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 0.1, FEASIBLE, &[3, 3], jones);

    let minf = 0.1;

    compare_dirchoose(
        &mut c, &r, 3, minf,
        0.5, 0.0,  // Very aggressive eps_rel to trigger epsilon elimination
        0, jones, 100,
        "Scenario 4 (Gablonsky): epsilon test eliminates hull point",
    );
}

#[test]
fn test_dirchoose_scenario4_epsilon_eliminates_with_epsabs() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Test with eps_abs instead of eps_rel
    insert_rect(&mut c, &mut r, 10.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[1, 1], jones);
    insert_rect(&mut c, &mut r, 0.01, FEASIBLE, &[4, 4], jones);

    let minf = 0.01;

    compare_dirchoose(
        &mut c, &r, 4, minf,
        0.0, 0.5,  // Use epsabs=0.5
        0, jones, 100,
        "Scenario 4 (Gablonsky): epsilon test with epsabs",
    );
}

// ====================================================================
// Scenario 5: Empty levels (gaps in depth distribution)
// ====================================================================
#[test]
fn test_dirchoose_scenario5_gaps_gablonsky() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Rects at levels 0, 3, 7 (gaps at 1, 2, 4, 5, 6)
    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[3, 3], jones);  // level 3
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[7, 7], jones);  // level 7

    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, 7, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 5 (Gablonsky): gaps in depth distribution",
    );
}

#[test]
fn test_dirchoose_scenario5_gaps_original() {
    let n = 3;
    let maxfunc = 50;
    let maxdeep = 30;
    let jones = 0;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // With n=3, Jones Original cube levels are 0, 3, 6, 9, ...
    // Insert at levels 0, 6 with gap at 3
    insert_rect(&mut c, &mut r, 8.0, FEASIBLE, &[0, 0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[2, 2, 2], jones);  // level 6

    let minf = 1.0;

    compare_dirchoose(
        &mut c, &r, 9, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 5 (Original n=3): gaps in depth distribution",
    );
}

// ====================================================================
// Scenario 6: All infeasible → picks first non-empty anchor
// ====================================================================
#[test]
fn test_dirchoose_all_infeasible() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Insert infeasible rects at levels 1 and 3
    // Level 0 is empty
    insert_rect(&mut c, &mut r, 100.0, INFEASIBLE, &[1, 1], jones);
    insert_rect(&mut c, &mut r, 50.0, INFEASIBLE, &[3, 3], jones);

    let minf = f64::MAX; // no feasible min

    compare_dirchoose(
        &mut c, &r, 3, minf,
        1e-4, 0.0,
        1, // ifeasible_f >= 1 means all infeasible
        jones, 100,
        "Scenario 6: all infeasible, pick first non-empty anchor",
    );
}

// ====================================================================
// Scenario 7: Mixed feasible/infeasible with infeasible anchor
// ====================================================================
#[test]
fn test_dirchoose_infeasible_anchor() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Feasible rects at levels 0 and 2
    insert_rect(&mut c, &mut r, 3.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[2, 2], jones);

    // Manually place an infeasible rect in anchor[-1] (anchor[0] in Rust)
    let inf_idx = c.alloc_rect();
    c.set_f(inf_idx, 100.0, INFEASIBLE);
    for j in 0..n {
        c.set_length(inf_idx, j, 1);
    }
    // Set anchor[-1] = inf_idx in C storage
    c.anchor[0] = inf_idx as i32; // anchor[-1] after ++anchor adjustment

    let r_inf_idx = r.alloc_rect().expect("Rust free list exhausted");
    assert_eq!(inf_idx, r_inf_idx);
    r.set_f(r_inf_idx, 100.0, INFEASIBLE);
    for j in 0..n {
        r.lengths[r_inf_idx * n + j] = 1;
    }
    r.anchor[0] = r_inf_idx as i32;

    let minf = 1.0;

    compare_dirchoose(
        &mut c, &r, 2, minf,
        1e-4, 0.0,
        0, // feasible points exist
        jones, 100,
        "Scenario 7: mixed feasible/infeasible with infeasible anchor",
    );
}

// ====================================================================
// Scenario 8: Larger configuration - 6 rects with complex hull
// ====================================================================
#[test]
fn test_dirchoose_complex_hull_gablonsky() {
    let n = 3;
    let maxfunc = 100;
    let maxdeep = 30;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Create a more complex scenario with 6 rects at various levels
    // Some on hull, some above, some eliminated by epsilon
    insert_rect(&mut c, &mut r, 20.0, FEASIBLE, &[0, 0, 0], jones);  // level 0, big rect
    insert_rect(&mut c, &mut r, 12.0, FEASIBLE, &[1, 1, 1], jones);  // level 1
    insert_rect(&mut c, &mut r, 15.0, FEASIBLE, &[2, 2, 2], jones);  // level 2 — likely above hull
    insert_rect(&mut c, &mut r, 5.0,  FEASIBLE, &[3, 3, 3], jones);  // level 3
    insert_rect(&mut c, &mut r, 3.0,  FEASIBLE, &[4, 4, 4], jones);  // level 4
    insert_rect(&mut c, &mut r, 0.1,  FEASIBLE, &[6, 6, 6], jones);  // level 6

    let minf = 0.1;

    compare_dirchoose(
        &mut c, &r, 6, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 8 (Gablonsky): complex hull with 6 rects",
    );
}

#[test]
fn test_dirchoose_complex_hull_original() {
    let n = 3;
    let maxfunc = 100;
    let maxdeep = 30;
    let jones = 0;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Jones Original with n=3: cube levels are 0, 3, 6, 9, ...
    insert_rect(&mut c, &mut r, 20.0, FEASIBLE, &[0, 0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 12.0, FEASIBLE, &[1, 1, 1], jones);  // level 3
    insert_rect(&mut c, &mut r, 15.0, FEASIBLE, &[2, 2, 2], jones);  // level 6 — above hull?
    insert_rect(&mut c, &mut r, 5.0,  FEASIBLE, &[3, 3, 3], jones);  // level 9
    insert_rect(&mut c, &mut r, 3.0,  FEASIBLE, &[4, 4, 4], jones);  // level 12
    insert_rect(&mut c, &mut r, 0.1,  FEASIBLE, &[6, 6, 6], jones);  // level 18

    let minf = 0.1;

    compare_dirchoose(
        &mut c, &r, 18, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 8 (Original): complex hull with 6 rects",
    );
}

// ====================================================================
// Scenario 9: Non-cube rectangles (mixed lengths) — Jones Original
// ====================================================================
#[test]
fn test_dirchoose_noncube_original() {
    let n = 3;
    let maxfunc = 50;
    let maxdeep = 30;
    let jones = 0;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Non-cube rects test different level computation in Jones Original
    // lengths [1,0,1]: k=0, help=1, k!=help → level = 0*3 + 1 = 1
    // lengths [2,1,0]: k=0, help=2, k!=help → level = 0*3 + 1 = 1
    // lengths [0,0,0]: cube, level = 0*3 + 0 = 0
    // lengths [2,2,2]: cube, level = 2*3 + 0 = 6
    insert_rect(&mut c, &mut r, 10.0, FEASIBLE, &[0, 0, 0], jones);  // level 0
    insert_rect(&mut c, &mut r, 7.0,  FEASIBLE, &[1, 0, 1], jones);  // level 1
    insert_rect(&mut c, &mut r, 3.0,  FEASIBLE, &[2, 2, 2], jones);  // level 6
    insert_rect(&mut c, &mut r, 0.5,  FEASIBLE, &[3, 3, 3], jones);  // level 9

    let minf = 0.5;

    compare_dirchoose(
        &mut c, &r, 9, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 9 (Original): non-cube rectangles",
    );
}

// ====================================================================
// Test dirdoubleinsert — Jones Original only
// ====================================================================
#[test]
fn test_dirdoubleinsert_basic() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 0; // Original only
    let maxdiv: i32 = 100;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Insert 3 rects at level 0 with very close f-values (within 1e-13)
    // and 1 rect with different f-value
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[0, 0], jones);       // anchor
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[0, 0], jones);       // tied
    insert_rect(&mut c, &mut r, 1.0 + 1e-14, FEASIBLE, &[0, 0], jones); // within tolerance
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[0, 0], jones);       // NOT tied

    // Insert a rect at level 2
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[1, 1], jones);

    let act_deep = 2;
    let minf = 0.5;

    // First run dirchoose on C side to get the s array
    let md = maxdiv as usize;
    let mut s_c = vec![0i32; md * 2];
    let mut maxpos_c: c_int = 0;
    let mut act_deep_c = act_deep;
    let mut minf_c = minf;
    let mut cheat: c_int = 0;
    let mut kmax: c_double = 1e10;
    let mut ifeasible: c_int = 0;

    unsafe {
        direct_dirchoose_(
            c.anchor.as_mut_ptr(),
            s_c.as_mut_ptr(),
            &mut act_deep_c,
            c.f.as_mut_ptr(),
            &mut minf_c,
            1e-4, 0.0,
            c.levels.as_mut_ptr(),
            &mut maxpos_c,
            c.length.as_mut_ptr(),
            &mut c.maxfunc,
            &c.maxdeep,
            &maxdiv,
            &mut c.n,
            std::ptr::null_mut(),
            &mut cheat,
            &mut kmax,
            &mut ifeasible,
            jones,
        );
    }

    // Now call dirdoubleinsert on C side
    let ierror = c.dirdoubleinsert(&mut s_c, &mut maxpos_c, maxdiv);
    assert_eq!(ierror, 0, "C dirdoubleinsert error: {}", ierror);

    // Collect C results after doubleinsert
    let mut c_indices: Vec<i32> = Vec::new();
    for k in 1..=maxpos_c {
        let idx = s_c[k as usize - 1];
        if idx > 0 {
            c_indices.push(idx);
        }
    }

    // Now Rust side: dirchoose + double_insert
    let mut po = PotentiallyOptimal::new(maxdiv as usize);
    po.select(&r, act_deep, minf, 1e-4, 0.0, 0, jones);
    let result = po.double_insert(&r);
    assert!(result.is_ok(), "Rust double_insert error");

    let mut r_indices: Vec<i32> = Vec::new();
    for i in 0..po.count {
        if po.indices[i] > 0 {
            r_indices.push(po.indices[i]);
        }
    }

    println!("--- dirdoubleinsert test ---");
    println!("  C: maxpos={}, indices={:?}", maxpos_c, c_indices);
    println!("  R: count={}, indices={:?}", po.count, r_indices);

    let mut c_sorted = c_indices.clone();
    c_sorted.sort();
    let mut r_sorted = r_indices.clone();
    r_sorted.sort();

    assert_eq!(
        c_sorted, r_sorted,
        "dirdoubleinsert: indices mismatch.\n  C={:?}\n  R={:?}",
        c_indices, r_indices
    );
}

// ====================================================================
// Scenario 10: Single rect — trivially selected
// ====================================================================
#[test]
fn test_dirchoose_single_rect() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    insert_rect(&mut c, &mut r, 3.0, FEASIBLE, &[0, 0], jones);

    compare_dirchoose(
        &mut c, &r, 0, 3.0,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 10: single rect, trivially selected",
    );
}

// ====================================================================
// Scenario 11: Two rects, deeper one has HIGHER f
// (lower convex hull slope ≤ 0 → deeper rect eliminated)
// ====================================================================
#[test]
fn test_dirchoose_deeper_higher_f() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    // Level 0: f=1.0, Level 2: f=5.0
    // The deeper rect has higher f → slope from level-0 to level-2 is negative
    // → level-2 rect should be eliminated (help2 ≤ 0)
    insert_rect(&mut c, &mut r, 1.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[2, 2], jones);

    let minf = 1.0;

    compare_dirchoose(
        &mut c, &r, 2, minf,
        1e-4, 0.0,
        0, jones, 100,
        "Scenario 11: deeper rect has higher f, eliminated",
    );
}

// ====================================================================
// Scenario 12: Varying eps values — boundary cases
// ====================================================================
#[test]
fn test_dirchoose_zero_epsilon() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[1, 1], jones);
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[3, 3], jones);

    let minf = 0.5;

    // With zero epsilon, the epsilon test is most permissive (threshold = minf)
    compare_dirchoose(
        &mut c, &r, 3, minf,
        0.0, 0.0,
        0, jones, 100,
        "Scenario 12: zero epsilon (most permissive)",
    );
}

#[test]
fn test_dirchoose_large_epsilon() {
    let n = 2;
    let maxfunc = 50;
    let maxdeep = 20;
    let jones = 1;

    let (mut c, mut r) = setup_pair(n, maxfunc, maxdeep);
    c.precompute_levels(jones);
    r.precompute_levels(jones);

    insert_rect(&mut c, &mut r, 5.0, FEASIBLE, &[0, 0], jones);
    insert_rect(&mut c, &mut r, 2.0, FEASIBLE, &[1, 1], jones);
    insert_rect(&mut c, &mut r, 0.5, FEASIBLE, &[3, 3], jones);

    let minf = 0.5;

    // With very large epsilon, most rects should be eliminated
    compare_dirchoose(
        &mut c, &r, 3, minf,
        1.0, 0.0,  // eps_rel=1.0 → threshold = min(0.5 - 1.0*0.5, 0.5) = min(0.0, 0.5) = 0.0
        0, jones, 100,
        "Scenario 12: large epsilon (aggressive filtering)",
    );
}
