//! SGJ C re-implementation of DIRECT using ordered trees.
//!
//! Faithfully mirrors NLOPT's `cdirect.c` by Steven G. Johnson,
//! which uses red-black trees for rectangle storage instead of the
//! Gablonsky translation's SoA + linked-list approach.
//!
//! # NLOPT C Correspondence
//!
//! | Rust                                | NLOPT C function               | File        |
//! |-------------------------------------|--------------------------------|-------------|
//! | `CDirect::optimize_unscaled()`      | `cdirect_unscaled()`           | cdirect.c   |
//! | `CDirect::optimize()`               | `cdirect()`                    | cdirect.c   |
//! | `CDirect::divide_rect()`            | `divide_rect()`                | cdirect.c   |
//! | `CDirect::convex_hull()`            | `convex_hull()`                | cdirect.c   |
//! | `CDirect::divide_good_rects()`      | `divide_good_rects()`          | cdirect.c   |
//! | `CDirect::rect_diameter()`          | `rect_diameter()`              | cdirect.c   |
//! | `CDirect::function_eval()`          | `function_eval()`              | cdirect.c   |

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{DirectError, DirectReturnCode, Result};
use crate::types::{Bounds, DirectOptions, DirectResult};

/// Arc-wrapped objective function type for internal use.
type ArcObjFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Arc-wrapped callback function type for internal use.
type ArcCallbackFn = Arc<dyn Fn(&[f64], f64, usize, usize) -> bool + Send + Sync>;

/// Tolerance for considering sides as equal length.
/// Matches `EQUAL_SIDE_TOL` in cdirect.c line 149.
const EQUAL_SIDE_TOL: f64 = 5e-2;

/// One third, matching `THIRD` in cdirect.c line 147.
const THIRD: f64 = 1.0 / 3.0;

// ──────────────────────────────────────────────────────────────────────────────
// HyperRect key for the BTreeMap
// ──────────────────────────────────────────────────────────────────────────────

/// A hyperrectangle stored as a flat array of length `L = 2*n + 3`.
///
/// Layout (matches cdirect.c):
/// - `[0]` = diameter (d)
/// - `[1]` = function value (f)
/// - `[2]` = age (tie-breaker)
/// - `[3 .. 3+n]` = center coordinates (c)
/// - `[3+n .. 3+2n]` = side widths (w)
#[derive(Debug, Clone)]
struct HyperRect {
    data: Vec<f64>,
}

impl HyperRect {
    fn new(n: usize) -> Self {
        Self {
            data: vec![0.0; 2 * n + 3],
        }
    }

    #[inline]
    fn diameter(&self) -> f64 {
        self.data[0]
    }

    #[inline]
    fn set_diameter(&mut self, d: f64) {
        self.data[0] = d;
    }

    #[inline]
    fn f_value(&self) -> f64 {
        self.data[1]
    }

    #[inline]
    fn set_f_value(&mut self, f: f64) {
        self.data[1] = f;
    }

    #[inline]
    fn age(&self) -> f64 {
        self.data[2]
    }

    #[inline]
    fn set_age(&mut self, age: f64) {
        self.data[2] = age;
    }

    #[inline]
    fn center(&self, n: usize) -> &[f64] {
        &self.data[3..3 + n]
    }

    #[inline]
    fn center_mut(&mut self, n: usize) -> &mut [f64] {
        &mut self.data[3..3 + n]
    }

    #[inline]
    fn widths(&self, n: usize) -> &[f64] {
        &self.data[3 + n..3 + 2 * n]
    }

    #[inline]
    fn widths_mut(&mut self, n: usize) -> &mut [f64] {
        &mut self.data[3 + n..3 + 2 * n]
    }
}

/// Lexicographic (d, f, age) ordering key for HyperRect.
///
/// Matches `cdirect_hyperrect_compare()` in cdirect.c lines 463–472.
/// Uses `OrderedFloat` semantics via total_cmp for NaN-safe ordering.
#[derive(Debug, Clone)]
struct RectKey {
    diameter: f64,
    f_value: f64,
    age: f64,
    /// Unique id for tie-breaking (matches pointer comparison in C).
    id: usize,
}

impl PartialEq for RectKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for RectKey {}

impl PartialOrd for RectKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RectKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lexicographic order: (diameter, f_value, age, id)
        // Matches cdirect_hyperrect_compare: compare d, then f, then age
        self.diameter
            .total_cmp(&other.diameter)
            .then_with(|| self.f_value.total_cmp(&other.f_value))
            .then_with(|| self.age.total_cmp(&other.age))
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CDirectParams: internal algorithm parameters
// ──────────────────────────────────────────────────────────────────────────────

/// Internal parameters matching the `params` struct in cdirect.c (lines 55–82).
///
/// The `which_alg` integer is decomposed into three sub-flags using base-3
/// encoding (cdirect.c lines 490–493):
/// - `which_diam = which_alg % 3` — diameter measure (0=Jones Euclidean, 1=Gablonsky max-side)
/// - `which_div  = (which_alg / 3) % 3` — division strategy (0=all longest, 1=Gablonsky, 2=random)
/// - `which_opt  = (which_alg / 9) % 3` — hull selection (0=all hull pts, 1=one per diameter, 2=randomized)
#[allow(dead_code)]
struct CDirectParams {
    n: usize,
    which_diam: i32,
    which_div: i32,
    which_opt: i32,
    magic_eps: f64,
    lb: Vec<f64>,
    ub: Vec<f64>,

    minf: f64,
    xmin: Vec<f64>,
    nfev: usize,
    age: usize,
    next_id: usize,

    /// Red-black tree equivalent: BTreeMap sorted by (d, f, age).
    rtree: BTreeMap<RectKey, HyperRect>,
}

impl CDirectParams {
    fn new(n: usize, lb: Vec<f64>, ub: Vec<f64>, magic_eps: f64, which_alg: i32) -> Self {
        Self {
            n,
            which_diam: which_alg % 3,
            which_div: (which_alg / 3) % 3,
            which_opt: (which_alg / 9) % 3,
            magic_eps,
            lb,
            ub,
            minf: f64::INFINITY,
            xmin: vec![0.0; n],
            nfev: 0,
            age: 0,
            next_id: 0,
            rtree: BTreeMap::new(),
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CDirect optimizer
// ──────────────────────────────────────────────────────────────────────────────

/// CDirect optimizer implementing SGJ's red-black tree DIRECT.
///
/// This struct provides the public API for running any of the cdirect-based
/// NLOPT algorithm variants (DIRECT, DIRECT-L, DIRECT-L-RAND, and their
/// unscaled counterparts).
///
/// # NLOPT C Correspondence
///
/// The public entry point `minimize()` dispatches to either:
/// - `optimize_scaled()` → `cdirect()` in cdirect.c (line 569)
/// - `optimize_unscaled()` → `cdirect_unscaled()` in cdirect.c (line 476)
///
/// # Deviation from NLOPT C
///
/// - Uses `BTreeMap<RectKey, HyperRect>` instead of NLOPT's `rb_tree` (red-black tree).
///   The ordering semantics are identical via `RectKey::cmp()`.
/// - Does not handle infeasible points (NaN/Inf) — matching cdirect.c behavior,
///   which unlike the Gablonsky translation has no `dirreplaceinf_()` equivalent.
pub struct CDirect {
    func: ArcObjFn,
    bounds: Bounds,
    options: DirectOptions,
    callback: Option<ArcCallbackFn>,
}

impl CDirect {
    /// Create a new CDirect optimizer.
    pub fn new(
        func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
        bounds: Bounds,
        options: DirectOptions,
    ) -> Self {
        Self {
            func: Arc::new(func),
            bounds,
            options,
            callback: None,
        }
    }

    /// Set a callback for progress monitoring and early stopping.
    pub fn with_callback(
        mut self,
        callback: impl Fn(&[f64], f64, usize, usize) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.callback = Some(Arc::new(callback));
        self
    }

    /// Run the optimizer, dispatching to scaled or unscaled variant.
    ///
    /// Matches `cdirect()` (scaled, cdirect.c line 569) or `cdirect_unscaled()`
    /// (unscaled, cdirect.c line 476) depending on the algorithm variant.
    pub fn minimize(&self) -> Result<DirectResult> {
        let n = self.bounds.len();
        if n == 0 {
            return Err(DirectError::InvalidArgs("dimension must be > 0".into()));
        }

        // Validate bounds
        for (i, &(lo, hi)) in self.bounds.iter().enumerate() {
            if lo >= hi {
                return Err(DirectError::InvalidBounds { dim: i });
            }
        }

        let which_alg = self.options.algorithm.which_alg().ok_or_else(|| {
            DirectError::InvalidArgs(
                "CDirect requires a cdirect algorithm variant, not Gablonsky".into(),
            )
        })?;

        if self.options.algorithm.is_unscaled() {
            let lb: Vec<f64> = self.bounds.iter().map(|&(lo, _)| lo).collect();
            let ub: Vec<f64> = self.bounds.iter().map(|&(_, hi)| hi).collect();
            self.optimize_unscaled(n, &lb, &ub, which_alg)
        } else {
            self.optimize_scaled(n, which_alg)
        }
    }

    /// Scaled variant: maps bounds to [0,1]^n then calls optimize_unscaled.
    ///
    /// Matches `cdirect()` in cdirect.c (lines 569–603).
    ///
    /// Wraps the user's objective function with an unscaling transform
    /// (`cdirect_uf()` in cdirect.c line 555): `x_actual = lb + xu * (ub - lb)`.
    /// After optimization, the result point is unscaled back to original coordinates.
    fn optimize_scaled(&self, n: usize, which_alg: i32) -> Result<DirectResult> {
        let lb: Vec<f64> = self.bounds.iter().map(|&(lo, _)| lo).collect();
        let ub: Vec<f64> = self.bounds.iter().map(|&(_, hi)| hi).collect();

        // Wrap function to unscale: x_actual = lb + xu * (ub - lb)
        //
        // Note: We allocate a fresh Vec per call rather than using a thread-local
        // RefCell buffer. A thread-local RefCell would panic if the user's objective
        // function itself uses rayon internally — rayon work-stealing can cause the
        // same thread to re-enter this closure while the RefCell borrow is still held.
        // The allocation cost of a small Vec<f64> is negligible compared to any
        // real objective function.
        let lb_clone = lb.clone();
        let ub_clone = ub.clone();
        let func = self.func.clone();
        let n_dims = n;
        let scaled_func = move |xu: &[f64]| -> f64 {
            let mut x_actual = vec![0.0; n_dims];
            for i in 0..n_dims {
                x_actual[i] = lb_clone[i] + xu[i] * (ub_clone[i] - lb_clone[i]);
            }
            func(&x_actual)
        };

        let unit_lb = vec![0.0; n];
        let unit_ub = vec![1.0; n];

        let scaled_func_arc: ArcObjFn = Arc::new(scaled_func);
        let result = self.optimize_unscaled_inner(&scaled_func_arc, n, &unit_lb, &unit_ub, which_alg)?;

        // Unscale the result point: x = lb + x_scaled * (ub - lb)
        let mut x_unscaled = result.x;
        for i in 0..n {
            x_unscaled[i] = lb[i] + x_unscaled[i] * (ub[i] - lb[i]);
        }

        Ok(DirectResult::new(
            x_unscaled,
            result.fun,
            result.nfev,
            result.nit,
            result.return_code,
        ))
    }

    /// Unscaled variant.
    ///
    /// Matches `cdirect_unscaled()` in cdirect.c lines 476–549.
    fn optimize_unscaled(
        &self,
        n: usize,
        lb: &[f64],
        ub: &[f64],
        which_alg: i32,
    ) -> Result<DirectResult> {
        self.optimize_unscaled_inner(&self.func, n, lb, ub, which_alg)
    }

    /// Core unscaled optimization loop.
    ///
    /// Matches `cdirect_unscaled()` in cdirect.c lines 476–549.
    fn optimize_unscaled_inner(
        &self,
        func: &ArcObjFn,
        n: usize,
        lb: &[f64],
        ub: &[f64],
        which_alg: i32,
    ) -> Result<DirectResult> {
        let mut p = CDirectParams::new(
            n,
            lb.to_vec(),
            ub.to_vec(),
            self.options.magic_eps,
            which_alg,
        );

        let max_feval = self.options.max_feval;
        let max_iter = self.options.max_iter;
        let max_time = self.options.max_time;
        let start_time = if max_time > 0.0 {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Create initial rectangle at center of domain
        let mut rnew = HyperRect::new(n);
        for i in 0..n {
            rnew.center_mut(n)[i] = 0.5 * (lb[i] + ub[i]);
            rnew.widths_mut(n)[i] = ub[i] - lb[i];
        }
        rnew.set_diameter(Self::rect_diameter(n, rnew.widths(n), p.which_diam));
        let fval = Self::function_eval(func, rnew.center(n), &mut p);
        rnew.set_f_value(fval);
        rnew.set_age(p.age as f64);
        p.age += 1;

        // Check stopping conditions after first eval
        let ret = self.check_stop_after_eval(&p, max_feval, max_time, start_time.as_ref());
        if let Some(code) = ret {
            return Ok(DirectResult::new(p.xmin.clone(), p.minf, p.nfev, 0, code));
        }

        let init_key = RectKey {
            diameter: rnew.diameter(),
            f_value: rnew.f_value(),
            age: rnew.age(),
            id: p.alloc_id(),
        };
        p.rtree.insert(init_key.clone(), rnew);

        // Divide the initial rectangle
        let ret = self.divide_rect_by_key(
            func,
            &mut p,
            &init_key,
            max_feval,
            max_time,
            start_time.as_ref(),
        );
        if let Some(code) = ret {
            if code.is_error() {
                return Ok(DirectResult::new(p.xmin.clone(), p.minf, p.nfev, 0, code));
            }
        }

        // Main loop
        let mut nit: usize = 0;
        loop {
            let minf0 = p.minf;

            let ret = if self.options.parallel {
                self.divide_good_rects_parallel(
                    func,
                    &mut p,
                    max_feval,
                    max_time,
                    start_time.as_ref(),
                )
            } else {
                self.divide_good_rects(
                    func,
                    &mut p,
                    max_feval,
                    max_time,
                    start_time.as_ref(),
                )
            };

            nit += 1;

            match ret {
                Ok(xtol_reached) => {
                    // Check ftol: if minf improved and we have ftol criteria
                    if p.minf < minf0 {
                        let fglobal = self.options.fglobal;
                        let fglobal_reltol = self.options.fglobal_reltol;
                        if fglobal > f64::NEG_INFINITY {
                            let threshold = if fglobal == 0.0 {
                                fglobal_reltol
                            } else {
                                fglobal + fglobal_reltol * fglobal.abs()
                            };
                            if p.minf <= threshold {
                                return Ok(DirectResult::new(
                                    p.xmin.clone(),
                                    p.minf,
                                    p.nfev,
                                    nit,
                                    DirectReturnCode::GlobalFound,
                                ));
                            }
                        }
                    }

                    // Check xtol
                    if xtol_reached {
                        return Ok(DirectResult::new(
                            p.xmin.clone(),
                            p.minf,
                            p.nfev,
                            nit,
                            DirectReturnCode::VolTol,
                        ));
                    }
                }
                Err(code) => {
                    return Ok(DirectResult::new(p.xmin.clone(), p.minf, p.nfev, nit, code));
                }
            }

            // Check callback
            if let Some(ref cb) = self.callback {
                if cb(&p.xmin, p.minf, p.nfev, nit) {
                    return Ok(DirectResult::new(
                        p.xmin.clone(),
                        p.minf,
                        p.nfev,
                        nit,
                        DirectReturnCode::ForcedStop,
                    ));
                }
            }

            // Check max iterations
            if max_iter > 0 && nit >= max_iter {
                return Ok(DirectResult::new(
                    p.xmin.clone(),
                    p.minf,
                    p.nfev,
                    nit,
                    DirectReturnCode::MaxIterExceeded,
                ));
            }
        }
    }

    /// Evaluate the objective function and update min tracking.
    ///
    /// Matches `function_eval()` in cdirect.c (lines 136–144).
    /// Updates `p.minf` and `p.xmin` if a new minimum is found.
    /// Increments `p.nfev` unconditionally.
    fn function_eval(
        func: &ArcObjFn,
        x: &[f64],
        p: &mut CDirectParams,
    ) -> f64 {
        let f = func(x);
        if f < p.minf {
            p.minf = f;
            p.xmin.copy_from_slice(x);
        }
        p.nfev += 1;
        f
    }

    /// Compute the rectangle diameter measure.
    ///
    /// Matches `rect_diameter()` in cdirect.c (lines 94–112).
    ///
    /// - `which_diam == 0` (Jones): Euclidean half-diagonal `sqrt(sum(w_i^2)) / 2`
    /// - `which_diam == 1` (Gablonsky): half-width of longest side `max(w_i) / 2`
    ///
    /// Both paths cast to `f32` then back to `f64` — this float-rounding trick
    /// (cdirect.c line 103/109) groups rectangles by diameter level, which is
    /// essential for convex_hull() performance.
    fn rect_diameter(n: usize, w: &[f64], which_diam: i32) -> f64 {
        if which_diam == 0 {
            // Jones measure: Euclidean distance from center to vertex
            let sum: f64 = w[..n].iter().map(|&wi| wi * wi).sum();
            (sum.sqrt() * 0.5) as f32 as f64
        } else {
            // Gablonsky measure: half-width of longest side
            let maxw = w[..n].iter().cloned().fold(0.0_f64, f64::max);
            (maxw * 0.5) as f32 as f64
        }
    }

    /// Divide a rectangle identified by its key.
    ///
    /// Matches `divide_rect()` in cdirect.c (lines 152–243).
    ///
    /// Two division paths:
    /// - **Path A** (lines 169–208): `which_div == 1` (Gablonsky) or all sides equal.
    ///   Trisects all longest sides in order of min(f+, f-). Evaluates all 2×nlongest
    ///   children first via `sort_fv()`, then creates and inserts them sorted.
    /// - **Path B** (lines 210–241): `which_div == 0` (Jones) with non-cube rect.
    ///   Trisects only one longest side (or random among longest if `which_div == 2`).
    ///   Evaluates children during creation.
    fn divide_rect_by_key(
        &self,
        func: &ArcObjFn,
        p: &mut CDirectParams,
        target_key: &RectKey,
        max_feval: usize,
        max_time: f64,
        start_time: Option<&std::time::Instant>,
    ) -> Option<DirectReturnCode> {
        let n = p.n;
        let which_diam = p.which_diam;
        let which_div = p.which_div;

        // Remove the rectangle from the tree
        let mut rdiv = match p.rtree.remove(target_key) {
            Some(r) => r,
            None => return Some(DirectReturnCode::InvalidArgs),
        };

        let widths: Vec<f64> = rdiv.widths(n).to_vec();
        let mut wmax = 0.0_f64;
        let mut imax = 0;
        for (i, &w) in widths.iter().enumerate() {
            if w > wmax {
                wmax = w;
                imax = i;
            }
        }

        let mut nlongest = 0;
        for i in 0..n {
            if wmax - widths[i] <= wmax * EQUAL_SIDE_TOL {
                nlongest += 1;
            }
        }

        if which_div == 1 || (which_div == 0 && nlongest == n as i32) {
            // Path A: Trisect all longest sides in order of min(f+,f-)
            // Matches cdirect.c lines 169–208
            let mut fv = vec![0.0; 2 * n];
            let mut isort: Vec<usize> = (0..n).collect();

            // Evaluate function along each longest dimension
            for i in 0..n {
                if wmax - widths[i] <= wmax * EQUAL_SIDE_TOL {
                    let csave = rdiv.center(n)[i];

                    // Evaluate at center - w*THIRD
                    rdiv.center_mut(n)[i] = csave - widths[i] * THIRD;
                    let fval = Self::function_eval(func, rdiv.center(n), p);
                    fv[2 * i] = fval;

                    // Check stopping after eval
                    if let Some(code) = self.check_stop_after_eval(p, max_feval, max_time, start_time) {
                        // Re-insert rectangle before returning
                        rdiv.center_mut(n)[i] = csave;
                        let key = RectKey {
                            diameter: rdiv.diameter(),
                            f_value: rdiv.f_value(),
                            age: rdiv.age(),
                            id: p.alloc_id(),
                        };
                        p.rtree.insert(key, rdiv);
                        return Some(code);
                    }

                    // Evaluate at center + w*THIRD
                    rdiv.center_mut(n)[i] = csave + widths[i] * THIRD;
                    let fval = Self::function_eval(func, rdiv.center(n), p);
                    fv[2 * i + 1] = fval;

                    rdiv.center_mut(n)[i] = csave;

                    if let Some(code) = self.check_stop_after_eval(p, max_feval, max_time, start_time) {
                        let key = RectKey {
                            diameter: rdiv.diameter(),
                            f_value: rdiv.f_value(),
                            age: rdiv.age(),
                            id: p.alloc_id(),
                        };
                        p.rtree.insert(key, rdiv);
                        return Some(code);
                    }
                } else {
                    fv[2 * i] = f64::INFINITY;
                    fv[2 * i + 1] = f64::INFINITY;
                }
            }

            // Sort dimensions by min(f+, f-)
            // Matches sort_fv() in cdirect.c lines 116–134
            isort.sort_by(|&a, &b| {
                let fa = fv[2 * a].min(fv[2 * a + 1]);
                let fb = fv[2 * b].min(fv[2 * b + 1]);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Re-insert parent with updated diameter and age, then create children
            // Process in sorted order: for each longest dim, trisect
            // The parent keeps its center but gets narrower widths
            for idx in 0..nlongest as usize {
                let si = isort[idx];
                rdiv.widths_mut(n)[si] *= THIRD;
                rdiv.set_diameter(Self::rect_diameter(n, rdiv.widths(n), which_diam));
                rdiv.set_age(p.age as f64);
                p.age += 1;

                // Create two children
                for k in 0..=1 {
                    let mut rnew = rdiv.clone();
                    // Offset: (2*k - 1) = -1 for k=0, +1 for k=1
                    rnew.center_mut(n)[si] += rdiv.widths(n)[si] * (2.0 * k as f64 - 1.0);
                    rnew.set_f_value(fv[2 * si + k]);
                    rnew.set_age(p.age as f64);
                    p.age += 1;
                    let key = RectKey {
                        diameter: rnew.diameter(),
                        f_value: rnew.f_value(),
                        age: rnew.age(),
                        id: p.alloc_id(),
                    };
                    p.rtree.insert(key, rnew);
                }
            }

            // Re-insert the modified parent
            let key = RectKey {
                diameter: rdiv.diameter(),
                f_value: rdiv.f_value(),
                age: rdiv.age(),
                id: p.alloc_id(),
            };
            p.rtree.insert(key, rdiv);
        } else {
            // Path B: Trisect only one longest side
            // Matches cdirect.c lines 210–241
            let mut i_div = imax;
            if nlongest > 1 && which_div == 2 {
                // Random choice among longest sides
                let rand_idx = p.age % nlongest as usize; // deterministic pseudo-random
                let mut count = 0;
                for k in 0..n {
                    if wmax - widths[k] <= wmax * EQUAL_SIDE_TOL {
                        if count == rand_idx {
                            i_div = k;
                            break;
                        }
                        count += 1;
                    }
                }
            }

            rdiv.widths_mut(n)[i_div] *= THIRD;
            rdiv.set_diameter(Self::rect_diameter(n, rdiv.widths(n), which_diam));
            rdiv.set_age(p.age as f64);
            p.age += 1;

            // Create two children with function evaluations
            for k in 0..=1 {
                let mut rnew = rdiv.clone();
                rnew.center_mut(n)[i_div] += rdiv.widths(n)[i_div] * (2.0 * k as f64 - 1.0);
                let fval = Self::function_eval(func, rnew.center(n), p);
                rnew.set_f_value(fval);
                rnew.set_age(p.age as f64);
                p.age += 1;

                if let Some(code) = self.check_stop_after_eval(p, max_feval, max_time, start_time) {
                    // Still insert the child and parent so state is consistent
                    let key = RectKey {
                        diameter: rnew.diameter(),
                        f_value: rnew.f_value(),
                        age: rnew.age(),
                        id: p.alloc_id(),
                    };
                    p.rtree.insert(key, rnew);
                    let pkey = RectKey {
                        diameter: rdiv.diameter(),
                        f_value: rdiv.f_value(),
                        age: rdiv.age(),
                        id: p.alloc_id(),
                    };
                    p.rtree.insert(pkey, rdiv);
                    return Some(code);
                }

                let key = RectKey {
                    diameter: rnew.diameter(),
                    f_value: rnew.f_value(),
                    age: rnew.age(),
                    id: p.alloc_id(),
                };
                p.rtree.insert(key, rnew);
            }

            // Re-insert modified parent
            let key = RectKey {
                diameter: rdiv.diameter(),
                f_value: rdiv.f_value(),
                age: rdiv.age(),
                id: p.alloc_id(),
            };
            p.rtree.insert(key, rdiv);
        }

        None
    }

    /// Find the lower convex hull of rectangles sorted by (diameter, f_value).
    ///
    /// Matches `convex_hull()` in cdirect.c (lines 261–378).
    ///
    /// Uses a monotone chain algorithm with two performance hacks from NLOPT:
    /// 1. Points above the line from (xmin, yminmin) to (xmax, ymaxmin) are
    ///    skipped immediately (cdirect.c line 302).
    /// 2. Vertical lines (duplicate diameters) are handled by keeping only the
    ///    entry with the lowest f-value at each diameter level (cdirect.c line 319).
    ///
    /// When `allow_dups` is true (DIRECT Original, `which_opt != 1`), all entries
    /// at xmin with f == yminmin are included on the hull. When false (DIRECT-L),
    /// only one entry per diameter level is kept.
    ///
    /// Returns keys of hull points in order from smallest to largest diameter.
    fn convex_hull(rtree: &BTreeMap<RectKey, HyperRect>, allow_dups: bool) -> Vec<RectKey> {
        if rtree.is_empty() {
            return vec![];
        }

        // Build (diameter, f_value, key_ref) from the sorted BTreeMap iteration.
        // Avoid cloning keys until we know they're on the hull.
        let entries: Vec<(&RectKey, f64, f64)> = rtree
            .iter()
            .map(|(k, _)| (k, k.diameter, k.f_value))
            .collect();

        if entries.is_empty() {
            return vec![];
        }

        let xmin = entries.first().unwrap().1;
        let yminmin = entries.first().unwrap().2;
        let xmax = entries.last().unwrap().1;

        // Use indices into entries[] for the hull, only clone at the end
        let mut hull_idx: Vec<usize> = Vec::new();

        if allow_dups {
            for (ei, &(_, d, f)) in entries.iter().enumerate() {
                if d == xmin && f == yminmin {
                    hull_idx.push(ei);
                } else {
                    break;
                }
            }
        } else {
            hull_idx.push(0);
        }

        if xmin == xmax {
            return hull_idx.iter().map(|&i| entries[i].0.clone()).collect();
        }

        // Find ymaxmin: minimum f at xmax
        let ymaxmin = entries
            .iter()
            .rev()
            .take_while(|&&(_, d, _)| d == xmax)
            .map(|&(_, _, f)| f)
            .fold(f64::INFINITY, f64::min);

        let minslope = (ymaxmin - yminmin) / (xmax - xmin);

        let start_idx = entries
            .iter()
            .position(|&(_, d, _)| d != xmin)
            .unwrap_or(entries.len());

        let nmax_start = entries
            .iter()
            .position(|&(_, d, _)| d == xmax)
            .unwrap_or(entries.len());

        let mut i = start_idx;
        while i < nmax_start {
            let (_, x, y) = entries[i];

            if y > yminmin + (x - xmin) * minslope {
                i += 1;
                continue;
            }

            // Performance hack: skip vertical lines
            if !hull_idx.is_empty() {
                let last_d = entries[*hull_idx.last().unwrap()].1;
                if x == last_d {
                    let last_f = entries[*hull_idx.last().unwrap()].2;
                    if y > last_f {
                        let cur_d = x;
                        while i < nmax_start && entries[i].1 == cur_d {
                            i += 1;
                        }
                        continue;
                    } else if allow_dups {
                        hull_idx.push(i);
                        i += 1;
                        continue;
                    }
                }
            }

            // Remove points until we make a "left turn" to entry i
            while hull_idx.len() > 1 {
                let t1_ei = *hull_idx.last().unwrap();
                let t1_d = entries[t1_ei].1;
                let t1_f = entries[t1_ei].2;

                let mut it2 = hull_idx.len() as i64 - 2;
                loop {
                    if it2 < 0 {
                        break;
                    }
                    let t2_ei = hull_idx[it2 as usize];
                    let t2_d_cand = entries[t2_ei].1;
                    let t2_f_cand = entries[t2_ei].2;
                    if t2_d_cand != t1_d || t2_f_cand != t1_f {
                        break;
                    }
                    it2 -= 1;
                }
                if it2 < 0 {
                    break;
                }
                let t2_ei = hull_idx[it2 as usize];
                let t2_d = entries[t2_ei].1;
                let t2_f = entries[t2_ei].2;

                let cross = (t1_d - t2_d) * (y - t2_f) - (t1_f - t2_f) * (x - t2_d);
                if cross >= 0.0 {
                    break;
                }
                hull_idx.pop();
            }
            hull_idx.push(i);
            i += 1;
        }

        // Add points at (xmax, ymaxmin)
        if allow_dups {
            for (j, &(_, d, f)) in entries[nmax_start..].iter().enumerate() {
                if d == xmax && f == ymaxmin {
                    hull_idx.push(nmax_start + j);
                } else if d != xmax {
                    break;
                }
            }
        } else if let Some(j) = entries[nmax_start..].iter().position(|&(_, d, f)| d == xmax && f == ymaxmin) {
            hull_idx.push(nmax_start + j);
        }

        // Only clone the keys that are actually on the hull
        hull_idx.iter().map(|&i| entries[i].0.clone()).collect()
    }

    /// Divide potentially optimal rectangles.
    ///
    /// Matches `divide_good_rects()` in cdirect.c (lines 392–458).
    ///
    /// For each point on the convex hull, computes the maximum slope `k` to
    /// adjacent hull points (cdirect.c lines 414–424), then applies the
    /// epsilon test: `f - k*d <= minf - eps*|minf|` (cdirect.c line 427).
    /// Rectangles passing this test are divided via `divide_rect_by_key()`.
    ///
    /// If no rectangles qualify (even with eps=0), falls back to dividing the
    /// largest rectangle with the smallest f-value (cdirect.c lines 442–454).
    ///
    /// For DIRECT-L (`which_opt == 1`), skips to the next distinct diameter
    /// after processing each hull point (cdirect.c line 436).
    ///
    /// Returns Ok(true) if xtol reached, Ok(false) if normal, Err(code) on stop/error.
    fn divide_good_rects(
        &self,
        func: &ArcObjFn,
        p: &mut CDirectParams,
        max_feval: usize,
        max_time: f64,
        start_time: Option<&std::time::Instant>,
    ) -> std::result::Result<bool, DirectReturnCode> {
        let magic_eps_orig = p.magic_eps;
        let mut magic_eps = magic_eps_orig;

        loop {
            let allow_dups = p.which_opt != 1;
            let hull = Self::convex_hull(&p.rtree, allow_dups);
            let nhull = hull.len();
            if nhull == 0 {
                return Err(DirectReturnCode::InvalidArgs);
            }

            let mut divided_some = false;
            let mut xtol_reached = true;

            let mut i = 0;
            while i < nhull {
                // Find unequal points before (im) and after (ip)
                let mut im: i64 = i as i64 - 1;
                while im >= 0 && hull[im as usize].diameter == hull[i].diameter {
                    im -= 1;
                }
                let mut ip = i + 1;
                while ip < nhull && hull[ip].diameter == hull[i].diameter {
                    ip += 1;
                }

                let mut k1 = f64::NEG_INFINITY;
                let mut k2 = f64::NEG_INFINITY;

                if im >= 0 {
                    k1 = (hull[i].f_value - hull[im as usize].f_value)
                        / (hull[i].diameter - hull[im as usize].diameter);
                }
                if ip < nhull {
                    k2 = (hull[i].f_value - hull[ip].f_value)
                        / (hull[i].diameter - hull[ip].diameter);
                }
                let k = k1.max(k2);

                // Potentially optimal test
                if hull[i].f_value - k * hull[i].diameter
                    <= p.minf - magic_eps * p.minf.abs()
                    || ip == nhull
                {
                    // Divide this rectangle
                    let target_key = hull[i].clone();
                    let ret = self.divide_rect_by_key(
                        func,
                        p,
                        &target_key,
                        max_feval,
                        max_time,
                        start_time,
                    );
                    divided_some = true;

                    if let Some(code) = ret {
                        if code.is_error() || code != DirectReturnCode::MaxFevalExceeded {
                            return Err(code);
                        }
                        return Err(code);
                    }

                    // Check xtol: are all widths small?
                    // We check the divided rect's widths via the hull entry
                    // In NLOPT this uses small() on the widths after division
                    // Since we've already divided and the rect is modified, we
                    // track xtol_reached as a heuristic
                    xtol_reached = false; // Simplified: we don't have xtol in our API
                }

                // For DIRECT-L: skip to next unequal point
                if p.which_opt == 1 {
                    i = ip;
                } else if p.which_opt == 2 {
                    // Randomized: possibly do another equal point
                    let skip = p.age % (ip - i).max(1);
                    i += skip + 1;
                    if i > ip {
                        i = ip;
                    }
                } else {
                    i += 1;
                }
            }

            if !divided_some {
                if magic_eps != 0.0 {
                    magic_eps = 0.0;
                    continue; // Retry with eps=0
                } else {
                    // Fallback: divide largest rectangle with smallest f
                    // Matches cdirect.c lines 442–454
                    if let Some((max_key, _)) = p.rtree.iter().next_back() {
                        let wmax_d = max_key.diameter;
                        // Find the rect with largest diameter but smallest f
                        let target_key = p
                            .rtree
                            .range(..)
                            .rev()
                            .take_while(|(k, _)| k.diameter == wmax_d)
                            .last()
                            .map(|(k, _)| k.clone());

                        if let Some(key) = target_key {
                            let ret = self.divide_rect_by_key(
                                func,
                                p,
                                &key,
                                max_feval,
                                max_time,
                                start_time,
                            );
                            if let Some(code) = ret {
                                return Err(code);
                            }
                        }
                    }
                }
                return Ok(xtol_reached);
            }

            return Ok(xtol_reached);
        }
    }

    /// Parallel version of `divide_good_rects()`.
    ///
    /// Uses a collect→parallel-eval→apply pattern:
    /// 1. Compute convex hull (read-only on tree)
    /// 2. Identify qualifying rectangles via epsilon test
    /// 3. For each qualifying rect, compute candidate evaluation points
    /// 4. Batch-evaluate all points in parallel via rayon
    /// 5. Apply results sequentially: dimension sorts, child creation, tree insertion
    ///
    /// The serial `divide_good_rects()` interleaves evaluation with tree mutation
    /// and checks stopping conditions after each eval. The parallel version instead
    /// evaluates all candidates for the current iteration in one batch, then applies
    /// results. This means:
    /// - `max_feval` may be slightly overshot (same behavior as Gablonsky parallel)
    /// - Stopping checks happen after the batch, not mid-eval
    /// - The dimension sort order within each rectangle is identical to serial
    ///   (it depends only on f-values within that rectangle)
    /// - `age` and `next_id` assignment order may differ from serial (affects only
    ///   tiebreaking in the BTreeMap, not correctness)
    fn divide_good_rects_parallel(
        &self,
        func: &ArcObjFn,
        p: &mut CDirectParams,
        max_feval: usize,
        max_time: f64,
        start_time: Option<&std::time::Instant>,
    ) -> std::result::Result<bool, DirectReturnCode> {
        let magic_eps_orig = p.magic_eps;
        let mut magic_eps = magic_eps_orig;
        let n = p.n;
        let which_diam = p.which_diam;
        let which_div = p.which_div;

        loop {
            let allow_dups = p.which_opt != 1;
            let hull = Self::convex_hull(&p.rtree, allow_dups);
            let nhull = hull.len();
            if nhull == 0 {
                return Err(DirectReturnCode::InvalidArgs);
            }

            // ── Phase 1: Identify qualifying rectangles ──
            // Walk the hull exactly as the serial path does, collecting keys of
            // rectangles that pass the potentially-optimal test.
            let mut qualifying_keys: Vec<RectKey> = Vec::new();

            let mut i = 0;
            while i < nhull {
                let mut im: i64 = i as i64 - 1;
                while im >= 0 && hull[im as usize].diameter == hull[i].diameter {
                    im -= 1;
                }
                let mut ip = i + 1;
                while ip < nhull && hull[ip].diameter == hull[i].diameter {
                    ip += 1;
                }

                let mut k1 = f64::NEG_INFINITY;
                let mut k2 = f64::NEG_INFINITY;

                if im >= 0 {
                    k1 = (hull[i].f_value - hull[im as usize].f_value)
                        / (hull[i].diameter - hull[im as usize].diameter);
                }
                if ip < nhull {
                    k2 = (hull[i].f_value - hull[ip].f_value)
                        / (hull[i].diameter - hull[ip].diameter);
                }
                let k = k1.max(k2);

                if hull[i].f_value - k * hull[i].diameter
                    <= p.minf - magic_eps * p.minf.abs()
                    || ip == nhull
                {
                    qualifying_keys.push(hull[i].clone());
                }

                if p.which_opt == 1 {
                    i = ip;
                } else if p.which_opt == 2 {
                    let skip = p.age % (ip - i).max(1);
                    i += skip + 1;
                    if i > ip {
                        i = ip;
                    }
                } else {
                    i += 1;
                }
            }

            if qualifying_keys.is_empty() {
                if magic_eps != 0.0 {
                    magic_eps = 0.0;
                    continue;
                } else {
                    // Fallback: divide largest rectangle with smallest f
                    if let Some((max_key, _)) = p.rtree.iter().next_back() {
                        let wmax_d = max_key.diameter;
                        let target_key = p
                            .rtree
                            .range(..)
                            .rev()
                            .take_while(|(k, _)| k.diameter == wmax_d)
                            .last()
                            .map(|(k, _)| k.clone());

                        if let Some(key) = target_key {
                            let ret = self.divide_rect_by_key(
                                func, p, &key, max_feval, max_time, start_time,
                            );
                            if let Some(code) = ret {
                                return Err(code);
                            }
                        }
                    }
                    return Ok(false);
                }
            }

            // If the number of candidate evaluations is below the parallel
            // threshold, fall back to serial subdivision for this iteration.
            // Estimate: each rect produces at least 2 evals.
            let est_evals: usize = qualifying_keys.len() * 2;
            if est_evals < self.options.min_parallel_evals {
                // Use serial path for this batch
                for key in &qualifying_keys {
                    let ret = self.divide_rect_by_key(
                        func, p, key, max_feval, max_time, start_time,
                    );
                    if let Some(code) = ret {
                        return Err(code);
                    }
                }
                return Ok(false);
            }

            // ── Phase 2: Remove rects from tree, compute candidate points ──
            //
            // For each qualifying rectangle we compute the coordinates of all
            // candidate points that need evaluation, WITHOUT calling the
            // objective function. We also record which division path (A or B)
            // each rectangle uses, and which dimensions are involved.

            /// Describes the division plan for one rectangle.
            struct RectDivisionPlan {
                /// The removed HyperRect (mutated: widths narrowed, etc.)
                rect: HyperRect,
                /// Which path: true = Path A (all longest), false = Path B (one side)
                is_path_a: bool,
                /// For Path A: the sorted dimension indices (all longest dims)
                /// For Path B: single-element vec with the chosen dimension
                dims: Vec<usize>,
                /// Global flat index where this rect's candidate f-values start
                fval_offset: usize,
            }

            let mut plans: Vec<RectDivisionPlan> = Vec::with_capacity(qualifying_keys.len());
            let mut all_points: Vec<Vec<f64>> = Vec::new();

            for key in &qualifying_keys {
                let rdiv = match p.rtree.remove(key) {
                    Some(r) => r,
                    None => continue, // Already removed (shouldn't happen)
                };

                let widths: Vec<f64> = rdiv.widths(n).to_vec();
                let mut wmax = 0.0_f64;
                let mut imax = 0;
                for (i_dim, &w) in widths.iter().enumerate() {
                    if w > wmax {
                        wmax = w;
                        imax = i_dim;
                    }
                }

                let mut nlongest = 0i32;
                for i_dim in 0..n {
                    if wmax - widths[i_dim] <= wmax * EQUAL_SIDE_TOL {
                        nlongest += 1;
                    }
                }

                let is_path_a = which_div == 1 || (which_div == 0 && nlongest == n as i32);
                let fval_offset = all_points.len();

                if is_path_a {
                    // Path A: evaluate 2 points per longest dimension
                    let mut candidate_points = Vec::with_capacity(2 * nlongest as usize);
                    let dims: Vec<usize> = (0..n)
                        .filter(|&i_dim| wmax - widths[i_dim] <= wmax * EQUAL_SIDE_TOL)
                        .collect();

                    for &i_dim in &dims {
                        let csave = rdiv.center(n)[i_dim];

                        // Point at center - w*THIRD
                        let mut pt_minus = rdiv.center(n).to_vec();
                        pt_minus[i_dim] = csave - widths[i_dim] * THIRD;
                        candidate_points.push(pt_minus);

                        // Point at center + w*THIRD
                        let mut pt_plus = rdiv.center(n).to_vec();
                        pt_plus[i_dim] = csave + widths[i_dim] * THIRD;
                        candidate_points.push(pt_plus);
                    }

                    all_points.extend(candidate_points);
                    plans.push(RectDivisionPlan {
                        rect: rdiv,
                        is_path_a: true,
                        dims,
                        fval_offset,
                    });
                } else {
                    // Path B: evaluate 2 points for one dimension
                    let mut i_div = imax;
                    if nlongest > 1 && which_div == 2 {
                        let rand_idx = p.age % nlongest as usize;
                        let mut count = 0;
                        for k in 0..n {
                            if wmax - widths[k] <= wmax * EQUAL_SIDE_TOL {
                                if count == rand_idx {
                                    i_div = k;
                                    break;
                                }
                                count += 1;
                            }
                        }
                    }

                    // Narrow the rect's width for the chosen dimension (same as serial)
                    let mut rdiv_narrowed = rdiv;
                    rdiv_narrowed.widths_mut(n)[i_div] *= THIRD;
                    rdiv_narrowed.set_diameter(
                        Self::rect_diameter(n, rdiv_narrowed.widths(n), which_diam),
                    );

                    let mut pt_minus = rdiv_narrowed.center(n).to_vec();
                    pt_minus[i_div] -= rdiv_narrowed.widths(n)[i_div];
                    let mut pt_plus = rdiv_narrowed.center(n).to_vec();
                    pt_plus[i_div] = rdiv_narrowed.center(n)[i_div] + rdiv_narrowed.widths(n)[i_div];

                    all_points.push(pt_minus);
                    all_points.push(pt_plus);

                    plans.push(RectDivisionPlan {
                        rect: rdiv_narrowed,
                        is_path_a: false,
                        dims: vec![i_div],
                        fval_offset,
                    });
                }
            }

            // ── Phase 3: Parallel evaluation ──
            let func_clone = Arc::clone(func);
            let f_values: Vec<f64> = all_points
                .par_iter()
                .map(|pt| func_clone(pt))
                .collect();

            p.nfev += f_values.len();

            // Update global minimum from all new evaluations
            for (idx, pt) in all_points.iter().enumerate() {
                if f_values[idx] < p.minf {
                    p.minf = f_values[idx];
                    p.xmin.copy_from_slice(pt);
                }
            }

            // ── Phase 4: Apply results — create children, insert into tree ──
            for plan in plans {
                let fvals = &f_values[plan.fval_offset..];

                if plan.is_path_a {
                    // Path A: sort dimensions by min(f+, f-), trisect in sorted order
                    let mut rdiv = plan.rect;
                    let nlongest = plan.dims.len();

                    // Build fv array matching serial: fv[2*dim_index] = f_minus,
                    // fv[2*dim_index+1] = f_plus. But our candidate_points are
                    // ordered by dimension: [dim0_minus, dim0_plus, dim1_minus, ...]
                    // and fvals are in the same order.

                    // Sort dims by min(f_minus, f_plus)
                    let mut dim_fvals: Vec<(usize, f64, f64)> = Vec::with_capacity(nlongest);
                    for (di, &dim_idx) in plan.dims.iter().enumerate() {
                        let f_minus = fvals[2 * di];
                        let f_plus = fvals[2 * di + 1];
                        dim_fvals.push((dim_idx, f_minus, f_plus));
                    }
                    dim_fvals.sort_by(|a, b| {
                        let fa = a.1.min(a.2);
                        let fb = b.1.min(b.2);
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Trisect in sorted order (matching serial Path A)
                    for &(si, f_minus, f_plus) in &dim_fvals {
                        rdiv.widths_mut(n)[si] *= THIRD;
                        rdiv.set_diameter(Self::rect_diameter(n, rdiv.widths(n), which_diam));
                        rdiv.set_age(p.age as f64);
                        p.age += 1;

                        // Create two children
                        for k in 0..=1usize {
                            let mut rnew = rdiv.clone();
                            rnew.center_mut(n)[si] +=
                                rdiv.widths(n)[si] * (2.0 * k as f64 - 1.0);
                            rnew.set_f_value(if k == 0 { f_minus } else { f_plus });
                            rnew.set_age(p.age as f64);
                            p.age += 1;
                            let key = RectKey {
                                diameter: rnew.diameter(),
                                f_value: rnew.f_value(),
                                age: rnew.age(),
                                id: p.alloc_id(),
                            };
                            p.rtree.insert(key, rnew);
                        }
                    }

                    // Re-insert modified parent
                    let key = RectKey {
                        diameter: rdiv.diameter(),
                        f_value: rdiv.f_value(),
                        age: rdiv.age(),
                        id: p.alloc_id(),
                    };
                    p.rtree.insert(key, rdiv);
                } else {
                    // Path B: one dimension, 2 children
                    let mut rdiv = plan.rect;
                    let i_div = plan.dims[0];
                    rdiv.set_age(p.age as f64);
                    p.age += 1;

                    for k in 0..=1usize {
                        let mut rnew = rdiv.clone();
                        rnew.center_mut(n)[i_div] +=
                            rdiv.widths(n)[i_div] * (2.0 * k as f64 - 1.0);
                        rnew.set_f_value(fvals[k]);
                        rnew.set_age(p.age as f64);
                        p.age += 1;
                        let key = RectKey {
                            diameter: rnew.diameter(),
                            f_value: rnew.f_value(),
                            age: rnew.age(),
                            id: p.alloc_id(),
                        };
                        p.rtree.insert(key, rnew);
                    }

                    // Re-insert modified parent
                    let key = RectKey {
                        diameter: rdiv.diameter(),
                        f_value: rdiv.f_value(),
                        age: rdiv.age(),
                        id: p.alloc_id(),
                    };
                    p.rtree.insert(key, rdiv);
                }
            }

            // Check stopping conditions after the batch
            let ret = self.check_stop_after_eval(p, max_feval, max_time, start_time);
            if let Some(code) = ret {
                return Err(code);
            }

            return Ok(false);
        }
    }

    /// Check stopping conditions after a function evaluation.
    /// Matches the FUNCTION_EVAL macro in cdirect.c line 145:
    /// checks force_stop, minf_max (fglobal), maxeval, maxtime.
    fn check_stop_after_eval(
        &self,
        p: &CDirectParams,
        max_feval: usize,
        max_time: f64,
        start_time: Option<&std::time::Instant>,
    ) -> Option<DirectReturnCode> {
        // Check fglobal (minf_max in NLOPT)
        let fglobal = self.options.fglobal;
        if fglobal > f64::NEG_INFINITY {
            let fglobal_reltol = self.options.fglobal_reltol;
            let threshold = if fglobal == 0.0 {
                fglobal_reltol
            } else {
                fglobal + fglobal_reltol * fglobal.abs()
            };
            if p.minf <= threshold {
                return Some(DirectReturnCode::GlobalFound);
            }
        }
        if max_feval > 0 && p.nfev >= max_feval {
            return Some(DirectReturnCode::MaxFevalExceeded);
        }
        if let Some(start) = start_time {
            if max_time > 0.0 && start.elapsed().as_secs_f64() >= max_time {
                return Some(DirectReturnCode::MaxTimeExceeded);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DirectAlgorithm, DirectOptions};

    /// Sphere function: f(x) = sum(x_i^2)
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    /// Rosenbrock function: f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)
    fn rosenbrock(x: &[f64]) -> f64 {
        let mut f = 0.0;
        for i in 0..x.len() - 1 {
            f += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        f
    }

    /// Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum: f64 = x
            .iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        10.0 * n + sum
    }

    #[test]
    fn test_rect_diameter_jones() {
        // Jones: sqrt(sum(w^2)) * 0.5, rounded to f32
        let w = vec![1.0, 1.0];
        let d = CDirect::rect_diameter(2, &w, 0);
        let expected = (2.0_f64.sqrt() * 0.5) as f32 as f64;
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rect_diameter_gablonsky() {
        // Gablonsky: max(w) * 0.5, rounded to f32
        let w = vec![1.0, 2.0, 0.5];
        let d = CDirect::rect_diameter(3, &w, 1);
        let expected = (2.0 * 0.5) as f32 as f64;
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_2d_direct_original() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::Original,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
        for &xi in &result.x {
            assert!(xi.abs() < 0.2, "x = {:?}", result.x);
        }
    }

    #[test]
    fn test_sphere_2d_direct_l() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-4, "fun = {}", result.fun);
        for &xi in &result.x {
            assert!(xi.abs() < 0.1, "x = {:?}", result.x);
        }
    }

    #[test]
    fn test_sphere_2d_unscaled() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::OriginalUnscaled,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
    }

    #[test]
    fn test_sphere_2d_locally_biased_unscaled() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiasedUnscaled,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-4, "fun = {}", result.fun);
    }

    #[test]
    fn test_sphere_3d_direct_l() {
        let bounds = vec![(-5.0, 5.0); 3];
        let opts = DirectOptions {
            max_feval: 1000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-3, "fun = {}", result.fun);
    }

    #[test]
    fn test_sphere_5d_direct_l() {
        let bounds = vec![(-5.0, 5.0); 5];
        let opts = DirectOptions {
            max_feval: 3000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
    }

    #[test]
    fn test_rosenbrock_2d_direct_l() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(rosenbrock, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1.0, "fun = {}", result.fun);
    }

    #[test]
    fn test_rosenbrock_2d_direct_original() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::Original,
            ..Default::default()
        };
        let optimizer = CDirect::new(rosenbrock, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 5.0, "fun = {}", result.fun);
    }

    #[test]
    fn test_rastrigin_2d_direct_l() {
        let bounds = vec![(-5.12, 5.12); 2];
        let opts = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(rastrigin, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 5.0, "fun = {}", result.fun);
    }

    #[test]
    fn test_maxfeval_termination() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 50,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(
            result.nfev <= 55, // May slightly exceed due to batch evaluations
            "nfev = {}",
            result.nfev
        );
    }

    #[test]
    fn test_maxiter_termination() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_iter: 5,
            max_feval: 10000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert_eq!(result.return_code, DirectReturnCode::MaxIterExceeded);
        assert_eq!(result.nit, 5);
    }

    #[test]
    fn test_fglobal_termination() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 5000,
            fglobal: 0.0,
            fglobal_reltol: 1e-4,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert_eq!(result.return_code, DirectReturnCode::GlobalFound);
        assert!(result.fun <= 1e-4, "fun = {}", result.fun);
    }

    #[test]
    fn test_callback_force_stop() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 5000,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts)
            .with_callback(|_x, f, _nfev, _nit| f < 1.0);
        let result = optimizer.minimize().unwrap();
        assert_eq!(result.return_code, DirectReturnCode::ForcedStop);
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_invalid_bounds() {
        let bounds = vec![(5.0, -5.0)]; // Invalid: lower > upper
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        assert!(optimizer.minimize().is_err());
    }

    #[test]
    fn test_gablonsky_variant_rejected() {
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyOriginal,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        assert!(optimizer.minimize().is_err());
    }

    #[test]
    fn test_1d_sphere() {
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            max_feval: 200,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(|x: &[f64]| x[0] * x[0], bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-4, "fun = {}", result.fun);
        assert!(result.x[0].abs() < 0.1);
    }

    #[test]
    fn test_asymmetric_bounds() {
        let bounds = vec![(2.0, 10.0), (-1.0, 100.0)];
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        // Minimum of sphere in [2,10]×[-1,100] is at (2, 0) with f=4
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 5.0, "fun = {}", result.fun);
    }

    #[test]
    fn test_convex_hull_basic() {
        // Build a small tree and verify hull computation
        let mut rtree = BTreeMap::new();
        let entries = vec![
            (1.0_f64, 5.0_f64, 0.0_f64),
            (2.0, 3.0, 1.0),
            (3.0, 2.0, 2.0),
            (4.0, 4.0, 3.0), // above hull
            (5.0, 1.0, 4.0),
        ];

        for (i, &(d, f, age)) in entries.iter().enumerate() {
            let key = RectKey {
                diameter: d,
                f_value: f,
                age,
                id: i,
            };
            let mut rect = HyperRect::new(1);
            rect.set_diameter(d);
            rect.set_f_value(f);
            rect.set_age(age);
            rtree.insert(key, rect);
        }

        let hull = CDirect::convex_hull(&rtree, false);
        // Hull should include points at d=1 (f=5), d=2 or 3 (on lower envelope), d=5 (f=1)
        assert!(hull.len() >= 2, "hull has {} points", hull.len());
        // First should be smallest diameter
        assert_eq!(hull.first().unwrap().diameter, 1.0);
        // Last should be largest diameter
        assert_eq!(hull.last().unwrap().diameter, 5.0);
    }

    #[test]
    fn test_convex_hull_duplicates() {
        let mut rtree = BTreeMap::new();
        let entries = vec![
            (1.0_f64, 3.0_f64, 0.0_f64),
            (1.0, 3.0, 1.0), // duplicate
            (2.0, 1.0, 2.0),
        ];

        for (i, &(d, f, age)) in entries.iter().enumerate() {
            let key = RectKey {
                diameter: d,
                f_value: f,
                age,
                id: i,
            };
            let mut rect = HyperRect::new(1);
            rect.set_diameter(d);
            rect.set_f_value(f);
            rect.set_age(age);
            rtree.insert(key, rect);
        }

        // With allow_dups = true, should include both d=1 points
        let hull_dups = CDirect::convex_hull(&rtree, true);
        let count_d1 = hull_dups.iter().filter(|k| k.diameter == 1.0).count();
        assert_eq!(count_d1, 2, "should include duplicate at d=1");

        // Without dups, should have just one
        let hull_no_dups = CDirect::convex_hull(&rtree, false);
        let count_d1 = hull_no_dups.iter().filter(|k| k.diameter == 1.0).count();
        assert_eq!(count_d1, 1);
    }

    #[test]
    fn test_all_cdirect_variants() {
        // Verify all cdirect algorithm variants can run
        let variants = vec![
            DirectAlgorithm::Original,
            DirectAlgorithm::LocallyBiased,
            DirectAlgorithm::OriginalUnscaled,
            DirectAlgorithm::LocallyBiasedUnscaled,
        ];

        for alg in variants {
            let bounds = vec![(-5.0, 5.0); 2];
            let opts = DirectOptions {
                max_feval: 200,
                algorithm: alg,
                ..Default::default()
            };
            let optimizer = CDirect::new(sphere, bounds, opts);
            let result = optimizer.minimize().unwrap();
            assert!(result.success, "Algorithm {:?} failed", alg);
            assert!(result.fun < 1.0, "Algorithm {:?}: fun = {}", alg, result.fun);
        }
    }

    #[test]
    fn test_magic_eps_effect() {
        // Test with magic_eps = 1e-4 (Jones recommendation)
        let bounds = vec![(-5.0, 5.0); 2];
        let opts = DirectOptions {
            max_feval: 500,
            magic_eps: 1e-4,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let optimizer = CDirect::new(sphere, bounds, opts);
        let result = optimizer.minimize().unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
    }

    #[test]
    fn test_scaled_vs_unscaled_same_problem() {
        // On symmetric bounds [-5,5]^2, scaled and unscaled should give similar results
        let bounds = vec![(-5.0, 5.0); 2];

        let opts_scaled = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let r1 = CDirect::new(sphere, bounds.clone(), opts_scaled)
            .minimize()
            .unwrap();

        let opts_unscaled = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiasedUnscaled,
            ..Default::default()
        };
        let r2 = CDirect::new(sphere, bounds, opts_unscaled)
            .minimize()
            .unwrap();

        // Both should find a good solution
        assert!(r1.fun < 1e-2, "scaled fun = {}", r1.fun);
        assert!(r2.fun < 1e-2, "unscaled fun = {}", r2.fun);
    }
}
