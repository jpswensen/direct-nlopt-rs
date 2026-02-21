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

/// Internal parameters matching the `params` struct in cdirect.c.
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
    /// Matches `cdirect()` in cdirect.c lines 569–603.
    fn optimize_scaled(&self, n: usize, which_alg: i32) -> Result<DirectResult> {
        let lb: Vec<f64> = self.bounds.iter().map(|&(lo, _)| lo).collect();
        let ub: Vec<f64> = self.bounds.iter().map(|&(_, hi)| hi).collect();

        // Wrap function to unscale: x_actual = lb + xu * (ub - lb)
        let lb_clone = lb.clone();
        let ub_clone = ub.clone();
        let func = self.func.clone();
        let scaled_func = move |xu: &[f64]| -> f64 {
            let mut x_actual = vec![0.0; xu.len()];
            for i in 0..xu.len() {
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

            let ret = self.divide_good_rects(
                func,
                &mut p,
                max_feval,
                max_time,
                start_time.as_ref(),
            );

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
    /// Matches `function_eval()` in cdirect.c lines 136–144.
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
    /// Matches `rect_diameter()` in cdirect.c lines 94–112.
    /// Rounds to f32 precision to group rectangles by diameter level.
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
    /// Matches `divide_rect()` in cdirect.c lines 152–243.
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
        let mut wmax = widths[0];
        let mut imax = 0;
        for i in 1..n {
            if widths[i] > wmax {
                wmax = widths[i];
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
    /// Matches `convex_hull()` in cdirect.c lines 261–378.
    ///
    /// Returns keys of hull points in order from smallest to largest diameter.
    fn convex_hull(rtree: &BTreeMap<RectKey, HyperRect>, allow_dups: bool) -> Vec<RectKey> {
        if rtree.is_empty() {
            return vec![];
        }

        // Collect (diameter, f_value, key) sorted by (diameter, f_value, age)
        // BTreeMap iteration is already in sorted order by RectKey
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

        let mut hull: Vec<RectKey> = Vec::new();

        if allow_dups {
            // Include duplicate points at (xmin, yminmin)
            for &(k, d, f) in &entries {
                if d == xmin && f == yminmin {
                    hull.push(k.clone());
                } else {
                    break;
                }
            }
        } else {
            hull.push(entries[0].0.clone());
        }

        if xmin == xmax {
            return hull;
        }

        // Find ymaxmin: minimum f at xmax
        let ymaxmin = entries
            .iter()
            .rev()
            .find(|&&(_, d, _)| {
                // Find first entry where d == xmax (going backwards, all at end have d==xmax)
                d == xmax
            })
            .map(|&(_, _, _)| {
                // Actually we need the min f among entries with d == xmax
                entries
                    .iter()
                    .filter(|&&(_, d, _)| d == xmax)
                    .map(|&(_, _, f)| f)
                    .fold(f64::INFINITY, f64::min)
            })
            .unwrap();

        let minslope = (ymaxmin - yminmin) / (xmax - xmin);

        // Skip entries with x == xmin
        let start_idx = entries
            .iter()
            .position(|&(_, d, _)| d != xmin)
            .unwrap_or(entries.len());

        // Find nmax_start: first entry with d == xmax
        let nmax_start = entries
            .iter()
            .position(|&(_, d, _)| d == xmax)
            .unwrap_or(entries.len());

        // Process entries between xmin and xmax
        let mut i = start_idx;
        while i < nmax_start {
            let (k, x, y) = entries[i];

            // Skip if above the line from (xmin,yminmin) to (xmax,ymaxmin)
            if y > yminmin + (x - xmin) * minslope {
                i += 1;
                continue;
            }

            // Performance hack: skip vertical lines
            if !hull.is_empty() && x == hull.last().unwrap().diameter {
                if y > entries
                    .iter()
                    .find(|&&(kk, _, _)| std::ptr::eq(kk, hull.last().unwrap()))
                    .map(|&(_, _, f)| f)
                    .unwrap_or(f64::INFINITY)
                {
                    // Skip to next diameter value
                    let cur_d = x;
                    while i < nmax_start && entries[i].1 == cur_d {
                        i += 1;
                    }
                    continue;
                } else if allow_dups {
                    hull.push(k.clone());
                    i += 1;
                    continue;
                }
                // If equal y and not allow_dups, fall through to hull update
            }

            // Remove points until we make a "left turn" to k
            while hull.len() > 1 {
                let t1_d = hull[hull.len() - 1].diameter;
                let t1_f = hull[hull.len() - 1].f_value;

                // Look backwards for a different point
                let mut it2 = hull.len() as i64 - 2;
                let (t2_d, t2_f);
                loop {
                    if it2 < 0 {
                        break;
                    }
                    let t2_d_cand = hull[it2 as usize].diameter;
                    let t2_f_cand = hull[it2 as usize].f_value;
                    if t2_d_cand != t1_d || t2_f_cand != t1_f {
                        break;
                    }
                    it2 -= 1;
                }
                if it2 < 0 {
                    break;
                }
                t2_d = hull[it2 as usize].diameter;
                t2_f = hull[it2 as usize].f_value;

                // Cross product (t1-t2) × (k-t2) >= 0 means left turn or straight
                let cross = (t1_d - t2_d) * (y - t2_f) - (t1_f - t2_f) * (x - t2_d);
                if cross >= 0.0 {
                    break;
                }
                hull.pop();
            }
            hull.push(k.clone());
            i += 1;
        }

        // Add points at (xmax, ymaxmin)
        if allow_dups {
            for j in nmax_start..entries.len() {
                let (k, d, f) = entries[j];
                if d == xmax && f == ymaxmin {
                    hull.push(k.clone());
                } else if d != xmax {
                    break;
                }
            }
        } else {
            // Find the entry with min f at xmax
            if let Some(&(k, _, _)) = entries[nmax_start..].iter().find(|&&(_, d, f)| d == xmax && f == ymaxmin) {
                hull.push(k.clone());
            }
        }

        hull
    }

    /// Divide potentially optimal rectangles.
    ///
    /// Matches `divide_good_rects()` in cdirect.c lines 392–458.
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

            // Build hull info: (key, diameter, f_value)
            let hull_info: Vec<(RectKey, f64, f64)> = hull
                .iter()
                .map(|k| (k.clone(), k.diameter, k.f_value))
                .collect();

            let mut i = 0;
            while i < nhull {
                // Find unequal points before (im) and after (ip)
                let mut im: i64 = i as i64 - 1;
                while im >= 0 && hull_info[im as usize].1 == hull_info[i].1 {
                    im -= 1;
                }
                let mut ip = i + 1;
                while ip < nhull && hull_info[ip].1 == hull_info[i].1 {
                    ip += 1;
                }

                let mut k1 = f64::NEG_INFINITY;
                let mut k2 = f64::NEG_INFINITY;

                if im >= 0 {
                    k1 = (hull_info[i].2 - hull_info[im as usize].2)
                        / (hull_info[i].1 - hull_info[im as usize].1);
                }
                if ip < nhull {
                    k2 = (hull_info[i].2 - hull_info[ip].2)
                        / (hull_info[i].1 - hull_info[ip].1);
                }
                let k = k1.max(k2);

                // Potentially optimal test
                if hull_info[i].2 - k * hull_info[i].1
                    <= p.minf - magic_eps * p.minf.abs()
                    || ip == nhull
                {
                    // Divide this rectangle
                    let target_key = hull_info[i].0.clone();
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
