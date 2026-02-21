//! Core DIRECT algorithm: preprocessing, scaling, initialization, and objective function evaluation.
//!
//! Faithfully mirrors NLOPT's Gablonsky Fortran→C translation from
//! `DIRect.c`, `DIRsubrout.c`, and `DIRserial.c`.
//!
//! # NLOPT C Correspondence
//!
//! | Rust                            | NLOPT C function              | File             |
//! |---------------------------------|-------------------------------|------------------|
//! | `Direct::new()`                 | `direct_dirpreprc_()`         | DIRsubrout.c     |
//! | `Direct::validate_inputs()`     | `direct_dirheader_()`         | DIRsubrout.c     |
//! | `Direct::to_actual()`           | unscaling in `direct_dirinfcn_()` | DIRsubrout.c |
//! | `Direct::to_normalized()`       | rescaling in `direct_dirinfcn_()` | DIRsubrout.c |
//! | `Direct::evaluate()`            | `direct_dirinfcn_()`          | DIRsubrout.c     |
//! | `Direct::initialize()`          | `direct_dirinit_()`           | DIRsubrout.c     |
//! | `Direct::sample_points()`       | `direct_dirsamplepoints_()`   | DIRsubrout.c     |
//! | `Direct::evaluate_sample_points()` | `direct_dirsamplef_()`     | DIRserial.c      |
//! | `Direct::divide_rectangle()`    | `direct_dirdivide_()`         | DIRsubrout.c     |

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{DirectError, Result};
use crate::storage::RectangleStorage;
use crate::types::{Bounds, CallbackFn, DirectOptions, ObjectiveFn, DIRECT_UNKNOWN_FGLOBAL};

/// Core DIRECT optimizer state for the Gablonsky Fortran→C translation path.
///
/// Holds the objective function, bounds, scaling factors, options, and
/// rectangle storage for the DIRECT algorithm.
pub struct Direct {
    /// Objective function: takes &[f64] (actual coordinates), returns f64.
    /// Returns f64::NAN or f64::INFINITY for infeasible points.
    func: Arc<ObjectiveFn>,

    /// Number of dimensions.
    pub dim: usize,

    /// Original lower bounds.
    pub lower: Vec<f64>,

    /// Original upper bounds.
    pub upper: Vec<f64>,

    /// Scaling factors: `xs1[i] = upper[i] - lower[i]`.
    ///
    /// Matches NLOPT's `c1` / `xs1` in `direct_dirpreprc_()` (DIRsubrout.c line 1437):
    /// ```c
    /// help = u[i__] - l[i__];
    /// xs1[i__] = help;
    /// ```
    pub xs1: Vec<f64>,

    /// Offset factors: `xs2[i] = lower[i] / (upper[i] - lower[i])`.
    ///
    /// Matches NLOPT's `c2` / `xs2` in `direct_dirpreprc_()` (DIRsubrout.c line 1438):
    /// ```c
    /// xs2[i__] = l[i__] / help;
    /// ```
    pub xs2: Vec<f64>,

    /// Optimizer options.
    pub options: DirectOptions,

    /// Epsilon for potentially-optimal test (may be modified by Jones' update formula).
    pub eps: f64,

    /// Fixed epsilon value stored when using Jones' update formula.
    /// Matches NLOPT's `epsfix` in `direct_dirheader_()`.
    pub eps_fix: f64,

    /// Flag: 1 if epsilon changes using Jones' formula, 0 if constant.
    /// Matches NLOPT's `iepschange` flag.
    pub ieps_change: i32,

    /// Rectangle storage.
    pub storage: RectangleStorage,

    /// Current minimum feasible function value.
    /// Matches NLOPT's `minf` in DIRect.c.
    pub minf: f64,

    /// Position (1-based) of rectangle with minimum function value.
    /// Matches NLOPT's `minpos` in DIRect.c.
    pub minpos: usize,

    /// Maximum feasible function value found so far.
    /// Matches NLOPT's `fmax` in DIRect.c.
    pub fmax: f64,

    /// Feasibility status: 0 if at least one feasible point found, 1 if all infeasible.
    /// Matches NLOPT's `ifeasiblef` in DIRect.c.
    pub ifeasible_f: i32,

    /// Maximum infeasibility flag seen (0=all feasible, ≥1=some infeasible).
    /// Matches NLOPT's `iinfesiblef` in DIRect.c.
    pub iinfeasible: i32,

    /// Current active maximum depth for the convex hull sweep.
    /// Matches NLOPT's `actmaxdeep` in DIRect.c.
    pub actmaxdeep: i32,

    /// Total function evaluations.
    pub nfev: usize,

    /// Total iterations.
    pub nit: usize,

    /// Force-stop flag: when set to true, subsequent evaluations are skipped.
    /// Matches NLOPT's `force_stop` pointer in `direct_dirsamplef_()`.
    /// Thread-safe for use in parallel evaluation.
    pub force_stop: Arc<AtomicBool>,
}

impl Direct {
    /// Create a new Direct optimizer with preprocessing matching `direct_dirpreprc_()`.
    ///
    /// # NLOPT C Correspondence
    ///
    /// Preprocessing matches `direct_dirpreprc_()` in DIRsubrout.c (lines 1403-1442):
    /// ```c
    /// for (i__ = 1; i__ <= n; ++i__) {
    ///     help = u[i__] - l[i__];
    ///     xs2[i__] = l[i__] / help;
    ///     xs1[i__] = help;
    /// }
    /// ```
    ///
    /// # Arguments
    /// * `func` - Objective function taking actual (unscaled) coordinates
    /// * `bounds` - Lower and upper bounds for each dimension
    /// * `options` - Optimizer configuration
    ///
    /// # Errors
    /// Returns `DirectError::InvalidBounds` if any lower bound >= upper bound.
    pub fn new(
        func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
        bounds: &Bounds,
        options: DirectOptions,
    ) -> Result<Self> {
        let dim = bounds.len();
        if dim == 0 {
            return Err(DirectError::InvalidArgs("dimension must be >= 1".into()));
        }

        let mut lower = Vec::with_capacity(dim);
        let mut upper = Vec::with_capacity(dim);
        let mut xs1 = Vec::with_capacity(dim);
        let mut xs2 = Vec::with_capacity(dim);

        // Preprocessing: matches direct_dirpreprc_() exactly
        for (i, &(l, u)) in bounds.iter().enumerate() {
            if u <= l {
                return Err(DirectError::InvalidBounds { dim: i });
            }
            let help = u - l;
            xs1.push(help);           // xs1[i] = u - l
            xs2.push(l / help);       // xs2[i] = l / (u - l)
            lower.push(l);
            upper.push(u);
        }

        // Create storage
        let storage = RectangleStorage::new(dim, options.max_feval, 0);

        Ok(Self {
            func: Arc::new(func),
            dim,
            lower,
            upper,
            xs1,
            xs2,
            options,
            eps: 0.0,     // set by validate_inputs
            eps_fix: 0.0,  // set by validate_inputs
            ieps_change: 0, // set by validate_inputs
            storage,
            minf: f64::INFINITY,
            minpos: 0,
            fmax: 0.0,
            ifeasible_f: 0,
            iinfeasible: 0,
            actmaxdeep: 0,
            nfev: 0,
            nit: 0,
            force_stop: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Validate inputs matching `direct_dirheader_()` in DIRsubrout.c (lines 1444-1565).
    ///
    /// Handles:
    /// - Bound validation (u > l for all dimensions)
    /// - Epsilon sign handling: negative eps → Jones update formula
    /// - Memory size checks (maxf + 20 > maxfunc)
    ///
    /// # NLOPT C Correspondence
    ///
    /// Epsilon handling (DIRsubrout.c lines 1484-1491):
    /// ```c
    /// if (*eps < 0.) {
    ///     *iepschange = 1;
    ///     *epsfix = -(*eps);
    ///     *eps = -(*eps);
    /// } else {
    ///     *iepschange = 0;
    ///     *epsfix = 1e100;
    /// }
    /// ```
    ///
    /// Volume/sigma tolerance conversion happens in `direct_wrap.c` (lines 67-74):
    /// ```c
    /// volume_reltol *= 100;
    /// sigma_reltol *= 100;
    /// if (volume_reltol <= 0) volume_reltol = -1;
    /// if (sigma_reltol <= 0) sigma_reltol = -1;
    /// ```
    pub fn validate_inputs(&mut self) -> Result<()> {
        // Bound validation (redundant since new() checks, but matches NLOPT structure)
        for i in 0..self.dim {
            if self.upper[i] <= self.lower[i] {
                return Err(DirectError::InvalidBounds { dim: i });
            }
        }

        // Epsilon handling matching direct_dirheader_()
        let mut eps = self.options.magic_eps;
        if eps < 0.0 {
            self.ieps_change = 1;
            self.eps_fix = -eps;
            eps = -eps;
        } else {
            self.ieps_change = 0;
            self.eps_fix = 1e100;
        }
        self.eps = eps;

        // Check maxfeval fits in storage (matching DIRsubrout.c line 1538)
        let maxf = self.options.max_feval;
        if maxf > 0 && maxf + 20 > self.storage.maxfunc {
            return Err(DirectError::MaxFevalTooBig(maxf));
        }

        Ok(())
    }

    /// Convert normalized coordinates [0,1]^n to actual coordinates.
    ///
    /// Matches the unscaling step in `direct_dirinfcn_()` (DIRsubrout.c lines 1071-1074):
    /// ```c
    /// for (i__ = 1; i__ <= n; ++i__) {
    ///     x[i__] = (x[i__] + c2[i__]) * c1[i__];
    /// }
    /// ```
    ///
    /// Formula: `x_actual[i] = (x_norm[i] + xs2[i]) * xs1[i]`
    ///
    /// Algebraic equivalence:
    ///   `(x + l/(u-l)) * (u-l) = x*(u-l) + l`
    /// So `to_actual(0.0) = l` and `to_actual(1.0) = u`.
    #[inline]
    pub fn to_actual(&self, x_norm: &[f64], x_actual: &mut [f64]) {
        debug_assert_eq!(x_norm.len(), self.dim);
        debug_assert_eq!(x_actual.len(), self.dim);
        for i in 0..self.dim {
            x_actual[i] = (x_norm[i] + self.xs2[i]) * self.xs1[i];
        }
    }

    /// Convert actual coordinates to normalized coordinates [0,1]^n.
    ///
    /// Matches the rescaling step in `direct_dirinfcn_()` (DIRsubrout.c lines 1084-1087):
    /// ```c
    /// for (i__ = 1; i__ <= n; ++i__) {
    ///     x[i__] = x[i__] / c1[i__] - c2[i__];
    /// }
    /// ```
    ///
    /// Formula: `x_norm[i] = x_actual[i] / xs1[i] - xs2[i]`
    #[inline]
    pub fn to_normalized(&self, x_actual: &[f64], x_norm: &mut [f64]) {
        debug_assert_eq!(x_actual.len(), self.dim);
        debug_assert_eq!(x_norm.len(), self.dim);
        for i in 0..self.dim {
            x_norm[i] = x_actual[i] / self.xs1[i] - self.xs2[i];
        }
    }

    /// Evaluate the objective function at a normalized point.
    ///
    /// Matches `direct_dirinfcn_()` in DIRsubrout.c (lines 1048-1088):
    /// 1. Unscale: x_actual = (x_norm + xs2) * xs1
    /// 2. Call user function with actual coordinates
    /// 3. Handle infeasible points (NaN/Inf → flag as infeasible)
    ///
    /// # Arguments
    /// * `x_norm` - Point in normalized [0,1]^n coordinates
    ///
    /// # Returns
    /// `(f_value, is_feasible)` where:
    /// - `f_value` is the function value (or f64::MAX if infeasible)
    /// - `is_feasible` is true if the point is feasible (finite f-value)
    pub fn evaluate(&self, x_norm: &[f64]) -> (f64, bool) {
        let mut x_actual = vec![0.0; self.dim];
        self.to_actual(x_norm, &mut x_actual);

        let f = (self.func)(&x_actual);

        if f.is_finite() {
            (f, true)
        } else {
            // Infeasible: NaN or Inf returned
            (f64::MAX, false)
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Initialization — matches direct_dirinit_() in DIRsubrout.c
    // ──────────────────────────────────────────────────────────────────────

    /// Initialize the DIRECT algorithm.
    ///
    /// Matches `direct_dirinit_()` in DIRsubrout.c (lines 1152-1321) exactly:
    /// 1. Precompute thirds[k] and levels[k]
    /// 2. Set center of rect 1 to (0.5, ..., 0.5)
    /// 3. Evaluate center point
    /// 4. For each dimension, sample at center ± thirds[1]
    /// 5. Evaluate all 2n sample points
    /// 6. Divide the initial rectangle (sort dims by min(f+,f-), trisect)
    /// 7. Insert initial rectangles into linked list storage
    ///
    /// After initialization, function evaluation count = 2*n + 1.
    pub fn initialize(&mut self) -> Result<()> {
        let jones = self.jones();
        let n = self.dim;

        // Initialize lists (matches direct_dirinitlist_ call in DIRect.c line 381)
        self.storage.init_lists();

        // Precompute levels and thirds (matches dirinit_ lines 1208-1243)
        self.storage.precompute_levels(jones);
        self.storage.precompute_thirds();

        // Set center of rect 1 to (0.5, ..., 0.5) with zero lengths
        // (matches dirinit_ lines 1244-1250)
        for j in 0..n {
            self.storage.set_center(1, j, 0.5);
            self.storage.set_length(1, j, 0);
        }

        // Evaluate center point (matches dirinit_ line 1251)
        let x_center = vec![0.5; n];
        let (f_center, feasible) = self.evaluate(&x_center);
        self.nfev += 1;

        // Store result (matches dirinit_ lines 1251-1267)
        // help = 0 if feasible, 1 if infeasible
        let help: i32 = if feasible { 0 } else { 1 };
        self.storage.set_f(1, f_center, help as f64);
        self.iinfeasible = help;
        self.fmax = f_center;

        if help > 0 {
            // Infeasible center: f = HUGE_VAL (matches dirinit_ lines 1261-1264)
            self.storage.f_values[2] = f64::INFINITY;
            self.fmax = f64::INFINITY;
            self.ifeasible_f = 1;
        } else {
            self.ifeasible_f = 0;
        }

        // Set initial state (matches dirinit_ lines 1269-1273)
        self.minf = self.storage.f_val(1);
        self.minpos = 1;
        self.storage.point[1] = 0;
        self.storage.free = 2;

        // Compute delta (matches dirinit_ line 1274)
        let delta = self.storage.thirds[1];

        // Get longest dimensions of rect 1 (matches dirinit_ line 1279)
        let (arrayi, maxi) = self.storage.get_longest_dims(1);

        // Sample points (matches dirinit_ lines 1280-1284)
        let new_start = self.sample_points(1, &arrayi, delta)?;

        // Evaluate sample points (matches dirinit_ lines 1296-1300)
        self.evaluate_sample_points(new_start, maxi)?;

        // Divide rectangle with currentlength=0 (matches dirinit_ lines 1316-1318)
        self.divide_rectangle(new_start, 0, 1, &arrayi, maxi);

        // Insert into linked lists (matches dirinit_ lines 1319-1320)
        let mut new_chain = new_start as i32;
        self.storage.insert_into_list(&mut new_chain, maxi, 1, jones);

        // Set actmaxdeep = 1 (matches DIRect.c line 426 after dirinit_ returns)
        self.actmaxdeep = 1;

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Sample points — matches direct_dirsamplepoints_() in DIRsubrout.c
    // ──────────────────────────────────────────────────────────────────────

    /// Create 2×maxi new sample points by copying parent center ± delta.
    ///
    /// Matches `direct_dirsamplepoints_()` in DIRsubrout.c (lines 870-936).
    ///
    /// Allocates 2×maxi new rectangle slots from the free list, copies the
    /// parent's center and lengths to each, then offsets the center along
    /// each longest dimension by ±delta.
    ///
    /// The new rectangles are chained via point[]: pos1→neg1→pos2→neg2→...→0
    ///
    /// # Arguments
    /// * `sample` — 1-based index of the parent rectangle
    /// * `arrayi` — dimensions with longest sides (1-based, from get_longest_dims)
    /// * `delta` — step size (typically thirds[depth+1])
    ///
    /// # Returns
    /// The 1-based index of the first new rectangle (start of chain).
    pub fn sample_points(
        &mut self,
        sample: usize,
        arrayi: &[usize],
        delta: f64,
    ) -> Result<usize> {
        let maxi = arrayi.len();

        // Record start of new chain (matches dirsamplepoints_ lines 900-901)
        let start = self.storage.free as usize;
        if start == 0 {
            return Err(DirectError::SamplePointsFailed(
                "No more free positions! Increase maxfunc.".into(),
            ));
        }

        // Allocate 2*maxi new rects, copy center and lengths from sample
        // (matches dirsamplepoints_ lines 903-922)
        let mut last_pos = start;
        for _k in 0..(2 * maxi) {
            let free_idx = self.storage.free as usize;
            self.storage.copy_center(free_idx, sample);
            self.storage.copy_lengths(free_idx, sample);
            last_pos = free_idx;
            self.storage.free = self.storage.point[free_idx];
            if self.storage.free == 0 {
                return Err(DirectError::SamplePointsFailed(
                    "No more free positions! Increase maxfunc.".into(),
                ));
            }
        }
        // Terminate chain (matches dirsamplepoints_ line 923)
        self.storage.point[last_pos] = 0;

        // Set offsets: positive then negative for each dimension
        // (matches dirsamplepoints_ lines 924-934)
        let mut pos = start;
        for &dim_1based in &arrayi[..maxi] {
            let dim_j = dim_1based - 1; // convert to 0-based
            let center_val = self.storage.center(sample, dim_j);

            // Positive offset
            self.storage.set_center(pos, dim_j, center_val + delta);
            let next = self.storage.point[pos] as usize;

            // Negative offset
            self.storage.set_center(next, dim_j, center_val - delta);
            pos = self.storage.point[next] as usize;
        }

        Ok(start)
    }

    // ──────────────────────────────────────────────────────────────────────
    // Evaluate sample points — matches direct_dirsamplef_() in DIRserial.c
    // ──────────────────────────────────────────────────────────────────────

    /// Evaluate objective function at all new sample points.
    ///
    /// Matches `direct_dirsamplef_()` in DIRserial.c (lines 17-150).
    ///
    /// Two passes:
    /// 1. Evaluate all 2×maxi points, tracking fmax and setting feasibility flags
    /// 2. Update minf/minpos for feasible points only
    ///
    /// When `options.parallel` is `true`, evaluations are performed in parallel
    /// using rayon. The storage updates remain sequential.
    ///
    /// # Force-stop handling
    /// Matches NLOPT's `force_stop` check in `direct_dirsamplef_()` (lines 86-92, 124-126):
    /// - If `force_stop` is set, the function value is set to `fmax` and the
    ///   feasibility flag is set to -1.0 (setup error), matching `kret = -1`.
    ///
    /// # Arguments
    /// * `new_start` — 1-based index of first new rectangle in chain
    /// * `maxi` — number of dimension splits (chain has 2×maxi rects)
    pub fn evaluate_sample_points(
        &mut self,
        new_start: usize,
        maxi: usize,
    ) -> Result<()> {
        let n = self.dim;
        let total = 2 * maxi;

        if self.options.parallel && total > 1 {
            // ── Parallel path ──
            // Collect all normalized coordinates from the linked list
            let mut indices: Vec<usize> = Vec::with_capacity(total);
            let mut points: Vec<Vec<f64>> = Vec::with_capacity(total);
            let mut pos = new_start;
            for _ in 0..total {
                indices.push(pos);
                let x_norm: Vec<f64> = (0..n)
                    .map(|i| self.storage.center(pos, i))
                    .collect();
                points.push(x_norm);
                pos = self.storage.point[pos] as usize;
            }

            // Evaluate all points in parallel
            let func = Arc::clone(&self.func);
            let force_stop = Arc::clone(&self.force_stop);
            let xs1 = &self.xs1;
            let xs2 = &self.xs2;
            let fmax_before = self.fmax;

            let results: Vec<(f64, i32)> = points
                .par_iter()
                .map(|x_norm| {
                    if force_stop.load(Ordering::Relaxed) {
                        (fmax_before, -1)
                    } else {
                        let mut x_actual = vec![0.0; n];
                        for i in 0..n {
                            x_actual[i] = (x_norm[i] + xs2[i]) * xs1[i];
                        }
                        let f = func(&x_actual);
                        if f.is_finite() {
                            (f, 0) // kret = 0: feasible
                        } else {
                            (f64::MAX, 1) // kret >= 1: infeasible
                        }
                    }
                })
                .collect();

            self.nfev += total;

            // Apply results to storage sequentially (matching C's ordering)
            for (j, &idx) in indices.iter().enumerate() {
                let (f_val, kret) = results[j];
                self.iinfeasible = self.iinfeasible.max(kret);

                if kret == 0 {
                    // Feasible (matches dirsamplef_ lines 97-109)
                    self.storage.set_f(idx, f_val, 0.0);
                    self.ifeasible_f = 0;
                    self.fmax = self.fmax.max(f_val);
                } else if kret >= 1 {
                    // Infeasible (matches dirsamplef_ lines 111-119)
                    self.storage.set_f(idx, self.fmax, 2.0);
                } else {
                    // kret == -1: force_stop / setup error (matches dirsamplef_ lines 124-126)
                    self.storage.set_f(idx, self.fmax, -1.0);
                }
            }

            // Second pass: update minf, minpos for feasible points only
            // (matches dirsamplef_ lines 141-149)
            for &idx in &indices {
                if self.storage.f_val(idx) < self.minf
                    && self.storage.f_flag(idx) == 0.0
                {
                    self.minf = self.storage.f_val(idx);
                    self.minpos = idx;
                }
            }
        } else {
            // ── Serial path — identical to NLOPT C evaluation order ──
            let mut pos = new_start;

            // First pass: evaluate all 2*maxi points
            // (matches dirsamplef_ lines 73-133)
            for _j in 0..total {
                // Check force_stop before evaluation (matches dirsamplef_ lines 86-87)
                if self.force_stop.load(Ordering::Relaxed) {
                    self.storage.set_f(pos, self.fmax, -1.0);
                    self.nfev += 1;
                    self.iinfeasible = self.iinfeasible.max(-1);
                    pos = self.storage.point[pos] as usize;
                    continue;
                }

                // Copy center to temporary buffer
                let x_norm: Vec<f64> = (0..n)
                    .map(|i| self.storage.center(pos, i))
                    .collect();

                // Evaluate (matches dirinfcn_ call in dirsamplef_ line 89)
                let (f_val, feasible) = self.evaluate(&x_norm);
                self.nfev += 1;

                // Check force_stop after evaluation (matches dirsamplef_ lines 91-92)
                if self.force_stop.load(Ordering::Relaxed) {
                    self.storage.set_f(pos, self.fmax, -1.0);
                    pos = self.storage.point[pos] as usize;
                    continue;
                }

                let kret: i32 = if feasible { 0 } else { 1 };
                self.iinfeasible = self.iinfeasible.max(kret);

                if kret == 0 {
                    // Feasible (matches dirsamplef_ lines 97-109)
                    self.storage.set_f(pos, f_val, 0.0);
                    self.ifeasible_f = 0;
                    self.fmax = self.fmax.max(f_val);
                } else {
                    // Infeasible: kret >= 1 (matches dirsamplef_ lines 111-119)
                    self.storage.set_f(pos, self.fmax, 2.0);
                }

                pos = self.storage.point[pos] as usize;
            }

            // Second pass: update minf, minpos for feasible points only
            // (matches dirsamplef_ lines 141-149)
            pos = new_start;
            for _j in 0..total {
                if self.storage.f_val(pos) < self.minf
                    && self.storage.f_flag(pos) == 0.0
                {
                    self.minf = self.storage.f_val(pos);
                    self.minpos = pos;
                }
                pos = self.storage.point[pos] as usize;
            }
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Divide rectangle — matches direct_dirdivide_() in DIRsubrout.c
    // ──────────────────────────────────────────────────────────────────────

    /// Divide a rectangle by trisecting along the longest dimensions.
    ///
    /// Matches `direct_dirdivide_()` in DIRsubrout.c (lines 944-1009).
    ///
    /// 1. For each longest dimension: compute w[j] = min(f+, f-) where f+ and f-
    ///    are the function values of the positive and negative children.
    /// 2. Sort dimensions by w[j] ascending (stable sort matching dirinsertlist_2__).
    /// 3. For each dimension in sorted order: increment the length index for the parent
    ///    and all children from that dimension onward.
    ///
    /// This means dimensions divided first (lower w) get their length incremented
    /// in ALL subsequent children, producing a finer-grained depth classification.
    ///
    /// # Arguments
    /// * `new_start` — 1-based index of first child rect in the chain
    /// * `current_length` — current length index (depth) to increment from
    /// * `sample` — 1-based index of the parent rectangle
    /// * `arrayi` — dimensions with longest sides (1-based)
    /// * `maxi` — number of dimensions being divided
    pub fn divide_rectangle(
        &mut self,
        new_start: usize,
        current_length: i32,
        sample: usize,
        arrayi: &[usize],
        maxi: usize,
    ) {
        // Step 1: Walk chain collecting pairs and computing w values
        // (matches dirdivide_ lines 972-987)
        let mut dim_info: Vec<(f64, usize, usize)> = Vec::with_capacity(maxi);
        // Each entry: (w_value, dimension_1based, positive_child_idx)

        let mut pos = new_start;
        for &dim_1based in &arrayi[..maxi] {
            let pos_child = pos;
            let f_pos = self.storage.f_val(pos_child);

            let neg_pos = self.storage.point[pos_child] as usize;
            let f_neg = self.storage.f_val(neg_pos);
            let w = f_pos.min(f_neg);

            dim_info.push((w, dim_1based, pos_child));

            pos = self.storage.point[neg_pos] as usize;
        }

        // Step 2: Sort by w value ascending (stable sort matching NLOPT's insertion sort)
        // (matches dirinsertlist_2__ behavior)
        dim_info.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Set lengths — dimensions divided first get their length incremented
        // in all subsequent children
        // (matches dirdivide_ lines 989-1008)
        let new_len = current_length + 1;
        for (i, &(_, dim_1based, _)) in dim_info.iter().enumerate() {
            let dim_k = dim_1based - 1; // convert to 0-based

            // Parent gets this dimension's length incremented
            self.storage.set_length(sample, dim_k, new_len);

            // All children from dimension i onward get this dimension's length incremented
            for &(_, _, pos_child) in &dim_info[i..] {
                let neg_child = self.storage.point[pos_child] as usize;
                self.storage.set_length(pos_child, dim_k, new_len);
                self.storage.set_length(neg_child, dim_k, new_len);
            }
        }
    }

    /// Get the volume tolerance in percentage form matching NLOPT's direct_wrap.c.
    ///
    /// NLOPT converts: `volume_reltol *= 100`, then `if (volume_reltol <= 0) volume_reltol = -1`
    pub fn volume_reltol_pct(&self) -> f64 {
        let v = self.options.volume_reltol * 100.0;
        if v <= 0.0 { -1.0 } else { v }
    }

    /// Get the sigma tolerance in percentage form matching NLOPT's direct_wrap.c.
    ///
    /// NLOPT converts: `sigma_reltol *= 100`, then `if (sigma_reltol <= 0) sigma_reltol = -1`
    pub fn sigma_reltol_pct(&self) -> f64 {
        let v = self.options.sigma_reltol * 100.0;
        if v <= 0.0 { -1.0 } else { v }
    }

    /// Get the fglobal_reltol in percentage form matching NLOPT's direct_wrap.c.
    ///
    /// NLOPT: `fglobal_reltol *= 100`, and if fglobal is unknown, set to 0.
    pub fn fglobal_reltol_pct(&self) -> f64 {
        if self.options.fglobal == DIRECT_UNKNOWN_FGLOBAL {
            0.0
        } else {
            self.options.fglobal_reltol * 100.0
        }
    }

    /// Update epsilon using Jones' formula, matching DIRect.c lines 671-675.
    ///
    /// ```c
    /// if (iepschange == 1) {
    ///     d__1 = fabs(*minf) * 1e-4;
    ///     *eps = MAX(d__1, epsfix);
    /// }
    /// ```
    pub fn update_epsilon(&mut self, minf: f64) {
        if self.ieps_change == 1 {
            let d = minf.abs() * 1e-4;
            self.eps = d.max(self.eps_fix);
        }
    }

    /// Get the algorithm method flag (jones parameter) for Gablonsky functions.
    ///
    /// Returns 0 for Original, 1 for Gablonsky (locally biased).
    pub fn jones(&self) -> i32 {
        self.options.algorithm.algmethod().unwrap_or(1)
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main iteration loop — matches direct_direct_() in DIRect.c
    // ──────────────────────────────────────────────────────────────────────

    /// Run the full DIRECT optimization.
    ///
    /// Matches `direct_direct_()` in DIRect.c (lines 449–768) exactly.
    ///
    /// # Algorithm Flow
    ///
    /// 1. `validate_inputs()` — input validation and epsilon setup
    /// 2. `initialize()` — center evaluation + 2n neighbors + first division
    /// 3. Main loop (each iteration):
    ///    a. `dirchoose_()` — select potentially optimal rectangles (convex hull)
    ///    b. `dirdoubleinsert_()` — (Jones Original only) add equal-valued rects
    ///    c. For each selected rectangle:
    ///       - `get_max_deep()` — compute depth for delta calculation
    ///       - Remove rectangle from anchor linked list
    ///       - `get_longest_dims()` — find dimensions with longest sides
    ///       - `sample_points()` — create 2×maxi new sample points
    ///       - `evaluate_sample_points()` — evaluate objective at new points
    ///       - `divide_rectangle()` — trisect along longest dimensions
    ///       - `insert_into_list()` — insert children into sorted linked lists
    ///    - d. Termination checks: volume_tol, sigma_tol, fglobal, maxfun, maxiter
    ///    - e. `replace_infeasible()` — replace infeasible point values
    ///    - f. Epsilon update (Jones formula)
    /// 4. Extract best point and build `DirectResult`
    ///
    /// # Callback
    ///
    /// If provided, the callback is called after each iteration with the current
    /// best point and value. If the callback returns `true`, optimization stops
    /// with `ForcedStop`.
    ///
    /// # Returns
    ///
    /// `DirectResult` containing the best point, function value, evaluation count,
    /// iteration count, and termination reason.
    pub fn minimize(
        &mut self,
        callback: Option<&CallbackFn>,
    ) -> Result<crate::types::DirectResult> {
        use crate::error::DirectReturnCode;
        use std::time::Instant;

        let start_time = Instant::now();

        // Step 1: Validate inputs (matching direct_dirheader_)
        self.validate_inputs()?;

        let jones = self.jones();
        let algmethod = jones; // algmethod=0 → Original, =1 → Gablonsky

        // Compute divfactor for fglobal test (DIRect.c lines 377-382)
        let divfactor = if self.options.fglobal == 0.0 {
            1.0
        } else {
            self.options.fglobal.abs()
        };

        // Save maxf budget for infeasible extension (DIRect.c lines 387-389)
        let oldmaxf = self.options.max_feval;
        let mut increase = false;

        // Compute max iteration count (DIRect.c line 60: maxt)
        let maxt = if self.options.max_iter == 0 {
            // NLOPT default: use max_feval / n as a fallback, or 1000000
            1_000_000usize
        } else {
            self.options.max_iter
        };

        // Step 2: Initialize (matching dirinit_)
        self.initialize()?;

        let mdeep = self.storage.maxdeep as i32;
        let mut numfunc = self.nfev;

        // Create PotentiallyOptimal selection buffer
        let mut po = crate::storage::PotentiallyOptimal::new(
            crate::storage::PotentiallyOptimal::DEFAULT_MAX_DIV,
        );

        // Get tolerance thresholds in NLOPT percentage form
        let volper = self.volume_reltol_pct();
        let sigmaper = self.sigma_reltol_pct();
        let fglper = self.fglobal_reltol_pct();
        let epsabs = self.options.magic_eps_abs;

        // Return code (matches NLOPT's ierror)
        let mut return_code: Option<DirectReturnCode> = None;

        // Step 3: Main loop (DIRect.c lines 448-449: i__1 = *maxt; for (t = tstart; t <= i__1; ++t))
        // tstart=2, loop goes from t=2 to t=maxt inclusive
        for t in 2..=maxt {
            self.nit = t - 1;

            // 3a. Select potentially optimal rectangles (dirchoose_)
            // NOTE: NLOPT passes MAXDEEP (maximum possible depth), not actmaxdeep
            po.select(
                &self.storage,
                self.storage.maxdeep as i32,
                self.minf,
                self.eps,
                epsabs,
                self.ifeasible_f,
                jones,
            );

            // 3b. Double insert for Jones Original (algmethod=0)
            // (DIRect.c lines 467-480)
            if algmethod == 0 {
                if let Err(_code) = po.double_insert(&self.storage) {
                    return Err(DirectError::DirectCode(
                        -6,
                        "Capacity of selection array S reached in DIRDoubleInsert".into(),
                    ));
                }
            }

            let oldpos = self.minpos;

            // 3c. Process each selected rectangle
            // (DIRect.c lines 487-607)
            let maxpos = po.count;
            for j in 0..maxpos {
                let actdeep = po.rect_levels[j];
                let help = po.indices[j];

                // Skip if this slot was eliminated (0 = empty)
                if help <= 0 {
                    continue;
                }

                let help_idx = help as usize;

                // Compute delta for sampling (DIRect.c lines 501-503)
                let actdeep_div = self.storage.get_max_deep(help_idx);
                let delta = self.storage.thirds[(actdeep_div + 1) as usize];

                // Check max depth (DIRect.c lines 510-515)
                if actdeep + 1 >= mdeep {
                    return_code = Some(DirectReturnCode::OutOfMemory);
                    break;
                }

                // Update actmaxdeep (DIRect.c line 517)
                self.actmaxdeep = self.actmaxdeep.max(actdeep);

                // Remove rectangle from anchor list (DIRect.c lines 518-528)
                self.storage.remove_from_list_at_depth(help_idx, actdeep);

                // Handle infeasible depth: if actdeep < 0, read from f-value
                // (DIRect.c lines 529-531)
                let _actdeep_for_division = if actdeep < 0 {
                    self.storage.f_val(help_idx) as i32
                } else {
                    actdeep
                };

                // Get longest dimensions (DIRect.c line 536)
                let (arrayi, maxi) = self.storage.get_longest_dims(help_idx);

                // Sample new points (DIRect.c lines 542-551)
                let new_start = self.sample_points(help_idx, &arrayi, delta)?;

                // Evaluate sample points (DIRect.c lines 556-568)
                self.evaluate_sample_points(new_start, maxi)?;

                // Check force_stop (DIRect.c lines 569-572)
                if self.force_stop.load(Ordering::Relaxed) {
                    return_code = Some(DirectReturnCode::ForcedStop);
                    break;
                }

                // Check max time (DIRect.c lines 573-576)
                if self.options.max_time > 0.0
                    && start_time.elapsed().as_secs_f64() >= self.options.max_time
                {
                    return_code = Some(DirectReturnCode::MaxTimeExceeded);
                    break;
                }

                // Divide the rectangle (DIRect.c lines 581-583)
                self.divide_rectangle(new_start, actdeep_div, help_idx, &arrayi, maxi);

                // Insert new rects into sorted lists (DIRect.c lines 588-590)
                let mut start_chain = new_start as i32;
                self.storage
                    .insert_into_list(&mut start_chain, maxi, help_idx, jones);

                // Update function evaluation count (DIRect.c lines 595-596)
                numfunc += 2 * maxi;
            }

            // Break out if we got a return code during rectangle processing
            if return_code.is_some() {
                break;
            }

            // Update nfev from numfunc
            self.nfev = numfunc;

            // Call callback if provided
            if let Some(cb) = callback {
                if oldpos < self.minpos {
                    // New minimum found — extract current best point
                    let x_best: Vec<f64> = (0..self.dim)
                        .map(|i| self.storage.center(self.minpos, i))
                        .collect();
                    let mut x_actual = vec![0.0; self.dim];
                    self.to_actual(&x_best, &mut x_actual);

                    if cb(&x_actual, self.minf, self.nfev, self.nit) {
                        self.force_stop.store(true, Ordering::Relaxed);
                        return_code = Some(DirectReturnCode::ForcedStop);
                        break;
                    }
                }
            }

            // ── Termination checks (DIRect.c lines 613-668) ──

            // Volume tolerance check (DIRect.c lines 613-626)
            // Use jones=0 to get level for volume check
            let level_for_vol =
                self.storage.get_level(self.minpos, 0) as usize;
            let vol_delta = self.storage.thirds[level_for_vol] * 100.0;
            if volper > 0.0 && vol_delta <= volper {
                return_code = Some(DirectReturnCode::VolTol);
                break;
            }

            // Sigma tolerance check (DIRect.c lines 631-640)
            let level_for_sigma =
                self.storage.get_level(self.minpos, jones) as usize;
            let sigma_delta = self.storage.levels[level_for_sigma];
            if sigmaper > 0.0 && sigma_delta <= sigmaper {
                return_code = Some(DirectReturnCode::SigmaTol);
                break;
            }

            // fglobal tolerance check (DIRect.c lines 645-652)
            if self.options.fglobal != DIRECT_UNKNOWN_FGLOBAL
                && fglper > 0.0
                && (self.minf - self.options.fglobal) * 100.0 / divfactor <= fglper
            {
                return_code = Some(DirectReturnCode::GlobalFound);
                break;
            }

            // Replace infeasible points (DIRect.c lines 657-661)
            if self.iinfeasible > 0 {
                self.storage.replace_infeasible(&self.xs1, &self.xs2, self.fmax, jones);
            }

            // Epsilon update using Jones formula (DIRect.c lines 666-670)
            self.update_epsilon(self.minf);

            // Infeasible budget extension (DIRect.c lines 678-694)
            if increase {
                if self.options.max_feval > 0 {
                    self.options.max_feval = numfunc + oldmaxf;
                }
                if self.ifeasible_f == 0 {
                    increase = false;
                }
            }

            // Max function evaluations check (DIRect.c lines 700-713)
            if self.options.max_feval > 0 && numfunc > self.options.max_feval {
                if self.ifeasible_f == 0 {
                    return_code = Some(DirectReturnCode::MaxFevalExceeded);
                    break;
                } else {
                    increase = true;
                    if oldmaxf > 0 {
                        self.options.max_feval = numfunc + oldmaxf;
                    }
                }
            }

            // Max time check at end of iteration
            if self.options.max_time > 0.0
                && start_time.elapsed().as_secs_f64() >= self.options.max_time
            {
                return_code = Some(DirectReturnCode::MaxTimeExceeded);
                break;
            }
        }

        // If loop finished without a return code, it's maxiter exceeded
        // (DIRect.c line 720)
        let return_code = return_code.unwrap_or(DirectReturnCode::MaxIterExceeded);

        // Step 4: Extract best point (DIRect.c lines 732-735)
        // C formula: x[i] = c[minpos,i] * l[i] + l[i] * u[i]
        // where l[i]=xs1[i] and u[i]=xs2[i] after dirpreprc_ aliasing.
        // This differs in floating-point order from to_actual's (c+xs2)*xs1.
        let x_actual: Vec<f64> = (0..self.dim)
            .map(|i| {
                let c = self.storage.center(self.minpos, i);
                c * self.xs1[i] + self.xs1[i] * self.xs2[i]
            })
            .collect();

        Ok(crate::types::DirectResult::new(
            x_actual,
            self.minf,
            self.nfev,
            self.nit,
            return_code,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DirectAlgorithm;

    // ────────────────────────────────────────────────────────────────
    // Preprocessing / Scaling tests
    // ────────────────────────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_preprocessing_symmetric_bounds() {
        // [-5, 5]: xs1 = 10, xs2 = -5/10 = -0.5
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        assert_eq!(d.xs1, vec![10.0, 10.0]);
        assert_eq!(d.xs2, vec![-0.5, -0.5]);
    }

    #[test]
    fn test_preprocessing_asymmetric_bounds() {
        // [2, 10]: xs1 = 8, xs2 = 2/8 = 0.25
        let bounds = vec![(2.0, 10.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        assert_eq!(d.xs1, vec![8.0]);
        assert_eq!(d.xs2, vec![0.25]);
    }

    #[test]
    fn test_preprocessing_zero_lower_bound() {
        // [0, 1]: xs1 = 1, xs2 = 0/1 = 0
        let bounds = vec![(0.0, 1.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        assert_eq!(d.xs1, vec![1.0]);
        assert_eq!(d.xs2, vec![0.0]);
    }

    #[test]
    fn test_preprocessing_invalid_bounds() {
        // Lower >= upper should fail
        let bounds = vec![(5.0, 5.0)];
        let opts = DirectOptions::default();
        let result = Direct::new(sphere, &bounds, opts);
        assert!(result.is_err());

        let bounds = vec![(10.0, 5.0)];
        let opts = DirectOptions::default();
        let result = Direct::new(sphere, &bounds, opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocessing_empty_bounds() {
        let bounds: Vec<(f64, f64)> = vec![];
        let opts = DirectOptions::default();
        let result = Direct::new(sphere, &bounds, opts);
        assert!(result.is_err());
    }

    // ────────────────────────────────────────────────────────────────
    // to_actual / to_normalized roundtrip tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_to_actual_symmetric() {
        // [-5, 5]: xs1=10, xs2=-0.5
        // to_actual(0.5) = (0.5 + (-0.5)) * 10 = 0.0 (center)
        // to_actual(0.0) = (0.0 + (-0.5)) * 10 = -5.0 (lower)
        // to_actual(1.0) = (1.0 + (-0.5)) * 10 = 5.0 (upper)
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        let mut actual = vec![0.0];

        d.to_actual(&[0.5], &mut actual);
        assert!((actual[0] - 0.0).abs() < 1e-15);

        d.to_actual(&[0.0], &mut actual);
        assert!((actual[0] - (-5.0)).abs() < 1e-15);

        d.to_actual(&[1.0], &mut actual);
        assert!((actual[0] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_to_actual_asymmetric() {
        // [2, 10]: xs1=8, xs2=0.25
        // to_actual(0.0) = (0.0 + 0.25) * 8 = 2.0
        // to_actual(1.0) = (1.0 + 0.25) * 8 = 10.0
        // to_actual(0.5) = (0.5 + 0.25) * 8 = 6.0
        let bounds = vec![(2.0, 10.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        let mut actual = vec![0.0];

        d.to_actual(&[0.0], &mut actual);
        assert!((actual[0] - 2.0).abs() < 1e-15);

        d.to_actual(&[1.0], &mut actual);
        assert!((actual[0] - 10.0).abs() < 1e-15);

        d.to_actual(&[0.5], &mut actual);
        assert!((actual[0] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_roundtrip_scaling() {
        // Verify: to_actual(to_normalized(x)) == x for various bounds and points
        let test_cases: Vec<(Vec<(f64, f64)>, Vec<f64>)> = vec![
            (vec![(-5.0, 5.0), (-5.0, 5.0)], vec![0.0, 0.0]),
            (vec![(-5.0, 5.0), (-5.0, 5.0)], vec![-3.0, 2.7]),
            (vec![(2.0, 10.0)], vec![6.0]),
            (vec![(0.0, 1.0), (0.0, 1.0)], vec![0.3, 0.8]),
            (vec![(-100.0, 1.0), (-1.0, 100.0)], vec![-50.0, 50.0]),
            (vec![(0.0, 1e-10)], vec![5e-11]),
            (vec![(-1e10, 1e10)], vec![1234.5678]),
        ];

        for (bounds, x_actual_orig) in &test_cases {
            let opts = DirectOptions {
                algorithm: DirectAlgorithm::GablonskyLocallyBiased,
                ..Default::default()
            };
            let d = Direct::new(sphere, bounds, opts).unwrap();
            let dim = bounds.len();

            let mut x_norm = vec![0.0; dim];
            let mut x_actual_back = vec![0.0; dim];

            d.to_normalized(x_actual_orig, &mut x_norm);
            d.to_actual(&x_norm, &mut x_actual_back);

            for i in 0..dim {
                let rel_err = if x_actual_orig[i].abs() > 1e-15 {
                    (x_actual_back[i] - x_actual_orig[i]).abs() / x_actual_orig[i].abs()
                } else {
                    (x_actual_back[i] - x_actual_orig[i]).abs()
                };
                // Use 1e-9 tolerance to accommodate floating-point rounding
                // with extreme bound ranges (e.g., [-1e10, 1e10])
                assert!(
                    rel_err < 1e-9,
                    "Roundtrip failed for bounds {:?}, x={:?}: got {:?} (err={})",
                    bounds, x_actual_orig, x_actual_back, rel_err,
                );
            }
        }
    }

    #[test]
    fn test_algebraic_equivalence() {
        // Verify: (x + l/(u-l)) * (u-l) = x*(u-l) + l
        let test_bounds: Vec<(f64, f64)> = vec![
            (-5.0, 5.0),
            (2.0, 10.0),
            (0.0, 1.0),
            (-100.0, 1.0),
        ];
        let x_norms = [0.0, 0.25, 0.5, 0.75, 1.0];

        for &(l, u) in &test_bounds {
            let xs1 = u - l;
            let xs2 = l / xs1;
            for &x in &x_norms {
                let formula1 = (x + xs2) * xs1;
                let formula2 = x * xs1 + l;
                assert!(
                    (formula1 - formula2).abs() < 1e-12,
                    "Algebraic equiv failed: l={}, u={}, x={}: {} != {}",
                    l, u, x, formula1, formula2,
                );
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Evaluate tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_feasible() {
        // Sphere function at center of [-5,5]^2 should be 0
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        // Normalized center = (0.5, 0.5) → actual = (0.0, 0.0) → sphere = 0.0
        let (f, feasible) = d.evaluate(&[0.5, 0.5]);
        assert!(feasible);
        assert!((f - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_evaluate_non_center() {
        // sphere at [2, 10] normalized 0.0 → actual 2.0 → sphere = 4.0
        let bounds = vec![(2.0, 10.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        let (f, feasible) = d.evaluate(&[0.0]);
        assert!(feasible);
        assert!((f - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_infeasible_nan() {
        // Function returning NaN → infeasible
        let bounds = vec![(0.0, 1.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(|_x: &[f64]| f64::NAN, &bounds, opts).unwrap();

        let (f, feasible) = d.evaluate(&[0.5]);
        assert!(!feasible);
        assert_eq!(f, f64::MAX);
    }

    #[test]
    fn test_evaluate_infeasible_infinity() {
        // Function returning Infinity → infeasible
        let bounds = vec![(0.0, 1.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(|_x: &[f64]| f64::INFINITY, &bounds, opts).unwrap();

        let (f, feasible) = d.evaluate(&[0.5]);
        assert!(!feasible);
        assert_eq!(f, f64::MAX);
    }

    #[test]
    fn test_evaluate_infeasible_neg_infinity() {
        // Function returning -Infinity → infeasible
        let bounds = vec![(0.0, 1.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(|_x: &[f64]| f64::NEG_INFINITY, &bounds, opts).unwrap();

        let (f, feasible) = d.evaluate(&[0.5]);
        assert!(!feasible);
        assert_eq!(f, f64::MAX);
    }

    // ────────────────────────────────────────────────────────────────
    // Validate inputs tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_inputs_eps_positive() {
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            magic_eps: 1e-4,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();
        d.validate_inputs().unwrap();

        assert_eq!(d.ieps_change, 0);
        assert_eq!(d.eps, 1e-4);
        assert_eq!(d.eps_fix, 1e100);
    }

    #[test]
    fn test_validate_inputs_eps_negative() {
        // Negative eps → Jones update formula
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            magic_eps: -0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();
        d.validate_inputs().unwrap();

        assert_eq!(d.ieps_change, 1);
        assert_eq!(d.eps, 0.01);
        assert_eq!(d.eps_fix, 0.01);
    }

    #[test]
    fn test_validate_inputs_eps_zero() {
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            magic_eps: 0.0,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();
        d.validate_inputs().unwrap();

        assert_eq!(d.ieps_change, 0);
        assert_eq!(d.eps, 0.0);
        assert_eq!(d.eps_fix, 1e100);
    }

    // ────────────────────────────────────────────────────────────────
    // Tolerance conversion tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_volume_reltol_pct() {
        let bounds = vec![(-5.0, 5.0)];

        // volume_reltol = 0.01 → 1.0%
        let opts = DirectOptions {
            volume_reltol: 0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert!((d.volume_reltol_pct() - 1.0).abs() < 1e-15);

        // volume_reltol = 0.0 → -1.0 (disabled)
        let opts = DirectOptions {
            volume_reltol: 0.0,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.volume_reltol_pct(), -1.0);

        // volume_reltol = -0.5 → negative → -1.0 (disabled)
        let opts = DirectOptions {
            volume_reltol: -0.5,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.volume_reltol_pct(), -1.0);
    }

    #[test]
    fn test_sigma_reltol_pct() {
        let bounds = vec![(-5.0, 5.0)];

        // sigma_reltol = 0.001 → 0.1%
        let opts = DirectOptions {
            sigma_reltol: 0.001,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert!((d.sigma_reltol_pct() - 0.1).abs() < 1e-15);

        // sigma_reltol = -1.0 (default) → -1.0 (disabled)
        let opts = DirectOptions {
            sigma_reltol: -1.0,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.sigma_reltol_pct(), -1.0);
    }

    #[test]
    fn test_fglobal_reltol_pct() {
        let bounds = vec![(-5.0, 5.0)];

        // fglobal unknown → 0.0
        let opts = DirectOptions {
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: 0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.fglobal_reltol_pct(), 0.0);

        // fglobal known → percentage
        let opts = DirectOptions {
            fglobal: 0.0,
            fglobal_reltol: 0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert!((d.fglobal_reltol_pct() - 1.0).abs() < 1e-15);
    }

    // ────────────────────────────────────────────────────────────────
    // Epsilon update tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_update_epsilon_jones() {
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            magic_eps: -0.001,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();
        d.validate_inputs().unwrap();

        // eps = max(|minf| * 1e-4, epsfix)
        // With minf = 100.0: max(0.01, 0.001) = 0.01
        d.update_epsilon(100.0);
        assert!((d.eps - 0.01).abs() < 1e-15);

        // With minf = 1.0: max(0.0001, 0.001) = 0.001
        d.update_epsilon(1.0);
        assert!((d.eps - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_update_epsilon_constant() {
        let bounds = vec![(-5.0, 5.0)];
        let opts = DirectOptions {
            magic_eps: 0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();
        d.validate_inputs().unwrap();

        // ieps_change == 0, eps should not change
        let original_eps = d.eps;
        d.update_epsilon(100.0);
        assert_eq!(d.eps, original_eps);
    }

    // ────────────────────────────────────────────────────────────────
    // Jones flag tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_jones_flag() {
        let bounds = vec![(-5.0, 5.0)];

        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyOriginal,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.jones(), 0); // Original → algmethod=0

        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.jones(), 1); // Gablonsky → algmethod=1

        // cdirect variants default to jones=1
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();
        assert_eq!(d.jones(), 1);
    }

    // ────────────────────────────────────────────────────────────────
    // Multi-dimensional scaling tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_multidim_scaling() {
        // 3D with different bounds per dimension
        let bounds = vec![(-5.0, 5.0), (2.0, 10.0), (0.0, 1.0)];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let d = Direct::new(sphere, &bounds, opts).unwrap();

        // xs1 = [10, 8, 1], xs2 = [-0.5, 0.25, 0.0]
        assert_eq!(d.xs1, vec![10.0, 8.0, 1.0]);
        assert_eq!(d.xs2, vec![-0.5, 0.25, 0.0]);

        // Normalized center (0.5, 0.5, 0.5) → actual (0.0, 6.0, 0.5)
        let mut actual = vec![0.0; 3];
        d.to_actual(&[0.5, 0.5, 0.5], &mut actual);
        assert!((actual[0] - 0.0).abs() < 1e-15);
        assert!((actual[1] - 6.0).abs() < 1e-15);
        assert!((actual[2] - 0.5).abs() < 1e-15);

        // Roundtrip
        let mut norm = vec![0.0; 3];
        d.to_normalized(&actual, &mut norm);
        for i in 0..3 {
            assert!((norm[i] - 0.5).abs() < 1e-15);
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Initialization tests
    // ──────────────────────────────────────────────────────────────────

    /// Helper to create a Direct instance with sphere function on [-5,5]^n
    fn make_sphere_direct(n: usize) -> Direct {
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); n];
        let func = |x: &[f64]| -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            ..Default::default()
        };
        Direct::new(func, &bounds, options).unwrap()
    }

    #[test]
    fn test_initialize_nfev_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();
        assert_eq!(d.nfev, 5); // 2*2 + 1
    }

    #[test]
    fn test_initialize_nfev_3d() {
        let mut d = make_sphere_direct(3);
        d.initialize().unwrap();
        assert_eq!(d.nfev, 7); // 2*3 + 1
    }

    #[test]
    fn test_initialize_nfev_5d() {
        let mut d = make_sphere_direct(5);
        d.initialize().unwrap();
        assert_eq!(d.nfev, 11); // 2*5 + 1
    }

    #[test]
    fn test_initialize_center_values_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        // Center at (0.5, 0.5) → actual (0, 0) → f = 0.0
        assert_eq!(d.storage.center(1, 0), 0.5);
        assert_eq!(d.storage.center(1, 1), 0.5);
        assert_eq!(d.storage.f_val(1), 0.0);
        assert_eq!(d.storage.f_flag(1), 0.0);
    }

    #[test]
    fn test_initialize_sample_centers_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        let third = 1.0 / 3.0;

        // Rect 2: positive offset along dim 0
        assert!((d.storage.center(2, 0) - (0.5 + third)).abs() < 1e-15);
        assert!((d.storage.center(2, 1) - 0.5).abs() < 1e-15);

        // Rect 3: negative offset along dim 0
        assert!((d.storage.center(3, 0) - (0.5 - third)).abs() < 1e-15);
        assert!((d.storage.center(3, 1) - 0.5).abs() < 1e-15);

        // Rect 4: positive offset along dim 1
        assert!((d.storage.center(4, 0) - 0.5).abs() < 1e-15);
        assert!((d.storage.center(4, 1) - (0.5 + third)).abs() < 1e-15);

        // Rect 5: negative offset along dim 1
        assert!((d.storage.center(5, 0) - 0.5).abs() < 1e-15);
        assert!((d.storage.center(5, 1) - (0.5 - third)).abs() < 1e-15);
    }

    #[test]
    fn test_initialize_function_values_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        // All 4 sample points: actual = (±10/3, 0) or (0, ±10/3) → f = 100/9
        let expected_f = 100.0 / 9.0;
        for idx in 2..=5 {
            assert!(
                (d.storage.f_val(idx) - expected_f).abs() < 1e-10,
                "rect {}: expected f={}, got f={}",
                idx, expected_f, d.storage.f_val(idx)
            );
            assert_eq!(d.storage.f_flag(idx), 0.0);
        }
    }

    #[test]
    fn test_initialize_minf_minpos_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        assert_eq!(d.minf, 0.0);
        assert_eq!(d.minpos, 1);
        assert!((d.fmax - 100.0 / 9.0).abs() < 1e-10);
        assert_eq!(d.ifeasible_f, 0);
        assert_eq!(d.iinfeasible, 0);
    }

    #[test]
    fn test_initialize_lengths_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        // After divide_rectangle with 2D cube:
        // Sorted by w (all equal for sphere, so stable sort keeps dim 1 first)
        // Rect 1 (center): [1, 1]
        assert_eq!(d.storage.length(1, 0), 1);
        assert_eq!(d.storage.length(1, 1), 1);

        // Rect 2 (dim 0 +): [1, 0] — only dim 0 was divided for this pair
        assert_eq!(d.storage.length(2, 0), 1);
        assert_eq!(d.storage.length(2, 1), 0);

        // Rect 3 (dim 0 -): [1, 0]
        assert_eq!(d.storage.length(3, 0), 1);
        assert_eq!(d.storage.length(3, 1), 0);

        // Rect 4 (dim 1 +): [1, 1] — both dims divided
        assert_eq!(d.storage.length(4, 0), 1);
        assert_eq!(d.storage.length(4, 1), 1);

        // Rect 5 (dim 1 -): [1, 1]
        assert_eq!(d.storage.length(5, 0), 1);
        assert_eq!(d.storage.length(5, 1), 1);
    }

    #[test]
    fn test_initialize_anchor_lists_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        // Gablonsky (jones=1): level = min(length values)
        // Rect 2,3: level = min(1,0) = 0 → anchor[1]
        // Rect 1,4,5: level = min(1,1) = 1 → anchor[2]

        // Anchor[1] (depth 0): 2 → 3 → 0
        assert_eq!(d.storage.anchor[1], 2);
        assert_eq!(d.storage.point[2], 3);
        assert_eq!(d.storage.point[3], 0);

        // Anchor[2] (depth 1): 1 → 4 → 5 → 0
        assert_eq!(d.storage.anchor[2], 1);
        assert_eq!(d.storage.point[1], 4);
        assert_eq!(d.storage.point[4], 5);
        assert_eq!(d.storage.point[5], 0);
    }

    #[test]
    fn test_initialize_free_pointer_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();

        // After allocating rects 2,3,4,5, free should point to 6
        assert_eq!(d.storage.free, 6);
    }

    #[test]
    fn test_initialize_actmaxdeep_2d() {
        let mut d = make_sphere_direct(2);
        d.initialize().unwrap();
        assert_eq!(d.actmaxdeep, 1);
    }

    #[test]
    fn test_initialize_1d() {
        // 1D sphere on [-5, 5]: center = 0.5 → actual = 0 → f = 0
        let bounds = vec![(-5.0, 5.0)];
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        assert_eq!(d.nfev, 3); // 2*1 + 1
        assert_eq!(d.minf, 0.0);
        assert_eq!(d.minpos, 1);

        // Rect 2: center = 0.5 + 1/3 → actual = 10/3 → f = 100/9
        // Rect 3: center = 0.5 - 1/3 → actual = -10/3 → f = 100/9
        let expected_f = 100.0 / 9.0;
        assert!((d.storage.f_val(2) - expected_f).abs() < 1e-10);
        assert!((d.storage.f_val(3) - expected_f).abs() < 1e-10);

        // Lengths: Rect 1=[1], Rect 2=[1], Rect 3=[1]
        assert_eq!(d.storage.length(1, 0), 1);
        assert_eq!(d.storage.length(2, 0), 1);
        assert_eq!(d.storage.length(3, 0), 1);

        // All at depth 1 → anchor[2]
        assert_eq!(d.storage.anchor[2], 1); // center first (f=0)
    }

    #[test]
    fn test_initialize_asymmetric_bounds() {
        // Rosenbrock on [0, 2] × [-1, 3]: minimum at (1, 1)
        let bounds = vec![(0.0, 2.0), (-1.0, 3.0)];
        let func = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        assert_eq!(d.nfev, 5);
        // Center at (0.5, 0.5) → actual = (1.0, 1.0) → f = 0.0
        assert_eq!(d.minf, 0.0);
        assert_eq!(d.minpos, 1);
    }

    #[test]
    fn test_initialize_jones_original() {
        // Test with Jones Original algorithm (jones=0)
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); 2];
        let func = |x: &[f64]| -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyOriginal,
            max_feval: 10000,
            max_iter: 100,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        assert_eq!(d.nfev, 5);
        assert_eq!(d.minf, 0.0);
        assert_eq!(d.minpos, 1);

        // Jones: level = sum(lengths), not min
        // Rect 2,3: sum(1,0) = 1 → anchor[2]
        // Rect 1,4,5: sum(1,1) = 2 → anchor[3]
        assert_eq!(d.storage.anchor[2], 2);
        assert_eq!(d.storage.anchor[3], 1);
    }

    // ──────────────────────────────────────────────────────────────────
    // Sample points tests
    // ──────────────────────────────────────────────────────────────────

    /// Helper: create a Direct instance, initialize it, then return it
    /// ready for a second iteration's sample_points + evaluate_sample_points.
    fn make_initialized_sphere(n: usize, parallel: bool) -> Direct {
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); n];
        let func = |x: &[f64]| -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();
        d
    }

    #[test]
    fn test_sample_points_allocation_2d() {
        // After init, free=6. Sample rect 1 (center) with 2 longest dims → 4 new rects.
        let mut d = make_initialized_sphere(2, false);
        let (arrayi, _maxi) = d.storage.get_longest_dims(1);
        let delta = d.storage.thirds[2]; // next deeper level

        let new_start = d.sample_points(1, &arrayi, delta).unwrap();
        // 4 new rects allocated from free list starting at 6
        assert_eq!(new_start, 6);
        // Chain: 6→7→8→9→0
        assert_eq!(d.storage.point[9], 0);
        // Free should advance past 4 slots
        assert_eq!(d.storage.free, 10);
    }

    #[test]
    fn test_sample_points_centers_2d() {
        let mut d = make_initialized_sphere(2, false);
        let (arrayi, _maxi) = d.storage.get_longest_dims(1);
        let delta = d.storage.thirds[2]; // 1/9

        let new_start = d.sample_points(1, &arrayi, delta).unwrap();

        let parent_c0 = d.storage.center(1, 0);
        let parent_c1 = d.storage.center(1, 1);

        // First dim: positive then negative
        let pos1 = new_start;
        let neg1 = d.storage.point[pos1] as usize;

        assert!((d.storage.center(pos1, arrayi[0] - 1) - (parent_c0 + delta)).abs() < 1e-15
            || (d.storage.center(pos1, arrayi[1] - 1) - (parent_c1 + delta)).abs() < 1e-15);

        // Second dim: positive then negative
        let pos2 = d.storage.point[neg1] as usize;
        let neg2 = d.storage.point[pos2] as usize;

        // Chain ends at neg2
        assert_eq!(d.storage.point[neg2], 0);
    }

    #[test]
    fn test_sample_points_1d() {
        let mut d = make_initialized_sphere(1, false);
        let (arrayi, maxi) = d.storage.get_longest_dims(1);
        assert_eq!(maxi, 1);
        let delta = d.storage.thirds[2];

        let new_start = d.sample_points(1, &arrayi, delta).unwrap();
        // 2 new rects (2*1)
        let pos1 = new_start;
        let neg1 = d.storage.point[pos1] as usize;
        assert_eq!(d.storage.point[neg1], 0); // chain ends

        // Check offsets
        let parent_c = d.storage.center(1, 0);
        assert!((d.storage.center(pos1, 0) - (parent_c + delta)).abs() < 1e-15);
        assert!((d.storage.center(neg1, 0) - (parent_c - delta)).abs() < 1e-15);
    }

    #[test]
    fn test_evaluate_sample_points_serial_order_2d() {
        // Verify that serial evaluation counts correct number of evals
        let mut d = make_initialized_sphere(2, false);
        let init_nfev = d.nfev; // 5
        let (arrayi, maxi) = d.storage.get_longest_dims(1);
        let delta = d.storage.thirds[2];
        let new_start = d.sample_points(1, &arrayi, delta).unwrap();
        d.evaluate_sample_points(new_start, maxi).unwrap();

        // 2*maxi = 4 new evaluations
        assert_eq!(d.nfev, init_nfev + 2 * maxi);
    }

    #[test]
    fn test_evaluate_sample_points_feasibility_flags() {
        // Use a function that returns NaN for some regions
        let bounds = vec![(0.0, 1.0)];
        let func = |x: &[f64]| -> f64 {
            if x[0] < 0.1 {
                f64::NAN
            } else {
                x[0] * x[0]
            }
        };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel: false,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        // Center=0.5 → actual=0.5 → feasible
        // Pos=5/6 → feasible, Neg=1/6 → feasible (> 0.1)
        assert_eq!(d.ifeasible_f, 0);
    }

    #[test]
    fn test_evaluate_sample_points_infeasible_tracking() {
        // Function that always returns NaN → all infeasible
        let bounds = vec![(0.0, 1.0)];
        let func = |_x: &[f64]| -> f64 { f64::NAN };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel: false,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        assert!(d.iinfeasible >= 1);
    }

    #[test]
    fn test_evaluate_sample_points_fmax_tracking() {
        let mut d = make_initialized_sphere(2, false);
        let fmax_after_init = d.fmax;

        // For sphere on [-5,5]^2, initial sample points at ± 1/3 from center
        // produce f = 100/9 ≈ 11.11, which should be fmax
        assert!((fmax_after_init - 100.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_sample_points_minf_minpos_update() {
        let bounds = vec![(0.0, 2.0)];
        let func = |x: &[f64]| -> f64 { (x[0] - 1.5).powi(2) };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel: false,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        // Center at 0.5 → actual = 1.0 → f = 0.25
        // Pos = 0.5+1/3 → actual ≈ 1.667 → f ≈ 0.028
        // minf should be the smallest feasible value
        assert!(d.minf < 0.25 + 1e-10);
    }

    // ──────────────────────────────────────────────────────────────────
    // Parallel vs serial comparison tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_parallel_vs_serial_sphere_2d() {
        let d_serial = make_initialized_sphere(2, false);
        let d_parallel = make_initialized_sphere(2, true);

        assert_eq!(d_serial.nfev, d_parallel.nfev);
        assert_eq!(d_serial.minf, d_parallel.minf);
        assert_eq!(d_serial.minpos, d_parallel.minpos);
        assert_eq!(d_serial.fmax, d_parallel.fmax);
        assert_eq!(d_serial.ifeasible_f, d_parallel.ifeasible_f);
        assert_eq!(d_serial.iinfeasible, d_parallel.iinfeasible);

        for idx in 1..=5 {
            assert_eq!(d_serial.storage.f_val(idx), d_parallel.storage.f_val(idx),
                "f_val mismatch at idx {}", idx);
            assert_eq!(d_serial.storage.f_flag(idx), d_parallel.storage.f_flag(idx),
                "f_flag mismatch at idx {}", idx);
            for j in 0..2 {
                assert!((d_serial.storage.center(idx, j) - d_parallel.storage.center(idx, j)).abs() < 1e-15,
                    "center mismatch at idx {}, dim {}", idx, j);
                assert_eq!(d_serial.storage.length(idx, j), d_parallel.storage.length(idx, j),
                    "length mismatch at idx {}, dim {}", idx, j);
            }
        }
    }

    #[test]
    fn test_parallel_vs_serial_sphere_3d() {
        let d_serial = make_initialized_sphere(3, false);
        let d_parallel = make_initialized_sphere(3, true);

        assert_eq!(d_serial.nfev, d_parallel.nfev);
        assert_eq!(d_serial.minf, d_parallel.minf);
        assert_eq!(d_serial.minpos, d_parallel.minpos);
        assert_eq!(d_serial.fmax, d_parallel.fmax);

        for idx in 1..=7 {
            assert_eq!(d_serial.storage.f_val(idx), d_parallel.storage.f_val(idx));
            assert_eq!(d_serial.storage.f_flag(idx), d_parallel.storage.f_flag(idx));
        }
    }

    #[test]
    fn test_parallel_vs_serial_sphere_5d() {
        let d_serial = make_initialized_sphere(5, false);
        let d_parallel = make_initialized_sphere(5, true);

        assert_eq!(d_serial.nfev, d_parallel.nfev);
        assert_eq!(d_serial.minf, d_parallel.minf);
        assert_eq!(d_serial.minpos, d_parallel.minpos);
        assert_eq!(d_serial.fmax, d_parallel.fmax);

        for idx in 1..=11 {
            assert_eq!(d_serial.storage.f_val(idx), d_parallel.storage.f_val(idx));
            assert_eq!(d_serial.storage.f_flag(idx), d_parallel.storage.f_flag(idx));
        }
    }

    #[test]
    fn test_parallel_vs_serial_rosenbrock_2d() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let rosenbrock = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };

        let make = |parallel: bool| -> Direct {
            let options = crate::types::DirectOptions {
                algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
                max_feval: 10000,
                max_iter: 100,
                parallel,
                ..Default::default()
            };
            let mut d = Direct::new(rosenbrock, &bounds, options).unwrap();
            d.initialize().unwrap();
            d
        };

        let d_serial = make(false);
        let d_parallel = make(true);

        assert_eq!(d_serial.nfev, d_parallel.nfev);
        assert!((d_serial.minf - d_parallel.minf).abs() < 1e-12);
        assert_eq!(d_serial.minpos, d_parallel.minpos);
    }

    #[test]
    fn test_parallel_second_iteration_2d() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let func = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };

        let run = |parallel: bool| -> (f64, usize, f64) {
            let options = crate::types::DirectOptions {
                algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
                max_feval: 10000,
                max_iter: 100,
                parallel,
                ..Default::default()
            };
            let mut d = Direct::new(func, &bounds, options).unwrap();
            d.initialize().unwrap();

            let (arrayi, maxi) = d.storage.get_longest_dims(2);
            let depth = d.storage.get_max_deep(2);
            let delta = d.storage.thirds[(depth + 1) as usize];
            let new_start = d.sample_points(2, &arrayi, delta).unwrap();
            d.evaluate_sample_points(new_start, maxi).unwrap();

            (d.minf, d.nfev, d.fmax)
        };

        let (minf_s, nfev_s, fmax_s) = run(false);
        let (minf_p, nfev_p, fmax_p) = run(true);

        assert_eq!(nfev_s, nfev_p);
        assert!((minf_s - minf_p).abs() < 1e-12);
        assert!((fmax_s - fmax_p).abs() < 1e-12);
    }

    #[test]
    fn test_force_stop_serial() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let func = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel: false,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        // Set force_stop before second sample evaluation
        d.force_stop.store(true, Ordering::Relaxed);

        let (arrayi, maxi) = d.storage.get_longest_dims(1);
        let delta = d.storage.thirds[2];
        let new_start = d.sample_points(1, &arrayi, delta).unwrap();
        d.evaluate_sample_points(new_start, maxi).unwrap();

        // All new points should have f_flag = -1.0 (force_stop)
        let mut pos = new_start;
        for _ in 0..(2 * maxi) {
            assert_eq!(d.storage.f_flag(pos), -1.0,
                "Expected f_flag=-1.0 for force_stopped point at idx {}", pos);
            pos = d.storage.point[pos] as usize;
        }
    }

    #[test]
    fn test_force_stop_parallel() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let func = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let options = crate::types::DirectOptions {
            algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            max_iter: 100,
            parallel: true,
            ..Default::default()
        };
        let mut d = Direct::new(func, &bounds, options).unwrap();
        d.initialize().unwrap();

        d.force_stop.store(true, Ordering::Relaxed);

        let (arrayi, maxi) = d.storage.get_longest_dims(1);
        let delta = d.storage.thirds[2];
        let new_start = d.sample_points(1, &arrayi, delta).unwrap();
        d.evaluate_sample_points(new_start, maxi).unwrap();

        let mut pos = new_start;
        for _ in 0..(2 * maxi) {
            assert_eq!(d.storage.f_flag(pos), -1.0,
                "Expected f_flag=-1.0 for force_stopped point at idx {}", pos);
            pos = d.storage.point[pos] as usize;
        }
    }

    #[test]
    fn test_parallel_with_infeasible_points() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let func = |x: &[f64]| -> f64 {
            let r: f64 = x.iter().map(|xi| xi * xi).sum();
            if r > 10.0 { f64::NAN } else { r }
        };

        let run = |parallel: bool| -> (f64, usize, i32, i32) {
            let options = crate::types::DirectOptions {
                algorithm: crate::types::DirectAlgorithm::GablonskyLocallyBiased,
                max_feval: 10000,
                max_iter: 100,
                parallel,
                ..Default::default()
            };
            let mut d = Direct::new(func, &bounds, options).unwrap();
            d.initialize().unwrap();
            (d.minf, d.nfev, d.ifeasible_f, d.iinfeasible)
        };

        let (minf_s, nfev_s, ifeas_s, iinfeas_s) = run(false);
        let (minf_p, nfev_p, ifeas_p, iinfeas_p) = run(true);

        assert_eq!(nfev_s, nfev_p);
        assert!((minf_s - minf_p).abs() < 1e-12);
        assert_eq!(ifeas_s, ifeas_p);
        assert_eq!(iinfeas_s, iinfeas_p);
    }

    // ──────────────────────────────────────────────────────────────────
    // divide_rectangle tests — matching direct_dirdivide_() in DIRsubrout.c
    // ──────────────────────────────────────────────────────────────────

    /// Helper: create a Direct instance with controlled f-values for divide testing.
    /// Sets up `count` child rects in a chain starting at `start_idx` with given
    /// f-values, then calls divide_rectangle and returns the resulting lengths.
    fn setup_divide_test(
        dim: usize,
        arrayi: &[usize],
        f_values_pairs: &[(f64, f64)], // (f_pos, f_neg) per dimension
        current_length: i32,
    ) -> Direct {
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); dim];
        let opts = DirectOptions {
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            max_feval: 10000,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, opts).unwrap();

        // Set up parent rect at index 1 with all lengths = current_length
        let sample = 1_usize;
        for j in 0..dim {
            d.storage.set_center(sample, j, 0.5);
            d.storage.set_length(sample, j, current_length);
        }
        d.storage.point[sample] = 0;
        d.storage.free = 2;

        // Allocate 2*maxi children manually
        let maxi = arrayi.len();
        let start = 2_usize;
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
        for (k, &(f_pos, f_neg)) in f_values_pairs.iter().enumerate() {
            let pos_idx = start + 2 * k;
            let neg_idx = start + 2 * k + 1;
            d.storage.set_f(pos_idx, f_pos, 0.0);
            d.storage.set_f(neg_idx, f_neg, 0.0);
        }

        // Call divide_rectangle
        d.divide_rectangle(start, current_length, sample, arrayi, maxi);
        d
    }

    #[test]
    fn test_divide_3d_unequal_w_sort() {
        // 3D, all dims longest: arrayi = [1, 2, 3]
        // f-values: dim1: (10.0, 12.0), dim2: (5.0, 8.0), dim3: (15.0, 20.0)
        // w = [min(10,12)=10, min(5,8)=5, min(15,20)=15]
        // Sort by w ascending: dim2(5) < dim1(10) < dim3(15)
        let arrayi = vec![1, 2, 3];
        let f_pairs = vec![(10.0, 12.0), (5.0, 8.0), (15.0, 20.0)];
        let d = setup_divide_test(3, &arrayi, &f_pairs, 0);

        let sample = 1;
        // Parent: all dims set to new_len=1
        assert_eq!(d.storage.length(sample, 0), 1);
        assert_eq!(d.storage.length(sample, 1), 1);
        assert_eq!(d.storage.length(sample, 2), 1);

        // Sorted dim_info:
        //   i=0: dim2 (dim_k=1), pos_child=4
        //   i=1: dim1 (dim_k=0), pos_child=2
        //   i=2: dim3 (dim_k=2), pos_child=6
        //
        // i=0 sets dim_k=1 in parent + ALL children
        // i=1 sets dim_k=0 in parent + children from i=1 (dim1, dim3 pairs)
        // i=2 sets dim_k=2 in parent + children from i=2 (dim3 pair only)

        // dim2 children (pos=4, neg=5) — sorted first (i=0):
        //   Only covered by i=0 → only dim1 set
        assert_eq!(d.storage.length(4, 0), 0); // dim0 NOT set
        assert_eq!(d.storage.length(4, 1), 1); // dim1 set at i=0
        assert_eq!(d.storage.length(4, 2), 0); // dim2 NOT set
        assert_eq!(d.storage.length(5, 0), 0);
        assert_eq!(d.storage.length(5, 1), 1);
        assert_eq!(d.storage.length(5, 2), 0);

        // dim1 children (pos=2, neg=3) — sorted second (i=1):
        //   Covered by i=0 (dim1) and i=1 (dim0) → dims 1 and 0 set
        assert_eq!(d.storage.length(2, 0), 1); // dim0 set at i=1
        assert_eq!(d.storage.length(2, 1), 1); // dim1 set at i=0
        assert_eq!(d.storage.length(2, 2), 0); // dim2 NOT set
        assert_eq!(d.storage.length(3, 0), 1);
        assert_eq!(d.storage.length(3, 1), 1);
        assert_eq!(d.storage.length(3, 2), 0);

        // dim3 children (pos=6, neg=7) — sorted last (i=2):
        //   Covered by all i=0,1,2 → all dims set
        assert_eq!(d.storage.length(6, 0), 1);
        assert_eq!(d.storage.length(6, 1), 1);
        assert_eq!(d.storage.length(6, 2), 1);
        assert_eq!(d.storage.length(7, 0), 1);
        assert_eq!(d.storage.length(7, 1), 1);
        assert_eq!(d.storage.length(7, 2), 1);
    }

    #[test]
    fn test_divide_2d_equal_w_stable_sort() {
        // 2D sphere: all w values equal → stable sort preserves original order
        // arrayi = [1, 2], both f-values identical
        let arrayi = vec![1, 2];
        let f_pairs = vec![(10.0, 10.0), (10.0, 10.0)];
        let d = setup_divide_test(2, &arrayi, &f_pairs, 0);

        let sample = 1;
        // Parent: both dims set to 1
        assert_eq!(d.storage.length(sample, 0), 1);
        assert_eq!(d.storage.length(sample, 1), 1);

        // Stable sort: dim1 stays first (i=0), dim2 stays second (i=1)
        // i=0 sets dim0 in parent + ALL children (dim_info[0..])
        // i=1 sets dim1 in parent + children from i=1 only (dim2 pair)

        // dim1 children (pos=2, neg=3) — sorted first, only get dim0 set
        assert_eq!(d.storage.length(2, 0), 1); // dim0 set at i=0
        assert_eq!(d.storage.length(2, 1), 0); // dim1 NOT set (i=1 doesn't cover)
        assert_eq!(d.storage.length(3, 0), 1);
        assert_eq!(d.storage.length(3, 1), 0);

        // dim2 children (pos=4, neg=5) — sorted second, get both dims
        assert_eq!(d.storage.length(4, 0), 1); // dim0 set at i=0 (covers all)
        assert_eq!(d.storage.length(4, 1), 1); // dim1 set at i=1
        assert_eq!(d.storage.length(5, 0), 1);
        assert_eq!(d.storage.length(5, 1), 1);
    }

    #[test]
    fn test_divide_1d_single_dim() {
        // 1D: only one dimension to divide
        let arrayi = vec![1];
        let f_pairs = vec![(7.0, 3.0)];
        let d = setup_divide_test(1, &arrayi, &f_pairs, 0);

        let sample = 1;
        // Parent: length = 1
        assert_eq!(d.storage.length(sample, 0), 1);
        // Both children: length = 1
        assert_eq!(d.storage.length(2, 0), 1);
        assert_eq!(d.storage.length(3, 0), 1);
    }

    #[test]
    fn test_divide_nonzero_current_length() {
        // Verify current_length is used properly (not always 0)
        let arrayi = vec![1, 2];
        let f_pairs = vec![(5.0, 8.0), (10.0, 12.0)];
        let d = setup_divide_test(2, &arrayi, &f_pairs, 3);

        let sample = 1;
        let new_len = 4; // current_length + 1
        // Parent: both dims set to new_len
        assert_eq!(d.storage.length(sample, 0), new_len);
        assert_eq!(d.storage.length(sample, 1), new_len);

        // w[dim1]=min(5,8)=5, w[dim2]=min(10,12)=10
        // Sorted: dim1(5) first, dim2(10) second
        // i=0 (dim1, dim_k=0): set in parent + ALL children
        // i=1 (dim2, dim_k=1): set in parent + dim2 children only

        // dim1 children (pos=2, neg=3): only dim0 set to new_len
        assert_eq!(d.storage.length(2, 0), new_len);
        assert_eq!(d.storage.length(2, 1), 3); // original length preserved
        assert_eq!(d.storage.length(3, 0), new_len);
        assert_eq!(d.storage.length(3, 1), 3);

        // dim2 children (pos=4, neg=5): both dims set to new_len
        assert_eq!(d.storage.length(4, 0), new_len); // set by i=0 (covers all)
        assert_eq!(d.storage.length(4, 1), new_len); // set by i=1
        assert_eq!(d.storage.length(5, 0), new_len);
        assert_eq!(d.storage.length(5, 1), new_len);
    }

    #[test]
    fn test_divide_parent_center_unchanged() {
        // Verify parent center is not modified by divide_rectangle
        let arrayi = vec![1, 2];
        let f_pairs = vec![(5.0, 8.0), (10.0, 12.0)];
        let d = setup_divide_test(2, &arrayi, &f_pairs, 0);

        let sample = 1;
        assert_eq!(d.storage.center(sample, 0), 0.5);
        assert_eq!(d.storage.center(sample, 1), 0.5);
    }

    #[test]
    fn test_divide_3d_two_dims_longest() {
        // 3D but only 2 dims are longest (e.g., after prior division)
        // arrayi = [1, 3] (dims 1 and 3 are longest, dim 2 already divided)
        let arrayi = vec![1, 3];
        let f_pairs = vec![(20.0, 15.0), (8.0, 12.0)];
        let d = setup_divide_test(3, &arrayi, &f_pairs, 1);

        let sample = 1;
        let new_len = 2; // current_length(1) + 1

        // w[dim1] = min(20,15) = 15, w[dim3] = min(8,12) = 8
        // Sort: dim3(8) < dim1(15)
        // Sorted: i=0: dim3 (dim_k=2, pos_child=4), i=1: dim1 (dim_k=0, pos_child=2)

        // i=0 (dim3, dim_k=2): set dim2 in parent + ALL children
        // i=1 (dim1, dim_k=0): set dim0 in parent + dim1 children only

        assert_eq!(d.storage.length(sample, 0), new_len);
        assert_eq!(d.storage.length(sample, 2), new_len);
        // dim 1 (not divided) keeps original length
        assert_eq!(d.storage.length(sample, 1), 1);

        // dim3 children (pos=4, neg=5) — sorted first (i=0):
        //   Only dim2 set
        assert_eq!(d.storage.length(4, 0), 1); // dim0 NOT set, keeps original
        assert_eq!(d.storage.length(4, 1), 1); // dim1 untouched
        assert_eq!(d.storage.length(4, 2), new_len); // dim2 set at i=0
        assert_eq!(d.storage.length(5, 0), 1);
        assert_eq!(d.storage.length(5, 1), 1);
        assert_eq!(d.storage.length(5, 2), new_len);

        // dim1 children (pos=2, neg=3) — sorted second (i=1):
        //   Covered by i=0 (dim2) and i=1 (dim0)
        assert_eq!(d.storage.length(2, 0), new_len); // dim0 set at i=1
        assert_eq!(d.storage.length(2, 1), 1); // dim1 untouched
        assert_eq!(d.storage.length(2, 2), new_len); // dim2 set at i=0
        assert_eq!(d.storage.length(3, 0), new_len);
        assert_eq!(d.storage.length(3, 1), 1);
        assert_eq!(d.storage.length(3, 2), new_len);
    }

    #[test]
    fn test_divide_w_uses_min_f() {
        // Verify w[j] = min(f_pos, f_neg) determines sort order
        // 2D: dim1 has f_pos=100, f_neg=1 → w=1
        //     dim2 has f_pos=2, f_neg=50 → w=2
        // Sort: dim1(1) < dim2(2), so dim1 is divided first
        let arrayi = vec![1, 2];
        let f_pairs = vec![(100.0, 1.0), (2.0, 50.0)];
        let d = setup_divide_test(2, &arrayi, &f_pairs, 0);

        // i=0 (dim1, dim_k=0): set dim0 in parent + ALL children
        // i=1 (dim2, dim_k=1): set dim1 in parent + dim2 children only

        // dim1 children (pos=2, neg=3): only dim0 set
        assert_eq!(d.storage.length(2, 0), 1); // dim0 set at i=0
        assert_eq!(d.storage.length(2, 1), 0); // dim1 NOT set
        assert_eq!(d.storage.length(3, 0), 1);
        assert_eq!(d.storage.length(3, 1), 0);

        // dim2 children (pos=4, neg=5): both dims set
        assert_eq!(d.storage.length(4, 0), 1); // dim0 set at i=0 (covers all)
        assert_eq!(d.storage.length(4, 1), 1); // dim1 set at i=1
        assert_eq!(d.storage.length(5, 0), 1);
        assert_eq!(d.storage.length(5, 1), 1);
    }

    #[test]
    fn test_divide_integrated_with_sample_points() {
        // End-to-end: initialize, pick a rect, sample, evaluate, divide
        // Verify results are consistent with the initialization test
        let mut d = make_initialized_sphere(2, false);

        // After initialization, rect 2 has lengths [1,0] (dim0 divided, dim1 not)
        // Get longest dims of rect 2: dim 2 (0-indexed: 1) has length 0 (longest)
        let (arrayi, maxi) = d.storage.get_longest_dims(2);
        assert_eq!(maxi, 1); // only 1 longest dim

        let depth = d.storage.get_max_deep(2);
        let delta = d.storage.thirds[(depth + 1) as usize];
        let new_start = d.sample_points(2, &arrayi, delta).unwrap();
        d.evaluate_sample_points(new_start, maxi).unwrap();
        d.divide_rectangle(new_start, depth, 2, &arrayi, maxi);

        // After dividing rect 2 along its longest dim:
        // New length = depth + 1
        let new_len = depth + 1;
        // Rect 2 (parent): dim with length 0 now = new_len
        let divided_dim = arrayi[0] - 1; // 0-based
        assert_eq!(d.storage.length(2, divided_dim), new_len);

        // Children should have same new length
        let pos_child = new_start;
        let neg_child = d.storage.point[pos_child] as usize;
        assert_eq!(d.storage.length(pos_child, divided_dim), new_len);
        assert_eq!(d.storage.length(neg_child, divided_dim), new_len);
    }

    #[test]
    fn test_divide_5d_all_dims() {
        // 5D with all dims longest, varying w values
        let arrayi = vec![1, 2, 3, 4, 5];
        let f_pairs = vec![
            (30.0, 25.0), // dim1: w=25
            (10.0, 15.0), // dim2: w=10
            (50.0, 45.0), // dim3: w=45
            (5.0, 8.0),   // dim4: w=5
            (20.0, 18.0), // dim5: w=18
        ];
        let d = setup_divide_test(5, &arrayi, &f_pairs, 0);

        let sample = 1;
        // Sort by w: dim4(5) < dim2(10) < dim5(18) < dim1(25) < dim3(45)
        // Sorted dim_info:
        //   i=0: dim4 (dim_k=3), pos_child=8
        //   i=1: dim2 (dim_k=1), pos_child=4
        //   i=2: dim5 (dim_k=4), pos_child=10
        //   i=3: dim1 (dim_k=0), pos_child=2
        //   i=4: dim3 (dim_k=2), pos_child=6

        // All parent dims set to 1
        for j in 0..5 {
            assert_eq!(d.storage.length(sample, j), 1);
        }

        // dim4 pair (8,9) — sorted first (i=0):
        //   Only covered by i=0 → only dim_k=3 set
        assert_eq!(d.storage.length(8, 3), 1); // dim3 set at i=0
        assert_eq!(d.storage.length(8, 0), 0);
        assert_eq!(d.storage.length(8, 1), 0);
        assert_eq!(d.storage.length(8, 2), 0);
        assert_eq!(d.storage.length(8, 4), 0);
        assert_eq!(d.storage.length(9, 3), 1);
        assert_eq!(d.storage.length(9, 0), 0);

        // dim2 pair (4,5) — sorted second (i=1):
        //   Covered by i=0 (dim3) and i=1 (dim1) → dims 3 and 1 set
        assert_eq!(d.storage.length(4, 3), 1); // dim3 set at i=0
        assert_eq!(d.storage.length(4, 1), 1); // dim1 set at i=1
        assert_eq!(d.storage.length(4, 0), 0);
        assert_eq!(d.storage.length(4, 2), 0);
        assert_eq!(d.storage.length(4, 4), 0);
        assert_eq!(d.storage.length(5, 3), 1);
        assert_eq!(d.storage.length(5, 1), 1);
        assert_eq!(d.storage.length(5, 0), 0);

        // dim5 pair (10,11) — sorted third (i=2):
        //   Covered by i=0 (dim3), i=1 (dim1), i=2 (dim4) → dims 3,1,4 set
        assert_eq!(d.storage.length(10, 3), 1);
        assert_eq!(d.storage.length(10, 1), 1);
        assert_eq!(d.storage.length(10, 4), 1);
        assert_eq!(d.storage.length(10, 0), 0);
        assert_eq!(d.storage.length(10, 2), 0);

        // dim1 pair (2,3) — sorted fourth (i=3):
        //   Covered by i=0,1,2,3 → dims 3,1,4,0 set
        assert_eq!(d.storage.length(2, 3), 1);
        assert_eq!(d.storage.length(2, 1), 1);
        assert_eq!(d.storage.length(2, 4), 1);
        assert_eq!(d.storage.length(2, 0), 1);
        assert_eq!(d.storage.length(2, 2), 0); // dim2 NOT set

        // dim3 pair (6,7) — sorted last (i=4):
        //   Covered by all i=0..4 → all 5 dims set
        for j in 0..5 {
            assert_eq!(d.storage.length(6, j), 1, "dim3+ dim{} should be 1", j);
            assert_eq!(d.storage.length(7, j), 1, "dim3- dim{} should be 1", j);
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Integration tests — minimize() main loop
    // ────────────────────────────────────────────────────────────────

    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mut sum = 10.0 * n;
        for &xi in x {
            sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
        }
        sum
    }

    #[test]
    fn test_minimize_sphere_2d_gablonsky() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 200,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success, "Optimization should succeed: {}", result.message);
        assert_eq!(result.return_code, crate::error::DirectReturnCode::MaxFevalExceeded);
        assert!(result.fun < 1.0, "Should find a good minimum, got f={}", result.fun);
        // Note: nfev may slightly exceed max_feval since the check happens after each iteration
        assert!(result.nfev >= 100, "Should do significant evaluations");
        assert!(result.nit >= 1, "Should do at least 1 iteration");
        // Sphere minimum is at origin
        for xi in &result.x {
            assert!(xi.abs() < 2.0, "x should be near origin, got {:?}", result.x);
        }
    }

    #[test]
    fn test_minimize_sphere_2d_original() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 200,
            algorithm: DirectAlgorithm::GablonskyOriginal,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success, "Optimization should succeed: {}", result.message);
        assert!(result.fun < 1.0, "Should find a good minimum, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_sphere_2d_maxiter() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_iter: 10,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert_eq!(result.return_code, crate::error::DirectReturnCode::MaxIterExceeded);
        assert!(result.nit <= 10, "Should respect maxiter, got nit={}", result.nit);
    }

    #[test]
    fn test_minimize_sphere_3d_gablonsky() {
        let bounds = vec![(-5.0, 5.0); 3];
        let options = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        assert!(result.fun < 1.0, "3D sphere should converge, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_sphere_1d_gablonsky() {
        let bounds = vec![(-5.0, 5.0)];
        let options = DirectOptions {
            max_feval: 100,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        assert!(result.fun < 0.1, "1D sphere should converge well, got f={}", result.fun);
        assert!(result.x[0].abs() < 1.0, "x should be near 0, got x={}", result.x[0]);
    }

    #[test]
    fn test_minimize_sphere_fglobal_termination() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 10000,
            fglobal: 0.0,
            fglobal_reltol: 0.01,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert_eq!(result.return_code, crate::error::DirectReturnCode::GlobalFound);
        assert!(result.fun < 0.01, "Should converge to near-global, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_rosenbrock_2d_gablonsky() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(rosenbrock, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        // Rosenbrock is harder, but should make progress
        assert!(result.fun < 100.0, "Rosenbrock should make progress, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_rosenbrock_2d_original() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::GablonskyOriginal,
            ..Default::default()
        };
        let mut d = Direct::new(rosenbrock, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        assert!(result.fun < 100.0, "Rosenbrock should make progress, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_rastrigin_2d_gablonsky() {
        let bounds = vec![(-5.12, 5.12); 2];
        let options = DirectOptions {
            max_feval: 1000,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(rastrigin, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        // Rastrigin global minimum is 0 at origin
        assert!(result.fun < 10.0, "Rastrigin should make progress, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_callback_force_stop() {
        // Use shifted sphere so minimum is NOT at domain center
        let shifted_sphere = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 10000,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(shifted_sphere, &bounds, options).unwrap();

        // Stop when f < 1.0
        let cb: Box<CallbackFn> = Box::new(|_x, fun, _nfev, _nit| fun < 1.0);
        let result = d.minimize(Some(&*cb)).unwrap();

        assert_eq!(result.return_code, crate::error::DirectReturnCode::ForcedStop);
        assert!(result.fun < 1.0, "Should have found f < 1.0 before stopping");
    }

    #[test]
    fn test_minimize_force_stop_atomic() {
        use std::sync::atomic::Ordering;

        let bounds = vec![(-5.0, 5.0); 2];
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter2 = Arc::clone(&counter);

        let mut d = Direct::new(
            move |x: &[f64]| {
                counter2.fetch_add(1, Ordering::Relaxed);
                x.iter().map(|xi| xi * xi).sum()
            },
            &bounds,
            DirectOptions {
                max_feval: 10000,
                algorithm: DirectAlgorithm::GablonskyLocallyBiased,
                ..Default::default()
            },
        ).unwrap();

        // Set force_stop after construction
        let stop_flag = Arc::clone(&d.force_stop);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            stop_flag.store(true, Ordering::Relaxed);
        });

        let result = d.minimize(None).unwrap();
        assert_eq!(result.return_code, crate::error::DirectReturnCode::ForcedStop);
    }

    #[test]
    fn test_minimize_asymmetric_bounds() {
        let bounds = vec![(2.0, 10.0), (-3.0, 7.0)];
        let options = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        // Minimum of sphere in [2,10]×[-3,7] is at (2, 0) with f=4
        assert!(result.fun < 10.0, "Should find near-boundary minimum, got f={}", result.fun);
        assert!(result.x[0] >= 2.0 && result.x[0] <= 10.0, "x[0] should be in bounds");
        assert!(result.x[1] >= -3.0 && result.x[1] <= 7.0, "x[1] should be in bounds");
    }

    #[test]
    fn test_minimize_jones_eps_update() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 500,
            magic_eps: -1e-4, // Negative → Jones update formula
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        assert!(result.fun < 1.0, "Should converge with Jones eps update");
    }

    #[test]
    fn test_minimize_volume_tol_termination() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 100000,
            volume_reltol: 1e-2,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        // Should stop due to volume tolerance
        assert_eq!(result.return_code, crate::error::DirectReturnCode::VolTol);
    }

    #[test]
    fn test_minimize_parallel_vs_serial_sphere() {
        // Run same problem with parallel=false and parallel=true
        let bounds = vec![(-5.0, 5.0); 2];

        // Serial
        let options_serial = DirectOptions {
            max_feval: 200,
            parallel: false,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d_serial = Direct::new(sphere, &bounds, options_serial).unwrap();
        let result_serial = d_serial.minimize(None).unwrap();

        // Parallel
        let options_parallel = DirectOptions {
            max_feval: 200,
            parallel: true,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d_parallel = Direct::new(sphere, &bounds, options_parallel).unwrap();
        let result_parallel = d_parallel.minimize(None).unwrap();

        // Both should produce valid results (may differ in exact values due to parallel eval order)
        assert!(result_serial.success);
        assert!(result_parallel.success);
        assert!(result_serial.fun < 1.0);
        assert!(result_parallel.fun < 1.0);
    }

    #[test]
    fn test_minimize_5d_sphere() {
        let bounds = vec![(-5.0, 5.0); 5];
        let options = DirectOptions {
            max_feval: 2000,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        assert!(result.success);
        // 5D requires more evaluations, but should make progress
        assert!(result.fun < 5.0, "5D sphere should converge, got f={}", result.fun);
    }

    #[test]
    fn test_minimize_result_fields() {
        let bounds = vec![(-5.0, 5.0); 2];
        let options = DirectOptions {
            max_feval: 100,
            algorithm: DirectAlgorithm::GablonskyLocallyBiased,
            ..Default::default()
        };
        let mut d = Direct::new(sphere, &bounds, options).unwrap();
        let result = d.minimize(None).unwrap();

        // Verify result fields are correctly populated
        assert_eq!(result.x.len(), 2);
        assert!(result.fun.is_finite());
        assert!(result.nfev > 0);
        assert!(result.nit > 0);
        assert!(!result.message.is_empty());
    }
}
