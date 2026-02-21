//! Core DIRECT algorithm: preprocessing, scaling, and objective function evaluation.
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

use std::sync::Arc;

use crate::error::{DirectError, Result};
use crate::storage::RectangleStorage;
use crate::types::{Bounds, DirectOptions, ObjectiveFn, DIRECT_UNKNOWN_FGLOBAL};

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
}
