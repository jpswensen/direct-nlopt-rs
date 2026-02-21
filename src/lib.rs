//! # DIRECT-NLOPT-RS: NLOPT DIRECT/DIRECT-L Global Optimization in Rust
//!
//! A faithful Rust-native implementation of the NLOPT DIRECT (DIviding RECTangles)
//! and DIRECT-L global optimization algorithms, with rayon parallelization where
//! appropriate.
//!
//! ## Overview
//!
//! This crate is a 100% faithful port of the NLOPT implementation of the DIRECT
//! algorithm family. The NLOPT implementation includes two separate codebases:
//!
//! 1. **Gablonsky Fortran→C translation** (`algs/direct/`): A translation of
//!    Gablonsky's original Fortran DIRECT implementation to C, done by Steven G.
//!    Johnson. Supports `DIRECT_ORIGINAL` (Jones 1993) and `DIRECT_GABLONSKY`
//!    (Gablonsky 2001, locally-biased).
//!
//! 2. **SGJ C re-implementation** (`algs/cdirect/`): A from-scratch C implementation
//!    by Steven G. Johnson using red-black trees. Supports DIRECT, DIRECT-L,
//!    randomized variants, and a hybrid DIRECT + local optimizer.
//!
//! This Rust crate implements BOTH codepaths faithfully, ensuring identical results
//! (without parallelization) to the original NLOPT C code.
//!
//! ## Algorithm Variants
//!
//! - **DIRECT (Original, Jones 1993)**: `DirectAlgorithm::Original`
//! - **DIRECT-L (Gablonsky 2001)**: `DirectAlgorithm::LocallyBiased`
//! - **DIRECT Randomized**: `DirectAlgorithm::Randomized`
//! - **DIRECT-L Randomized**: `DirectAlgorithm::LocallyBiasedRandomized`
//!
//! ## References
//!
//! - Jones, D.R., Perttunen, C.D. & Stuckman, B.E. "Lipschitzian optimization
//!   without the Lipschitz constant." J Optim Theory Appl 79, 157–181 (1993).
//! - Gablonsky, J.M. & Kelley, C.T. "A Locally-Biased form of the DIRECT Algorithm."
//!   Journal of Global Optimization 21, 27–37 (2001).
//! - NLOPT: <https://github.com/stevengj/nlopt>

pub mod cdirect;
pub mod direct;
pub mod error;
pub mod storage;
pub mod types;

// Re-export main types
pub use cdirect::CDirect;
pub use error::{DirectError, DirectReturnCode, Result};
pub use types::{
    Bounds, CallbackFn, DirectAlgorithm, DirectOptions, DirectResult, ObjectiveFn,
    DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};

// ──────────────────────────────────────────────────────────────────────────────
// High-level public API matching NLOPT's direct_optimize() wrapper
// ──────────────────────────────────────────────────────────────────────────────

/// High-level optimization function matching NLOPT's `direct_optimize()` wrapper.
///
/// Dispatches to the Gablonsky Fortran→C translation or the SGJ cdirect.c
/// re-implementation based on the algorithm variant in `options`.
///
/// # NLOPT C Correspondence
///
/// Matches `direct_optimize()` in `direct_wrap.c`:
/// - Tolerance conversion (ratios → percentages) is handled internally
/// - `DIRECT_UNKNOWN_FGLOBAL` handling matches NLOPT's default
/// - Algorithm dispatch: Gablonsky variants use `Direct`, others use `CDirect`
///
/// # Arguments
/// * `func` - Objective function: `f(x) -> f64`. Return `f64::NAN` for infeasible.
/// * `bounds` - Lower and upper bounds for each dimension.
/// * `options` - Optimizer configuration including algorithm variant.
///
/// # Example
/// ```
/// use direct_nlopt::{direct_optimize, DirectOptions, DirectAlgorithm};
///
/// let result = direct_optimize(
///     |x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
///     &vec![(-5.0, 5.0), (-5.0, 5.0)],
///     DirectOptions {
///         max_feval: 500,
///         algorithm: DirectAlgorithm::LocallyBiased,
///         ..Default::default()
///     },
/// ).unwrap();
///
/// assert!(result.fun < 1e-4);
/// ```
pub fn direct_optimize(
    func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    bounds: &Bounds,
    options: DirectOptions,
) -> Result<DirectResult> {
    DirectBuilder::new(func, bounds.to_vec())
        .options(options)
        .minimize()
}

/// Builder for configuring and running DIRECT optimization.
///
/// Provides a fluent API for setting options, algorithm, callback, and running
/// the optimization. Dispatches to the appropriate backend (Gablonsky or CDirect)
/// based on the selected algorithm.
///
/// # Example
/// ```
/// use direct_nlopt::{DirectBuilder, DirectAlgorithm};
///
/// let result = DirectBuilder::new(
///     |x: &[f64]| x[0] * x[0] + x[1] * x[1],
///     vec![(-5.0, 5.0), (-5.0, 5.0)],
/// )
/// .algorithm(DirectAlgorithm::LocallyBiased)
/// .max_feval(500)
/// .minimize()
/// .unwrap();
///
/// assert!(result.fun < 1e-4);
/// ```
pub struct DirectBuilder {
    func: Box<ObjectiveFn>,
    bounds: Bounds,
    opts: DirectOptions,
    callback: Option<Box<CallbackFn>>,
}

impl DirectBuilder {
    /// Create a new builder with an objective function and bounds.
    pub fn new(
        func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
        bounds: Bounds,
    ) -> Self {
        Self {
            func: Box::new(func),
            bounds,
            opts: DirectOptions::default(),
            callback: None,
        }
    }

    /// Set all options at once.
    pub fn options(mut self, opts: DirectOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Set the algorithm variant.
    pub fn algorithm(mut self, alg: DirectAlgorithm) -> Self {
        self.opts.algorithm = alg;
        self
    }

    /// Set maximum number of function evaluations. 0 = unlimited.
    pub fn max_feval(mut self, n: usize) -> Self {
        self.opts.max_feval = n;
        self
    }

    /// Set maximum number of iterations. 0 = unlimited.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.opts.max_iter = n;
        self
    }

    /// Set maximum wall-clock time in seconds. 0.0 = unlimited.
    pub fn max_time(mut self, t: f64) -> Self {
        self.opts.max_time = t;
        self
    }

    /// Set Jones' epsilon parameter for the potentially-optimal test.
    pub fn magic_eps(mut self, eps: f64) -> Self {
        self.opts.magic_eps = eps;
        self
    }

    /// Set the absolute epsilon for the potentially-optimal test.
    pub fn magic_eps_abs(mut self, eps: f64) -> Self {
        self.opts.magic_eps_abs = eps;
        self
    }

    /// Set the volume relative tolerance.
    pub fn volume_reltol(mut self, tol: f64) -> Self {
        self.opts.volume_reltol = tol;
        self
    }

    /// Set the sigma relative tolerance.
    pub fn sigma_reltol(mut self, tol: f64) -> Self {
        self.opts.sigma_reltol = tol;
        self
    }

    /// Set the known global minimum and relative tolerance.
    pub fn fglobal(mut self, fglobal: f64, reltol: f64) -> Self {
        self.opts.fglobal = fglobal;
        self.opts.fglobal_reltol = reltol;
        self
    }

    /// Enable or disable parallel function evaluation.
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.opts.parallel = enabled;
        self
    }

    /// Set a callback for progress monitoring and early stopping.
    ///
    /// The callback receives `(x_best, f_best, nfev, nit)` and returns
    /// `true` to force stop.
    pub fn with_callback(
        mut self,
        callback: impl Fn(&[f64], f64, usize, usize) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }

    /// Run the optimization and return the result.
    ///
    /// Dispatches to the Gablonsky translation (`Direct`) for `GablonskyOriginal`
    /// and `GablonskyLocallyBiased` variants, or to the SGJ re-implementation
    /// (`CDirect`) for all other variants.
    pub fn minimize(self) -> Result<DirectResult> {
        if self.opts.algorithm.is_gablonsky_translation() {
            self.minimize_gablonsky()
        } else {
            self.minimize_cdirect()
        }
    }

    /// Gablonsky Fortran→C translation path.
    fn minimize_gablonsky(self) -> Result<DirectResult> {
        let mut solver = direct::Direct::new(self.func, &self.bounds, self.opts)?;
        let cb = self.callback;
        match cb {
            Some(callback) => solver.minimize(Some(callback.as_ref())),
            None => solver.minimize(None),
        }
    }

    /// SGJ cdirect.c re-implementation path.
    fn minimize_cdirect(self) -> Result<DirectResult> {
        let mut solver = CDirect::new(self.func, self.bounds, self.opts);
        if let Some(callback) = self.callback {
            solver = solver.with_callback(callback);
        }
        solver.minimize()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ──────────────────────────────────────────────────────────────────────────────

/// Minimize using DIRECT-L (locally biased, SGJ cdirect implementation).
///
/// This is the most commonly used variant, equivalent to NLOPT's `NLOPT_GN_DIRECT_L`.
///
/// # Arguments
/// * `func` - Objective function
/// * `bounds` - Variable bounds
/// * `max_feval` - Maximum function evaluations (0 = unlimited)
pub fn minimize(
    func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    bounds: &Bounds,
    max_feval: usize,
) -> Result<DirectResult> {
    DirectBuilder::new(func, bounds.to_vec())
        .algorithm(DirectAlgorithm::LocallyBiased)
        .max_feval(max_feval)
        .minimize()
}

/// Minimize using DIRECT-L (locally biased, SGJ cdirect implementation).
///
/// Alias for [`minimize`] — the default and recommended variant.
pub fn minimize_l(
    func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    bounds: &Bounds,
    max_feval: usize,
) -> Result<DirectResult> {
    minimize(func, bounds, max_feval)
}

/// Minimize using DIRECT-L with randomized tie-breaking.
///
/// Equivalent to NLOPT's `NLOPT_GN_DIRECT_L_RAND`.
pub fn minimize_randomized(
    func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    bounds: &Bounds,
    max_feval: usize,
) -> Result<DirectResult> {
    DirectBuilder::new(func, bounds.to_vec())
        .algorithm(DirectAlgorithm::Randomized)
        .max_feval(max_feval)
        .minimize()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod api_tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    // ── direct_optimize() tests ──

    #[test]
    fn test_direct_optimize_sphere_default() {
        let result = direct_optimize(
            sphere,
            &vec![(-5.0, 5.0), (-5.0, 5.0)],
            DirectOptions {
                max_feval: 500,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
        assert!(result.nfev <= 500);
    }

    #[test]
    fn test_direct_optimize_gablonsky_original() {
        let result = direct_optimize(
            sphere,
            &vec![(-5.0, 5.0), (-5.0, 5.0)],
            DirectOptions {
                max_feval: 500,
                algorithm: DirectAlgorithm::GablonskyOriginal,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
    }

    #[test]
    fn test_direct_optimize_gablonsky_locally_biased() {
        let result = direct_optimize(
            sphere,
            &vec![(-5.0, 5.0), (-5.0, 5.0)],
            DirectOptions {
                max_feval: 500,
                algorithm: DirectAlgorithm::GablonskyLocallyBiased,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-4, "fun = {}", result.fun);
    }

    #[test]
    fn test_direct_optimize_cdirect_variants() {
        for alg in [
            DirectAlgorithm::Original,
            DirectAlgorithm::LocallyBiased,
            DirectAlgorithm::OriginalUnscaled,
            DirectAlgorithm::LocallyBiasedUnscaled,
        ] {
            let result = direct_optimize(
                sphere,
                &vec![(-5.0, 5.0), (-5.0, 5.0)],
                DirectOptions {
                    max_feval: 500,
                    algorithm: alg,
                    ..Default::default()
                },
            )
            .unwrap();

            assert!(result.success, "alg={:?} failed: {:?}", alg, result.return_code);
            assert!(result.fun < 1e-2, "alg={:?} fun={}", alg, result.fun);
        }
    }

    // ── Builder pattern tests ──

    #[test]
    fn test_builder_basic() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .max_feval(500)
            .minimize()
            .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn test_builder_with_algorithm() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
            .max_feval(500)
            .minimize()
            .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_builder_with_options() {
        let opts = DirectOptions {
            max_feval: 500,
            algorithm: DirectAlgorithm::LocallyBiased,
            ..Default::default()
        };

        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .options(opts)
            .minimize()
            .unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_builder_all_setters() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .algorithm(DirectAlgorithm::LocallyBiased)
            .max_feval(500)
            .max_iter(200)
            .max_time(30.0)
            .magic_eps(1e-4)
            .magic_eps_abs(0.0)
            .volume_reltol(0.0)
            .sigma_reltol(-1.0)
            .fglobal(DIRECT_UNKNOWN_FGLOBAL, 0.0)
            .parallel(false)
            .minimize()
            .unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_builder_with_callback() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .max_feval(500)
            .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
            .with_callback(|_x, _f, _nfev, _nit| false) // never stop
            .minimize()
            .unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_builder_callback_force_stop() {
        // Use CDirect which reliably calls the callback
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .max_feval(5000)
            .algorithm(DirectAlgorithm::LocallyBiased)
            .with_callback(|_x, f, _nfev, _nit| f < 5.0)
            .minimize()
            .unwrap();

        assert_eq!(result.return_code, DirectReturnCode::ForcedStop);
        assert!(result.fun < 5.0);
    }

    #[test]
    fn test_builder_callback_cdirect_force_stop() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .max_feval(5000)
            .algorithm(DirectAlgorithm::LocallyBiased)
            .with_callback(|_x, f, _nfev, _nit| f < 1.0)
            .minimize()
            .unwrap();

        // CDirect may use ForcedStop or MaxFevalExceeded depending on when callback fires
        assert!(result.fun < 1.0 || result.success);
    }

    #[test]
    fn test_builder_fglobal() {
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
            .max_feval(5000)
            .fglobal(0.0, 1e-4)
            .minimize()
            .unwrap();

        assert!(result.success);
        assert!(result.fun < 1e-2, "fun = {}", result.fun);
    }

    // ── Convenience function tests ──

    #[test]
    fn test_minimize_sphere() {
        let result = minimize(sphere, &vec![(-5.0, 5.0), (-5.0, 5.0)], 500).unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn test_minimize_l_sphere() {
        let result = minimize_l(sphere, &vec![(-5.0, 5.0), (-5.0, 5.0)], 500).unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-2);
    }

    #[test]
    fn test_minimize_randomized_sphere() {
        let result = minimize_randomized(sphere, &vec![(-5.0, 5.0), (-5.0, 5.0)], 500).unwrap();
        assert!(result.success);
        assert!(result.fun < 1e-1);
    }

    #[test]
    fn test_minimize_rosenbrock() {
        let result = minimize(rosenbrock, &vec![(-5.0, 5.0), (-5.0, 5.0)], 2000).unwrap();
        assert!(result.success);
        assert!(result.fun < 1.0, "fun = {}", result.fun);
    }

    // ── Error handling tests ──

    #[test]
    fn test_invalid_bounds() {
        let result = direct_optimize(
            sphere,
            &vec![(5.0, -5.0)], // lower > upper
            DirectOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_bounds() {
        let result = direct_optimize(sphere, &vec![], DirectOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_bounds() {
        let result = DirectBuilder::new(sphere, vec![(5.0, -5.0)])
            .max_feval(100)
            .minimize();
        assert!(result.is_err());
    }

    // ── DIRECT_UNKNOWN_FGLOBAL handling tests ──

    #[test]
    fn test_unknown_fglobal_default() {
        // When fglobal is unknown, fglobal_reltol should be ignored
        let result = DirectBuilder::new(sphere, vec![(-5.0, 5.0), (-5.0, 5.0)])
            .max_feval(200)
            .fglobal(DIRECT_UNKNOWN_FGLOBAL, 1e-4)
            .minimize()
            .unwrap();

        // Should not terminate with GlobalFound since fglobal is unknown
        assert_ne!(result.return_code, DirectReturnCode::GlobalFound);
    }

    // ── Display tests ──

    #[test]
    fn test_result_display() {
        let result = DirectResult::new(
            vec![0.001, -0.002],
            1.5e-6,
            350,
            25,
            DirectReturnCode::MaxFevalExceeded,
        );
        let display = format!("{}", result);
        assert!(display.contains("success: true"));
        assert!(display.contains("nfev: 350"));
        assert!(display.contains("nit: 25"));
        assert!(display.contains("1.500000000000000e-6"));
    }

    // ── Dispatch consistency test ──

    #[test]
    fn test_dispatch_gablonsky_vs_cdirect() {
        // Verify that Gablonsky and CDirect variants both work through the unified API
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let gab_result = direct_optimize(
            sphere,
            &bounds,
            DirectOptions {
                max_feval: 500,
                algorithm: DirectAlgorithm::GablonskyLocallyBiased,
                ..Default::default()
            },
        )
        .unwrap();

        let cd_result = direct_optimize(
            sphere,
            &bounds,
            DirectOptions {
                max_feval: 500,
                algorithm: DirectAlgorithm::LocallyBiased,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(gab_result.success);
        assert!(cd_result.success);
        // Both should find near-optimal for sphere
        assert!(gab_result.fun < 1e-2);
        assert!(cd_result.fun < 1e-2);
    }
}
