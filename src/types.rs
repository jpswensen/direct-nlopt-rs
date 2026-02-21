//! Core type definitions for the DIRECT-NLOPT-RS implementation.
//!
//! Maps to NLOPT's API types, algorithm variants, options, and result structures.

use std::fmt;

use crate::error::DirectReturnCode;

// ──────────────────────────────────────────────────────────────────────────────
// Algorithm Variants
// ──────────────────────────────────────────────────────────────────────────────

/// DIRECT algorithm variant selection.
///
/// NLOPT provides two separate DIRECT implementations:
/// - **ORIG_DIRECT / ORIG_DIRECT_L**: Gablonsky Fortran→C translation using SoA + linked lists
/// - **DIRECT / DIRECT_L / DIRECT_L_RAND (+ NOSCAL)**: SGJ C re-implementation using red-black trees
///
/// Each variant corresponds to a specific NLOPT algorithm enum value and internal
/// configuration (algmethod for Gablonsky, which_alg for SGJ).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DirectAlgorithm {
    /// Jones' original DIRECT (1993) via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT`, cdirect `which_alg=0`.
    Original,

    /// DIRECT-L (Gablonsky 2001, locally-biased) via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT_L`, cdirect `which_alg=13`.
    #[default]
    LocallyBiased,

    /// DIRECT-L Randomized via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT_L_RAND`, cdirect `which_alg=16`.
    Randomized,

    /// Jones' original DIRECT, unscaled via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT_NOSCAL`, cdirect_unscaled `which_alg=0`.
    OriginalUnscaled,

    /// DIRECT-L, unscaled via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT_L_NOSCAL`, cdirect_unscaled `which_alg=13`.
    LocallyBiasedUnscaled,

    /// DIRECT-L Randomized, unscaled via SGJ cdirect.c.
    /// NLOPT: `NLOPT_GN_DIRECT_L_RAND_NOSCAL`, cdirect_unscaled `which_alg=16`.
    LocallyBiasedRandomizedUnscaled,

    /// Jones' original DIRECT via Gablonsky Fortran→C translation.
    /// NLOPT: `NLOPT_GN_ORIG_DIRECT`, `algmethod=0` (DIRECT_ORIGINAL).
    GablonskyOriginal,

    /// DIRECT-L via Gablonsky Fortran→C translation.
    /// NLOPT: `NLOPT_GN_ORIG_DIRECT_L`, `algmethod=1` (DIRECT_GABLONSKY).
    GablonskyLocallyBiased,
}

impl DirectAlgorithm {
    /// Returns the `which_alg` integer used by NLOPT's cdirect.c implementation.
    ///
    /// Encoding: `which_alg = which_diam + 3*which_div + 9*which_opt`
    /// - `which_diam`: 0=Jones Euclidean, 1=Gablonsky max-side
    /// - `which_div`: 0=Jones all longest, 1=Gablonsky, 2=random
    /// - `which_opt`: 0=all hull pts, 1=one per diameter, 2=randomized
    ///
    /// Returns `None` for Gablonsky translation variants (they use `algmethod` instead).
    pub fn which_alg(&self) -> Option<i32> {
        match self {
            Self::Original | Self::OriginalUnscaled => Some(0),
            Self::LocallyBiased | Self::LocallyBiasedUnscaled => Some(13),
            Self::Randomized | Self::LocallyBiasedRandomizedUnscaled => Some(16),
            Self::GablonskyOriginal | Self::GablonskyLocallyBiased => None,
        }
    }

    /// Returns the `algmethod` flag used by NLOPT's Gablonsky translation (DIRect.c).
    ///
    /// - `0` = DIRECT_ORIGINAL (Jones 1993)
    /// - `1` = DIRECT_GABLONSKY (Gablonsky 2001)
    ///
    /// Returns `None` for SGJ cdirect variants (they use `which_alg` instead).
    pub fn algmethod(&self) -> Option<i32> {
        match self {
            Self::GablonskyOriginal => Some(0),
            Self::GablonskyLocallyBiased => Some(1),
            _ => None,
        }
    }

    /// Returns true if this variant uses the Gablonsky Fortran→C translation.
    pub fn is_gablonsky_translation(&self) -> bool {
        matches!(self, Self::GablonskyOriginal | Self::GablonskyLocallyBiased)
    }

    /// Returns true if this variant uses the SGJ cdirect.c re-implementation.
    pub fn is_cdirect(&self) -> bool {
        !self.is_gablonsky_translation()
    }

    /// Returns true if this variant uses unscaled coordinates (cdirect_unscaled).
    pub fn is_unscaled(&self) -> bool {
        matches!(
            self,
            Self::OriginalUnscaled
                | Self::LocallyBiasedUnscaled
                | Self::LocallyBiasedRandomizedUnscaled
        )
    }

    /// Returns the NLOPT algorithm name string.
    pub fn nlopt_name(&self) -> &'static str {
        match self {
            Self::Original => "GN_DIRECT",
            Self::LocallyBiased => "GN_DIRECT_L",
            Self::Randomized => "GN_DIRECT_L_RAND",
            Self::OriginalUnscaled => "GN_DIRECT_NOSCAL",
            Self::LocallyBiasedUnscaled => "GN_DIRECT_L_NOSCAL",
            Self::LocallyBiasedRandomizedUnscaled => "GN_DIRECT_L_RAND_NOSCAL",
            Self::GablonskyOriginal => "GN_ORIG_DIRECT",
            Self::GablonskyLocallyBiased => "GN_ORIG_DIRECT_L",
        }
    }
}

impl fmt::Display for DirectAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.nlopt_name())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Type Aliases
// ──────────────────────────────────────────────────────────────────────────────

/// Bounds for each dimension: `Vec<(lower, upper)>`.
pub type Bounds = Vec<(f64, f64)>;

/// Objective function signature.
///
/// Matches NLOPT's `direct_objective_func`:
/// - `x`: current point (dimension n)
/// - Returns: function value (return `f64::NAN` or `f64::INFINITY` for infeasible points)
pub type ObjectiveFn = dyn Fn(&[f64]) -> f64 + Send + Sync;

/// Callback function for progress monitoring and early stopping.
///
/// Called after each iteration with the current best point and value.
/// - `x`: current best point
/// - `fun`: current best function value
/// - `nfev`: total function evaluations so far
/// - `nit`: total iterations so far
/// - Returns: `true` to force stop, `false` to continue
pub type CallbackFn = dyn Fn(&[f64], f64, usize, usize) -> bool + Send + Sync;

// ──────────────────────────────────────────────────────────────────────────────
// Options
// ──────────────────────────────────────────────────────────────────────────────

/// The NLOPT sentinel value for unknown global minimum: `-HUGE_VAL` in C.
pub const DIRECT_UNKNOWN_FGLOBAL: f64 = f64::NEG_INFINITY;

/// Default relative tolerance for fglobal (0.0 means exact match required).
pub const DIRECT_UNKNOWN_FGLOBAL_RELTOL: f64 = 0.0;

/// Configuration options for the DIRECT optimizer.
///
/// Matches the parameters accepted by NLOPT's `direct_optimize()` and `cdirect()`.
#[derive(Debug, Clone)]
pub struct DirectOptions {
    /// Maximum number of function evaluations. 0 means no limit.
    pub max_feval: usize,

    /// Maximum number of iterations. 0 means no limit.
    /// For Gablonsky translation, -1 means unlimited (matching NLOPT).
    pub max_iter: usize,

    /// Maximum wall-clock time in seconds. 0.0 means no limit.
    pub max_time: f64,

    /// Jones' epsilon parameter for the potentially-optimal test.
    /// Controls the balance between global and local search.
    /// Default: 0.0 (from `nlopt_get_param(opt, "magic_eps", 0.0)`).
    pub magic_eps: f64,

    /// Absolute epsilon for the potentially-optimal test.
    /// Default: 0.0 (from `nlopt_get_param(opt, "magic_eps_abs", 0.0)`).
    pub magic_eps_abs: f64,

    /// Volume relative tolerance. Optimization stops when the volume of the
    /// smallest rectangle is less than this fraction of the original volume.
    /// Default: 0.0 (disabled). Internally mapped to -1.0 when <= 0 by NLOPT.
    pub volume_reltol: f64,

    /// Sigma relative tolerance. Optimization stops when the measure of the
    /// potentially-optimal rectangles is below this threshold.
    /// Default: -1.0 (disabled), matching `nlopt_get_param(opt, "sigma_reltol", -1.0)`.
    pub sigma_reltol: f64,

    /// Known global minimum value. Set to `DIRECT_UNKNOWN_FGLOBAL` if unknown.
    /// When known, optimization stops when `f <= fglobal + fglobal_reltol * |fglobal|`.
    pub fglobal: f64,

    /// Relative tolerance for the global minimum test.
    /// Default: 0.0 (from `nlopt_get_param(opt, "fglobal_reltol", 0.0)`).
    pub fglobal_reltol: f64,

    /// Algorithm variant to use.
    pub algorithm: DirectAlgorithm,

    /// Enable parallel function evaluation using rayon.
    /// When `false`, evaluation order is identical to NLOPT C (bit-exact results).
    pub parallel: bool,

    /// Enable batch parallelization across multiple selected rectangles.
    /// When `true` (and `parallel` is also `true`), ALL sample points from
    /// ALL selected rectangles in an iteration are collected and evaluated
    /// in a single parallel batch, rather than per-rectangle.
    /// This can improve throughput when the objective function is expensive.
    /// Results may differ from sequential mode due to evaluation order.
    pub parallel_batch: bool,

    /// Minimum number of function evaluations required to use parallel evaluation.
    /// When the number of points to evaluate is below this threshold, the serial
    /// path is used even if `parallel` is `true`. This avoids rayon thread-pool
    /// overhead for small batches where parallelism doesn't pay off.
    ///
    /// Typical rayon overhead is 1–5 µs per task spawn. For objective functions
    /// that take less than a few microseconds, the overhead dominates unless
    /// the batch is large enough. A threshold of 4 works well in practice:
    /// it falls back to serial for 1D and 2D problems (which produce only 2–4
    /// sample points per rectangle) while still parallelizing higher-dimensional
    /// problems and batch evaluations.
    ///
    /// Set to 1 to always parallelize when `parallel` is `true`.
    /// Default: 4.
    pub min_parallel_evals: usize,
}

impl Default for DirectOptions {
    /// Default options matching NLOPT's defaults.
    fn default() -> Self {
        Self {
            max_feval: 0,
            max_iter: 0,
            max_time: 0.0,
            magic_eps: 0.0,
            magic_eps_abs: 0.0,
            volume_reltol: 0.0,
            sigma_reltol: -1.0,
            fglobal: DIRECT_UNKNOWN_FGLOBAL,
            fglobal_reltol: DIRECT_UNKNOWN_FGLOBAL_RELTOL,
            algorithm: DirectAlgorithm::default(),
            parallel: false,
            parallel_batch: false,
            min_parallel_evals: 4,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Result
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a DIRECT optimization run.
#[derive(Debug, Clone)]
pub struct DirectResult {
    /// Best point found (dimension n).
    pub x: Vec<f64>,

    /// Best function value found.
    pub fun: f64,

    /// Total number of function evaluations.
    pub nfev: usize,

    /// Total number of iterations.
    pub nit: usize,

    /// Whether the optimization terminated successfully.
    pub success: bool,

    /// The return code indicating why optimization stopped.
    pub return_code: DirectReturnCode,

    /// Human-readable message describing the termination reason.
    pub message: String,
}

impl DirectResult {
    /// Create a new result from optimization output.
    pub fn new(
        x: Vec<f64>,
        fun: f64,
        nfev: usize,
        nit: usize,
        return_code: DirectReturnCode,
    ) -> Self {
        let success = return_code.is_success();
        let message = format!("{}", return_code);
        Self {
            x,
            fun,
            nfev,
            nit,
            success,
            return_code,
            message,
        }
    }
}

impl fmt::Display for DirectResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DirectResult {{")?;
        writeln!(f, "  success: {}", self.success)?;
        writeln!(f, "  message: {}", self.message)?;
        writeln!(f, "  fun: {:.15e}", self.fun)?;
        write!(f, "  x: [")?;
        for (i, xi) in self.x.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.15e}", xi)?;
        }
        writeln!(f, "]")?;
        writeln!(f, "  nfev: {}", self.nfev)?;
        writeln!(f, "  nit: {}", self.nit)?;
        writeln!(f, "  return_code: {:?}", self.return_code)?;
        write!(f, "}}")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Display for DirectReturnCode
// ──────────────────────────────────────────────────────────────────────────────

impl fmt::Display for DirectReturnCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds => write!(f, "Invalid bounds: lower >= upper"),
            Self::MaxFevalTooBig => write!(f, "Maximum evaluations too large for memory"),
            Self::InitFailed => write!(f, "Initialization failed"),
            Self::SamplePointsFailed => write!(f, "Sample points creation failed"),
            Self::SampleFailed => write!(f, "Function evaluation failed"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::InvalidArgs => write!(f, "Invalid arguments"),
            Self::ForcedStop => write!(f, "Optimization forced to stop"),
            Self::MaxFevalExceeded => write!(f, "Maximum function evaluations reached"),
            Self::MaxIterExceeded => write!(f, "Maximum iterations reached"),
            Self::GlobalFound => write!(f, "Global minimum found within tolerance"),
            Self::VolTol => write!(f, "Volume tolerance reached"),
            Self::SigmaTol => write!(f, "Sigma tolerance reached"),
            Self::MaxTimeExceeded => write!(f, "Maximum time exceeded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_which_alg() {
        // NLOPT_GN_DIRECT → which_alg=0
        assert_eq!(DirectAlgorithm::Original.which_alg(), Some(0));
        // NLOPT_GN_DIRECT_L → which_alg=13 (1+3*1+9*1)
        assert_eq!(DirectAlgorithm::LocallyBiased.which_alg(), Some(13));
        // NLOPT_GN_DIRECT_L_RAND → which_alg=16 (1+3*2+9*1)
        assert_eq!(DirectAlgorithm::Randomized.which_alg(), Some(16));
        // Unscaled variants use same which_alg
        assert_eq!(DirectAlgorithm::OriginalUnscaled.which_alg(), Some(0));
        assert_eq!(DirectAlgorithm::LocallyBiasedUnscaled.which_alg(), Some(13));
        assert_eq!(
            DirectAlgorithm::LocallyBiasedRandomizedUnscaled.which_alg(),
            Some(16)
        );
        // Gablonsky variants don't use which_alg
        assert_eq!(DirectAlgorithm::GablonskyOriginal.which_alg(), None);
        assert_eq!(DirectAlgorithm::GablonskyLocallyBiased.which_alg(), None);
    }

    #[test]
    fn test_algorithm_algmethod() {
        assert_eq!(DirectAlgorithm::GablonskyOriginal.algmethod(), Some(0));
        assert_eq!(DirectAlgorithm::GablonskyLocallyBiased.algmethod(), Some(1));
        // cdirect variants don't use algmethod
        assert_eq!(DirectAlgorithm::Original.algmethod(), None);
        assert_eq!(DirectAlgorithm::LocallyBiased.algmethod(), None);
    }

    #[test]
    fn test_algorithm_classification() {
        assert!(DirectAlgorithm::GablonskyOriginal.is_gablonsky_translation());
        assert!(DirectAlgorithm::GablonskyLocallyBiased.is_gablonsky_translation());
        assert!(!DirectAlgorithm::Original.is_gablonsky_translation());

        assert!(DirectAlgorithm::Original.is_cdirect());
        assert!(DirectAlgorithm::LocallyBiased.is_cdirect());
        assert!(DirectAlgorithm::Randomized.is_cdirect());
        assert!(!DirectAlgorithm::GablonskyOriginal.is_cdirect());

        assert!(DirectAlgorithm::OriginalUnscaled.is_unscaled());
        assert!(DirectAlgorithm::LocallyBiasedUnscaled.is_unscaled());
        assert!(DirectAlgorithm::LocallyBiasedRandomizedUnscaled.is_unscaled());
        assert!(!DirectAlgorithm::Original.is_unscaled());
        assert!(!DirectAlgorithm::LocallyBiased.is_unscaled());
    }

    #[test]
    fn test_which_alg_decomposition() {
        // Verify the base-3 encoding: which_alg = which_diam + 3*which_div + 9*which_opt
        for alg in [
            DirectAlgorithm::Original,
            DirectAlgorithm::LocallyBiased,
            DirectAlgorithm::Randomized,
            DirectAlgorithm::OriginalUnscaled,
            DirectAlgorithm::LocallyBiasedUnscaled,
            DirectAlgorithm::LocallyBiasedRandomizedUnscaled,
        ] {
            let wa = alg.which_alg().unwrap();
            let which_diam = wa % 3;
            let which_div = (wa / 3) % 3;
            let which_opt = (wa / 9) % 3;

            match alg {
                DirectAlgorithm::Original | DirectAlgorithm::OriginalUnscaled => {
                    assert_eq!(which_diam, 0, "Jones Euclidean diameter");
                    assert_eq!(which_div, 0, "Jones division");
                    assert_eq!(which_opt, 0, "All hull points");
                }
                DirectAlgorithm::LocallyBiased | DirectAlgorithm::LocallyBiasedUnscaled => {
                    assert_eq!(which_diam, 1, "Gablonsky max-side diameter");
                    assert_eq!(which_div, 1, "Gablonsky division");
                    assert_eq!(which_opt, 1, "One per diameter");
                }
                DirectAlgorithm::Randomized
                | DirectAlgorithm::LocallyBiasedRandomizedUnscaled => {
                    assert_eq!(which_diam, 1, "Gablonsky max-side diameter");
                    assert_eq!(which_div, 2, "Random division");
                    assert_eq!(which_opt, 1, "One per diameter");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_algorithm_nlopt_names() {
        assert_eq!(DirectAlgorithm::Original.nlopt_name(), "GN_DIRECT");
        assert_eq!(DirectAlgorithm::LocallyBiased.nlopt_name(), "GN_DIRECT_L");
        assert_eq!(
            DirectAlgorithm::Randomized.nlopt_name(),
            "GN_DIRECT_L_RAND"
        );
        assert_eq!(
            DirectAlgorithm::OriginalUnscaled.nlopt_name(),
            "GN_DIRECT_NOSCAL"
        );
        assert_eq!(
            DirectAlgorithm::LocallyBiasedUnscaled.nlopt_name(),
            "GN_DIRECT_L_NOSCAL"
        );
        assert_eq!(
            DirectAlgorithm::LocallyBiasedRandomizedUnscaled.nlopt_name(),
            "GN_DIRECT_L_RAND_NOSCAL"
        );
        assert_eq!(
            DirectAlgorithm::GablonskyOriginal.nlopt_name(),
            "GN_ORIG_DIRECT"
        );
        assert_eq!(
            DirectAlgorithm::GablonskyLocallyBiased.nlopt_name(),
            "GN_ORIG_DIRECT_L"
        );
    }

    #[test]
    fn test_default_options() {
        let opts = DirectOptions::default();
        assert_eq!(opts.max_feval, 0);
        assert_eq!(opts.max_iter, 0);
        assert_eq!(opts.max_time, 0.0);
        assert_eq!(opts.magic_eps, 0.0);
        assert_eq!(opts.magic_eps_abs, 0.0);
        assert_eq!(opts.volume_reltol, 0.0);
        assert_eq!(opts.sigma_reltol, -1.0);
        assert_eq!(opts.fglobal, DIRECT_UNKNOWN_FGLOBAL);
        assert!(opts.fglobal.is_infinite() && opts.fglobal.is_sign_negative());
        assert_eq!(opts.fglobal_reltol, DIRECT_UNKNOWN_FGLOBAL_RELTOL);
        assert_eq!(opts.algorithm, DirectAlgorithm::LocallyBiased);
        assert!(!opts.parallel);
        assert!(!opts.parallel_batch);
        assert_eq!(opts.min_parallel_evals, 4);
    }

    #[test]
    fn test_direct_unknown_fglobal() {
        // DIRECT_UNKNOWN_FGLOBAL = -HUGE_VAL in C = f64::NEG_INFINITY in Rust
        assert_eq!(DIRECT_UNKNOWN_FGLOBAL, f64::NEG_INFINITY);
        assert_eq!(DIRECT_UNKNOWN_FGLOBAL_RELTOL, 0.0);
    }

    #[test]
    fn test_return_code_display() {
        assert_eq!(
            format!("{}", DirectReturnCode::GlobalFound),
            "Global minimum found within tolerance"
        );
        assert_eq!(
            format!("{}", DirectReturnCode::MaxFevalExceeded),
            "Maximum function evaluations reached"
        );
        assert_eq!(
            format!("{}", DirectReturnCode::ForcedStop),
            "Optimization forced to stop"
        );
    }

    #[test]
    fn test_direct_result_new() {
        let result = DirectResult::new(
            vec![1.0, 2.0],
            42.0,
            100,
            10,
            DirectReturnCode::MaxFevalExceeded,
        );
        assert_eq!(result.x, vec![1.0, 2.0]);
        assert_eq!(result.fun, 42.0);
        assert_eq!(result.nfev, 100);
        assert_eq!(result.nit, 10);
        assert!(result.success);
        assert_eq!(result.return_code, DirectReturnCode::MaxFevalExceeded);
    }

    #[test]
    fn test_direct_result_error() {
        let result = DirectResult::new(
            vec![0.0],
            f64::INFINITY,
            0,
            0,
            DirectReturnCode::InvalidBounds,
        );
        assert!(!result.success);
        assert_eq!(result.return_code, DirectReturnCode::InvalidBounds);
    }

    #[test]
    fn test_direct_result_display() {
        let result = DirectResult::new(
            vec![1.0, 2.0],
            3.0,
            50,
            5,
            DirectReturnCode::GlobalFound,
        );
        let display = format!("{}", result);
        assert!(display.contains("success: true"));
        assert!(display.contains("nfev: 50"));
        assert!(display.contains("nit: 5"));
    }

    #[test]
    fn test_algorithm_default() {
        assert_eq!(DirectAlgorithm::default(), DirectAlgorithm::LocallyBiased);
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", DirectAlgorithm::Original), "GN_DIRECT");
        assert_eq!(
            format!("{}", DirectAlgorithm::GablonskyLocallyBiased),
            "GN_ORIG_DIRECT_L"
        );
    }
}
