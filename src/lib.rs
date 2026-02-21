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

pub mod error;
pub mod storage;
pub mod types;

// Re-export main types
pub use error::{DirectError, DirectReturnCode, Result};
pub use types::{
    Bounds, CallbackFn, DirectAlgorithm, DirectOptions, DirectResult, ObjectiveFn,
    DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
};
