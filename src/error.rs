//! Error types for the DIRECT-NLOPT-RS implementation.
//!
//! Maps to NLOPT's `direct_return_code` enum and provides Rust-idiomatic error handling.

use thiserror::Error;

/// Return codes matching NLOPT's `direct_return_code` enum.
///
/// Negative values indicate errors, positive values indicate successful termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectReturnCode {
    /// Invalid bounds (lower >= upper for some dimension)
    InvalidBounds = -1,
    /// maxfeval too large for available memory
    MaxFevalTooBig = -2,
    /// Initialization failed
    InitFailed = -3,
    /// Sample points creation failed
    SamplePointsFailed = -4,
    /// Function evaluation failed
    SampleFailed = -5,
    /// Out of memory
    OutOfMemory = -100,
    /// Invalid arguments
    InvalidArgs = -101,
    /// Forced stop via callback or signal
    ForcedStop = -102,

    /// Maximum function evaluations exceeded
    MaxFevalExceeded = 1,
    /// Maximum iterations exceeded
    MaxIterExceeded = 2,
    /// Global minimum found (within tolerance)
    GlobalFound = 3,
    /// Volume tolerance reached
    VolTol = 4,
    /// Sigma tolerance reached
    SigmaTol = 5,
    /// Maximum time exceeded
    MaxTimeExceeded = 6,
}

impl DirectReturnCode {
    /// Returns true if this is a successful termination (positive code).
    pub fn is_success(&self) -> bool {
        (*self as i32) > 0
    }

    /// Returns true if this is an error (negative code).
    pub fn is_error(&self) -> bool {
        (*self as i32) < 0
    }

    /// Convert from integer error code (matching NLOPT convention).
    pub fn from_i32(code: i32) -> Option<Self> {
        match code {
            -1 => Some(Self::InvalidBounds),
            -2 => Some(Self::MaxFevalTooBig),
            -3 => Some(Self::InitFailed),
            -4 => Some(Self::SamplePointsFailed),
            -5 => Some(Self::SampleFailed),
            -100 => Some(Self::OutOfMemory),
            -101 => Some(Self::InvalidArgs),
            -102 => Some(Self::ForcedStop),
            1 => Some(Self::MaxFevalExceeded),
            2 => Some(Self::MaxIterExceeded),
            3 => Some(Self::GlobalFound),
            4 => Some(Self::VolTol),
            5 => Some(Self::SigmaTol),
            6 => Some(Self::MaxTimeExceeded),
            _ => None,
        }
    }
}

/// Errors that can occur during DIRECT optimization.
#[derive(Error, Debug)]
pub enum DirectError {
    #[error("Invalid bounds: lower bound >= upper bound in dimension {dim}")]
    InvalidBounds { dim: usize },

    #[error("Maximum function evaluations ({0}) too large for available memory")]
    MaxFevalTooBig(usize),

    #[error("Initialization failed: {0}")]
    InitFailed(String),

    #[error("Sample points creation failed: {0}")]
    SamplePointsFailed(String),

    #[error("Function evaluation failed: {0}")]
    SampleFailed(String),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("Optimization forced to stop")]
    ForcedStop,

    #[error("DIRECT error code {0}: {1}")]
    DirectCode(i32, String),
}

impl From<DirectReturnCode> for DirectError {
    fn from(code: DirectReturnCode) -> Self {
        match code {
            DirectReturnCode::InvalidBounds => DirectError::InvalidBounds { dim: 0 },
            DirectReturnCode::MaxFevalTooBig => DirectError::MaxFevalTooBig(0),
            DirectReturnCode::InitFailed => DirectError::InitFailed("unknown".into()),
            DirectReturnCode::SamplePointsFailed => DirectError::SamplePointsFailed("unknown".into()),
            DirectReturnCode::SampleFailed => DirectError::SampleFailed("unknown".into()),
            DirectReturnCode::OutOfMemory => DirectError::OutOfMemory,
            DirectReturnCode::InvalidArgs => DirectError::InvalidArgs("unknown".into()),
            DirectReturnCode::ForcedStop => DirectError::ForcedStop,
            _ => DirectError::DirectCode(code as i32, format!("{:?}", code)),
        }
    }
}

/// Result type alias for DIRECT operations.
pub type Result<T> = std::result::Result<T, DirectError>;
