//! Tracing infrastructure for step-by-step algorithm comparison.
//!
//! When the `trace` feature is enabled, key algorithm events are written to a
//! `TraceWriter`. This allows line-by-line comparison of Rust output with NLOPT C
//! output produced by a matching tracing shim.
//!
//! The trace output format is a series of tagged lines:
//! ```text
//! TRACE INIT center_f=<val> minf=<val> minpos=<idx> nfev=<n>
//! TRACE ITER t=<n> selected=<count> minf=<val> nfev=<n>
//! TRACE SELECT j=<n> rect=<idx> level=<lev> f=<val>
//! TRACE SAMPLE rect=<parent> dim=<d> pos_idx=<p> pos_f=<f> neg_idx=<n> neg_f=<f>
//! TRACE DIVIDE rect=<parent> dim_order=[d1,d2,...] lengths_after=[l1,l2,...]
//! TRACE ENDITER t=<n> minf=<val> minpos=<idx> nfev=<n>
//! ```

use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::sync::Mutex;

/// A thread-safe buffer that collects trace lines.
pub struct TraceWriter {
    buffer: Mutex<String>,
}

impl TraceWriter {
    pub fn new() -> Self {
        Self {
            buffer: Mutex::new(String::with_capacity(64 * 1024)),
        }
    }

    /// Write a formatted trace line.
    pub fn write_line(&self, line: &str) {
        let mut buf = self.buffer.lock().unwrap();
        buf.push_str(line);
        buf.push('\n');
    }

    /// Write a formatted trace line using format args.
    pub fn write_fmt(&self, args: std::fmt::Arguments<'_>) {
        let mut buf = self.buffer.lock().unwrap();
        let _ = buf.write_fmt(args);
        buf.push('\n');
    }

    /// Get all collected trace output.
    pub fn get_output(&self) -> String {
        self.buffer.lock().unwrap().clone()
    }

    /// Get trace output as a vector of lines.
    pub fn get_lines(&self) -> Vec<String> {
        self.buffer
            .lock()
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    /// Also write trace output to stderr for debugging.
    pub fn dump_to_stderr(&self) {
        let buf = self.buffer.lock().unwrap();
        let _ = std::io::stderr().write_all(buf.as_bytes());
    }
}

impl Default for TraceWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for conditional trace output (only active with `trace` feature).
#[cfg(feature = "trace")]
#[macro_export]
macro_rules! trace_write {
    ($tracer:expr, $($arg:tt)*) => {
        if let Some(ref tw) = $tracer {
            tw.write_fmt(format_args!($($arg)*));
        }
    };
}

/// No-op when trace feature is disabled.
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! trace_write {
    ($tracer:expr, $($arg:tt)*) => {};
}
