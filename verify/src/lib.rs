//! Real 3D guarantee checks (G0 to G5) over per-eye captures.

pub mod checks;
pub mod counter;
pub mod error;
pub mod image;

pub use checks::{Capture, Finding, Measurement, Outcome, Test, run};
pub use error::{Error, Result};
