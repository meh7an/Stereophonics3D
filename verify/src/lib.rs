//! Real 3D guarantee checks (G0 to G5) over per-eye captures, plus the tools that build
//! per-eye dumps from single-eye runs and compare their depth.

pub mod checks;
pub mod combine;
pub mod counter;
pub mod depth;
pub mod error;
pub mod image;

pub use checks::{Capture, Finding, Measurement, Outcome, Test, run};
pub use error::{Error, Result};
