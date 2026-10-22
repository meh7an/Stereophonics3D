//! SPIR-V analysis and per-eye shear injection.

pub mod analysis;
pub mod error;
pub mod module;

pub use analysis::{Analysis, PositionForm, PositionOutput, PositionStore, StoreTarget};
pub use error::{Error, Result};
pub use module::{EntryPoint, Instruction, ShaderModule, Stage, last_pre_raster, words_from_bytes};
