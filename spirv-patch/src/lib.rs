//! SPIR-V analysis and per-eye shear injection.

pub mod analysis;
mod edit;
pub mod error;
pub mod module;
pub mod patch;

pub use analysis::{Analysis, PositionForm, PositionOutput, PositionStore, StoreTarget};
pub use error::{Error, Result};
pub use module::{EntryPoint, Instruction, ShaderModule, Stage, last_pre_raster, words_from_bytes};
pub use patch::{
    ConstantSource, DEFAULT_SPEC_ID, EyeSource, Insertion, PatchOptions, PatchReport, ShearMode,
};
