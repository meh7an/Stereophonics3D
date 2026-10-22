//! Types, file formats and errors shared by every Stereophonics3D crate.

pub mod anaglyph;
pub mod capture;
pub mod config;
pub mod error;
pub mod ffi;
pub mod logging;
pub mod shader_hash;
pub mod stereo;

pub use anaglyph::{AnaglyphMatrix, AnaglyphPreset, AnaglyphSettings, MatrixChoice};
pub use config::{AnaglyphOverrides, Config, Profile, ShaderOverride, StereoOverrides};
pub use error::{Error, Result};
pub use shader_hash::ShaderHash;
pub use stereo::{Eye, StereoParams, StereoPushConstants};
