//! Error type for `s3d-spirv-patch`.

/// Errors raised by `s3d-spirv-patch`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The bytes are not a well-formed SPIR-V module.
    #[error("cannot parse SPIR-V: {0}")]
    Parse(String),

    /// A byte length that is not a whole number of 32-bit words.
    #[error("SPIR-V must be a whole number of words, got {0} bytes")]
    Length(usize),

    /// The module has no entry point with this name.
    #[error("no entry point named {0:?}")]
    NoEntryPoint(String),
}

/// Result alias for `s3d-spirv-patch`.
pub type Result<T> = std::result::Result<T, Error>;
