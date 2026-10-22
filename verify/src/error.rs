//! Error type for `s3d-verify`.

/// Errors raised by `s3d-verify`. A failed test is a [`crate::Finding`], not an error;
/// these mean the test could not be run at all.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Reading a capture file failed.
    #[error(transparent)]
    Capture(#[from] s3d_shared::Error),

    /// The sidecar has no `stereo` section, so it is not a per-eye dump.
    #[error("{0} is a mono capture; the verifier needs a per-eye dump")]
    NotStereo(String),

    /// The two eye images differ in size from each other or from the sidecar.
    #[error("eye images do not match the sidecar's {width}x{height}")]
    Size { width: u32, height: u32 },

    /// The capture lacks something a test needs, such as its markers.
    #[error("{test} cannot run: {reason}")]
    NotApplicable { test: &'static str, reason: String },
}

/// Result alias for `s3d-verify`.
pub type Result<T> = std::result::Result<T, Error>;
