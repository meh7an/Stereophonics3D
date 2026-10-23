//! Error type for `s3d-shared`.
//!
//! Every library crate follows this shape: one `thiserror` enum named `Error` in
//! `error.rs` plus a `Result` alias. Binaries convert these into `anyhow` at their edge.

use std::path::PathBuf;

use tracing::dispatcher::SetGlobalDefaultError;
use tracing_appender::rolling::InitError;

/// Errors raised by `s3d-shared`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// `%LOCALAPPDATA%` is unset, so there is no default log directory.
    #[error("LOCALAPPDATA is not set; no default log directory")]
    NoLocalAppData,

    /// `%APPDATA%` is unset, so there is no default configuration directory.
    #[error("APPDATA is not set; no default configuration directory")]
    NoAppData,

    /// The configuration directory could not be watched for changes.
    #[error("cannot watch the configuration directory")]
    Watch(#[from] notify::Error),

    /// The log directory could not be created.
    #[error("failed to create log directory {dir}")]
    LogDir {
        /// Directory that could not be created.
        dir: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// The rolling log file could not be created.
    #[error("failed to open a log file in {dir}")]
    LogFile {
        /// Directory the log file was meant to live in.
        dir: PathBuf,
        /// Underlying appender failure.
        #[source]
        source: InitError,
    },

    /// Another global tracing subscriber is already installed in this process.
    #[error("a global log subscriber is already installed")]
    LoggerAlreadySet(#[source] SetGlobalDefaultError),

    /// A file could not be read or written.
    #[error("cannot access {path}")]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// A sidecar did not match the JSON schema.
    #[error("invalid sidecar {path}")]
    Json {
        /// File that failed.
        path: PathBuf,
        /// Underlying parse or serialization failure.
        #[source]
        source: serde_json::Error,
    },

    /// Pixels could not be encoded as PNG.
    #[error("cannot encode PNG")]
    PngEncode(#[from] png::EncodingError),

    /// A PNG could not be decoded.
    #[error("cannot decode PNG")]
    PngDecode(#[from] png::DecodingError),

    /// A PNG decoded, but not as 8-bit RGBA.
    #[error("expected an 8-bit RGBA PNG, found {found}")]
    UnsupportedPng {
        /// Colour type and depth found.
        found: String,
    },

    /// A depth dump does not hold one float per pixel.
    #[error("depth dump {path} has {found} bytes, expected {expected}")]
    DepthSize {
        /// File that failed.
        path: PathBuf,
        /// Bytes found.
        found: usize,
        /// Bytes a width x height dump holds.
        expected: usize,
    },

    /// Text is not 16 hex digits.
    #[error("invalid shader hash {text:?}; expected 16 hex digits")]
    InvalidShaderHash {
        /// The rejected text.
        text: String,
    },
}

/// Result alias for `s3d-shared`.
pub type Result<T> = std::result::Result<T, Error>;
