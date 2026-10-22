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
}

/// Result alias for `s3d-shared`.
pub type Result<T> = std::result::Result<T, Error>;
