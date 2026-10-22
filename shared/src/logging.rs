//! File logging shared by the layer and the tools.
//!
//! The layer runs inside a game with no console, so every program logs to daily rolling
//! files under `%LOCALAPPDATA%\Stereophonics3D\logs`, filtered by [`LOG_ENV_VAR`].

use std::path::{Path, PathBuf};

use tracing::Dispatch;
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_subscriber::{EnvFilter, filter::LevelFilter};

use crate::{Error, Result};

/// Environment variable holding the log filter, for example `debug` or `s3d_layer=trace`.
pub const LOG_ENV_VAR: &str = "S3D_LOG";

/// Daily files kept per program before the oldest is deleted.
pub const MAX_LOG_FILES: usize = 7;

/// Returns `%LOCALAPPDATA%\Stereophonics3D\logs`.
///
/// # Errors
///
/// Returns [`Error::NoLocalAppData`] when `LOCALAPPDATA` is unset.
pub fn default_log_dir() -> Result<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .map(|base| PathBuf::from(base).join("Stereophonics3D").join("logs"))
        .ok_or(Error::NoLocalAppData)
}

/// Builds a dispatcher writing to `<program>.<date>.log` in `dir`, rotated daily.
///
/// The returned guard flushes buffered lines when dropped, so it must outlive every
/// event logged through the dispatcher.
///
/// # Errors
///
/// Returns [`Error::LogDir`] or [`Error::LogFile`] when the directory or file cannot be
/// created.
pub fn file_dispatch(dir: &Path, program: &str) -> Result<(Dispatch, WorkerGuard)> {
    // The appender prunes old files before creating the directory and prints an error to
    // stderr when it is missing, so the directory must exist first.
    std::fs::create_dir_all(dir).map_err(|source| Error::LogDir {
        dir: dir.to_path_buf(),
        source,
    })?;
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(program)
        .filename_suffix("log")
        .max_log_files(MAX_LOG_FILES)
        .build(dir)
        .map_err(|source| Error::LogFile {
            dir: dir.to_path_buf(),
            source,
        })?;
    let (writer, guard) = tracing_appender::non_blocking(appender);
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .with_env_var(LOG_ENV_VAR)
        .from_env_lossy();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_ansi(false)
        .with_env_filter(filter)
        .finish();
    Ok((Dispatch::new(subscriber), guard))
}

/// Installs file logging for `program` in [`default_log_dir`] as the global subscriber.
///
/// # Errors
///
/// Returns an error when the log directory is unknown or unwritable, or when another
/// global subscriber is already installed.
pub fn init(program: &str) -> Result<WorkerGuard> {
    let (dispatch, guard) = file_dispatch(&default_log_dir()?, program)?;
    tracing::dispatcher::set_global_default(dispatch).map_err(Error::LoggerAlreadySet)?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_file_dispatch_logged_event_lands_in_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (dispatch, guard) = file_dispatch(dir.path(), "unit").expect("dispatch");
        tracing::dispatcher::with_default(&dispatch, || tracing::info!("marker-7f3a"));
        drop(guard);

        let files: Vec<_> = fs::read_dir(dir.path())
            .expect("read dir")
            .map(|entry| entry.expect("dir entry").path())
            .collect();
        assert_eq!(files.len(), 1);
        let name = files[0]
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf-8 name");
        assert!(
            name.starts_with("unit.") && name.ends_with(".log"),
            "{name}"
        );
        let contents = fs::read_to_string(&files[0]).expect("read log");
        assert!(contents.contains("marker-7f3a"), "{contents}");
    }

    #[test]
    fn test_file_dispatch_missing_dir_is_created() {
        let dir = tempfile::tempdir().expect("temp dir");
        let nested = dir.path().join("a").join("b");
        let (_dispatch, _guard) = file_dispatch(&nested, "unit").expect("dispatch");
        assert!(nested.is_dir());
    }
}
