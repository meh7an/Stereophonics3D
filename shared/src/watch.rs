//! Hot reload: a background thread reloads the configuration when its files change and
//! publishes each new [`Snapshot`] for lock-free reading.

use std::{
    path::Path,
    sync::{
        Arc,
        mpsc::{self, RecvTimeoutError},
    },
    thread,
    time::Duration,
};

use arc_swap::ArcSwap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use crate::{
    Error, Result,
    settings::{Loader, Snapshot},
};

/// Quiet time after the last file event before reloading; editors often save a file in
/// several steps, and loading a half-written file would only produce a spurious error.
const DEBOUNCE: Duration = Duration::from_millis(200);

/// Cheap, cloneable read access to the current snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotHandle(Arc<ArcSwap<Snapshot>>);

impl SnapshotHandle {
    /// The current snapshot; never blocks and never waits for a reload.
    pub fn current(&self) -> Arc<Snapshot> {
        self.0.load_full()
    }
}

/// Watches the configuration directory for one executable while it lives.
pub struct ConfigWatcher {
    handle: SnapshotHandle,
    // Dropping the watcher closes the event channel, which ends the reload thread.
    _watcher: RecommendedWatcher,
}

impl ConfigWatcher {
    /// Loads once, then reloads on every change under `dir`, creating `dir` if missing.
    ///
    /// The reload thread is detached rather than joined on drop: inside a DLL the drop may
    /// run under the loader lock, where joining a thread can deadlock.
    ///
    /// # Errors
    ///
    /// Returns an error when `dir` cannot be created or watched.
    pub fn start(dir: &Path, executable: &str) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let mut loader = Loader::new(dir, executable);
        let handle = SnapshotHandle(Arc::new(ArcSwap::from_pointee(loader.load())));

        let (sender, events) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(sender)?;
        watcher.watch(dir, RecursiveMode::Recursive)?;

        let published = handle.clone();
        thread::Builder::new()
            .name("s3d-config-watch".to_owned())
            .spawn(move || reload_loop(&events, &mut loader, &published))
            .map_err(|source| Error::Io {
                path: dir.to_path_buf(),
                source,
            })?;

        Ok(Self {
            handle,
            _watcher: watcher,
        })
    }

    pub fn handle(&self) -> SnapshotHandle {
        self.handle.clone()
    }
}

fn reload_loop(
    events: &mpsc::Receiver<notify::Result<notify::Event>>,
    loader: &mut Loader,
    published: &SnapshotHandle,
) {
    while events.recv().is_ok() {
        loop {
            match events.recv_timeout(DEBOUNCE) {
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        let snapshot = loader.load();
        for diagnostic in &snapshot.diagnostics {
            tracing::warn!(%diagnostic, "configuration problem");
        }
        tracing::info!(
            generation = snapshot.generation,
            profile = ?snapshot.settings.as_ref().map(|s| &s.profile),
            "configuration reloaded"
        );
        published.0.store(Arc::new(snapshot));
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::Instant};

    use super::*;

    /// Polls until `done` holds, so the test tolerates slow file notifications.
    fn wait_for(handle: &SnapshotHandle, done: impl Fn(&Snapshot) -> bool) -> Arc<Snapshot> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let snapshot = handle.current();
            if done(&snapshot) || Instant::now() > deadline {
                return snapshot;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn test_watcher_publishes_edited_profile() {
        let dir = tempfile::tempdir().expect("temp dir");
        let profiles = dir.path().join("profiles");
        fs::create_dir_all(&profiles).expect("mkdir");
        let path = profiles.join("game.toml");
        fs::write(&path, "executable = \"Game.exe\"\n").expect("write");

        let watcher = ConfigWatcher::start(dir.path(), "Game.exe").expect("start");
        let handle = watcher.handle();
        let first = handle.current();
        assert_eq!(first.generation, 1);
        assert!(!first.settings.as_ref().expect("profile").stereo.swap_eyes);

        fs::write(
            &path,
            "executable = \"Game.exe\"\n[stereo]\nswap_eyes = true\n",
        )
        .expect("edit");
        let edited = wait_for(&handle, |s| {
            s.settings
                .as_ref()
                .is_some_and(|settings| settings.stereo.swap_eyes)
        });
        assert!(edited.generation > 1, "no reload within 10 s");

        fs::write(
            &path,
            "executable = \"Game.exe\"\n[stereo]\nconvergence = 0\n",
        )
        .expect("break");
        let broken = wait_for(&handle, |s| !s.diagnostics.is_empty());
        assert!(!broken.diagnostics.is_empty(), "no reload within 10 s");
        assert!(broken.settings.as_ref().expect("kept").stereo.swap_eyes);
    }
}
