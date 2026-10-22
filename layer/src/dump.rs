//! Shader dumps for inspection and hunting, 3Dmigoto style: `<hash>.spv` plus a
//! `<hash>.spvasm` disassembly per shader, in one folder per executable.

use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, PoisonError},
};

use s3d_shared::{ShaderHash, settings::default_dump_dir};

use crate::activation;

/// The process's dumper, created on first use; `None` when no dump directory exists.
static DUMPER: OnceLock<Option<Dumper>> = OnceLock::new();

/// Dumps `code` when the active settings ask for it. Failures are logged, never fatal.
pub fn shader(hash: ShaderHash, code: &[u32]) {
    let activation = activation::get();
    let enabled = activation
        .snapshot()
        .and_then(|snapshot| snapshot.settings.as_ref().map(|s| s.dump.shaders))
        .unwrap_or(false);
    if !enabled {
        return;
    }
    let dumper = DUMPER.get_or_init(|| match default_dump_dir() {
        Ok(dir) => Some(Dumper::new(dir.join(&activation.executable))),
        Err(error) => {
            tracing::warn!(%error, "no dump directory, shaders are not dumped");
            None
        }
    });
    let Some(dumper) = dumper else {
        return;
    };
    match dumper.write(hash, code) {
        Ok(true) => tracing::debug!(%hash, "dumped shader"),
        Ok(false) => {}
        Err(error) => tracing::warn!(%hash, %error, "cannot dump shader"),
    }
}

/// Writes each shader once per process into one directory.
#[derive(Debug)]
pub struct Dumper {
    dir: PathBuf,
    written: Mutex<HashSet<ShaderHash>>,
}

impl Dumper {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            written: Mutex::new(HashSet::new()),
        }
    }

    /// Writes `<hash>.spv` and `<hash>.spvasm`, returning whether anything new was
    /// written. Files already on disk are kept: the name is the content's hash, so they
    /// are identical, and dumps stay stable across runs.
    ///
    /// # Errors
    ///
    /// Returns the I/O error when the directory or the `.spv` file cannot be written. A
    /// module the disassembler rejects still gets its `.spv`; the failure is logged.
    pub fn write(&self, hash: ShaderHash, code: &[u32]) -> io::Result<bool> {
        let first = self
            .written
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(hash);
        if !first {
            return Ok(false);
        }
        fs::create_dir_all(&self.dir)?;
        let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
        if !create_new(&self.dir.join(format!("{hash}.spv")), &bytes)? {
            return Ok(false);
        }
        match s3d_spirv_val::disassemble(code) {
            Ok(text) => {
                create_new(&self.dir.join(format!("{hash}.spvasm")), text.as_bytes())?;
            }
            Err(error) => tracing::warn!(%hash, %error, "cannot disassemble shader"),
        }
        Ok(true)
    }
}

/// Writes `bytes` to a file that must not exist yet; `false` when it already does.
fn create_new(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => file.write_all(bytes).map(|()| true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/spirv/glsl_push.spv");

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }

    #[test]
    fn test_dump_names_files_by_the_m2_hash_and_writes_once() {
        let dir = std::env::temp_dir().join(format!("s3d-dump-test-{}", std::process::id()));
        let code = words(FIXTURE);
        let hash = ShaderHash::of(FIXTURE);
        let dumper = Dumper::new(dir.clone());

        assert!(dumper.write(hash, &code).expect("first write"));
        assert!(!dumper.write(hash, &code).expect("second write"));
        assert_eq!(
            fs::read(dir.join(format!("{hash}.spv"))).expect("spv"),
            FIXTURE
        );
        let text = fs::read_to_string(dir.join(format!("{hash}.spvasm"))).expect("spvasm");
        assert!(text.contains("OpEntryPoint"), "{text}");

        // A new process finds the files and leaves them alone.
        let again = Dumper::new(dir.clone());
        assert!(!again.write(hash, &code).expect("rerun"));
        fs::remove_dir_all(&dir).expect("cleanup");
    }
}
