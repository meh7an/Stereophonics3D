//! `s3d-install`: writes the layer manifest next to the DLL and registers it with the
//! Vulkan loader as an implicit layer.
//!
//! Registration is a `REG_DWORD 0` value named after the manifest's absolute path under
//! `SOFTWARE\Khronos\Vulkan\ImplicitLayers`, in HKCU for the current user or HKLM
//! (`--machine`, needs elevation). The loader ignores HKCU for elevated applications, so
//! games running as administrator need the HKLM registration.

use std::{
    io,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use s3d_shared::layer::{
    DISABLE_ENV, DLL_FILE, LAYER_API_VERSION, LAYER_DESCRIPTION, LAYER_NAME, MANIFEST_FILE,
};
use windows::{
    Win32::{
        Foundation::{ERROR_FILE_NOT_FOUND, ERROR_NO_MORE_ITEMS, ERROR_SUCCESS, WIN32_ERROR},
        System::Registry::{
            HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, KEY_WRITE, REG_DWORD,
            REG_OPTION_NON_VOLATILE, RegCloseKey, RegCreateKeyExW, RegDeleteKeyW,
            RegDeleteValueW, RegEnumValueW, RegOpenKeyExW, RegQueryInfoKeyW, RegSetValueExW,
        },
    },
    core::{HSTRING, PCWSTR, PWSTR},
};

const IMPLICIT_LAYERS_KEY: &str = r"SOFTWARE\Khronos\Vulkan\ImplicitLayers";
/// Runtime DLLs the layer needs because it links the C runtime dynamically.
const RUNTIME_DLLS: [&str; 2] = ["vcruntime140.dll", "msvcp140.dll"];

#[derive(Debug, Parser)]
#[command(version, about = "Installs the Stereophonics3D Vulkan layer")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Write the manifest next to the layer DLL.
    Manifest {
        /// Directory holding s3d_layer.dll (default: this program's directory).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Write the manifest and register the layer with the Vulkan loader.
    Register {
        /// Directory holding s3d_layer.dll (default: this program's directory).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Register for every user (HKLM); needs an elevated prompt.
        #[arg(long)]
        machine: bool,
    },
    /// Remove every registration of this layer.
    Unregister {
        /// Remove the all-users (HKLM) registrations; needs an elevated prompt.
        #[arg(long)]
        machine: bool,
    },
    /// Show registrations and check that their files and the C runtime are present.
    Status,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Manifest { dir } => {
            let manifest = write_manifest(&layer_dir(dir)?)?;
            println!("wrote {}", manifest.display());
        }
        Command::Register { dir, machine } => {
            let manifest = write_manifest(&layer_dir(dir)?)?;
            let scope = Scope::new(machine);
            scope
                .open(Access::Create)?
                .set_dword(&manifest.to_string_lossy(), 0)
                .with_context(|| format!("cannot register in {}", scope.name))?;
            println!("registered {} in {}", manifest.display(), scope.name);
            warn_missing_runtime();
        }
        Command::Unregister { machine } => {
            let scope = Scope::new(machine);
            let Some(key) = scope.open_existing(Access::Write)? else {
                println!("nothing registered in {}", scope.name);
                return Ok(());
            };
            let ours = key.our_values()?;
            for name in &ours {
                key.delete(name)
                    .with_context(|| format!("cannot remove {name} from {}", scope.name))?;
                println!("removed {name} from {}", scope.name);
            }
            if ours.is_empty() {
                println!("nothing registered in {}", scope.name);
            }
            drop(key);
            scope.prune_empty()?;
        }
        Command::Status => {
            for scope in [Scope::new(false), Scope::new(true)] {
                let ours = match scope.open_existing(Access::Read)? {
                    Some(key) => key.our_values()?,
                    None => Vec::new(),
                };
                if ours.is_empty() {
                    println!("{}: not registered", scope.name);
                }
                for name in ours {
                    let manifest = Path::new(&name);
                    let dll = manifest.with_file_name(DLL_FILE);
                    println!(
                        "{}: {name} (manifest {}, dll {})",
                        scope.name,
                        presence(manifest),
                        presence(&dll)
                    );
                }
            }
            warn_missing_runtime();
        }
    }
    Ok(())
}

/// The directory holding the layer DLL, which must exist there.
fn layer_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = match dir {
        Some(dir) => dir,
        None => std::env::current_exe()?
            .parent()
            .context("this program has no parent directory")?
            .to_path_buf(),
    };
    let dir = std::path::absolute(&dir)?;
    if !dir.join(DLL_FILE).is_file() {
        bail!("{} not found in {}", DLL_FILE, dir.display());
    }
    Ok(dir)
}

/// Writes the manifest into `dir` and returns its path.
fn write_manifest(dir: &Path) -> Result<PathBuf> {
    let manifest = serde_json::json!({
        "file_format_version": "1.2.0",
        "layer": {
            "name": LAYER_NAME,
            "type": "GLOBAL",
            // Relative to the manifest, which always sits next to the DLL.
            "library_path": format!(".\\{DLL_FILE}"),
            "api_version": LAYER_API_VERSION,
            "implementation_version": "1",
            "description": LAYER_DESCRIPTION,
            "disable_environment": { DISABLE_ENV: "1" },
        }
    });
    let path = dir.join(MANIFEST_FILE);
    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(path)
}

fn presence(path: &Path) -> &'static str {
    if path.is_file() { "present" } else { "MISSING" }
}

fn warn_missing_runtime() {
    let Some(system) =
        std::env::var_os("SystemRoot").map(|root| PathBuf::from(root).join("System32"))
    else {
        return;
    };
    for dll in RUNTIME_DLLS {
        if !system.join(dll).is_file() {
            println!(
                "warning: {dll} not found; install the Microsoft Visual C++ Redistributable (x64)"
            );
        }
    }
}

/// HKCU or HKLM.
struct Scope {
    root: HKEY,
    name: &'static str,
}

impl Scope {
    fn new(machine: bool) -> Self {
        if machine {
            Self {
                root: HKEY_LOCAL_MACHINE,
                name: "HKLM",
            }
        } else {
            Self {
                root: HKEY_CURRENT_USER,
                name: "HKCU",
            }
        }
    }

    /// Opens `path` below the root; `Access::Create` also creates it.
    fn open_path(&self, path: &str, access: Access) -> Result<Key> {
        let mut key = HKEY::default();
        let wide = HSTRING::from(path);
        let rights = match access {
            Access::Read => KEY_READ,
            Access::Write | Access::Create => KEY_READ | KEY_WRITE,
        };
        // SAFETY: valid root, nul-terminated path and out-pointer.
        let status = unsafe {
            match access {
                Access::Create => RegCreateKeyExW(
                    self.root,
                    &wide,
                    None,
                    PCWSTR::null(),
                    REG_OPTION_NON_VOLATILE,
                    rights,
                    None,
                    &mut key,
                    None,
                ),
                Access::Read | Access::Write => {
                    RegOpenKeyExW(self.root, &wide, None, rights, &mut key)
                }
            }
        };
        check(status).with_context(|| format!("cannot open {}\\{path}", self.name))?;
        Ok(Key(key))
    }

    fn open(&self, access: Access) -> Result<Key> {
        self.open_path(IMPLICIT_LAYERS_KEY, access)
    }

    /// The key when it exists; `None` when nothing was ever registered there.
    fn open_existing(&self, access: Access) -> Result<Option<Key>> {
        match self.open(access) {
            Ok(key) => Ok(Some(key)),
            Err(error) if is_not_found(&error) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Deletes the implicit-layer key and its parents up to `SOFTWARE` while they are
    /// empty, so unregistering leaves no keys this program may have created. A key that
    /// holds anything else stops the walk.
    fn prune_empty(&self) -> Result<()> {
        let mut path = IMPLICIT_LAYERS_KEY;
        while let Some((parent, _)) = path.rsplit_once('\\') {
            let empty = match self.open_path(path, Access::Read) {
                Ok(key) => key.is_empty()?,
                Err(error) if is_not_found(&error) => false,
                Err(error) => return Err(error),
            };
            if !empty {
                break;
            }
            // SAFETY: valid root and nul-terminated path of a key without subkeys.
            check(unsafe { RegDeleteKeyW(self.root, &HSTRING::from(path)) })
                .with_context(|| format!("cannot remove empty key {}\\{path}", self.name))?;
            println!("removed empty key {}\\{path}", self.name);
            path = parent;
        }
        Ok(())
    }
}

/// How a key is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    Create,
}

/// An open registry key, closed on drop.
struct Key(HKEY);

impl Key {
    fn set_dword(&self, name: &str, value: u32) -> Result<()> {
        // SAFETY: open key, nul-terminated name, 4-byte data.
        let status = unsafe {
            RegSetValueExW(
                self.0,
                &HSTRING::from(name),
                None,
                REG_DWORD,
                Some(&value.to_le_bytes()),
            )
        };
        check(status)
    }

    /// Whether the key has neither values nor subkeys.
    fn is_empty(&self) -> Result<bool> {
        let (mut subkeys, mut values) = (0u32, 0u32);
        // SAFETY: open key and valid out-pointers; other outputs are not requested.
        let status = unsafe {
            RegQueryInfoKeyW(
                self.0,
                None,
                None,
                None,
                Some(&mut subkeys),
                None,
                None,
                Some(&mut values),
                None,
                None,
                None,
                None,
            )
        };
        check(status)?;
        Ok(subkeys == 0 && values == 0)
    }

    fn delete(&self, name: &str) -> Result<()> {
        // SAFETY: open key and nul-terminated name.
        check(unsafe { RegDeleteValueW(self.0, &HSTRING::from(name)) })
    }

    /// Value names that point at this layer's manifest.
    fn our_values(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for index in 0.. {
            let mut buffer = vec![0u16; 32_768];
            let mut length = buffer.len() as u32;
            // SAFETY: the buffer holds `length` UTF-16 units; data is not requested.
            let status = unsafe {
                RegEnumValueW(
                    self.0,
                    index,
                    Some(PWSTR(buffer.as_mut_ptr())),
                    &mut length,
                    None,
                    None,
                    None,
                    None,
                )
            };
            if status == ERROR_NO_MORE_ITEMS {
                break;
            }
            check(status)?;
            let name = String::from_utf16_lossy(&buffer[..length as usize]);
            let file = Path::new(&name)
                .file_name()
                .map(|f| f.to_string_lossy().to_lowercase());
            if file.as_deref() == Some(MANIFEST_FILE) {
                names.push(name);
            }
        }
        Ok(names)
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: the key was opened by this program and is closed once. A failed close
        // leaks one handle in a short-lived process.
        let _ = unsafe { RegCloseKey(self.0) };
    }
}

fn check(status: WIN32_ERROR) -> Result<()> {
    if status == ERROR_SUCCESS {
        return Ok(());
    }
    Err(io::Error::from_raw_os_error(status.0 as i32).into())
}

fn is_not_found(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<io::Error>())
        .any(|io| io.raw_os_error() == Some(ERROR_FILE_NOT_FOUND.0 as i32))
}
