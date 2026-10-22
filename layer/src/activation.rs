//! The process-wide decision whether the layer does anything at all.
//!
//! An implicit layer loads into every Vulkan application, so the default is pure
//! pass-through: no hooks beyond what the loader chain requires, no threads, no file
//! writes. The layer activates only when a profile names the executable and no
//! anti-cheat module is loaded; only then does it start the configuration watcher.
//!
//! The loader frees the layer DLL after `vkDestroyInstance`, so applications that create
//! a short-lived instance (device enumeration) would unmap code the log writer and
//! watcher threads are still running. The DLL therefore pins itself before starting any
//! thread and stays resident until the process exits.

use std::{
    any::Any,
    sync::{Mutex, OnceLock, PoisonError},
};

use s3d_shared::{
    ConfigWatcher, Loader,
    layer::anti_cheat_for,
    logging,
    settings::{current_executable_name, default_config_dir},
};
use windows::{
    Win32::{
        Foundation::HMODULE,
        System::{
            LibraryLoader::{
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN,
                GetModuleHandleExW,
            },
            ProcessStatus::{EnumProcessModules, GetModuleBaseNameW},
            Threading::GetCurrentProcess,
        },
    },
    core::PCWSTR,
};

/// The outcome of the gate, decided once per process.
pub struct Activation {
    pub active: bool,
    /// Keeps live settings flowing while active; read from M3.3 on.
    _watcher: Option<ConfigWatcher>,
}

static ACTIVATION: OnceLock<Activation> = OnceLock::new();
/// Keeps the log writer alive for the life of the process.
static LOG_GUARD: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);

/// The gate's outcome, deciding on first use.
pub fn get() -> &'static Activation {
    ACTIVATION.get_or_init(decide)
}

fn decide() -> Activation {
    let pinned = pin_module();
    start_logging();
    let executable = current_executable_name().unwrap_or_else(|| "<unknown>".to_owned());
    let pid = std::process::id();
    tracing::info!(pid, executable, "s3d-layer loaded");
    if let Err(error) = pinned {
        tracing::warn!(%error, "cannot pin the layer DLL; an unload may crash the process");
    }

    let pass_through = Activation {
        active: false,
        _watcher: None,
    };
    if let Some((module, product)) = loaded_anti_cheat() {
        tracing::warn!(
            module,
            product,
            "refused: anti-cheat module loaded, pass-through"
        );
        return pass_through;
    }
    let dir = match default_config_dir() {
        Ok(dir) => dir,
        Err(error) => {
            tracing::warn!(%error, "no configuration directory, pass-through");
            return pass_through;
        }
    };
    let probe = Loader::new(&dir, &executable).load();
    for diagnostic in &probe.diagnostics {
        tracing::warn!(%diagnostic, "configuration problem");
    }
    let Some(settings) = probe.settings else {
        tracing::info!(dir = %dir.display(), "no profile matches, pass-through");
        return pass_through;
    };
    match ConfigWatcher::start(&dir, &executable) {
        Ok(watcher) => {
            tracing::info!(profile = %settings.profile.display(), "active");
            Activation {
                active: true,
                _watcher: Some(watcher),
            }
        }
        Err(error) => {
            tracing::error!(%error, "cannot watch the configuration, pass-through");
            pass_through
        }
    }
}

/// Keeps this DLL loaded until the process exits, whatever the loader frees.
fn pin_module() -> windows::core::Result<()> {
    let mut module = HMODULE::default();
    let address = pin_module as *const () as *const u16;
    // SAFETY: with FROM_ADDRESS the name is read as an address inside this module, and
    // `module` is a valid out-pointer. A pinned module needs no matching FreeLibrary.
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            PCWSTR(address),
            &mut module,
        )
    }
}

/// File logging; a game that already installed a global subscriber keeps it.
fn start_logging() {
    if let Ok(guard) = logging::init("s3d-layer") {
        let mut slot = LOG_GUARD.lock().unwrap_or_else(PoisonError::into_inner);
        *slot = Some(Box::new(guard));
    }
}

/// The first loaded module that belongs to an anti-cheat product, with the product.
fn loaded_anti_cheat() -> Option<(String, &'static str)> {
    module_names()
        .into_iter()
        .find_map(|name| anti_cheat_for(&name).map(|product| (name, product)))
}

/// File names of every module loaded in this process.
fn module_names() -> Vec<String> {
    let mut modules = vec![HMODULE::default(); 1024];
    let mut needed = 0u32;
    let capacity = (modules.len() * size_of::<HMODULE>()) as u32;
    // SAFETY: the buffer holds `capacity` bytes and `needed` is a valid out-pointer.
    let listed = unsafe {
        EnumProcessModules(
            GetCurrentProcess(),
            modules.as_mut_ptr(),
            capacity,
            &mut needed,
        )
    };
    if listed.is_err() {
        return Vec::new();
    }
    modules.truncate((needed as usize / size_of::<HMODULE>()).min(modules.len()));
    modules
        .into_iter()
        .filter_map(|module| {
            let mut name = [0u16; 260];
            // SAFETY: `module` came from EnumProcessModules for this process.
            let length =
                unsafe { GetModuleBaseNameW(GetCurrentProcess(), Some(module), &mut name) };
            (length > 0).then(|| String::from_utf16_lossy(&name[..length as usize]))
        })
        .collect()
}
