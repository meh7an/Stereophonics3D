//! SPIR-V validation behind a single API.
//!
//! Wraps the C API of SPIRV-Tools, statically linked from the Vulkan SDK, so validation
//! runs in-process: tests check every patched fixture, and the layer can check a patched
//! module at run time and fall back to the original when the patch is invalid.

use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    ptr,
};

/// Errors raised by `s3d-spirv-val`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// SPIRV-Tools does not know the requested environment name.
    #[error("unknown SPIR-V target environment {0:?}")]
    UnknownEnvironment(String),

    /// SPIRV-Tools could not allocate a validation context.
    #[error("cannot create a SPIRV-Tools context")]
    Context,

    /// The module failed validation.
    #[error("invalid SPIR-V at word {word}: {message}")]
    Invalid {
        /// Word offset SPIRV-Tools blames, when it reports one.
        word: usize,
        /// The validator's explanation.
        message: String,
    },
}

/// Result alias for `s3d-spirv-val`.
pub type Result<T> = std::result::Result<T, Error>;

/// Environment for Vulkan 1.3 modules, which also accepts every older SPIR-V version.
pub const VULKAN_1_3: &str = "vulkan1.3";

/// Validates `words` for `environment` (a SPIRV-Tools name such as `vulkan1.3`).
///
/// # Errors
///
/// Returns [`Error::Invalid`] with the validator's message when the module is invalid.
pub fn validate(words: &[u32], environment: &str) -> Result<()> {
    let context = Context::new(environment)?;
    let mut diagnostic: *mut Diagnostic = ptr::null_mut();
    // SAFETY: the context is live, `words` is a valid slice for the call, and the
    // diagnostic out-pointer is either left null or set to an object freed below.
    let result =
        unsafe { spvValidateBinary(context.0, words.as_ptr(), words.len(), &mut diagnostic) };
    if result == SPV_SUCCESS {
        // SAFETY: destroying a null diagnostic is a documented no-op.
        unsafe { spvDiagnosticDestroy(diagnostic) };
        return Ok(());
    }
    // SAFETY: on failure SPIRV-Tools either leaves the pointer null or points it at a
    // diagnostic whose `error` is a NUL-terminated string; both are freed right after.
    let (word, message) = unsafe {
        let details = diagnostic.as_ref().map(|d| {
            let message = if d.error.is_null() {
                String::new()
            } else {
                CStr::from_ptr(d.error).to_string_lossy().into_owned()
            };
            (d.position.index, message)
        });
        spvDiagnosticDestroy(diagnostic);
        details.unwrap_or((0, format!("validator returned code {result}")))
    };
    Err(Error::Invalid { word, message })
}

/// Owns a `spv_context` for one call.
struct Context(*mut c_void);

impl Context {
    fn new(environment: &str) -> Result<Self> {
        let unknown = || Error::UnknownEnvironment(environment.to_owned());
        let name = CString::new(environment).map_err(|_| unknown())?;
        let mut env: c_int = 0;
        // SAFETY: `name` is NUL-terminated and `env` is a valid out-pointer.
        if !unsafe { spvParseTargetEnv(name.as_ptr(), &mut env) } {
            return Err(unknown());
        }
        // SAFETY: `env` came from SPIRV-Tools itself.
        let context = unsafe { spvContextCreate(env) };
        if context.is_null() {
            return Err(Error::Context);
        }
        Ok(Self(context))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `spvContextCreate` and is destroyed once.
        unsafe { spvContextDestroy(self.0) };
    }
}

const SPV_SUCCESS: c_int = 0;

/// `spv_position_t` from `libspirv.h`.
#[repr(C)]
struct Position {
    line: usize,
    column: usize,
    index: usize,
}

/// `spv_diagnostic_t` from `libspirv.h`.
#[repr(C)]
struct Diagnostic {
    position: Position,
    error: *mut c_char,
    is_text_source: bool,
}

// Declarations copied from `libspirv.h` of the SDK the crate links against.
unsafe extern "C" {
    fn spvParseTargetEnv(name: *const c_char, env: *mut c_int) -> bool;
    fn spvContextCreate(env: c_int) -> *mut c_void;
    fn spvContextDestroy(context: *mut c_void);
    fn spvValidateBinary(
        context: *const c_void,
        words: *const u32,
        num_words: usize,
        diagnostic: *mut *mut Diagnostic,
    ) -> c_int;
    fn spvDiagnosticDestroy(diagnostic: *mut Diagnostic);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vertex shader whose `main` only returns, encoded by hand.
    const MINIMAL: [u32; 26] = [
        // Header: magic, SPIR-V 1.0, generator, id bound 5, schema.
        0x0723_0203,
        0x0001_0000,
        0,
        5,
        0,
        // OpCapability Shader
        0x0002_0011,
        1,
        // OpMemoryModel Logical GLSL450
        0x0003_000e,
        0,
        1,
        // OpEntryPoint Vertex %1 "main"
        0x0005_000f,
        0,
        1,
        0x6e69_616d,
        0,
        // %2 = OpTypeVoid
        0x0002_0013,
        2,
        // %3 = OpTypeFunction %2
        0x0003_0021,
        3,
        2,
        // %1 = OpFunction %2 None %3
        0x0005_0036,
        2,
        1,
        0,
        3,
        // %4 = OpLabel
        0x0002_00f8,
    ];

    fn minimal() -> Vec<u32> {
        let mut words = MINIMAL.to_vec();
        // %4, then OpReturn and OpFunctionEnd.
        words.extend([4, 0x0001_00fd, 0x0001_0038]);
        words
    }

    #[test]
    fn test_minimal_module_is_valid() {
        validate(&minimal(), VULKAN_1_3).expect("valid");
    }

    #[test]
    fn test_module_without_function_end_is_invalid_with_message() {
        let words = minimal();
        let error = validate(&words[..words.len() - 1], VULKAN_1_3).expect_err("invalid");
        assert!(matches!(error, Error::Invalid { ref message, .. } if !message.is_empty()));
    }

    #[test]
    fn test_unknown_environment_is_rejected() {
        assert!(matches!(
            validate(&minimal(), "vulkan9.9"),
            Err(Error::UnknownEnvironment(_))
        ));
    }
}
