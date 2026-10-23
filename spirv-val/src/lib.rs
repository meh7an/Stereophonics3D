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

/// Block-layout rules a device enables, which widen what a module may declare.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayoutRules {
    /// `VK_KHR_relaxed_block_layout`, core since Vulkan 1.1.
    pub relaxed: bool,
    /// The `uniformBufferStandardLayout` feature.
    pub uniform_buffer_standard: bool,
    /// The `scalarBlockLayout` feature.
    pub scalar: bool,
}

/// Validates `words` for `environment` (a SPIRV-Tools name such as `vulkan1.3`) with the
/// default block-layout rules.
///
/// # Errors
///
/// Returns [`Error::Invalid`] with the validator's message when the module is invalid.
pub fn validate(words: &[u32], environment: &str) -> Result<()> {
    validate_with(words, environment, LayoutRules::default())
}

/// Validates `words` as a device enabling `rules` would accept them.
///
/// # Errors
///
/// Returns [`Error::Invalid`] with the validator's message when the module is invalid.
pub fn validate_with(words: &[u32], environment: &str, rules: LayoutRules) -> Result<()> {
    let context = Context::new(environment)?;
    let options = Options::new(rules)?;
    let binary = Binary {
        code: words.as_ptr(),
        word_count: words.len(),
    };
    let mut diagnostic: *mut Diagnostic = ptr::null_mut();
    // SAFETY: the context and options are live, `binary` points at `words` for the
    // call, and the diagnostic out-pointer is either left null or set to an object freed
    // below.
    let result = unsafe { spvValidateWithOptions(context.0, options.0, &binary, &mut diagnostic) };
    // SAFETY: `diagnostic` is null or was produced by the call above.
    unsafe { check(result, diagnostic) }
}

/// Disassembles `words` into indented text with friendly names, as `spirv-dis` prints it.
///
/// # Errors
///
/// Returns [`Error::Invalid`] when the words cannot be decoded.
pub fn disassemble(words: &[u32]) -> Result<String> {
    let context = Context::new(VULKAN_1_3)?;
    let mut text: *mut Text = ptr::null_mut();
    let mut diagnostic: *mut Diagnostic = ptr::null_mut();
    // SAFETY: as in `validate`; `text` is an out-pointer to an object freed below.
    let result = unsafe {
        spvBinaryToText(
            context.0,
            words.as_ptr(),
            words.len(),
            BINARY_TO_TEXT_INDENT | BINARY_TO_TEXT_FRIENDLY_NAMES,
            &mut text,
            &mut diagnostic,
        )
    };
    // SAFETY: on success `text` points at `length` bytes of UTF-8; it is freed once.
    let disassembly = unsafe {
        let content = text.as_ref().map(|t| {
            let bytes = std::slice::from_raw_parts(t.str.cast::<u8>(), t.length);
            String::from_utf8_lossy(bytes)
                .trim_end_matches('\0')
                .to_owned()
        });
        spvTextDestroy(text);
        content
    };
    // SAFETY: `diagnostic` is null or was produced by the call above.
    unsafe { check(result, diagnostic)? };
    Ok(disassembly.unwrap_or_default())
}

/// Turns a SPIRV-Tools result and diagnostic into `Ok` or [`Error::Invalid`], freeing the
/// diagnostic.
///
/// # Safety
///
/// `diagnostic` must be null or a diagnostic returned by SPIRV-Tools and not yet freed.
unsafe fn check(result: c_int, diagnostic: *mut Diagnostic) -> Result<()> {
    // SAFETY: guaranteed by the caller; `error` is a NUL-terminated string when set.
    let details = unsafe {
        let details = diagnostic.as_ref().map(|d| {
            let message = if d.error.is_null() {
                String::new()
            } else {
                CStr::from_ptr(d.error).to_string_lossy().into_owned()
            };
            (d.position.index, message)
        });
        spvDiagnosticDestroy(diagnostic);
        details
    };
    if result == SPV_SUCCESS {
        return Ok(());
    }
    let (word, message) = details.unwrap_or((0, format!("SPIRV-Tools returned code {result}")));
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

/// Owns a `spv_validator_options` for one call.
struct Options(*mut c_void);

impl Options {
    fn new(rules: LayoutRules) -> Result<Self> {
        // SAFETY: no arguments; a null result is handled below.
        let options = unsafe { spvValidatorOptionsCreate() };
        if options.is_null() {
            return Err(Error::Context);
        }
        // SAFETY: `options` is live until dropped.
        unsafe {
            spvValidatorOptionsSetRelaxBlockLayout(options, rules.relaxed);
            spvValidatorOptionsSetUniformBufferStandardLayout(
                options,
                rules.uniform_buffer_standard,
            );
            spvValidatorOptionsSetScalarBlockLayout(options, rules.scalar);
        }
        Ok(Self(options))
    }
}

impl Drop for Options {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `spvValidatorOptionsCreate` and is destroyed once.
        unsafe { spvValidatorOptionsDestroy(self.0) };
    }
}

const SPV_SUCCESS: c_int = 0;
/// `SPV_BINARY_TO_TEXT_OPTION_INDENT`, `SPV_BIT(3)`.
const BINARY_TO_TEXT_INDENT: u32 = 1 << 3;
/// `SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES`, `SPV_BIT(6)`.
const BINARY_TO_TEXT_FRIENDLY_NAMES: u32 = 1 << 6;

/// `spv_binary_t` from `libspirv.h`; SPIRV-Tools only reads it during validation.
#[repr(C)]
struct Binary {
    code: *const u32,
    word_count: usize,
}

/// `spv_text_t` from `libspirv.h`.
#[repr(C)]
struct Text {
    str: *const c_char,
    length: usize,
}

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
    fn spvValidatorOptionsCreate() -> *mut c_void;
    fn spvValidatorOptionsDestroy(options: *mut c_void);
    fn spvValidatorOptionsSetRelaxBlockLayout(options: *mut c_void, value: bool);
    fn spvValidatorOptionsSetUniformBufferStandardLayout(options: *mut c_void, value: bool);
    fn spvValidatorOptionsSetScalarBlockLayout(options: *mut c_void, value: bool);
    fn spvValidateWithOptions(
        context: *const c_void,
        options: *const c_void,
        binary: *const Binary,
        diagnostic: *mut *mut Diagnostic,
    ) -> c_int;
    fn spvDiagnosticDestroy(diagnostic: *mut Diagnostic);
    fn spvBinaryToText(
        context: *const c_void,
        binary: *const u32,
        word_count: usize,
        options: u32,
        text: *mut *mut Text,
        diagnostic: *mut *mut Diagnostic,
    ) -> c_int;
    fn spvTextDestroy(text: *mut Text);
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

    /// A vertex shader with a push-constant block `{ float; vec4 at offset 4 }`, valid
    /// only under scalar block layout.
    const SCALAR_PUSH_BLOCK: [u32; 57] = [
        // Header: magic, SPIR-V 1.0, generator, id bound 10, schema.
        0x0723_0203,
        0x0001_0000,
        0,
        10,
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
        // OpDecorate %6 Block
        0x0003_0047,
        6,
        2,
        // OpMemberDecorate %6 0 Offset 0
        0x0005_0048,
        6,
        0,
        35,
        0,
        // OpMemberDecorate %6 1 Offset 4
        0x0005_0048,
        6,
        1,
        35,
        4,
        // %2 = OpTypeVoid
        0x0002_0013,
        2,
        // %3 = OpTypeFunction %2
        0x0003_0021,
        3,
        2,
        // %4 = OpTypeFloat 32
        0x0003_0016,
        4,
        32,
        // %5 = OpTypeVector %4 4
        0x0004_0017,
        5,
        4,
        4,
        // %6 = OpTypeStruct %4 %5
        0x0004_001e,
        6,
        4,
        5,
        // %7 = OpTypePointer PushConstant %6
        0x0004_0020,
        7,
        9,
        6,
        // %8 = OpVariable %7 PushConstant
        0x0004_003b,
        7,
        8,
        9,
        // %1 = OpFunction %2 None %3
        0x0005_0036,
        2,
        1,
        0,
        3,
    ];

    #[test]
    fn test_scalar_block_layout_is_accepted_only_when_enabled() {
        let mut words = SCALAR_PUSH_BLOCK.to_vec();
        // %9 = OpLabel, OpReturn, OpFunctionEnd.
        words.extend([0x0002_00f8, 9, 0x0001_00fd, 0x0001_0038]);
        let error = validate(&words, "vulkan1.0").expect_err("needs scalar layout");
        assert!(matches!(error, Error::Invalid { .. }));
        let relaxed = LayoutRules {
            relaxed: true,
            ..LayoutRules::default()
        };
        assert!(validate_with(&words, "vulkan1.0", relaxed).is_err());
        let scalar = LayoutRules {
            scalar: true,
            ..LayoutRules::default()
        };
        validate_with(&words, "vulkan1.0", scalar).expect("valid under scalar layout");
    }

    #[test]
    fn test_unknown_environment_is_rejected() {
        assert!(matches!(
            validate(&minimal(), "vulkan9.9"),
            Err(Error::UnknownEnvironment(_))
        ));
    }

    #[test]
    fn test_disassemble_names_entry_point() {
        let text = disassemble(&minimal()).expect("disassemble");
        assert!(text.contains("OpEntryPoint Vertex"), "{text}");
        assert!(text.contains("OpFunctionEnd"), "{text}");
    }
}
