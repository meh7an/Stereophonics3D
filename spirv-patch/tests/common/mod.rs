//! Fixture loading shared by the integration tests.

use std::{fs, path::PathBuf};

use s3d_spirv_patch::{ShaderModule, words_from_bytes};

pub const FIXTURES: [&str; 12] = [
    "glsl_basic",
    "glsl_piecewise",
    "glsl_redeclared",
    "glsl_function",
    "glsl_tese",
    "glsl_geom",
    "glsl_mesh",
    "glsl_frag",
    "glsl_push",
    "hlsl_basic",
    "hlsl_struct",
    "multi_entry",
];

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fixtures")
        .join("spirv")
}

pub fn words(name: &str) -> Vec<u32> {
    let path = fixtures_dir().join(format!("{name}.spv"));
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    words_from_bytes(&bytes).expect("whole words")
}

pub fn module(name: &str) -> ShaderModule {
    ShaderModule::parse(&words(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}
