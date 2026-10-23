//! Compiles the compositor GLSL shaders to SPIR-V with the Vulkan SDK's `glslc`.

use std::{env, path::PathBuf, process::Command};

const SHADERS: &[&str] = &["fullscreen.vert", "identity.frag"];

fn main() {
    let sdk = env::var_os("VULKAN_SDK")
        .expect("VULKAN_SDK is not set; install the LunarG Vulkan SDK to build shaders");
    let glslc = PathBuf::from(sdk).join("Bin").join("glslc.exe");
    let source_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"))
        .join("..")
        .join("shaders")
        .join("compositor");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("set by cargo"));

    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    for shader in SHADERS {
        let source = source_dir.join(shader);
        println!("cargo:rerun-if-changed={}", source.display());
        let output = out_dir.join(format!("{shader}.spv"));
        // Vulkan 1.0: the layer runs inside whatever device the game created.
        let status = Command::new(&glslc)
            .arg("--target-env=vulkan1.0")
            .arg("-O")
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .status()
            .unwrap_or_else(|error| panic!("failed to run {}: {error}", glslc.display()));
        assert!(status.success(), "glslc failed on {}", source.display());
    }
}
