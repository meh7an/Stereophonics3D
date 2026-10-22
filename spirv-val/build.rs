//! Links the static SPIRV-Tools library shipped with the LunarG Vulkan SDK.

use std::{env, path::PathBuf};

fn main() {
    let sdk = env::var_os("VULKAN_SDK")
        .expect("VULKAN_SDK is not set; install the LunarG Vulkan SDK to link SPIRV-Tools");
    let lib_dir = PathBuf::from(sdk).join("Lib");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // The release build: Rust links the release CRT even in debug builds, so the `d`
    // variant would mismatch.
    println!("cargo:rustc-link-lib=static=SPIRV-Tools");
}
