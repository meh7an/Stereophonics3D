//! Names shared by the layer DLL and its installer, and the anti-cheat refusal list.

/// Layer name in the manifest and in `VK_INSTANCE_LAYERS`.
pub const LAYER_NAME: &str = "VK_LAYER_S3D_stereo";
pub const LAYER_DESCRIPTION: &str = "Stereophonics3D same-frame geometric stereo";
/// Vulkan API version the layer is built against (the `ash` headers).
pub const LAYER_API_VERSION: &str = "1.3.281";
/// File name of the layer DLL.
pub const DLL_FILE: &str = "s3d_layer.dll";
/// File name of the manifest, written next to the DLL.
pub const MANIFEST_FILE: &str = "s3d_layer.json";
/// Setting this to `1` makes the Vulkan loader skip the layer entirely.
pub const DISABLE_ENV: &str = "DISABLE_S3D_LAYER";
/// Overrides `%APPDATA%\Stereophonics3D` as the configuration directory.
pub const CONFIG_DIR_ENV: &str = "S3D_CONFIG_DIR";
/// Overrides `%LOCALAPPDATA%\Stereophonics3D\logs` as the log directory.
pub const LOG_DIR_ENV: &str = "S3D_LOG_DIR";

/// User-mode modules of anti-cheat systems, as (lower-case file-name prefix, product).
/// Kernel-only systems leave nothing to detect here; extend this list as products are
/// identified.
const ANTI_CHEAT_MODULES: [(&str, &str); 5] = [
    ("easyanticheat", "Easy Anti-Cheat"),
    ("eaanticheat", "EA Javelin Anti-Cheat"),
    ("beclient", "BattlEye"),
    ("npggnt", "nProtect GameGuard"),
    ("x3.xem", "XIGNCODE3"),
];

/// The anti-cheat product that loaded module `file_name` belongs to, if any.
pub fn anti_cheat_for(file_name: &str) -> Option<&'static str> {
    let name = file_name.to_ascii_lowercase();
    ANTI_CHEAT_MODULES
        .iter()
        .find(|(prefix, _)| name.starts_with(prefix))
        .map(|(_, product)| *product)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anti_cheat_modules_are_recognised_case_insensitively() {
        assert_eq!(
            anti_cheat_for("EasyAntiCheat_x64.dll"),
            Some("Easy Anti-Cheat")
        );
        assert_eq!(
            anti_cheat_for("EasyAntiCheat_EOS.dll"),
            Some("Easy Anti-Cheat")
        );
        assert_eq!(anti_cheat_for("BEClient_x64.dll"), Some("BattlEye"));
        assert_eq!(anti_cheat_for("npggNT.des"), Some("nProtect GameGuard"));
        assert_eq!(anti_cheat_for("x3.xem"), Some("XIGNCODE3"));
    }

    #[test]
    fn test_ordinary_modules_are_not_anti_cheat() {
        for name in [
            "vulkan-1.dll",
            "X3DAudio1_7.dll",
            "kernel32.dll",
            "s3d_layer.dll",
        ] {
            assert_eq!(anti_cheat_for(name), None, "{name}");
        }
    }
}
