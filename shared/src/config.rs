//! Configuration file schemas.
//!
//! `config.toml` holds global defaults; each file in `profiles\` opts one game in and may
//! override any of them, plus per-shader behaviour. Every table rejects unknown keys so a
//! typo is reported instead of silently ignored. Loading, validation and merging live in
//! M1.2.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AnaglyphSettings, MatrixChoice, ShaderHash, StereoParams};

/// Global settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub stereo: StereoParams,
    pub anaglyph: AnaglyphSettings,
}

/// One game's settings. A process without a matching profile stays pass-through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// Executable file name the profile applies to, compared case-insensitively.
    pub executable: String,
    #[serde(default)]
    pub stereo: StereoOverrides,
    #[serde(default)]
    pub anaglyph: AnaglyphOverrides,
    /// Per-shader behaviour keyed by `ShaderHash`; wins over automatic classification.
    #[serde(default)]
    pub shaders: BTreeMap<ShaderHash, ShaderOverride>,
}

/// Stereo fields a profile may replace; `None` keeps the global value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StereoOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallax_percent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub convergence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_eyes: Option<bool>,
}

/// Anaglyph fields a profile may replace; `None` keeps the global value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AnaglyphOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix: Option<MatrixChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brightness: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f32>,
}

/// How draws using one pre-raster shader are treated.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ShaderOverride {
    /// Do not issue the draw (hide the object).
    Skip,
    /// Draw identically in both eyes, without a shear.
    Mono,
    /// Always shear, regardless of classification heuristics.
    Stereo,
    /// Screen-space content: shear as if it sat at view depth `w`.
    HudDepth(f32),
    /// Place at infinite depth (full parallax at infinity).
    Sky,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnaglyphPreset;

    fn hash(text: &str) -> ShaderHash {
        text.parse().expect("valid hash")
    }

    fn sample_profile() -> Profile {
        Profile {
            executable: "Game.exe".to_owned(),
            stereo: StereoOverrides {
                convergence: Some(12.5),
                ..StereoOverrides::default()
            },
            anaglyph: AnaglyphOverrides {
                matrix: Some(MatrixChoice::Preset(AnaglyphPreset::ZhangLcd)),
                ..AnaglyphOverrides::default()
            },
            shaders: BTreeMap::from([
                (hash("00000000000000aa"), ShaderOverride::Skip),
                (hash("00000000000000bb"), ShaderOverride::HudDepth(1.5)),
                (hash("00000000000000cc"), ShaderOverride::Sky),
            ]),
        }
    }

    #[test]
    fn test_profile_toml_round_trip() {
        let profile = sample_profile();
        let text = toml::to_string(&profile).expect("serialize");
        assert_eq!(toml::from_str::<Profile>(&text).expect("parse"), profile);
    }

    #[test]
    fn test_config_toml_round_trip() {
        let config = Config::default();
        let text = toml::to_string(&config).expect("serialize");
        assert_eq!(toml::from_str::<Config>(&text).expect("parse"), config);
    }

    #[test]
    fn test_empty_config_uses_defaults() {
        assert_eq!(
            toml::from_str::<Config>("").expect("parse"),
            Config::default()
        );
    }

    #[test]
    fn test_unknown_keys_are_rejected() {
        assert!(toml::from_str::<Config>("[stereo]\nconvergense = 3.0").is_err());
        assert!(toml::from_str::<Profile>("executable = \"a.exe\"\nshader = {}").is_err());
    }

    #[test]
    fn test_profile_requires_executable() {
        assert!(toml::from_str::<Profile>("[stereo]\nconvergence = 3.0").is_err());
    }

    #[test]
    fn test_malformed_shader_key_is_rejected() {
        let text = "executable = \"a.exe\"\n[shaders]\nnot-a-hash = \"skip\"";
        assert!(toml::from_str::<Profile>(text).is_err());
    }
}
