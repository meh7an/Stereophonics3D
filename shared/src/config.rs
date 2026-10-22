//! Configuration file schemas.
//!
//! `config.toml` holds global defaults; each file in `profiles\` opts one game in and may
//! override any of them, plus per-shader behaviour. Every table rejects unknown keys so a
//! typo is reported instead of silently ignored. Value checks and merging live here;
//! loading from disk lives in `settings` and hot reload in `watch`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AnaglyphSettings, MatrixChoice, ShaderHash, StereoParams};

/// Global settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub stereo: StereoParams,
    pub anaglyph: AnaglyphSettings,
    pub dump: DumpSettings,
}

/// Debug output written while a game runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DumpSettings {
    /// Write every shader the game creates as `.spv` plus `.spvasm` disassembly.
    pub shaders: bool,
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
    #[serde(default)]
    pub dump: DumpOverrides,
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

/// Dump fields a profile may replace; `None` keeps the global value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DumpOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shaders: Option<bool>,
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

/// A value outside its allowed range, named by its dotted TOML path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    /// Dotted path such as `stereo.convergence` or `shaders.<hash>`.
    pub field: String,
    pub message: String,
}

impl Config {
    /// Range checks for every value; empty when the file is usable.
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        let stereo = &self.stereo;
        check_parallax(&mut issues, stereo.max_parallax_percent);
        check_convergence(&mut issues, stereo.convergence);
        let anaglyph = &self.anaglyph;
        check_matrix(&mut issues, anaglyph.matrix);
        check_brightness(&mut issues, anaglyph.brightness);
        check_gamma(&mut issues, anaglyph.gamma);
        issues
    }

    /// Global settings with `profile`'s overrides applied.
    pub fn merged(&self, profile: &Profile) -> Config {
        let (stereo, anaglyph) = (&profile.stereo, &profile.anaglyph);
        Config {
            stereo: StereoParams {
                max_parallax_percent: stereo
                    .max_parallax_percent
                    .unwrap_or(self.stereo.max_parallax_percent),
                convergence: stereo.convergence.unwrap_or(self.stereo.convergence),
                swap_eyes: stereo.swap_eyes.unwrap_or(self.stereo.swap_eyes),
            },
            anaglyph: AnaglyphSettings {
                matrix: anaglyph.matrix.unwrap_or(self.anaglyph.matrix),
                brightness: anaglyph.brightness.unwrap_or(self.anaglyph.brightness),
                gamma: anaglyph.gamma.unwrap_or(self.anaglyph.gamma),
            },
            dump: DumpSettings {
                shaders: profile.dump.shaders.unwrap_or(self.dump.shaders),
            },
        }
    }
}

impl Profile {
    /// Range checks for the overrides present, the executable name and every shader entry.
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        let name = self.executable.trim();
        if name.is_empty() || name.contains(['/', '\\']) {
            issues.push(Issue {
                field: "executable".to_owned(),
                message: "must be a file name such as `Game.exe`, without a path".to_owned(),
            });
        }
        let (stereo, anaglyph) = (&self.stereo, &self.anaglyph);
        if let Some(value) = stereo.max_parallax_percent {
            check_parallax(&mut issues, value);
        }
        if let Some(value) = stereo.convergence {
            check_convergence(&mut issues, value);
        }
        if let Some(value) = anaglyph.matrix {
            check_matrix(&mut issues, value);
        }
        if let Some(value) = anaglyph.brightness {
            check_brightness(&mut issues, value);
        }
        if let Some(value) = anaglyph.gamma {
            check_gamma(&mut issues, value);
        }
        for (hash, entry) in &self.shaders {
            if let ShaderOverride::HudDepth(w) = entry
                && !(w.is_finite() && *w > 0.0)
            {
                issues.push(Issue {
                    field: format!("shaders.{hash}"),
                    message: format!("hud_depth must be a positive depth, found {w}"),
                });
            }
        }
        issues
    }

    /// Whether this profile applies to the executable file name `name`.
    pub fn matches(&self, name: &str) -> bool {
        self.executable.trim().to_lowercase() == name.to_lowercase()
    }
}

const MAX_PARALLAX_PERCENT: f32 = 10.0;
const BRIGHTNESS_RANGE: (f32, f32) = (0.1, 4.0);
const GAMMA_RANGE: (f32, f32) = (0.2, 5.0);
const MAX_MATRIX_COEFFICIENT: f32 = 4.0;

fn check_parallax(issues: &mut Vec<Issue>, value: f32) {
    if !(value.is_finite() && value > 0.0 && value <= MAX_PARALLAX_PERCENT) {
        issues.push(Issue {
            field: "stereo.max_parallax_percent".to_owned(),
            message: format!("must be in (0, {MAX_PARALLAX_PERCENT}], found {value}"),
        });
    }
}

fn check_convergence(issues: &mut Vec<Issue>, value: f32) {
    if !(value.is_finite() && value > 0.0) {
        issues.push(Issue {
            field: "stereo.convergence".to_owned(),
            message: format!("must be a positive depth, found {value}"),
        });
    }
}

fn check_range(issues: &mut Vec<Issue>, field: &str, value: f32, (low, high): (f32, f32)) {
    if !(value.is_finite() && (low..=high).contains(&value)) {
        issues.push(Issue {
            field: field.to_owned(),
            message: format!("must be in [{low}, {high}], found {value}"),
        });
    }
}

fn check_brightness(issues: &mut Vec<Issue>, value: f32) {
    check_range(issues, "anaglyph.brightness", value, BRIGHTNESS_RANGE);
}

fn check_gamma(issues: &mut Vec<Issue>, value: f32) {
    check_range(issues, "anaglyph.gamma", value, GAMMA_RANGE);
}

fn check_matrix(issues: &mut Vec<Issue>, choice: MatrixChoice) {
    let MatrixChoice::Custom(matrix) = choice else {
        return;
    };
    let valid = matrix
        .0
        .iter()
        .flatten()
        .all(|c| c.is_finite() && c.abs() <= MAX_MATRIX_COEFFICIENT);
    if !valid {
        issues.push(Issue {
            field: "anaglyph.matrix".to_owned(),
            message: format!(
                "every coefficient must be finite with magnitude at most {MAX_MATRIX_COEFFICIENT}"
            ),
        });
    }
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
            dump: DumpOverrides {
                shaders: Some(true),
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

    fn fields(issues: &[Issue]) -> Vec<&str> {
        issues.iter().map(|issue| issue.field.as_str()).collect()
    }

    #[test]
    fn test_default_config_and_sample_profile_are_valid() {
        assert!(Config::default().validate().is_empty());
        assert!(sample_profile().validate().is_empty());
    }

    #[test]
    fn test_config_out_of_range_values_name_their_fields() {
        let mut config = Config::default();
        config.stereo.max_parallax_percent = 12.0;
        config.stereo.convergence = 0.0;
        config.anaglyph.brightness = f32::NAN;
        config.anaglyph.gamma = 0.1;
        config.anaglyph.matrix = MatrixChoice::Custom(crate::AnaglyphMatrix([[5.0; 6]; 3]));
        assert_eq!(
            fields(&config.validate()),
            [
                "stereo.max_parallax_percent",
                "stereo.convergence",
                "anaglyph.matrix",
                "anaglyph.brightness",
                "anaglyph.gamma",
            ]
        );
    }

    #[test]
    fn test_profile_checks_executable_overrides_and_hud_depth() {
        let mut profile = sample_profile();
        profile.executable = r"C:\Games\Game.exe".to_owned();
        profile.stereo.max_parallax_percent = Some(-1.0);
        profile
            .shaders
            .insert(hash("00000000000000dd"), ShaderOverride::HudDepth(0.0));
        assert_eq!(
            fields(&profile.validate()),
            [
                "executable",
                "stereo.max_parallax_percent",
                "shaders.00000000000000dd",
            ]
        );
    }

    #[test]
    fn test_merged_applies_only_present_overrides() {
        let config = Config::default();
        let merged = config.merged(&sample_profile());
        assert_eq!(merged.stereo.convergence, 12.5);
        assert_eq!(
            merged.stereo.max_parallax_percent,
            config.stereo.max_parallax_percent
        );
        assert_eq!(
            merged.anaglyph.matrix,
            MatrixChoice::Preset(AnaglyphPreset::ZhangLcd)
        );
        assert_eq!(merged.anaglyph.gamma, config.anaglyph.gamma);
        assert!(merged.dump.shaders);
    }

    #[test]
    fn test_profile_matches_executable_case_insensitively() {
        let profile = sample_profile();
        assert!(profile.matches("game.EXE"));
        assert!(!profile.matches("other.exe"));
    }

    #[test]
    fn test_malformed_shader_key_is_rejected() {
        let text = "executable = \"a.exe\"\n[shaders]\nnot-a-hash = \"skip\"";
        assert!(toml::from_str::<Profile>(text).is_err());
    }
}
