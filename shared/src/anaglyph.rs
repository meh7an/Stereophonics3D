//! Red/cyan anaglyph matrices and the player's anaglyph settings.
//!
//! A matrix maps `[L.R, L.G, L.B, R.R, R.G, R.B]` (left eye, then right eye, linear light)
//! to one output channel per row. Values come from the research report: Dubois 2001 (as
//! stored by three.js), Dubois 2009 (as used by Dolphin), the LCD set Romeuf attributes to
//! Zhang and McAllister, and the simple FFmpeg methods.

use serde::{Deserialize, Serialize};

/// Rows R, G, B; columns left R, G, B then right R, G, B.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnaglyphMatrix(pub [[f32; 6]; 3]);

/// Published matrices selectable by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AnaglyphPreset {
    /// Dubois least-squares, 2001 (report Set A). The default.
    // kebab-case puts no hyphen before digits, so the year needs an explicit name.
    #[serde(rename = "dubois-2001")]
    Dubois2001,
    /// Dubois least-squares, 2009, designed for CRT and plasma (report Set B).
    #[serde(rename = "dubois-2009")]
    Dubois2009,
    /// LCD monitor with 5450C filters, attributed to Zhang and McAllister (report Set C).
    ZhangLcd,
    /// Red from the left eye, green and blue from the right.
    Color,
    /// Left luma into red, right green and blue unchanged.
    HalfColor,
    /// Left luma into red, right luma into green and blue.
    Gray,
}

const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

impl AnaglyphPreset {
    pub const ALL: [AnaglyphPreset; 6] = [
        Self::Dubois2001,
        Self::Dubois2009,
        Self::ZhangLcd,
        Self::Color,
        Self::HalfColor,
        Self::Gray,
    ];

    /// The name the preset has in TOML.
    pub fn name(self) -> &'static str {
        match self {
            Self::Dubois2001 => "dubois-2001",
            Self::Dubois2009 => "dubois-2009",
            Self::ZhangLcd => "zhang-lcd",
            Self::Color => "color",
            Self::HalfColor => "half-color",
            Self::Gray => "gray",
        }
    }

    /// The preset after this one in [`AnaglyphPreset::ALL`], wrapping around.
    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|&p| p == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    pub fn matrix(self) -> AnaglyphMatrix {
        let [ly, cy, by] = LUMA;
        AnaglyphMatrix(match self {
            Self::Dubois2001 => [
                [
                    0.456100,
                    0.500484,
                    0.176381,
                    -0.0434706,
                    -0.0879388,
                    -0.00155529,
                ],
                [
                    -0.0400822, -0.0378246, -0.0157589, 0.378476, 0.73364, -0.0184503,
                ],
                [
                    -0.0152161,
                    -0.0205971,
                    -0.00546856,
                    -0.0721527,
                    -0.112961,
                    1.2264,
                ],
            ],
            Self::Dubois2009 => [
                [0.437, 0.449, 0.164, -0.011, -0.032, -0.007],
                [-0.062, -0.062, -0.024, 0.377, 0.761, 0.009],
                [-0.048, -0.050, -0.017, -0.026, -0.093, 1.234],
            ],
            Self::ZhangLcd => [
                [0.4154, 0.4710, 0.1669, -0.0109, -0.0364, -0.0060],
                [-0.0458, -0.0484, -0.0257, 0.3756, 0.7333, 0.0111],
                [-0.0547, -0.0615, 0.0128, -0.0651, -0.1287, 1.2971],
            ],
            Self::Color => [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            Self::HalfColor => [
                [ly, cy, by, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            Self::Gray => [
                [ly, cy, by, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, ly, cy, by],
                [0.0, 0.0, 0.0, ly, cy, by],
            ],
        })
    }
}

/// A preset name or an explicit matrix, as written in TOML.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MatrixChoice {
    Preset(AnaglyphPreset),
    Custom(AnaglyphMatrix),
}

impl MatrixChoice {
    pub fn matrix(self) -> AnaglyphMatrix {
        match self {
            Self::Preset(preset) => preset.matrix(),
            Self::Custom(matrix) => matrix,
        }
    }
}

impl Default for MatrixChoice {
    fn default() -> Self {
        Self::Preset(AnaglyphPreset::Dubois2001)
    }
}

/// Anaglyph output settings. Least-squares output is dark, hence the brightness and
/// gamma controls applied after the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AnaglyphSettings {
    pub matrix: MatrixChoice,
    pub brightness: f32,
    pub gamma: f32,
}

impl Default for AnaglyphSettings {
    fn default() -> Self {
        Self {
            matrix: MatrixChoice::default(),
            brightness: 1.0,
            gamma: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_preset_row_sums_to_one() {
        for preset in AnaglyphPreset::ALL {
            for (index, row) in preset.matrix().0.iter().enumerate() {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() <= 0.001,
                    "{preset:?} row {index} sums to {sum}"
                );
            }
        }
    }

    #[test]
    fn test_presets_are_distinct() {
        for (i, a) in AnaglyphPreset::ALL.iter().enumerate() {
            for b in &AnaglyphPreset::ALL[i + 1..] {
                assert_ne!(a.matrix(), b.matrix(), "{a:?} equals {b:?}");
            }
        }
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Wrapper {
        matrix: MatrixChoice,
    }

    #[test]
    fn test_matrix_choice_parses_preset_name() {
        let wrapper: Wrapper = toml::from_str("matrix = \"half-color\"").expect("parse");
        assert_eq!(
            wrapper.matrix,
            MatrixChoice::Preset(AnaglyphPreset::HalfColor)
        );
    }

    #[test]
    fn test_matrix_choice_parses_custom_rows_and_round_trips() {
        let text = "matrix = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]";
        let wrapper: Wrapper = toml::from_str(text).expect("parse");
        assert_eq!(wrapper.matrix.matrix(), AnaglyphPreset::Color.matrix());
        let again: Wrapper =
            toml::from_str(&toml::to_string(&wrapper).expect("serialize")).expect("reparse");
        assert_eq!(again, wrapper);
    }

    #[test]
    fn test_preset_names_match_documentation() {
        let names = [
            "dubois-2001",
            "dubois-2009",
            "zhang-lcd",
            "color",
            "half-color",
            "gray",
        ];
        for (preset, name) in AnaglyphPreset::ALL.into_iter().zip(names) {
            let text = toml::to_string(&Wrapper {
                matrix: MatrixChoice::Preset(preset),
            })
            .expect("serialize");
            assert_eq!(text.trim(), format!("matrix = \"{name}\""), "{preset:?}");
            assert_eq!(preset.name(), name);
        }
    }

    #[test]
    fn test_next_cycles_through_every_preset() {
        let mut preset = AnaglyphPreset::Dubois2001;
        for expected in AnaglyphPreset::ALL.into_iter().skip(1) {
            preset = preset.next();
            assert_eq!(preset, expected);
        }
        assert_eq!(preset.next(), AnaglyphPreset::Dubois2001);
    }

    #[test]
    fn test_unknown_preset_name_is_rejected() {
        assert!(toml::from_str::<Wrapper>("matrix = \"dubois\"").is_err());
    }
}
