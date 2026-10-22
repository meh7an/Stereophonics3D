//! CPU reference of the anaglyph compositor.
//!
//! Pipeline per pixel: decode both eyes from sRGB to linear light, apply the 3x6 matrix,
//! clamp to [0, 1], multiply by brightness, clamp again, raise to `1 / gamma`, encode to
//! sRGB. The GPU compositor (M5.3) must match this within one 8-bit step.

use s3d_shared::{AnaglyphMatrix, AnaglyphSettings};

/// sRGB transfer function, decoding one channel in [0, 1] to linear light.
pub fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer function, encoding one linear channel in [0, 1].
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// `matrix * [left, right]`, unclamped; negative coefficients can push it out of range.
pub fn apply_matrix(matrix: &AnaglyphMatrix, left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    let input = [left[0], left[1], left[2], right[0], right[1], right[2]];
    matrix.0.map(|row| {
        row.iter()
            .zip(input)
            .map(|(coefficient, value)| coefficient * value)
            .sum()
    })
}

/// Brightness and gamma on one linear channel, clamping before and after the gain so the
/// result stays displayable.
pub fn adjust(linear: f32, brightness: f32, gamma: f32) -> f32 {
    (linear.clamp(0.0, 1.0) * brightness)
        .clamp(0.0, 1.0)
        .powf(1.0 / gamma)
}

/// The full pipeline on one pair of 8-bit sRGB pixels.
pub fn compose(settings: &AnaglyphSettings, left: [u8; 3], right: [u8; 3]) -> [u8; 3] {
    let matrix = settings.matrix.matrix();
    let mixed = apply_matrix(&matrix, left.map(decode), right.map(decode));
    mixed.map(|channel| encode(adjust(channel, settings.brightness, settings.gamma)))
}

fn decode(channel: u8) -> f32 {
    srgb_to_linear(f32::from(channel) / 255.0)
}

fn encode(linear: f32) -> u8 {
    (linear_to_srgb(linear.clamp(0.0, 1.0)) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use s3d_shared::{AnaglyphPreset, MatrixChoice};

    use super::*;

    fn settings(preset: AnaglyphPreset) -> AnaglyphSettings {
        AnaglyphSettings {
            matrix: MatrixChoice::Preset(preset),
            ..AnaglyphSettings::default()
        }
    }

    #[test]
    fn test_srgb_round_trips_every_8_bit_value() {
        for value in 0..=255u8 {
            assert_eq!(encode(decode(value)), value);
        }
    }

    #[test]
    fn test_srgb_known_midpoint() {
        assert!((srgb_to_linear(0.5) - 0.2140).abs() < 1e-4);
        assert!((linear_to_srgb(0.2140) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_romeuf_red_example_in_encoded_space() {
        let matrix = AnaglyphPreset::ZhangLcd.matrix();
        let mixed = apply_matrix(&matrix, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let bytes = mixed.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
        assert_eq!(bytes, [103, 84, 0]);
    }

    #[test]
    fn test_grey_in_both_eyes_maps_to_itself_for_every_preset() {
        for preset in AnaglyphPreset::ALL {
            for value in 0..=255u8 {
                let [r, g, b] = compose(&settings(preset), [value; 3], [value; 3]);
                for channel in [r, g, b] {
                    assert!(
                        channel.abs_diff(value) <= 1,
                        "{preset:?}: grey {value} became {channel}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_left_eye_feeds_red_and_right_eye_feeds_cyan() {
        let dubois = settings(AnaglyphPreset::Dubois2001);
        let [r, g, b] = compose(&dubois, [255; 3], [0; 3]);
        assert!(r > 200 && g == 0 && b == 0, "{:?}", [r, g, b]);
        let [r, g, b] = compose(&dubois, [0; 3], [255; 3]);
        assert!(r == 0 && g > 200 && b > 200, "{:?}", [r, g, b]);
    }

    #[test]
    fn test_adjust_identity_brightness_and_gamma() {
        assert_eq!(adjust(0.25, 1.0, 1.0), 0.25);
        assert_eq!(adjust(0.25, 2.0, 1.0), 0.5);
        assert_eq!(adjust(0.75, 2.0, 1.0), 1.0);
        assert_eq!(adjust(-0.2, 1.0, 1.0), 0.0);
        assert!((adjust(0.25, 1.0, 2.0) - 0.5).abs() < 1e-6);
    }
}
