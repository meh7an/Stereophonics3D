//! Eyes, user-facing stereo parameters and the push-constant block patched shaders read.

use std::mem::{offset_of, size_of};

use serde::{Deserialize, Serialize};

/// One of the two views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Eye {
    Left,
    Right,
}

impl Eye {
    /// Both eyes, left first, matching their view indices.
    pub const BOTH: [Eye; 2] = [Eye::Left, Eye::Right];

    /// Sign of this eye's clip-space shear.
    ///
    /// The left eye takes `-s`, so a vertex nearer than convergence (`w < conv`) moves
    /// right in the left image. This follows the research report's derivation and
    /// contradicts NVIDIA's text; test G0 is what confirms it.
    pub fn sign(self) -> f32 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }

    /// Multiview `ViewIndex` (Vulkan) or `SV_ViewID` (D3D12) that renders this eye.
    pub fn view_index(self) -> u32 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }

    /// Lowercase name used in file names.
    pub fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// Stereo settings as the player controls them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StereoParams {
    /// Disparity between the eyes at infinite depth, in percent of screen width.
    pub max_parallax_percent: f32,
    /// Clip-space `w` (view depth) at which both eyes coincide.
    pub convergence: f32,
    /// Exchange the eyes, for glasses worn the other way round.
    pub swap_eyes: bool,
    /// Single-eye mode: the whole frame renders as this eye. Absent renders mono.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eye: Option<Eye>,
}

impl Default for StereoParams {
    /// Conservative starting values until reconnaissance (M0.3) supplies measured ones:
    /// 3 % stays under the report's 1.5 degree comfort limit (about 3.45 % on a 24-inch
    /// screen at 700 mm), and convergence 4 matches the test scene.
    fn default() -> Self {
        Self {
            max_parallax_percent: 3.0,
            convergence: 4.0,
            swap_eyes: false,
            eye: None,
        }
    }
}

impl StereoParams {
    /// Per-eye shear coefficient `s`: the parallax at infinity as a fraction of screen
    /// width. Negated when the eyes are swapped, so swapping works under multiview too.
    pub fn separation(&self) -> f32 {
        let s = self.max_parallax_percent / 100.0;
        if self.swap_eyes { -s } else { s }
    }

    /// Push constants rendering `eye`, or mono when `eye` is `None`.
    pub fn push_constants(&self, eye: Option<Eye>) -> StereoPushConstants {
        StereoPushConstants {
            separation: self.separation(),
            convergence: self.convergence,
            eye_sign: eye.map_or(0.0, Eye::sign),
            reserved: 0.0,
        }
    }
}

/// Block the layer appends to a pipeline layout's push constants.
///
/// A patched last pre-raster stage computes
/// `position.x += eye_sign * separation * (position.w - convergence)`. Under multiview the
/// shader derives the sign from `ViewIndex` and ignores `eye_sign`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPushConstants {
    pub separation: f32,
    pub convergence: f32,
    /// -1 left, +1 right, 0 mono.
    pub eye_sign: f32,
    pub reserved: f32,
}

impl StereoPushConstants {
    pub const SIZE: u32 = 16;
    /// One `vec4`: keeps the block std430-aligned wherever it is appended.
    pub const ALIGN: u32 = 16;
    pub const SEPARATION_OFFSET: u32 = offset_of!(Self, separation) as u32;
    pub const CONVERGENCE_OFFSET: u32 = offset_of!(Self, convergence) as u32;
    pub const EYE_SIGN_OFFSET: u32 = offset_of!(Self, eye_sign) as u32;

    /// Offset of the block when the application's push-constant ranges end at `app_end`.
    pub const fn append_offset(app_end: u32) -> u32 {
        app_end.next_multiple_of(Self::ALIGN)
    }

    /// Bytes for `vkCmdPushConstants`, in field order.
    pub fn to_bytes(&self) -> [u8; Self::SIZE as usize] {
        let mut bytes = [0; Self::SIZE as usize];
        let fields = [
            self.separation,
            self.convergence,
            self.eye_sign,
            self.reserved,
        ];
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (chunk, value) in chunks.iter_mut().zip(fields) {
            *chunk = value.to_ne_bytes();
        }
        bytes
    }
}

const _: () = {
    assert!(size_of::<StereoPushConstants>() == StereoPushConstants::SIZE as usize);
    assert!(StereoPushConstants::SEPARATION_OFFSET == 0);
    assert!(StereoPushConstants::CONVERGENCE_OFFSET == 4);
    assert!(StereoPushConstants::EYE_SIGN_OFFSET == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_sign_left_negative_right_positive() {
        assert_eq!(Eye::Left.sign(), -1.0);
        assert_eq!(Eye::Right.sign(), 1.0);
        assert_eq!(Eye::BOTH.map(Eye::view_index), [0, 1]);
    }

    #[test]
    fn test_left_eye_near_vertex_moves_right() {
        let params = StereoParams::default();
        let block = params.push_constants(Some(Eye::Left));
        let (x, w) = (0.0, params.convergence / 2.0);
        let shifted = x + block.eye_sign * block.separation * (w - block.convergence);
        assert!(shifted > x);
    }

    #[test]
    fn test_push_constants_mono_has_zero_sign() {
        let block = StereoParams::default().push_constants(None);
        assert_eq!(block.eye_sign, 0.0);
        assert_eq!(block.separation, 0.03);
    }

    #[test]
    fn test_swap_eyes_negates_separation() {
        let params = StereoParams {
            swap_eyes: true,
            ..StereoParams::default()
        };
        assert_eq!(params.separation(), -0.03);
    }

    #[test]
    fn test_append_offset_rounds_up_to_vec4() {
        assert_eq!(StereoPushConstants::append_offset(0), 0);
        assert_eq!(StereoPushConstants::append_offset(64), 64);
        assert_eq!(StereoPushConstants::append_offset(68), 80);
    }

    #[test]
    fn test_to_bytes_follows_field_offsets() {
        let block = StereoPushConstants {
            separation: 1.0,
            convergence: 2.0,
            eye_sign: -1.0,
            reserved: 0.0,
        };
        let bytes = block.to_bytes();
        let at = |offset: u32| {
            let start = offset as usize;
            f32::from_ne_bytes(bytes[start..start + 4].try_into().expect("4 bytes"))
        };
        assert_eq!(at(StereoPushConstants::SEPARATION_OFFSET), 1.0);
        assert_eq!(at(StereoPushConstants::CONVERGENCE_OFFSET), 2.0);
        assert_eq!(at(StereoPushConstants::EYE_SIGN_OFFSET), -1.0);
    }

    #[test]
    fn test_stereo_params_toml_round_trip() {
        let params = StereoParams {
            max_parallax_percent: 2.5,
            convergence: 7.0,
            swap_eyes: true,
            eye: Some(Eye::Left),
        };
        let text = toml::to_string(&params).expect("serialize");
        assert_eq!(
            toml::from_str::<StereoParams>(&text).expect("parse"),
            params
        );
    }
}
