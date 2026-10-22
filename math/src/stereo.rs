//! Stereo geometry: the per-eye clip-space shear and its effect on screen.
//!
//! Conventions follow the research report. `separation` is the per-eye shear coefficient
//! `s`: the parallax at infinity as a fraction of screen width
//! ([`StereoParams::separation`](s3d_shared::StereoParams::separation)). `w` is clip-space
//! w, equal to view depth. NDC spans 2 units across the screen, so a shift of `k` screen
//! widths is `2k` in NDC.

use s3d_shared::StereoPushConstants;

/// Clip-space x after the shear a patched shader applies with `block`; the formula in
/// the shader and in this function must stay identical.
pub fn shear_clip_x(x: f32, w: f32, block: &StereoPushConstants) -> f32 {
    x + block.eye_sign * block.separation * (w - block.convergence)
}

/// Right-eye minus left-eye position of a point at depth `w`, in screen widths. Zero at
/// convergence, negative (crossed, in front of the screen) nearer, approaching
/// `separation` at infinity.
pub fn disparity(separation: f32, convergence: f32, w: f32) -> f32 {
    separation * (1.0 - convergence / w)
}

/// Distance between the two equivalent off-axis cameras, in world units, for a
/// horizontal field of view `hfov` in radians.
pub fn interaxial(separation: f32, convergence: f32, hfov: f32) -> f32 {
    2.0 * separation * convergence * (hfov / 2.0).tan()
}

/// Horizontal field of view in radians from the vertical one and the aspect ratio.
pub fn horizontal_fov(vfov: f32, aspect: f32) -> f32 {
    2.0 * ((vfov / 2.0).tan() * aspect).atan()
}

/// Width in millimetres of a screen given its diagonal in inches and its aspect ratio.
pub fn screen_width_mm(diagonal_inches: f32, aspect_width: f32, aspect_height: f32) -> f32 {
    diagonal_inches * 25.4 * aspect_width / aspect_width.hypot(aspect_height)
}

/// The physical setup that turns on-screen parallax into an angle at the eye, which is
/// what comfort limits are stated in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewing {
    pub screen_width_mm: f32,
    pub distance_mm: f32,
}

impl Viewing {
    /// On-screen separation that subtends `degrees` at the viewer.
    pub fn parallax_mm(&self, degrees: f32) -> f32 {
        2.0 * self.distance_mm * (degrees.to_radians() / 2.0).tan()
    }

    /// Angle at the viewer subtended by an on-screen separation of `mm`.
    pub fn parallax_degrees(&self, mm: f32) -> f32 {
        (2.0 * (mm / (2.0 * self.distance_mm)).atan()).to_degrees()
    }

    /// `max_parallax_percent` that keeps the parallax at infinity at `degrees`.
    pub fn percent_for_degrees(&self, degrees: f32) -> f32 {
        self.parallax_mm(degrees) / self.screen_width_mm * 100.0
    }

    /// Angle at the viewer produced by a `max_parallax_percent` of `percent`.
    pub fn degrees_for_percent(&self, percent: f32) -> f32 {
        self.parallax_degrees(percent / 100.0 * self.screen_width_mm)
    }
}

#[cfg(test)]
mod tests {
    use s3d_shared::{Eye, StereoParams};

    use super::*;

    const DESK: Viewing = Viewing {
        screen_width_mm: 531.0,
        distance_mm: 700.0,
    };

    fn close(actual: f32, expected: f32, tolerance: f32) -> bool {
        (actual - expected).abs() <= tolerance
    }

    #[test]
    fn test_24_inch_16_9_screen_is_531_mm_wide() {
        let width = screen_width_mm(24.0, 16.0, 9.0);
        assert!(close(width, 531.3, 0.5), "{width}");
    }

    #[test]
    fn test_comfort_limit_matches_report_figures() {
        let mm = DESK.parallax_mm(1.5);
        assert!(close(mm, 18.3, 0.05), "{mm}");
        let s = DESK.percent_for_degrees(1.5) / 100.0;
        assert!(close(s, 0.0345, 0.0002), "{s}");
        assert!(close(DESK.degrees_for_percent(s * 100.0), 1.5, 1e-4));
    }

    #[test]
    fn test_physical_ipd_at_infinity_is_five_degrees() {
        let degrees = DESK.parallax_degrees(65.0);
        assert!(close(degrees, 5.31, 0.02), "{degrees}");
        assert!(close(degrees / 1.5, 3.5, 0.05));
    }

    #[test]
    fn test_disparity_zero_at_convergence_crossed_near_uncrossed_far() {
        let (s, conv) = (0.0345, 4.0);
        assert_eq!(disparity(s, conv, conv), 0.0);
        assert!(disparity(s, conv, 2.0) < 0.0);
        assert!(disparity(s, conv, 8.0) > 0.0);
        assert!(close(disparity(s, conv, 1e9), s, 1e-6));
    }

    #[test]
    fn test_near_plane_at_fifth_of_convergence_gives_minus_four_s() {
        let (s, conv) = (0.0345, 4.0);
        assert!(close(disparity(s, conv, conv / 5.0), -4.0 * s, 1e-6));
    }

    #[test]
    fn test_left_eye_moves_near_vertex_right() {
        let params = StereoParams::default();
        let block = params.push_constants(Some(Eye::Left));
        assert!(shear_clip_x(0.0, params.convergence / 2.0, &block) > 0.0);
        let mono = params.push_constants(None);
        assert_eq!(shear_clip_x(0.3, 9.0, &mono), 0.3);
    }

    #[test]
    fn test_shear_of_both_eyes_reproduces_disparity_formula() {
        let params = StereoParams::default();
        let [left, right] = Eye::BOTH.map(|eye| params.push_constants(Some(eye)));
        for w in [0.5, 2.0, 4.0, 13.0, 400.0] {
            let x = 0.2 * w;
            let ndc_shift = (shear_clip_x(x, w, &right) - shear_clip_x(x, w, &left)) / w;
            let expected = disparity(params.separation(), params.convergence, w);
            assert!(close(ndc_shift / 2.0, expected, 1e-6), "w = {w}");
        }
    }

    #[test]
    fn test_interaxial_for_test_scene_camera() {
        let hfov = horizontal_fov(60f32.to_radians(), 16.0 / 9.0);
        let e = interaxial(0.0345, 4.0, hfov);
        assert!(close(e, 0.2833, 0.0005), "{e}");
    }
}
