//! Backend-independent scene definitions.
//!
//! Scenes describe geometry, camera and ground truth only; the Vulkan backend here and the
//! later D3D12 backend both consume them unchanged, so captures from either API can be
//! checked against the same sidecar data. A scene is rebuilt for every frame from the frame
//! index alone, which keeps animation deterministic.

mod builders;
mod geometry;

use clap::ValueEnum;
use glam::{Mat3, Mat4, Vec3, camera::rh};
pub use s3d_shared::capture::MarkerKind;

/// Background colour of every scene, chosen to differ from all marker colours.
pub const CLEAR_COLOR: [u8; 3] = [16, 16, 16];

/// Fixed time step: frame `n` shows time `n / FRAME_RATE` seconds.
pub const FRAME_RATE: f32 = 60.0;

/// Exposure of the tone-map pass, handed to `tonemap.frag` as specialization constant 0.
pub const TONEMAP_EXPOSURE: f32 = 1.6;

/// An 8-bit colour as it appears after the tone-map pass (Reinhard on exposed input).
pub fn tonemapped(color: [u8; 3]) -> [u8; 3] {
    color.map(|channel| {
        let exposed = f32::from(channel) / 255.0 * TONEMAP_EXPOSURE;
        (exposed / (1.0 + exposed) * 255.0).round() as u8
    })
}

/// Scenes the test application can render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SceneId {
    /// Marker grid at known depths (G0 geometry sanity).
    G0,
    /// Pillar in front of a wall hiding a high-frequency marker (G1 disocclusion).
    G1,
    /// Alpha-blended quad stack and particles (G2 transparents).
    G2,
    /// Lateral mover at convergence depth plus a frame counter (G3 same-frame).
    G3,
    /// Glossy sphere and reflector under a fixed light (G5 view-dependent shading).
    G5,
    /// Depth-only shadow-map pass feeding a lit scene.
    Shadow,
    /// Screen-space HUD pass drawn at w = 1.
    Hud,
    /// Skybox drawn on the far plane.
    Sky,
    /// Scene rendered to an HDR target and tone-mapped to the swapchain.
    Post,
    /// Every element above in one frame, for smoke runs.
    All,
}

impl SceneId {
    /// Short identifier used in file names and sidecars.
    pub fn name(self) -> &'static str {
        match self {
            Self::G0 => "g0",
            Self::G1 => "g1",
            Self::G2 => "g2",
            Self::G3 => "g3",
            Self::G5 => "g5",
            Self::Shadow => "shadow",
            Self::Hud => "hud",
            Self::Sky => "sky",
            Self::Post => "post",
            Self::All => "all",
        }
    }
}

/// Right-handed, Y-up camera looking along `target - position`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub vertical_fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// World-to-view transform.
    pub fn view(&self) -> Mat4 {
        rh::view::look_at_mat4(self.position, self.target, self.up)
    }

    /// Vulkan-convention projection (Y down, depth in [0, 1]).
    pub fn projection(&self, aspect: f32) -> Mat4 {
        rh::proj::vulkan::perspective(
            self.vertical_fov_deg.to_radians(),
            aspect,
            self.near,
            self.far,
        )
    }
}

/// Which pipeline draws a mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    /// Vertex colour, opaque.
    Unlit,
    /// Fine checker between the vertex colour and green, opaque.
    Pattern,
    /// Blinn-Phong with shadows from the frame's light.
    Lit,
    /// Mirror-like reflection of the procedural environment.
    Reflective,
    /// Vertex colour with alpha blending and no depth writes; drawn after the skybox.
    Transparent,
}

/// One vertex of any mesh. HUD vertices carry clip-space positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    /// Linear RGBA in [0, 1]; written unchanged to UNORM targets.
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

/// Triangle list drawn with one pipeline and one model matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub material: Material,
    pub model: Mat4,
    pub vertices: Vec<Vertex>,
}

/// Directional light, optionally casting shadows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// Unit vector pointing toward the light.
    pub direction: Vec3,
    pub color: Vec3,
    pub ambient: f32,
    pub shadow: Option<ShadowVolume>,
}

/// Box covered by the orthographic shadow map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowVolume {
    pub center: Vec3,
    pub half_extent: f32,
    /// Distance from `center` to the light's eye along the light direction.
    pub distance: f32,
}

impl ShadowVolume {
    /// Light view-projection for a light shining from `direction`.
    pub fn view_proj(&self, direction: Vec3) -> Mat4 {
        let eye = self.center + direction * self.distance;
        let up = if direction.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let view = rh::view::look_at_mat4(eye, self.center, up);
        let h = self.half_extent;
        let projection = rh::proj::vulkan::orthographic(-h, h, -h, h, 0.1, self.distance * 2.0);
        projection * view
    }
}

/// Ground truth for one identifiable object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marker {
    pub id: u32,
    pub kind: MarkerKind,
    pub color: [u8; 3],
    pub alpha: f32,
    /// World position, or clip-space position with z = 0 for HUD markers.
    pub center: Vec3,
    /// Half extents along the camera's right and up axes, in world (or clip) units.
    pub half_extent: [f32; 2],
    /// Distance from the camera plane, equal to clip-space `w`; 1 for HUD markers.
    pub depth: f32,
}

/// One cell of the frame counter, read most significant bit first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    pub bit: u32,
    /// Vulkan NDC of the cell centre (y down).
    pub ndc: [f32; 2],
    pub on: [u8; 3],
    pub off: [u8; 3],
}

/// A complete frame of one scene.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    pub id: SceneId,
    pub frame: u32,
    pub camera: Camera,
    pub aspect: f32,
    /// Depth at which the stereo milestones place zero parallax.
    pub convergence: f32,
    pub light: Option<Light>,
    pub sky: bool,
    /// Render to an HDR target and tone-map it to the swapchain.
    pub post: bool,
    /// Opaque meshes in draw order, then transparent meshes back to front.
    pub meshes: Vec<Mesh>,
    /// Clip-space triangles for the HUD pass.
    pub hud: Vec<Vertex>,
    pub markers: Vec<Marker>,
    pub probes: Vec<Probe>,
    /// Value the counter probes encode, when the scene has a counter.
    pub counter: Option<u32>,
}

impl Scene {
    /// Builds frame `frame` of scene `id` for a target of the given aspect ratio.
    pub fn build(id: SceneId, aspect: f32, frame: u32) -> Self {
        builders::build(id, aspect, frame)
    }

    /// Combined view and projection transform.
    pub fn view_proj(&self) -> Mat4 {
        self.camera.projection(self.aspect) * self.camera.view()
    }

    /// View rotation (no translation) and projection, for the skybox.
    pub fn sky_view_proj(&self) -> Mat4 {
        let rotation = Mat4::from_mat3(Mat3::from_mat4(self.camera.view()));
        self.camera.projection(self.aspect) * rotation
    }

    /// Light view-projection when the light casts shadows.
    pub fn light_view_proj(&self) -> Option<Mat4> {
        let light = self.light?;
        light.shadow.map(|volume| volume.view_proj(light.direction))
    }

    /// Unit cube drawn around the camera when `sky` is set.
    pub fn sky_vertices(&self) -> Vec<Vertex> {
        geometry::cuboid(Vec3::ZERO, Vec3::ONE, [0.0; 4])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const ASPECT: f32 = 1280.0 / 720.0;

    fn ndc(scene: &Scene, point: Vec3) -> Vec3 {
        let clip = scene.view_proj() * point.extend(1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn test_every_scene_builds_deterministically_with_unique_ids() {
        for id in SceneId::value_variants() {
            for frame in [0, 37] {
                let scene = Scene::build(*id, ASPECT, frame);
                assert_eq!(scene, Scene::build(*id, ASPECT, frame), "{id:?}");
                assert!(!scene.meshes.is_empty(), "{id:?}");
                let ids: HashSet<_> = scene.markers.iter().map(|m| m.id).collect();
                assert_eq!(ids.len(), scene.markers.len(), "{id:?}");
            }
        }
    }

    #[test]
    fn test_every_scene_marker_colors_distinct_from_clear() {
        for id in SceneId::value_variants() {
            let scene = Scene::build(*id, ASPECT, 0);
            assert!(
                scene.markers.iter().all(|m| m.color != CLEAR_COLOR),
                "{id:?}"
            );
        }
    }

    #[test]
    fn test_g0_markers_unique_colors_and_six_nearer_than_convergence() {
        let scene = Scene::build(SceneId::G0, ASPECT, 0);
        let colors: HashSet<_> = scene.markers.iter().map(|m| m.color).collect();
        assert_eq!(colors.len(), 15);
        let near = scene
            .markers
            .iter()
            .filter(|m| m.depth < scene.convergence)
            .count();
        assert_eq!(near, 6);
    }

    #[test]
    fn test_g0_markers_equal_screen_size_and_no_overlap() {
        let scene = Scene::build(SceneId::G0, ASPECT, 0);
        let rects: Vec<_> = scene
            .markers
            .iter()
            .map(|m| {
                let [hx, hy] = m.half_extent;
                let low = ndc(&scene, m.center - Vec3::new(hx, hy, 0.0));
                let high = ndc(&scene, m.center + Vec3::new(hx, hy, 0.0));
                (low.x, high.x, high.y.min(low.y), high.y.max(low.y))
            })
            .collect();
        let height = rects[0].3 - rects[0].2;
        for rect in &rects {
            assert!((rect.3 - rect.2 - height).abs() < 1e-5);
        }
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                let disjoint = a.1 < b.0 || b.1 < a.0 || a.3 < b.2 || b.3 < a.2;
                assert!(disjoint, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn test_g1_pattern_hidden_behind_pillar_in_mono() {
        let scene = Scene::build(SceneId::G1, ASPECT, 0);
        let marker = |id| scene.markers.iter().find(|m| m.id == id).expect("marker");
        let (pattern, pillar) = (marker(100), marker(101));
        let x_range = |m: &Marker| {
            let half = Vec3::new(m.half_extent[0], 0.0, 0.0);
            let near_face = m.center + Vec3::new(0.0, 0.0, m.half_extent[0]);
            let center = if m.id == 101 { near_face } else { m.center };
            (ndc(&scene, center - half).x, ndc(&scene, center + half).x)
        };
        let (pattern_min, pattern_max) = x_range(pattern);
        let (pillar_min, pillar_max) = x_range(pillar);
        assert!(pattern_min > pillar_min && pattern_max < pillar_max);
        assert!(pattern_min > 0.0, "pattern must sit on one side only");
    }

    #[test]
    fn test_g2_transparent_mesh_sorted_back_to_front() {
        let scene = Scene::build(SceneId::G2, ASPECT, 5);
        let transparent: Vec<_> = scene
            .meshes
            .iter()
            .filter(|m| m.material == Material::Transparent)
            .collect();
        assert_eq!(transparent.len(), 1);
        let depths: Vec<f32> = transparent[0]
            .vertices
            .chunks(6)
            .map(|quad| -quad[0].position.z)
            .collect();
        assert!(depths.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn test_g3_counter_and_mover_follow_frame() {
        let frame = 0b1010_0000_0000_0101;
        let scene = Scene::build(SceneId::G3, ASPECT, frame);
        assert_eq!(scene.counter, Some(frame));
        assert_eq!(scene.probes.len(), 16);
        let mover = |frame| {
            let scene = Scene::build(SceneId::G3, ASPECT, frame);
            let marker = scene.markers.iter().find(|m| m.id == 400).expect("mover");
            ndc(&scene, marker.center).x
        };
        assert!((mover(1) - mover(0) - 2.0 / FRAME_RATE).abs() < 1e-5);
    }

    #[test]
    fn test_post_scene_probes_expect_tonemapped_colors() {
        assert_eq!(tonemapped([255, 255, 255]), [157, 157, 157]);
        let plain = Scene::build(SceneId::G3, ASPECT, 0);
        let post = Scene::build(SceneId::All, ASPECT, 0);
        assert_eq!(post.probes[0].on, tonemapped(plain.probes[0].on));
        assert_eq!(post.probes[0].off, tonemapped(plain.probes[0].off));
    }

    #[test]
    fn test_shadow_volume_contains_casters() {
        let scene = Scene::build(SceneId::Shadow, ASPECT, 0);
        let light = scene.light_view_proj().expect("shadowed light");
        for marker in &scene.markers {
            let p = light.project_point3(marker.center);
            assert!(p.x.abs() < 1.0 && p.y.abs() < 1.0, "{marker:?}");
            assert!((0.0..1.0).contains(&p.z), "{marker:?}");
        }
    }

    #[test]
    fn test_hud_markers_are_clip_space() {
        let scene = Scene::build(SceneId::Hud, ASPECT, 0);
        let hud: Vec<_> = scene
            .markers
            .iter()
            .filter(|m| m.kind == MarkerKind::Hud)
            .collect();
        assert!(!hud.is_empty() && !scene.hud.is_empty());
        assert!(hud.iter().all(|m| m.depth == 1.0 && m.center.z == 0.0));
    }
}
