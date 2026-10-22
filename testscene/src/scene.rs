//! Backend-independent scene definitions.
//!
//! Scenes describe geometry, camera and ground truth only; the Vulkan backend here and the
//! later D3D12 backend both consume them unchanged, so captures from either API can be
//! checked against the same sidecar data.

use clap::ValueEnum;
use glam::{Mat4, Vec3, camera::rh};

/// Background colour of every scene, chosen to differ from all marker colours.
pub const CLEAR_COLOR: [u8; 3] = [16, 16, 16];

/// Scenes the test application can render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SceneId {
    /// Marker grid at known depths for the G0 geometry sanity test.
    G0,
}

impl SceneId {
    /// Short identifier used in file names and sidecars.
    pub fn name(self) -> &'static str {
        match self {
            Self::G0 => "g0",
        }
    }
}

/// Right-handed, Y-up camera looking along `target - position`.
#[derive(Debug, Clone, Copy)]
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

/// An unlit, camera-facing square with a unique colour.
#[derive(Debug, Clone, Copy)]
pub struct Marker {
    pub id: u32,
    pub color: [u8; 3],
    /// Centre in world space.
    pub center: Vec3,
    /// Half the edge length in world units.
    pub half_size: f32,
    /// Distance from the camera plane, equal to clip-space `w` at the centre.
    pub depth: f32,
}

/// One vertex of the scene mesh.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    /// Linear colour in [0, 1]; written unchanged to a UNORM target.
    pub color: [f32; 3],
}

/// A complete scene: camera, stereo reference depth and geometry.
#[derive(Debug, Clone)]
pub struct Scene {
    pub id: SceneId,
    pub camera: Camera,
    pub aspect: f32,
    /// Depth at which the stereo milestones place zero parallax.
    pub convergence: f32,
    pub markers: Vec<Marker>,
}

impl Scene {
    /// Builds the scene `id` for a target of the given aspect ratio.
    pub fn build(id: SceneId, aspect: f32) -> Self {
        match id {
            SceneId::G0 => g0(aspect),
        }
    }

    /// Combined view and projection transform.
    pub fn view_proj(&self) -> Mat4 {
        self.camera.projection(self.aspect) * self.camera.view()
    }

    /// Triangle list covering every marker, two triangles per marker.
    pub fn vertices(&self) -> Vec<Vertex> {
        self.markers.iter().flat_map(marker_vertices).collect()
    }
}

const G0_DEPTHS: [f32; 5] = [1.5, 2.5, 4.0, 8.0, 20.0];
const G0_COLUMNS_NDC: [f32; 5] = [-0.72, -0.36, 0.0, 0.36, 0.72];
const G0_ROWS_NDC: [f32; 3] = [0.5, 0.0, -0.5];
const G0_HALF_SIZE_NDC: f32 = 0.06;

/// Places each depth in its own column, so no marker can hide another, and scales every
/// marker by its depth so all of them cover the same screen area.
fn g0(aspect: f32) -> Scene {
    let camera = Camera {
        position: Vec3::ZERO,
        target: Vec3::NEG_Z,
        up: Vec3::Y,
        vertical_fov_deg: 60.0,
        near: 0.1,
        far: 100.0,
    };
    let tan_half_v = (camera.vertical_fov_deg.to_radians() / 2.0).tan();
    let tan_half_h = tan_half_v * aspect;

    let mut markers = Vec::with_capacity(G0_ROWS_NDC.len() * G0_DEPTHS.len());
    for (row, &v) in G0_ROWS_NDC.iter().enumerate() {
        for (column, (&u, &depth)) in G0_COLUMNS_NDC.iter().zip(&G0_DEPTHS).enumerate() {
            markers.push(Marker {
                id: (row * G0_DEPTHS.len() + column) as u32,
                color: g0_color(row, column),
                center: Vec3::new(u * depth * tan_half_h, v * depth * tan_half_v, -depth),
                half_size: G0_HALF_SIZE_NDC * depth * tan_half_v,
                depth,
            });
        }
    }

    Scene {
        id: SceneId::G0,
        camera,
        aspect,
        convergence: 4.0,
        markers,
    }
}

/// Unique per (row, column): red and blue encode the column, green the row.
fn g0_color(row: usize, column: usize) -> [u8; 3] {
    let column = column as u8;
    let row = row as u8;
    [55 + 50 * column, 60 + 70 * row, 220 - 40 * column]
}

fn marker_vertices(marker: &Marker) -> [Vertex; 6] {
    let color = marker.color.map(|channel| f32::from(channel) / 255.0);
    let h = marker.half_size;
    let corner = |dx: f32, dy: f32| Vertex {
        position: marker.center + Vec3::new(dx * h, dy * h, 0.0),
        color,
    };
    [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ]
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
    fn test_g0_markers_unique_colors_distinct_from_clear() {
        let scene = Scene::build(SceneId::G0, ASPECT);
        let colors: HashSet<_> = scene.markers.iter().map(|m| m.color).collect();
        assert_eq!(colors.len(), scene.markers.len());
        assert!(!colors.contains(&CLEAR_COLOR));
    }

    #[test]
    fn test_g0_markers_some_nearer_than_convergence() {
        let scene = Scene::build(SceneId::G0, ASPECT);
        let near = scene
            .markers
            .iter()
            .filter(|m| m.depth < scene.convergence)
            .count();
        let at = scene
            .markers
            .iter()
            .filter(|m| m.depth == scene.convergence)
            .count();
        assert_eq!(near, 6);
        assert_eq!(at, 3);
    }

    #[test]
    fn test_g0_centers_project_to_layout_with_y_down() {
        let scene = Scene::build(SceneId::G0, ASPECT);
        for marker in &scene.markers {
            let row = marker.id as usize / G0_DEPTHS.len();
            let column = marker.id as usize % G0_DEPTHS.len();
            let projected = ndc(&scene, marker.center);
            assert!(
                (projected.x - G0_COLUMNS_NDC[column]).abs() < 1e-5,
                "{marker:?}"
            );
            assert!((projected.y + G0_ROWS_NDC[row]).abs() < 1e-5, "{marker:?}");
            assert!((0.0..1.0).contains(&projected.z), "{marker:?}");
        }
    }

    #[test]
    fn test_g0_markers_equal_screen_size_and_no_overlap() {
        let scene = Scene::build(SceneId::G0, ASPECT);
        let rects: Vec<_> = scene
            .markers
            .iter()
            .map(|m| {
                let low = ndc(&scene, m.center - Vec3::new(m.half_size, m.half_size, 0.0));
                let high = ndc(&scene, m.center + Vec3::new(m.half_size, m.half_size, 0.0));
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
    fn test_scene_vertices_six_per_marker() {
        let scene = Scene::build(SceneId::G0, ASPECT);
        assert_eq!(scene.vertices().len(), scene.markers.len() * 6);
    }
}
