//! Mesh primitives. All are triangle lists with outward normals.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

use super::Vertex;

/// Linear RGBA from an 8-bit colour and an alpha.
pub fn rgba(color: [u8; 3], alpha: f32) -> [f32; 4] {
    let [r, g, b] = color.map(|channel| f32::from(channel) / 255.0);
    [r, g, b, alpha]
}

/// Rectangle spanned by `axis_u` and `axis_v` (unit vectors) around `center`, facing
/// `axis_u x axis_v`. Texture coordinates run from 0 to `uv_cells` across it.
pub fn quad(
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    half: [f32; 2],
    color: [f32; 4],
    uv_cells: [f32; 2],
) -> [Vertex; 6] {
    let normal = axis_u.cross(axis_v).normalize();
    let corner = |su: f32, sv: f32| Vertex {
        position: center + axis_u * (su * half[0]) + axis_v * (sv * half[1]),
        normal,
        color,
        uv: [
            (su + 1.0) * 0.5 * uv_cells[0],
            (sv + 1.0) * 0.5 * uv_cells[1],
        ],
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

/// View-space rectangle parallel to the screen, facing the camera.
pub fn facing_quad(center: Vec3, half: [f32; 2], color: [f32; 4]) -> [Vertex; 6] {
    quad(center, Vec3::X, Vec3::Y, half, color, [1.0, 1.0])
}

/// Clip-space HUD rectangle between `min` and `max` (Vulkan NDC, y down).
pub fn hud_rect(min: [f32; 2], max: [f32; 2], color: [f32; 4]) -> [Vertex; 6] {
    let center = Vec3::new((min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, 0.0);
    let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5];
    quad(center, Vec3::X, Vec3::Y, half, color, [1.0, 1.0])
}

/// Axis-aligned box.
pub fn cuboid(center: Vec3, half: Vec3, color: [f32; 4]) -> Vec<Vertex> {
    // (normal, u, v) with u x v = normal, so every face points outward.
    let faces = [
        (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
    ];
    faces
        .into_iter()
        .flat_map(|(normal, u, v)| {
            let face_center = center + normal * normal.abs().dot(half);
            let extent = [u.abs().dot(half), v.abs().dot(half)];
            quad(face_center, u, v, extent, color, [1.0, 1.0])
        })
        .collect()
}

/// UV sphere centred on the origin.
pub fn sphere(radius: f32, segments: u32, rings: u32, color: [f32; 4]) -> Vec<Vertex> {
    let point = |segment: u32, ring: u32| {
        let theta = segment as f32 / segments as f32 * TAU;
        let phi = ring as f32 / rings as f32 * PI;
        Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin())
    };
    let vertex = |normal: Vec3| Vertex {
        position: normal * radius,
        normal,
        color,
        uv: [0.0, 0.0],
    };
    let mut vertices = Vec::with_capacity((segments * rings * 6) as usize);
    for ring in 0..rings {
        for segment in 0..segments {
            let a = point(segment, ring);
            let b = point(segment + 1, ring);
            let c = point(segment + 1, ring + 1);
            let d = point(segment, ring + 1);
            vertices.extend([a, b, c, a, c, d].map(vertex));
        }
    }
    vertices
}

/// Deterministic hash of `seed` to [0, 1), so particles depend on the frame index only.
pub fn hash01(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277_803_737);
    x ^= x >> 22;
    (x >> 8) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuboid_normals_point_outward() {
        let vertices = cuboid(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 1.0, 2.0), [1.0; 4]);
        assert_eq!(vertices.len(), 36);
        for vertex in &vertices {
            let offset = vertex.position - Vec3::new(1.0, 2.0, 3.0);
            assert!(offset.dot(vertex.normal) > 0.0, "{vertex:?}");
        }
    }

    #[test]
    fn test_quad_normal_is_u_cross_v() {
        let vertices = quad(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            [1.0, 1.0],
            [1.0; 4],
            [1.0, 1.0],
        );
        assert!(vertices.iter().all(|v| v.normal == Vec3::Z));
    }

    #[test]
    fn test_hash01_in_unit_range_and_stable() {
        for seed in 0..1000 {
            let value = hash01(seed);
            assert!((0.0..1.0).contains(&value));
            assert_eq!(value, hash01(seed));
        }
    }
}
