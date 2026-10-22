//! Scene recipes: camera, light and content for each `SceneId`.
//!
//! Content is built in the camera's view space wherever it describes screen layout, so the
//! same pieces work under the elevated camera of the `all` scene. Marker ids are grouped
//! by content (grid 0.., G1 100.., G2 200.. and 300.., G3 400.., G5 500.., shadow 600..,
//! HUD 700..) so they stay unique when pieces are combined.

use glam::{Mat4, Vec3};

use super::{
    Camera, FRAME_RATE, Light, Marker, MarkerKind, Material, Mesh, Probe, Scene, SceneId,
    ShadowVolume, Vertex,
    geometry::{cuboid, facing_quad, hash01, hud_rect, quad, rgba, sphere},
    tonemapped,
};

const FRONT: Camera = Camera {
    position: Vec3::ZERO,
    target: Vec3::NEG_Z,
    up: Vec3::Y,
    vertical_fov_deg: 60.0,
    near: 0.1,
    far: 100.0,
};

const ELEVATED: Camera = Camera {
    position: Vec3::new(0.0, 1.6, 0.0),
    target: Vec3::new(0.0, 0.6, -5.0),
    ..FRONT
};

const CONVERGENCE: f32 = 4.0;

pub(super) fn build(id: SceneId, aspect: f32, frame: u32) -> Scene {
    let camera = match id {
        SceneId::Shadow | SceneId::All => ELEVATED,
        _ => FRONT,
    };
    let light = match id {
        SceneId::G5 => Some(Light {
            direction: Vec3::new(-0.4, 0.8, 0.45).normalize(),
            color: Vec3::ONE,
            ambient: 0.15,
            shadow: None,
        }),
        SceneId::Shadow | SceneId::All => Some(Light {
            direction: Vec3::new(0.5, 1.0, 0.3).normalize(),
            color: Vec3::ONE,
            ambient: 0.2,
            shadow: Some(ShadowVolume {
                center: Vec3::new(0.0, 0.0, -6.0),
                half_extent: 7.0,
                distance: 15.0,
            }),
        }),
        _ => None,
    };

    let ctx = Ctx::new(camera, aspect, frame);
    let mut parts = Parts::default();
    match id {
        SceneId::G0 | SceneId::Sky | SceneId::Post => parts.add(marker_grid(&ctx)),
        SceneId::G1 => {
            parts.add(backdrop(&ctx, 6.0, [90, 90, 90]));
            parts.add(occlusion(&ctx));
        }
        SceneId::G2 => {
            parts.add(backdrop(&ctx, 10.0, [40, 40, 40]));
            parts.add(transparents(&ctx));
        }
        SceneId::G3 => {
            parts.add(backdrop(&ctx, 10.0, [30, 30, 36]));
            parts.add(motion(&ctx, 0.2, -0.6));
        }
        SceneId::G5 => {
            parts.add(backdrop(&ctx, 10.0, [60, 60, 70]));
            parts.add(view_dependent(&ctx));
        }
        SceneId::Shadow => parts.add(shadow_casters(&ctx)),
        SceneId::Hud => {
            parts.add(marker_grid(&ctx));
            parts.add(hud_overlay());
        }
        SceneId::All => {
            parts.add(shadow_casters(&ctx));
            parts.add(view_dependent(&ctx));
            parts.add(transparents(&ctx));
            parts.add(motion(&ctx, 0.62, 0.85));
            parts.add(hud_overlay());
        }
    }

    let (mut meshes, transparent): (Vec<_>, Vec<_>) = parts
        .meshes
        .into_iter()
        .partition(|mesh| mesh.material != Material::Transparent);
    meshes.extend(transparent);

    let post = matches!(id, SceneId::Post | SceneId::All);
    if post {
        for probe in &mut parts.probes {
            probe.on = tonemapped(probe.on);
            probe.off = tonemapped(probe.off);
        }
    }

    Scene {
        id,
        frame,
        camera,
        aspect,
        convergence: CONVERGENCE,
        light,
        sky: matches!(id, SceneId::Sky | SceneId::All),
        post,
        meshes,
        hud: parts.hud,
        markers: parts.markers,
        probes: parts.probes,
        counter: parts.counter,
    }
}

/// Camera and time for one frame, with view-space placement helpers.
struct Ctx {
    aspect: f32,
    frame: u32,
    tan_half_v: f32,
    view: Mat4,
    view_to_world: Mat4,
    view_proj: Mat4,
}

impl Ctx {
    fn new(camera: Camera, aspect: f32, frame: u32) -> Self {
        let view = camera.view();
        Self {
            aspect,
            frame,
            tan_half_v: (camera.vertical_fov_deg.to_radians() / 2.0).tan(),
            view,
            view_to_world: view.inverse(),
            view_proj: camera.projection(aspect) * view,
        }
    }

    fn time(&self) -> f32 {
        self.frame as f32 / FRAME_RATE
    }

    /// View-space point that projects to NDC (`u` right, `v` up) at `depth`.
    fn at(&self, u: f32, v: f32, depth: f32) -> Vec3 {
        let half_v = depth * self.tan_half_v;
        Vec3::new(u * half_v * self.aspect, v * half_v, -depth)
    }

    /// World length covering `ndc` NDC-y units at `depth`.
    fn size(&self, ndc: f32, depth: f32) -> f32 {
        ndc * depth * self.tan_half_v
    }

    fn world(&self, view_point: Vec3) -> Vec3 {
        self.view_to_world.transform_point3(view_point)
    }

    fn depth_of(&self, world: Vec3) -> f32 {
        -self.view.transform_point3(world).z
    }

    /// Vulkan NDC (y down) of a world point.
    fn ndc(&self, world: Vec3) -> [f32; 2] {
        let p = self.view_proj.project_point3(world);
        [p.x, p.y]
    }
}

/// Content produced by one builder, merged into the scene.
#[derive(Default)]
struct Parts {
    meshes: Vec<Mesh>,
    hud: Vec<Vertex>,
    markers: Vec<Marker>,
    probes: Vec<Probe>,
    counter: Option<u32>,
}

impl Parts {
    fn add(&mut self, other: Parts) {
        self.meshes.extend(other.meshes);
        self.hud.extend(other.hud);
        self.markers.extend(other.markers);
        self.probes.extend(other.probes);
        self.counter = self.counter.or(other.counter);
    }

    /// Mesh whose vertices are in the camera's view space.
    fn view_mesh(&mut self, ctx: &Ctx, material: Material, vertices: Vec<Vertex>) {
        self.meshes.push(Mesh {
            material,
            model: ctx.view_to_world,
            vertices,
        });
    }

    fn world_marker(
        &mut self,
        ctx: &Ctx,
        id: u32,
        color: [u8; 3],
        alpha: f32,
        view_center: Vec3,
        half_extent: [f32; 2],
    ) {
        self.markers.push(Marker {
            id,
            kind: MarkerKind::World,
            color,
            alpha,
            center: ctx.world(view_center),
            half_extent,
            depth: -view_center.z,
        });
    }
}

/// Opaque screen-filling wall at `depth`.
fn backdrop(ctx: &Ctx, depth: f32, color: [u8; 3]) -> Parts {
    let half_v = ctx.size(1.05, depth);
    let mut parts = Parts::default();
    let vertices = facing_quad(
        Vec3::new(0.0, 0.0, -depth),
        [half_v * ctx.aspect, half_v],
        rgba(color, 1.0),
    );
    parts.view_mesh(ctx, Material::Unlit, vertices.to_vec());
    parts
}

const GRID_DEPTHS: [f32; 5] = [1.5, 2.5, 4.0, 8.0, 20.0];
const GRID_COLUMNS: [f32; 5] = [-0.72, -0.36, 0.0, 0.36, 0.72];
const GRID_ROWS: [f32; 3] = [0.5, 0.0, -0.5];
const GRID_HALF_NDC: f32 = 0.06;

/// G0: each depth in its own column so no marker hides another; every marker scaled by its
/// depth to cover the same screen area. Red and blue encode the column, green the row.
fn marker_grid(ctx: &Ctx) -> Parts {
    let mut parts = Parts::default();
    let mut vertices = Vec::new();
    for (row, &v) in GRID_ROWS.iter().enumerate() {
        for (column, (&u, &depth)) in GRID_COLUMNS.iter().zip(&GRID_DEPTHS).enumerate() {
            let (c, r) = (column as u8, row as u8);
            let color = [55 + 50 * c, 60 + 70 * r, 220 - 40 * c];
            let center = ctx.at(u, v, depth);
            let half = ctx.size(GRID_HALF_NDC, depth);
            vertices.extend(facing_quad(center, [half, half], rgba(color, 1.0)));
            let id = (row * GRID_DEPTHS.len() + column) as u32;
            parts.world_marker(ctx, id, color, 1.0, center, [half, half]);
        }
    }
    parts.view_mesh(ctx, Material::Unlit, vertices);
    parts
}

const PATTERN_COLOR: [u8; 3] = [255, 0, 255];
const PATTERN_CELL: f32 = 0.02;

/// G1: a pillar at depth 3 hides the wall at depth 6 for |x| < 0.30 in mono. The checker
/// sits at x in [0.18, 0.29], so only an eye shifted right sees it.
fn occlusion(ctx: &Ctx) -> Parts {
    let mut parts = Parts::default();
    let pillar_center = Vec3::new(0.0, 0.0, -3.0);
    let pillar_half = Vec3::new(0.15, 3.0, 0.15);
    let pillar_color = [200, 120, 40];
    parts.view_mesh(
        ctx,
        Material::Unlit,
        cuboid(pillar_center, pillar_half, rgba(pillar_color, 1.0)),
    );

    let pattern_center = Vec3::new(0.235, 0.0, -5.99);
    let pattern_half = [0.055, 0.5];
    let cells = pattern_half.map(|half| half * 2.0 / PATTERN_CELL);
    let pattern = quad(
        pattern_center,
        Vec3::X,
        Vec3::Y,
        pattern_half,
        rgba(PATTERN_COLOR, 1.0),
        cells,
    );
    parts.view_mesh(ctx, Material::Pattern, pattern.to_vec());

    parts.world_marker(ctx, 100, PATTERN_COLOR, 1.0, pattern_center, pattern_half);
    parts.world_marker(
        ctx,
        101,
        pillar_color,
        1.0,
        pillar_center,
        [pillar_half.x, pillar_half.y],
    );
    parts
}

const STACK: [(f32, f32, f32, [u8; 3]); 4] = [
    (-0.55, 0.35, 2.5, [230, 60, 60]),
    (-0.4, 0.2, 3.5, [60, 200, 90]),
    (-0.25, 0.05, 5.0, [70, 110, 240]),
    (-0.1, -0.1, 7.0, [240, 210, 60]),
];
const PARTICLES: u32 = 48;

/// G2: staggered half-transparent quads, each with an area no other quad covers, plus a
/// drifting particle field; all sorted back to front in one mesh.
fn transparents(ctx: &Ctx) -> Parts {
    let mut parts = Parts::default();
    let mut quads: Vec<(f32, [Vertex; 6])> = Vec::new();

    for (index, &(u, v, depth, color)) in STACK.iter().enumerate() {
        let center = ctx.at(u, v, depth);
        let half = ctx.size(0.18, depth);
        quads.push((depth, facing_quad(center, [half, half], rgba(color, 0.5))));
        parts.world_marker(ctx, 200 + index as u32, color, 0.5, center, [half, half]);
    }

    for particle in 0..PARTICLES {
        let [h0, h1, h2] = [0, 1, 2].map(|k| hash01(particle * 3 + k));
        let depth = 2.0 + 7.0 * h2;
        let v = (h1 + ctx.time() * 0.25).fract() - 0.5;
        let center = ctx.at(0.2 + 0.6 * h0, v, depth);
        let half = ctx.size(0.025, depth);
        let color = [255, 150 + (100.0 * h0) as u8, 60];
        quads.push((depth, facing_quad(center, [half, half], rgba(color, 0.6))));
        parts.world_marker(ctx, 300 + particle, color, 0.6, center, [half, half]);
    }

    quads.sort_by(|a, b| b.0.total_cmp(&a.0));
    let vertices = quads.into_iter().flat_map(|(_, quad)| quad).collect();
    parts.view_mesh(ctx, Material::Transparent, vertices);
    parts
}

const MOVER_COLOR: [u8; 3] = [0, 200, 255];
const COUNTER_ON: [u8; 3] = [255, 255, 255];
const COUNTER_OFF: [u8; 3] = [70, 70, 70];
const COUNTER_BITS: u32 = 16;

/// G3: a quad crossing the screen at one screen width per second at convergence depth,
/// moved by its model matrix, and a 16-bit frame counter on the same plane.
fn motion(ctx: &Ctx, mover_v: f32, counter_v: f32) -> Parts {
    let mut parts = Parts::default();

    let u = -0.8 + (2.0 * ctx.time()).rem_euclid(1.6);
    let center = ctx.at(u, mover_v, CONVERGENCE);
    let half = ctx.size(0.08, CONVERGENCE);
    parts.meshes.push(Mesh {
        material: Material::Unlit,
        model: ctx.view_to_world * Mat4::from_translation(center),
        vertices: facing_quad(Vec3::ZERO, [half, half], rgba(MOVER_COLOR, 1.0)).to_vec(),
    });
    parts.world_marker(ctx, 400, MOVER_COLOR, 1.0, center, [half, half]);

    let value = ctx.frame & ((1 << COUNTER_BITS) - 1);
    let half = ctx.size(0.025, CONVERGENCE);
    let mut vertices = Vec::new();
    for bit in 0..COUNTER_BITS {
        let set = (value >> (COUNTER_BITS - 1 - bit)) & 1 == 1;
        let color = if set { COUNTER_ON } else { COUNTER_OFF };
        let center = ctx.at(-0.6 + bit as f32 * 0.08, counter_v, CONVERGENCE);
        vertices.extend(facing_quad(center, [half, half], rgba(color, 1.0)));
        parts.world_marker(ctx, 410 + bit, color, 1.0, center, [half, half]);
        parts.probes.push(Probe {
            bit,
            ndc: ctx.ndc(ctx.world(center)),
            on: COUNTER_ON,
            off: COUNTER_OFF,
        });
    }
    parts.view_mesh(ctx, Material::Unlit, vertices);
    parts.counter = Some(value);
    parts
}

/// G5: a glossy sphere and a reflector whose shading depends on the camera position.
fn view_dependent(ctx: &Ctx) -> Parts {
    let mut parts = Parts::default();

    let sphere_center = Vec3::new(-0.9, 0.0, -4.0);
    let sphere_color = [180, 40, 40];
    parts.meshes.push(Mesh {
        material: Material::Lit,
        model: ctx.view_to_world * Mat4::from_translation(sphere_center),
        vertices: sphere(0.6, 48, 24, rgba(sphere_color, 1.0)),
    });
    parts.world_marker(ctx, 500, sphere_color, 1.0, sphere_center, [0.6, 0.6]);

    let reflector_center = Vec3::new(0.9, 0.0, -4.5);
    let reflector_color = [200, 200, 210];
    let reflector = facing_quad(reflector_center, [0.6, 0.8], rgba(reflector_color, 1.0));
    parts.view_mesh(ctx, Material::Reflective, reflector.to_vec());
    parts.world_marker(ctx, 501, reflector_color, 1.0, reflector_center, [0.6, 0.8]);
    parts
}

/// World-space ground, box and pole lit by a shadow-casting light.
fn shadow_casters(ctx: &Ctx) -> Parts {
    let mut parts = Parts::default();
    let mut add = |id: u32, color: [u8; 3], center: Vec3, half: [f32; 2], vertices| {
        parts.meshes.push(Mesh {
            material: Material::Lit,
            model: Mat4::IDENTITY,
            vertices,
        });
        parts.markers.push(Marker {
            id,
            kind: MarkerKind::World,
            color,
            alpha: 1.0,
            center,
            half_extent: half,
            depth: ctx.depth_of(center),
        });
    };

    let box_center = Vec3::new(0.0, 0.4, -5.0);
    let box_color = [60, 120, 200];
    add(
        600,
        box_color,
        box_center,
        [0.4, 0.4],
        cuboid(box_center, Vec3::splat(0.4), rgba(box_color, 1.0)),
    );

    let pole_center = Vec3::new(1.5, 1.0, -6.5);
    let pole_half = Vec3::new(0.1, 1.0, 0.1);
    let pole_color = [200, 170, 60];
    add(
        601,
        pole_color,
        pole_center,
        [pole_half.x, pole_half.y],
        cuboid(pole_center, pole_half, rgba(pole_color, 1.0)),
    );

    let ground_center = Vec3::new(0.0, 0.0, -6.5);
    let ground_half = [6.0, 5.5];
    let ground_color = [150, 150, 150];
    let ground = quad(
        ground_center,
        Vec3::X,
        Vec3::NEG_Z,
        ground_half,
        rgba(ground_color, 1.0),
        [1.0, 1.0],
    );
    add(
        602,
        ground_color,
        ground_center,
        ground_half,
        ground.to_vec(),
    );
    parts
}

struct HudElement {
    id: u32,
    min: [f32; 2],
    max: [f32; 2],
    color: [u8; 3],
    alpha: f32,
}

const fn hud_element(
    id: u32,
    min: [f32; 2],
    max: [f32; 2],
    color: [u8; 3],
    alpha: f32,
) -> HudElement {
    HudElement {
        id,
        min,
        max,
        color,
        alpha,
    }
}

const HUD_ELEMENTS: [HudElement; 5] = [
    hud_element(700, [-0.04, -0.006], [0.04, 0.006], [255, 255, 255], 0.9),
    hud_element(701, [-0.0035, -0.07], [0.0035, 0.07], [255, 255, 255], 0.9),
    hud_element(702, [-0.95, 0.85], [-0.45, 0.92], [60, 60, 60], 1.0),
    hud_element(703, [-0.94, 0.86], [-0.6, 0.91], [220, 50, 50], 1.0),
    hud_element(704, [0.7, -0.95], [0.95, -0.5], [40, 90, 60], 0.85),
];

/// Crosshair, health bar and minimap in clip space (Vulkan NDC, y down).
fn hud_overlay() -> Parts {
    let mut parts = Parts::default();
    for HudElement {
        id,
        min,
        max,
        color,
        alpha,
    } in HUD_ELEMENTS
    {
        parts.hud.extend(hud_rect(min, max, rgba(color, alpha)));
        parts.markers.push(Marker {
            id,
            kind: MarkerKind::Hud,
            color,
            alpha,
            center: Vec3::new((min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, 0.0),
            half_extent: [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5],
            depth: 1.0,
        });
    }
    parts
}
