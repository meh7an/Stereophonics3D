//! Synthetic image pairs with known geometry: correct stereo must pass each test, and the
//! failure modes the report names (toe-in, mirrored sign, alternate-frame offset, a
//! copied eye, a depth-warped eye, smeared disocclusions) must fail the test meant to
//! catch them.

use s3d_shared::{
    Eye,
    capture::{
        CameraRecord, Image, MarkerKind, MarkerRecord, ProbeRecord, Sidecar, StereoMethod,
        StereoRecord,
    },
};
use s3d_verify::{
    Capture, Outcome, Test,
    image::{Rgb, ndc_to_pixel},
    run,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
const SEPARATION: f32 = 0.03;
const CONVERGENCE: f32 = 4.0;
const CLEAR: Rgb = [16, 16, 16];
const TAN_HALF_V: f32 = 0.577_350_3;

/// Vulkan right-handed perspective (Y down, depth 0..1), vertical FOV 60 degrees.
fn projection() -> [f32; 16] {
    let (near, far) = (0.1, 100.0);
    let f = 1.0 / TAN_HALF_V;
    let mut m = [0.0; 16];
    m[0] = f / ASPECT;
    m[5] = -f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

fn identity() -> [f32; 16] {
    let mut m = [0.0; 16];
    for i in 0..4 {
        m[i * 5] = 1.0;
    }
    m
}

/// World point (camera at the origin looking down -Z) that projects to NDC (`u`, `v` up).
fn at(u: f32, v: f32, depth: f32) -> [f32; 3] {
    [
        u * depth * TAN_HALF_V * ASPECT,
        v * depth * TAN_HALF_V,
        -depth,
    ]
}

fn marker(id: u32, color: Rgb, alpha: f32, u: f32, v: f32, depth: f32, half: f32) -> MarkerRecord {
    MarkerRecord {
        id,
        kind: MarkerKind::World,
        color,
        alpha,
        center: at(u, v, depth),
        half_extent: [half * depth * TAN_HALF_V; 2],
        depth,
    }
}

fn sidecar(scene: &str, markers: Vec<MarkerRecord>) -> Sidecar {
    Sidecar {
        scene: scene.to_owned(),
        frame: 0,
        width: WIDTH,
        height: HEIGHT,
        rendering: "synthetic".to_owned(),
        camera: CameraRecord {
            position: [0.0; 3],
            target: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            vertical_fov_deg: 60.0,
            near: 0.1,
            far: 100.0,
        },
        view: identity(),
        projection: projection(),
        convergence: CONVERGENCE,
        clear_color: CLEAR,
        backdrop_color: None,
        light: None,
        sky: false,
        post: false,
        markers,
        probes: Vec::new(),
        counter: None,
        stereo: Some(StereoRecord {
            method: StereoMethod::SingleEye,
            separation: SEPARATION,
            convergence: CONVERGENCE,
            eyes: vec![Eye::Left, Eye::Right],
        }),
    }
}

fn canvas(color: Rgb) -> Image {
    Image {
        width: WIDTH,
        height: HEIGHT,
        rgba: (0..WIDTH * HEIGHT)
            .flat_map(|_| [color[0], color[1], color[2], 255])
            .collect(),
    }
}

/// Blends `color` over every pixel whose centre lies in the pixel-space rectangle.
fn fill(image: &mut Image, (x0, y0, x1, y1): (f64, f64, f64, f64), color: Rgb, alpha: f32) {
    for y in 0..image.height {
        for x in 0..image.width {
            let (cx, cy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            if cx < x0 || cx >= x1 || cy < y0 || cy >= y1 {
                continue;
            }
            let offset = (y as usize * image.width as usize + x as usize) * 4;
            let under = [
                image.rgba[offset],
                image.rgba[offset + 1],
                image.rgba[offset + 2],
            ];
            let out = s3d_verify::image::blend(color, alpha, under);
            image.rgba[offset..offset + 3].copy_from_slice(&out);
        }
    }
}

/// Pixel rectangle of a screen-aligned square at NDC (`u`, `v` up) with half-size `half`
/// in NDC-y units, moved right by `shift_px`.
fn square(u: f32, v: f32, half: f32, shift_px: f64) -> (f64, f64, f64, f64) {
    let (cx, cy) = ndc_to_pixel([u, -v], WIDTH, HEIGHT);
    let half_px = f64::from(half) * f64::from(HEIGHT) / 2.0;
    (
        cx - half_px + shift_px,
        cy - half_px,
        cx + half_px + shift_px,
        cy + half_px,
    )
}

/// Per-eye horizontal shift in pixels for a point at `depth`: the clip-space shear
/// `eye_sign * s * (w - conv)` divided by `w`, from NDC to pixels.
fn shift(eye_sign: f32, depth: f32) -> f64 {
    f64::from(eye_sign * SEPARATION * (1.0 - CONVERGENCE / depth)) * f64::from(WIDTH) / 2.0
}

fn outcome(test: Test, sidecar: Sidecar, left: Image, right: Image) -> Outcome {
    let capture = Capture::new(sidecar, left, right).expect("capture");
    let finding = run(test, &capture).expect("test runs");
    println!("{finding}");
    finding.outcome
}

// G0 ------------------------------------------------------------------------------------

const GRID: [(f32, f32, f32); 10] = [
    (-0.72, 0.4, 1.5),
    (-0.36, 0.4, 2.5),
    (0.0, 0.4, 4.0),
    (0.36, 0.4, 8.0),
    (0.72, 0.4, 20.0),
    (-0.72, -0.4, 20.0),
    (-0.36, -0.4, 8.0),
    (0.0, -0.4, 4.0),
    (0.36, -0.4, 2.5),
    (0.72, -0.4, 1.5),
];
const GRID_HALF: f32 = 0.06;

#[derive(Clone, Copy, PartialEq)]
enum G0Mode {
    Correct,
    MirroredSign,
    ToeIn,
    CopiedEye,
}

fn g0_pair(mode: G0Mode) -> (Sidecar, Image, Image) {
    let markers: Vec<_> = GRID
        .iter()
        .enumerate()
        .map(|(id, &(u, v, depth))| {
            let color = [40 + 20 * id as u8, 200 - 15 * id as u8, 90];
            marker(id as u32, color, 1.0, u, v, depth, GRID_HALF)
        })
        .collect();
    let mut left = canvas(CLEAR);
    let mut right = canvas(CLEAR);
    for (marker, &(u, v, depth)) in markers.iter().zip(&GRID) {
        let sign = if mode == G0Mode::MirroredSign {
            -1.0
        } else {
            1.0
        };
        let (mut l, mut r) = (
            square(u, v, GRID_HALF, shift(-sign, depth)),
            square(u, v, GRID_HALF, shift(sign, depth)),
        );
        if mode == G0Mode::ToeIn {
            // Converged cameras add vertical parallax growing toward the edges.
            let keystone = 0.02 * f64::from(HEIGHT) * f64::from(u) * f64::from(v.signum());
            l = (l.0, l.1 - keystone, l.2, l.3 - keystone);
            r = (r.0, r.1 + keystone, r.2, r.3 + keystone);
        }
        fill(&mut left, l, marker.color, 1.0);
        fill(&mut right, r, marker.color, 1.0);
    }
    if mode == G0Mode::CopiedEye {
        right = left.clone();
    }
    (sidecar("g0", markers), left, right)
}

#[test]
fn test_g0_correct_pair_passes() {
    let (sidecar, left, right) = g0_pair(G0Mode::Correct);
    assert_eq!(outcome(Test::G0, sidecar, left, right), Outcome::Pass);
}

#[test]
fn test_g0_rejects_mirrored_toe_in_and_copied() {
    for mode in [G0Mode::MirroredSign, G0Mode::ToeIn, G0Mode::CopiedEye] {
        let (sidecar, left, right) = g0_pair(mode);
        assert_eq!(outcome(Test::G0, sidecar, left, right), Outcome::Fail);
    }
}

// G1 ------------------------------------------------------------------------------------

const MAGENTA: Rgb = [255, 0, 255];
const GREEN: Rgb = [0, 255, 0];
const WALL: Rgb = [90, 90, 90];
const PILLAR: Rgb = [200, 120, 40];

#[derive(Clone, Copy, PartialEq)]
enum G1Mode {
    Correct,
    DepthWarped,
    Smeared,
}

fn g1_pair(mode: G1Mode) -> (Sidecar, Image, Image) {
    let mut left = canvas(WALL);
    fill(&mut left, (150.0, 0.0, 170.0, 180.0), PILLAR, 1.0);
    let mut right = canvas(WALL);
    fill(&mut right, (146.0, 0.0, 166.0, 180.0), PILLAR, 1.0);
    match mode {
        G1Mode::Correct => {
            for x in 0..6 {
                for y in 0..20 {
                    let color = if (x + y) % 2 == 0 { MAGENTA } else { GREEN };
                    let (px, py) = (f64::from(166 + x * 2), f64::from(80 + y * 2));
                    fill(&mut right, (px, py, px + 2.0, py + 2.0), color, 1.0);
                }
            }
        }
        // Hole-filling stretches the wall over the revealed strip: no checker anywhere.
        G1Mode::DepthWarped => {}
        G1Mode::Smeared => {
            for y in 0..20 {
                let py = f64::from(80 + y * 2);
                fill(&mut right, (166.0, py, 172.0, py + 1.0), MAGENTA, 1.0);
                fill(&mut right, (166.0, py + 1.0, 172.0, py + 2.0), GREEN, 1.0);
            }
        }
    }
    let mut sidecar = sidecar("g1", vec![marker(100, MAGENTA, 1.0, 0.07, 0.0, 6.0, 0.05)]);
    sidecar.backdrop_color = Some(WALL);
    (sidecar, left, right)
}

#[test]
fn test_g1_revealed_checker_in_one_eye_passes() {
    let (sidecar, left, right) = g1_pair(G1Mode::Correct);
    assert_eq!(outcome(Test::G1, sidecar, left, right), Outcome::Pass);
}

#[test]
fn test_g1_rejects_depth_warp_and_smear() {
    for mode in [G1Mode::DepthWarped, G1Mode::Smeared] {
        let (sidecar, left, right) = g1_pair(mode);
        assert_eq!(outcome(Test::G1, sidecar, left, right), Outcome::Fail);
    }
}

// G2 ------------------------------------------------------------------------------------

const STACK: [(f32, f32, f32, Rgb); 4] = [
    (-0.55, 0.35, 2.5, [230, 60, 60]),
    (-0.4, 0.2, 3.5, [60, 200, 90]),
    (-0.25, 0.05, 5.0, [70, 110, 240]),
    (-0.1, -0.1, 7.0, [240, 210, 60]),
];
const BACKDROP: Rgb = [40, 40, 40];
const BACKDROP_DEPTH: f32 = 10.0;

fn g2_pair(warped: bool) -> (Sidecar, Image, Image) {
    let mut left = canvas(BACKDROP);
    let mut right = canvas(BACKDROP);
    for &(u, v, depth, color) in STACK.iter().rev() {
        // A depth warp moves each pixel by the depth stored under it: the wall's.
        let moved = if warped { BACKDROP_DEPTH } else { depth };
        fill(
            &mut left,
            square(u, v, 0.18, shift(-1.0, moved)),
            color,
            0.5,
        );
        fill(
            &mut right,
            square(u, v, 0.18, shift(1.0, moved)),
            color,
            0.5,
        );
    }
    let markers = STACK
        .iter()
        .enumerate()
        .map(|(i, &(u, v, depth, color))| marker(200 + i as u32, color, 0.5, u, v, depth, 0.18))
        .collect();
    let mut sidecar = sidecar("g2", markers);
    sidecar.backdrop_color = Some(BACKDROP);
    (sidecar, left, right)
}

#[test]
fn test_g2_layers_at_own_depth_pass() {
    let (sidecar, left, right) = g2_pair(false);
    assert_eq!(outcome(Test::G2, sidecar, left, right), Outcome::Pass);
}

#[test]
fn test_g2_rejects_depth_warped_transparents() {
    let (sidecar, left, right) = g2_pair(true);
    assert_eq!(outcome(Test::G2, sidecar, left, right), Outcome::Fail);
}

// G3 ------------------------------------------------------------------------------------

const MOVER: Rgb = [0, 200, 255];
const ON: Rgb = [255, 255, 255];
const OFF: Rgb = [70, 70, 70];

/// One eye of the G3 scene at `frame`: the mover at convergence depth and the counter.
fn g3_eye(frame: u32) -> Image {
    let mut image = canvas([30, 30, 36]);
    let u = -0.8 + (2.0 * frame as f32 / 60.0).rem_euclid(1.6);
    fill(&mut image, square(u, 0.2, 0.08, 0.0), MOVER, 1.0);
    for bit in 0..16u32 {
        let set = (frame >> (15 - bit)) & 1 == 1;
        let color = if set { ON } else { OFF };
        fill(
            &mut image,
            square(-0.6 + bit as f32 * 0.08, -0.6, 0.025, 0.0),
            color,
            1.0,
        );
    }
    image
}

fn g3_sidecar(frame: u32) -> Sidecar {
    let u = -0.8 + (2.0 * frame as f32 / 60.0).rem_euclid(1.6);
    let mut sidecar = sidecar(
        "g3",
        vec![marker(400, MOVER, 1.0, u, 0.2, CONVERGENCE, 0.08)],
    );
    sidecar.counter = Some(frame);
    sidecar.probes = (0..16)
        .map(|bit| ProbeRecord {
            bit,
            ndc: [-0.6 + bit as f32 * 0.08, 0.6],
            on: ON,
            off: OFF,
        })
        .collect();
    sidecar
}

#[test]
fn test_g3_same_frame_eyes_pass() {
    assert_eq!(
        outcome(Test::G3, g3_sidecar(37), g3_eye(37), g3_eye(37)),
        Outcome::Pass
    );
}

#[test]
fn test_g3_rejects_alternate_frame_eyes() {
    assert_eq!(
        outcome(Test::G3, g3_sidecar(37), g3_eye(37), g3_eye(38)),
        Outcome::Fail
    );
}

// G4 ------------------------------------------------------------------------------------

#[test]
fn test_g4_suppressed_right_eye_with_overlay_passes() {
    let (sidecar, left, _) = g0_pair(G0Mode::Correct);
    let mut right = canvas(CLEAR);
    fill(&mut right, (4.0, 4.0, 14.0, 14.0), [250, 250, 250], 1.0);
    assert_eq!(outcome(Test::G4, sidecar, left, right), Outcome::Pass);
}

#[test]
fn test_g4_rejects_right_eye_derived_from_left() {
    let (sidecar, left, _) = g0_pair(G0Mode::Correct);
    let right = left.clone();
    assert_eq!(outcome(Test::G4, sidecar, left, right), Outcome::Fail);
}

#[test]
fn test_g4_rejects_copied_eye_of_sparse_scene() {
    let (mut sidecar, _, _) = g0_pair(G0Mode::Correct);
    sidecar.markers.truncate(1);
    let (u, v, depth) = GRID[0];
    let mut left = canvas(CLEAR);
    let color = sidecar.markers[0].color;
    fill(
        &mut left,
        square(u, v, GRID_HALF, shift(-1.0, depth)),
        color,
        1.0,
    );
    let right = left.clone();
    assert_eq!(outcome(Test::G4, sidecar, left, right), Outcome::Fail);
}

// G5 ------------------------------------------------------------------------------------

fn g5_pair(highlight_shift: f64) -> (Sidecar, Image, Image) {
    let eye = |offset: f64| {
        let mut image = canvas([60, 60, 70]);
        fill(&mut image, square(0.0, 0.0, 0.16, 0.0), [120, 30, 30], 1.0);
        let (cx, cy) = ndc_to_pixel([0.0, 0.0], WIDTH, HEIGHT);
        let (hx, hy) = (cx - 5.0 + offset, cy - 5.0);
        fill(
            &mut image,
            (hx - 1.5, hy - 1.5, hx + 1.5, hy + 1.5),
            [255, 255, 255],
            1.0,
        );
        image
    };
    let sidecar = sidecar(
        "g5",
        vec![marker(500, [180, 40, 40], 1.0, 0.0, 0.0, CONVERGENCE, 0.16)],
    );
    (
        sidecar,
        eye(-highlight_shift / 2.0),
        eye(highlight_shift / 2.0),
    )
}

#[test]
fn test_g5_grades_view_dependent_highlight() {
    let (sidecar, left, right) = g5_pair(4.0);
    assert_eq!(
        outcome(Test::G5, sidecar, left, right),
        Outcome::ViewDependent
    );
}

#[test]
fn test_g5_grades_mono_shading() {
    let (sidecar, left, right) = g5_pair(0.0);
    assert_eq!(
        outcome(Test::G5, sidecar, left, right),
        Outcome::MonoShading
    );
}

// Plumbing ------------------------------------------------------------------------------

#[test]
fn test_mono_capture_is_rejected() {
    let (mut sidecar, left, right) = g0_pair(G0Mode::Correct);
    sidecar.stereo = None;
    assert!(Capture::new(sidecar, left, right).is_err());
}

#[test]
fn test_missing_markers_make_test_not_applicable() {
    let (sidecar, left, right) = g0_pair(G0Mode::Correct);
    let capture = Capture::new(sidecar, left, right).expect("capture");
    assert!(run(Test::G2, &capture).is_err());
}
