//! Capture output: one PNG plus one JSON sidecar per captured frame.
//!
//! The sidecar schema is provisional; M1.1 moves it into `s3d-shared`, where the layer
//! and the verifier read the same definition.

use std::{
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use serde::Serialize;

use crate::{
    scene::{CLEAR_COLOR, MarkerKind, Scene},
    vulkan::RenderingMode,
};

/// Ground truth written next to every captured image.
#[derive(Debug, Serialize)]
pub struct Sidecar {
    pub scene: &'static str,
    pub frame: u32,
    pub width: u32,
    pub height: u32,
    pub rendering: RenderingMode,
    pub camera: CameraRecord,
    /// Column-major, world to view.
    pub view: [f32; 16],
    /// Column-major, Vulkan clip convention (Y down, depth in [0, 1]).
    pub projection: [f32; 16],
    pub convergence: f32,
    pub clear_color: [u8; 3],
    pub light: Option<LightRecord>,
    pub sky: bool,
    pub post: bool,
    pub markers: Vec<MarkerRecord>,
    pub probes: Vec<ProbeRecord>,
    /// Value the counter probes encode, when the scene has a counter.
    pub counter: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct CameraRecord {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub vertical_fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Serialize)]
pub struct LightRecord {
    /// Unit vector toward the light.
    pub direction: [f32; 3],
    pub shadows: bool,
}

#[derive(Debug, Serialize)]
pub struct MarkerRecord {
    pub id: u32,
    /// `world` or `hud` (clip space, w = 1).
    pub kind: &'static str,
    pub color: [u8; 3],
    pub alpha: f32,
    pub center: [f32; 3],
    pub half_extent: [f32; 2],
    pub depth: f32,
}

#[derive(Debug, Serialize)]
pub struct ProbeRecord {
    pub bit: u32,
    /// Vulkan NDC of the probe pixel (y down).
    pub ndc: [f32; 2],
    pub on: [u8; 3],
    pub off: [u8; 3],
}

impl Sidecar {
    pub fn new(scene: &Scene, size: (u32, u32), rendering: RenderingMode) -> Self {
        let camera = &scene.camera;
        Self {
            scene: scene.id.name(),
            frame: scene.frame,
            width: size.0,
            height: size.1,
            rendering,
            camera: CameraRecord {
                position: camera.position.to_array(),
                target: camera.target.to_array(),
                up: camera.up.to_array(),
                vertical_fov_deg: camera.vertical_fov_deg,
                near: camera.near,
                far: camera.far,
            },
            view: camera.view().to_cols_array(),
            projection: camera.projection(scene.aspect).to_cols_array(),
            convergence: scene.convergence,
            clear_color: CLEAR_COLOR,
            light: scene.light.map(|light| LightRecord {
                direction: light.direction.to_array(),
                shadows: light.shadow.is_some(),
            }),
            sky: scene.sky,
            post: scene.post,
            markers: scene
                .markers
                .iter()
                .map(|marker| MarkerRecord {
                    id: marker.id,
                    kind: match marker.kind {
                        MarkerKind::World => "world",
                        MarkerKind::Hud => "hud",
                    },
                    color: marker.color,
                    alpha: marker.alpha,
                    center: marker.center.to_array(),
                    half_extent: marker.half_extent,
                    depth: marker.depth,
                })
                .collect(),
            probes: scene
                .probes
                .iter()
                .map(|probe| ProbeRecord {
                    bit: probe.bit,
                    ndc: probe.ndc,
                    on: probe.on,
                    off: probe.off,
                })
                .collect(),
            counter: scene.counter,
        }
    }
}

/// Writes `<scene>-f<frame>.png` and `.json` into `dir`, creating it if needed, and
/// returns the PNG path.
///
/// # Errors
///
/// Returns an error when the directory or either file cannot be written.
pub fn write(dir: &Path, rgba: &[u8], sidecar: &Sidecar) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let stem = format!("{}-f{:05}", sidecar.scene, sidecar.frame);
    let png_path = dir.join(format!("{stem}.png"));
    fs::write(&png_path, encode_png(sidecar.width, sidecar.height, rgba)?)
        .with_context(|| format!("cannot write {}", png_path.display()))?;
    let json_path = dir.join(format!("{stem}.json"));
    let mut json = serde_json::to_string_pretty(sidecar)?;
    json.push('\n');
    fs::write(&json_path, json).with_context(|| format!("cannot write {}", json_path.display()))?;
    Ok(png_path)
}

/// Encodes RGBA8 pixels as PNG. The encoder writes no timestamps, so equal pixels give
/// byte-identical files.
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(BufWriter::new(&mut bytes), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
        writer.finish()?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneId;

    #[test]
    fn test_encode_png_same_pixels_identical_bytes() {
        let pixels: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
        let first = encode_png(4, 4, &pixels).expect("encode");
        let second = encode_png(4, 4, &pixels).expect("encode");
        assert_eq!(first, second);
    }

    #[test]
    fn test_encode_png_wrong_length_fails() {
        assert!(encode_png(4, 4, &[0; 10]).is_err());
    }

    #[test]
    fn test_write_creates_png_and_sidecar() {
        let dir = tempfile::tempdir().expect("temp dir");
        let scene = Scene::build(SceneId::G3, 2.0, 3);
        let sidecar = Sidecar::new(&scene, (2, 1), RenderingMode::Pass);
        let png = write(&dir.path().join("out"), &[0; 8], &sidecar).expect("write");
        assert_eq!(
            png.file_name().and_then(|n| n.to_str()),
            Some("g3-f00003.png")
        );
        let json = fs::read_to_string(png.with_extension("json")).expect("sidecar");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["rendering"], "pass");
        assert_eq!(value["counter"], 3);
        assert_eq!(value["probes"].as_array().map(Vec::len), Some(16));
        assert_eq!(value["markers"][0]["kind"], "world");
    }
}
