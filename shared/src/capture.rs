//! Capture files: RGBA8 PNG images plus a JSON sidecar of ground truth.
//!
//! The test scene writes mono captures, the layer writes per-eye dumps of the same frame,
//! and the verifier reads both. One frame produces `<scene>-f<frame:05>.json` plus either
//! `<stem>.png` (mono) or `<stem>-left.png` and `<stem>-right.png` (per eye).

use std::{
    fs,
    io::{BufWriter, Cursor},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{Error, Eye, Result};

/// Ground truth for one captured frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    pub scene: String,
    pub frame: u32,
    pub width: u32,
    pub height: u32,
    /// Backend-defined render path label, such as `pass` or `dynamic`.
    pub rendering: String,
    pub camera: CameraRecord,
    /// Column-major, world to view.
    pub view: [f32; 16],
    /// Column-major, Vulkan clip convention (Y down, depth in [0, 1]).
    pub projection: [f32; 16],
    pub convergence: f32,
    pub clear_color: [u8; 3],
    pub light: Option<LightRecord>,
    pub sky: bool,
    /// The frame went through a tone-map pass, so absolute colours differ.
    pub post: bool,
    pub markers: Vec<MarkerRecord>,
    pub probes: Vec<ProbeRecord>,
    /// Value the counter probes encode, when the scene has a counter.
    pub counter: Option<u32>,
    /// Present on per-eye dumps written by the injector; absent on mono captures.
    pub stereo: Option<StereoRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraRecord {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub vertical_fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightRecord {
    /// Unit vector toward the light.
    pub direction: [f32; 3],
    pub shadows: bool,
}

/// Where a marker's centre is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkerKind {
    /// World space, rasterised through the camera.
    World,
    /// Clip space with w = 1, drawn by the HUD pass.
    Hud,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerRecord {
    pub id: u32,
    pub kind: MarkerKind,
    pub color: [u8; 3],
    pub alpha: f32,
    pub center: [f32; 3],
    /// Half extents along the camera's right and up axes.
    pub half_extent: [f32; 2],
    /// Clip-space `w` at the centre; 1 for HUD markers.
    pub depth: f32,
}

/// One frame-counter cell, most significant bit first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeRecord {
    pub bit: u32,
    /// Vulkan NDC of the probe pixel (y down).
    pub ndc: [f32; 2],
    pub on: [u8; 3],
    pub off: [u8; 3],
}

/// How the injector produced the per-eye images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum StereoMethod {
    /// One eye per run of a deterministic scene (M3.3).
    SingleEye,
    Multiview,
    Duplication,
    /// Labelled prototype only; fails G3 by construction and is never released.
    AlternateFrame,
}

/// Stereo settings in effect for a per-eye dump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StereoRecord {
    pub method: StereoMethod,
    /// Per-eye shear coefficient `s`, negative when the eyes are swapped.
    pub separation: f32,
    pub convergence: f32,
    /// Eyes written for this frame.
    pub eyes: Vec<Eye>,
}

/// File names of one captured frame inside a capture directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturePaths {
    stem: PathBuf,
}

impl CapturePaths {
    pub fn new(dir: &Path, scene: &str, frame: u32) -> Self {
        Self {
            stem: dir.join(format!("{scene}-f{frame:05}")),
        }
    }

    /// The mono image when `eye` is `None`, otherwise that eye's image.
    pub fn image(&self, eye: Option<Eye>) -> PathBuf {
        let suffix = eye.map_or(String::new(), |eye| format!("-{}", eye.name()));
        self.with_suffix(&format!("{suffix}.png"))
    }

    pub fn sidecar(&self) -> PathBuf {
        self.with_suffix(".json")
    }

    fn with_suffix(&self, suffix: &str) -> PathBuf {
        let mut path = self.stem.clone().into_os_string();
        path.push(suffix);
        path.into()
    }
}

/// A decoded RGBA8 image, rows top to bottom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Encodes RGBA8 pixels as PNG. The encoder writes no timestamps, so equal pixels give
/// byte-identical files.
///
/// # Errors
///
/// Returns [`Error::PngEncode`] when `rgba` does not hold `width * height` pixels.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
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

/// Decodes a PNG written by [`encode_png`].
///
/// # Errors
///
/// Returns [`Error::PngDecode`] for corrupt data and [`Error::UnsupportedPng`] for
/// anything other than 8-bit RGBA.
pub fn decode_png(bytes: &[u8]) -> Result<Image> {
    let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info()?;
    let (color, depth) = reader.output_color_type();
    if color != png::ColorType::Rgba || depth != png::BitDepth::Eight {
        return Err(Error::UnsupportedPng {
            found: format!("{color:?} {depth:?}"),
        });
    }
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| Error::UnsupportedPng {
            found: "image too large".to_owned(),
        })?;
    let mut rgba = vec![0; size];
    let info = reader.next_frame(&mut rgba)?;
    rgba.truncate(info.buffer_size());
    Ok(Image {
        width: info.width,
        height: info.height,
        rgba,
    })
}

/// Writes a PNG, creating parent directories.
///
/// # Errors
///
/// Returns an error when encoding fails or the file cannot be written.
pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    write_file(path, &encode_png(width, height, rgba)?)
}

/// Reads an RGBA8 PNG.
///
/// # Errors
///
/// Returns an error when the file cannot be read or decoded.
pub fn read_png(path: &Path) -> Result<Image> {
    decode_png(&read_file(path)?)
}

/// Writes a sidecar as pretty JSON with a trailing newline, creating parent directories.
///
/// # Errors
///
/// Returns an error when the file cannot be written.
pub fn write_sidecar(path: &Path, sidecar: &Sidecar) -> Result<()> {
    let mut json = serde_json::to_string_pretty(sidecar).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })?;
    json.push('\n');
    write_file(path, json.as_bytes())
}

/// Reads a sidecar.
///
/// # Errors
///
/// Returns an error when the file cannot be read or does not match the schema.
pub fn read_sidecar(path: &Path) -> Result<Sidecar> {
    serde_json::from_slice(&read_file(path)?).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let io_error = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    fs::write(path, bytes).map_err(io_error)
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar() -> Sidecar {
        Sidecar {
            scene: "g3".to_owned(),
            frame: 7,
            width: 2,
            height: 1,
            rendering: "pass".to_owned(),
            camera: CameraRecord {
                position: [0.0; 3],
                target: [0.0, 0.0, -1.0],
                up: [0.0, 1.0, 0.0],
                vertical_fov_deg: 60.0,
                near: 0.1,
                far: 100.0,
            },
            view: [0.0; 16],
            projection: [0.0; 16],
            convergence: 4.0,
            clear_color: [16, 16, 16],
            light: None,
            sky: false,
            post: false,
            markers: vec![MarkerRecord {
                id: 400,
                kind: MarkerKind::World,
                color: [0, 200, 255],
                alpha: 1.0,
                center: [0.1, 0.2, -4.0],
                half_extent: [0.2, 0.2],
                depth: 4.0,
            }],
            probes: vec![ProbeRecord {
                bit: 0,
                ndc: [-0.6, 0.6],
                on: [255, 255, 255],
                off: [70, 70, 70],
            }],
            counter: Some(7),
            stereo: Some(StereoRecord {
                method: StereoMethod::SingleEye,
                separation: 0.03,
                convergence: 4.0,
                eyes: vec![Eye::Left],
            }),
        }
    }

    #[test]
    fn test_capture_paths_mono_eyes_and_sidecar() {
        let paths = CapturePaths::new(Path::new("out"), "g0", 59);
        assert_eq!(paths.image(None), Path::new("out").join("g0-f00059.png"));
        assert_eq!(
            paths.image(Some(Eye::Left)),
            Path::new("out").join("g0-f00059-left.png")
        );
        assert_eq!(
            paths.image(Some(Eye::Right)),
            Path::new("out").join("g0-f00059-right.png")
        );
        assert_eq!(paths.sidecar(), Path::new("out").join("g0-f00059.json"));
    }

    #[test]
    fn test_png_encode_decode_round_trip_and_determinism() {
        let rgba: Vec<u8> = (0..3 * 2 * 4).map(|i| (i * 7) as u8).collect();
        let bytes = encode_png(3, 2, &rgba).expect("encode");
        assert_eq!(bytes, encode_png(3, 2, &rgba).expect("encode"));
        let image = decode_png(&bytes).expect("decode");
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.rgba, rgba);
    }

    #[test]
    fn test_encode_png_wrong_length_fails() {
        assert!(encode_png(4, 4, &[0; 10]).is_err());
    }

    #[test]
    fn test_files_round_trip_through_disk() {
        let dir = tempfile::tempdir().expect("temp dir");
        let paths = CapturePaths::new(&dir.path().join("nested"), "g3", 7);
        let rgba = [1, 2, 3, 255, 4, 5, 6, 255];
        write_png(&paths.image(Some(Eye::Left)), 2, 1, &rgba).expect("write png");
        write_sidecar(&paths.sidecar(), &sidecar()).expect("write sidecar");
        assert_eq!(
            read_png(&paths.image(Some(Eye::Left))).expect("read").rgba,
            rgba
        );
        assert_eq!(read_sidecar(&paths.sidecar()).expect("read"), sidecar());
    }

    #[test]
    fn test_sidecar_json_uses_lowercase_enums() {
        let json = serde_json::to_value(sidecar()).expect("serialize");
        assert_eq!(json["markers"][0]["kind"], "world");
        assert_eq!(json["stereo"]["method"], "single-eye");
        assert_eq!(json["stereo"]["eyes"][0], "left");
    }

    #[test]
    fn test_read_missing_file_reports_path() {
        let error = read_png(Path::new("does/not/exist.png")).expect_err("missing");
        assert!(error.to_string().contains("exist.png"), "{error}");
    }
}
