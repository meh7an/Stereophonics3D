//! Depth checks between two captures of one deterministic scene.
//!
//! The shear changes clip-space x only, never z or w, so a sheared eye moves every marker
//! sideways but leaves its depth exactly as in mono; with the eye sign at zero the patched
//! run must match the unpatched one bit for bit.

use std::path::Path;

use s3d_shared::capture::{
    CapturePaths, Image, MarkerKind, Sidecar, read_depth, read_png, read_sidecar,
};

use crate::{
    Measurement, Result,
    image::{Rgb, find_color},
};

/// Colour tolerance when locating a marker.
const COLOR_TOLERANCE: u8 = 2;
/// Largest depth difference accepted for a marker; the values are computed identically,
/// so this only absorbs printing round trips.
const DEPTH_TOLERANCE: f64 = 1e-6;

/// A mono capture with its depth dump.
struct DepthCapture {
    sidecar: Sidecar,
    image: Image,
    depth: Vec<f32>,
}

impl DepthCapture {
    fn load(sidecar_path: &Path) -> Result<Self> {
        let sidecar = read_sidecar(sidecar_path)?;
        let dir = sidecar_path.parent().unwrap_or(Path::new("."));
        let paths = CapturePaths::new(dir, &sidecar.scene, sidecar.frame);
        let image = read_png(&paths.image(None))?;
        let depth = read_depth(&paths.depth(), sidecar.width, sidecar.height)?;
        Ok(Self {
            sidecar,
            image,
            depth,
        })
    }

    /// Depth at the centre of the marker drawn in `color`, if it is visible.
    fn marker_depth(&self, color: Rgb) -> Option<f32> {
        let blob = find_color(&self.image, color, COLOR_TOLERANCE)?;
        let (x, y) = blob.centroid;
        // Centroids are pixel-centre coordinates: pixel i spans [i, i + 1).
        let index = y.floor() as usize * self.image.width as usize + x.floor() as usize;
        self.depth.get(index).copied()
    }
}

/// Every opaque world marker has the same depth in `other` as in `reference`, measured
/// at wherever each capture shows it. Markers located by colour need a colour no other
/// marker shares and must show in the reference; lit or tone-mapped scenes leave few, and
/// the check is meant for flat-coloured scenes such as G0.
///
/// # Errors
///
/// Returns an error when a capture, image or depth dump cannot be read.
pub fn marker_depths(reference: &Path, other: &Path) -> Result<Vec<Measurement>> {
    let reference = DepthCapture::load(reference)?;
    let other = DepthCapture::load(other)?;
    let markers = &reference.sidecar.markers;
    let unique = |color: Rgb| markers.iter().filter(|m| m.color == color).count() == 1;
    let measurements = markers
        .iter()
        .filter(|m| m.kind == MarkerKind::World && m.alpha >= 1.0 && unique(m.color))
        .filter_map(|marker| {
            let expected = reference.marker_depth(marker.color)?;
            Some(match other.marker_depth(marker.color) {
                Some(measured) => Measurement::within(
                    format!("marker {} depth", marker.id),
                    f64::from(measured),
                    f64::from(expected),
                    DEPTH_TOLERANCE,
                ),
                None => {
                    Measurement::check(format!("marker {} visible", marker.id), false, 0.0, 1.0)
                }
            })
        })
        .collect();
    Ok(measurements)
}

/// Whether two captures hold identical pixels and identical depth.
///
/// # Errors
///
/// Returns an error when a capture, image or depth dump cannot be read.
pub fn identical(reference: &Path, other: &Path) -> Result<Vec<Measurement>> {
    let reference = DepthCapture::load(reference)?;
    let other = DepthCapture::load(other)?;
    let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let pixels = differing(&reference.image.rgba, &other.image.rgba);
    let depths = reference
        .depth
        .iter()
        .zip(&other.depth)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let same_size = reference.image.rgba.len() == other.image.rgba.len()
        && reference.depth.len() == other.depth.len();
    Ok(vec![
        Measurement::check("same size".to_owned(), same_size, 0.0, 0.0),
        Measurement::at_most("differing colour bytes".to_owned(), pixels as f64, 0.0),
        Measurement::at_most("differing depth values".to_owned(), depths as f64, 0.0),
    ])
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use s3d_shared::capture::{MarkerRecord, write_depth, write_png, write_sidecar};

    use super::*;

    fn scratch(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("s3d-depth-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A 4x1 capture: one red marker pixel at `x` with depth `marker_depth`, the rest at
    /// depth 1.
    fn write_capture(dir: &Path, x: usize, marker_depth: f32) -> PathBuf {
        let mut sidecar = Sidecar::injector("g0", 0, (4, 1), None);
        sidecar.markers.push(MarkerRecord {
            id: 7,
            kind: MarkerKind::World,
            color: [255, 0, 0],
            alpha: 1.0,
            center: [0.0; 3],
            half_extent: [0.1; 2],
            depth: 2.0,
        });
        let mut rgba = vec![0; 16];
        rgba[x * 4..x * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
        let mut depth = vec![1.0; 4];
        depth[x] = marker_depth;
        let paths = CapturePaths::new(dir, "g0", 0);
        write_png(&paths.image(None), 4, 1, &rgba).expect("png");
        write_depth(&paths.depth(), &depth).expect("depth");
        write_sidecar(&paths.sidecar(), &sidecar).expect("json");
        paths.sidecar()
    }

    #[test]
    fn test_moved_marker_keeps_its_depth() {
        let root = scratch("moved");
        let mono = write_capture(&root.join("mono"), 1, 0.5);
        let eye = write_capture(&root.join("eye"), 2, 0.5);
        let warped = write_capture(&root.join("warped"), 2, 0.6);
        assert!(
            marker_depths(&mono, &eye)
                .expect("eye")
                .iter()
                .all(|m| m.ok)
        );
        assert!(
            !marker_depths(&mono, &warped)
                .expect("warped")
                .iter()
                .all(|m| m.ok)
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn test_identical_detects_any_difference() {
        let root = scratch("identical");
        let a = write_capture(&root.join("a"), 1, 0.5);
        let b = write_capture(&root.join("b"), 1, 0.5);
        let moved = write_capture(&root.join("moved"), 2, 0.5);
        assert!(identical(&a, &b).expect("same").iter().all(|m| m.ok));
        assert!(!identical(&a, &moved).expect("moved").iter().all(|m| m.ok));
        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}
