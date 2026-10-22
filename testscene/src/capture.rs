//! Captures of the presented frame, written with the shared capture format.

use std::path::{Path, PathBuf};

use anyhow::Result;
use s3d_shared::capture::{
    CameraRecord, CapturePaths, LightRecord, MarkerRecord, ProbeRecord, Sidecar, write_png,
    write_sidecar,
};

use crate::scene::{CLEAR_COLOR, Scene};

/// Ground truth for `scene` rendered at `size` through the `rendering` path.
pub fn sidecar(scene: &Scene, size: (u32, u32), rendering: &str) -> Sidecar {
    let camera = &scene.camera;
    Sidecar {
        scene: scene.id.name().to_owned(),
        frame: scene.frame,
        width: size.0,
        height: size.1,
        rendering: rendering.to_owned(),
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
                kind: marker.kind,
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
        stereo: None,
    }
}

/// Writes the mono image and sidecar of one frame into `dir` and returns the image path.
///
/// # Errors
///
/// Returns an error when either file cannot be written.
pub fn write(dir: &Path, rgba: &[u8], sidecar: &Sidecar) -> Result<PathBuf> {
    let paths = CapturePaths::new(dir, &sidecar.scene, sidecar.frame);
    let image = paths.image(None);
    write_png(&image, sidecar.width, sidecar.height, rgba)?;
    write_sidecar(&paths.sidecar(), sidecar)?;
    Ok(image)
}

#[cfg(test)]
mod tests {
    use s3d_shared::capture::MarkerKind;

    use super::*;
    use crate::scene::SceneId;

    #[test]
    fn test_sidecar_carries_scene_ground_truth() {
        let scene = Scene::build(SceneId::G3, 2.0, 3);
        let sidecar = sidecar(&scene, (2, 1), "pass");
        assert_eq!(sidecar.scene, "g3");
        assert_eq!(sidecar.frame, 3);
        assert_eq!(sidecar.rendering, "pass");
        assert_eq!(sidecar.counter, Some(3));
        assert_eq!(sidecar.probes.len(), 16);
        assert_eq!(sidecar.markers.len(), scene.markers.len());
        assert!(sidecar.markers.iter().all(|m| m.kind == MarkerKind::World));
        assert!(sidecar.stereo.is_none());
    }

    #[test]
    fn test_hud_scene_marks_hud_markers() {
        let scene = Scene::build(SceneId::Hud, 2.0, 0);
        let sidecar = sidecar(&scene, (2, 1), "dynamic");
        assert!(sidecar.markers.iter().any(|m| m.kind == MarkerKind::Hud));
    }
}
