//! Builds a per-eye dump from two single-eye runs of a deterministic scene.
//!
//! The injector captures each eye but knows nothing about the scene, while the scene's
//! own mono capture of the same run carries the ground truth. Combining takes the
//! markers from the scene, the images and stereo settings from the injector, and checks
//! that both captured the same pixels.

use std::path::{Path, PathBuf};

use s3d_shared::{
    Eye,
    capture::{
        CapturePaths, Sidecar, StereoMethod, StereoRecord, read_png, read_sidecar, write_png,
        write_sidecar,
    },
};

use crate::{Error, Result};

/// One single-eye run: the scene's mono capture and the injector's capture of it.
#[derive(Debug, Clone)]
pub struct Run {
    /// Sidecar the scene wrote, next to its mono PNG.
    pub scene: PathBuf,
    /// Sidecar the injector wrote, next to its per-eye PNG.
    pub injector: PathBuf,
}

/// Writes `<scene>-f<frame>` with both eyes into `out` and returns the sidecar path.
///
/// # Errors
///
/// Returns [`Error::Mismatch`] when the runs are not two eyes of one frame with the same
/// settings, or when the injector's image differs from the scene's own capture.
pub fn combine(left: &Run, right: &Run, out: &Path) -> Result<PathBuf> {
    let (left_scene, left_image) = load(left, Eye::Left)?;
    let (right_scene, right_image) = load(right, Eye::Right)?;
    if (&left_scene.scene, left_scene.frame) != (&right_scene.scene, right_scene.frame) {
        return Err(Error::Mismatch(format!(
            "left is {} frame {}, right is {} frame {}",
            left_scene.scene, left_scene.frame, right_scene.scene, right_scene.frame
        )));
    }
    if left_scene.markers != right_scene.markers {
        return Err(Error::Mismatch(
            "the scene's markers differ between the runs".to_owned(),
        ));
    }
    let (left_stereo, right_stereo) = (stereo(&left.injector)?, stereo(&right.injector)?);
    if (left_stereo.separation, left_stereo.convergence)
        != (right_stereo.separation, right_stereo.convergence)
    {
        return Err(Error::Mismatch(
            "separation or convergence differs between the runs".to_owned(),
        ));
    }

    let mut sidecar = left_scene;
    sidecar.stereo = Some(StereoRecord {
        method: StereoMethod::SingleEye,
        separation: left_stereo.separation,
        convergence: left_stereo.convergence,
        eyes: Eye::BOTH.to_vec(),
    });
    let paths = CapturePaths::new(out, &sidecar.scene, sidecar.frame);
    for (eye, image) in [(Eye::Left, &left_image), (Eye::Right, &right_image)] {
        write_png(
            &paths.image(Some(eye)),
            image.width,
            image.height,
            &image.rgba,
        )?;
    }
    write_sidecar(&paths.sidecar(), &sidecar)?;
    Ok(paths.sidecar())
}

/// The scene sidecar of `run` and the injector's image of `eye`, after checking that the
/// injector captured exactly what the scene did.
fn load(run: &Run, eye: Eye) -> Result<(Sidecar, s3d_shared::capture::Image)> {
    let scene = read_sidecar(&run.scene)?;
    let injector = read_sidecar(&run.injector)?;
    let recorded = injector.stereo.as_ref().map(|s| s.eyes.as_slice());
    if recorded != Some(&[eye][..]) {
        return Err(Error::Mismatch(format!(
            "{} is not a {} capture",
            run.injector.display(),
            eye.name()
        )));
    }
    if injector.frame != scene.frame {
        return Err(Error::Mismatch(format!(
            "the injector captured frame {}, the scene frame {}",
            injector.frame, scene.frame
        )));
    }
    let image = read_png(&beside(&run.injector, &injector.scene, injector.frame).image(Some(eye)))?;
    let own = read_png(&beside(&run.scene, &scene.scene, scene.frame).image(None))?;
    if image != own {
        return Err(Error::Mismatch(format!(
            "the injector's {} image differs from the scene's own capture",
            eye.name()
        )));
    }
    Ok((scene, image))
}

/// Capture paths in the directory holding `sidecar`.
fn beside(sidecar: &Path, scene: &str, frame: u32) -> CapturePaths {
    CapturePaths::new(sidecar.parent().unwrap_or(Path::new(".")), scene, frame)
}

fn stereo(path: &Path) -> Result<StereoRecord> {
    read_sidecar(path)?
        .stereo
        .ok_or_else(|| Error::NotStereo(path.display().to_string()))
}

#[cfg(test)]
mod tests {
    use s3d_shared::capture::{Image, write_png};

    use super::*;

    /// A fresh directory for one test, removed by the test when it is done.
    fn scratch(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("s3d-combine-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn write_run(dir: &Path, eye: Eye, pixel: u8, frame: u32) -> Run {
        let mut scene = Sidecar::injector("g0", frame, (2, 1), None);
        scene.rendering = "pass".to_owned();
        let rgba = vec![pixel; 8];
        let scene_paths = CapturePaths::new(dir, "g0", frame);
        write_png(&scene_paths.image(None), 2, 1, &rgba).expect("scene png");
        write_sidecar(&scene_paths.sidecar(), &scene).expect("scene json");

        let injector_dir = dir.join("injector");
        let stereo = StereoRecord {
            method: StereoMethod::SingleEye,
            separation: 0.03,
            convergence: 4.0,
            eyes: vec![eye],
        };
        let injector = Sidecar::injector("app", frame, (2, 1), Some(stereo));
        let paths = CapturePaths::new(&injector_dir, "app", frame);
        write_png(&paths.image(Some(eye)), 2, 1, &rgba).expect("injector png");
        write_sidecar(&paths.sidecar(), &injector).expect("injector json");
        Run {
            scene: scene_paths.sidecar(),
            injector: paths.sidecar(),
        }
    }

    #[test]
    fn test_combine_merges_two_single_eye_runs() {
        let root = scratch("merge");
        let left = write_run(&root.join("left"), Eye::Left, 10, 5);
        let right = write_run(&root.join("right"), Eye::Right, 20, 5);
        let out = root.join("out");
        let sidecar = combine(&left, &right, &out).expect("combine");
        let merged = read_sidecar(&sidecar).expect("merged");
        let stereo = merged.stereo.expect("stereo");
        assert_eq!(stereo.eyes, Eye::BOTH);
        assert_eq!(stereo.method, StereoMethod::SingleEye);
        let paths = CapturePaths::new(&out, "g0", 5);
        let right_image = read_png(&paths.image(Some(Eye::Right))).expect("right");
        assert_eq!(
            right_image,
            Image {
                width: 2,
                height: 1,
                rgba: vec![20; 8]
            }
        );
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn test_combine_rejects_swapped_or_mismatched_runs() {
        let root = scratch("mismatch");
        let left = write_run(&root.join("left"), Eye::Left, 10, 5);
        let other_frame = write_run(&root.join("right"), Eye::Right, 20, 6);
        let out = root.join("out");
        assert!(matches!(
            combine(&left, &left, &out),
            Err(Error::Mismatch(_))
        ));
        assert!(matches!(
            combine(&left, &other_frame, &out),
            Err(Error::Mismatch(_))
        ));
        std::fs::remove_dir_all(&root).expect("cleanup");
    }

    #[test]
    fn test_combine_rejects_an_injector_image_that_differs() {
        let root = scratch("tamper");
        let left = write_run(&root.join("left"), Eye::Left, 10, 5);
        let right = write_run(&root.join("right"), Eye::Right, 20, 5);
        let tampered = CapturePaths::new(&root.join("right/injector"), "app", 5);
        write_png(&tampered.image(Some(Eye::Right)), 2, 1, &[99; 8]).expect("tamper");
        assert!(matches!(
            combine(&left, &right, &root.join("out")),
            Err(Error::Mismatch(_))
        ));
        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}
