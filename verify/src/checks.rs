//! The Real 3D guarantee tests, G0 to G5, run on one per-eye dump.
//!
//! Markers are found by their unique colours from the sidecar. Test-scene conventions the
//! checks rely on (see the scene builders): G0 grid markers have ids below 100, the G1
//! checker is marker 100 alternating with pure green, G2 stack layers are ids 200-299,
//! the G3 mover is 400, the G5 sphere is 500. Expected disparity comes from
//! `s3d_math::stereo::disparity`, the same formula the shader patch implements.

use std::{fmt, path::Path};

use s3d_math::stereo::disparity;
use s3d_shared::Eye;
use s3d_shared::capture::{
    CapturePaths, Image, MarkerKind, MarkerRecord, Sidecar, read_png, read_sidecar,
};
use serde::Serialize;

use crate::{
    Error, Result, counter,
    image::{
        Rect, Rgb, blend, blob_where, find_color, fraction_of, luma, matches, ndc_to_pixel, pixel,
        project,
    },
};

/// Allowed difference between measured and expected disparity, in pixels.
pub const DISPARITY_TOLERANCE_PX: f64 = 1.0;
/// Largest vertical disparity of a correct parallel camera pair, in pixels.
pub const VERTICAL_TOLERANCE_PX: f64 = 0.5;
/// Largest horizontal offset of the G3 mover between eyes, in pixels.
pub const MOVER_TOLERANCE_PX: f64 = 0.5;
/// Smallest share of clear-colour pixels in a G4 right eye; the rest may be an overlay.
pub const CLEAR_FRACTION: f64 = 0.995;
/// Per-channel colour tolerance when matching markers.
pub const COLOR_TOLERANCE: u8 = 2;
/// G1: pixels the revealing eye must show, and the most the other eye may show.
pub const G1_MIN_PIXELS: usize = 20;
pub const G1_MAX_HIDDEN_PIXELS: usize = 3;
/// G1: longest same-colour horizontal run inside the checker before it counts as smeared.
pub const G1_MAX_RUN: usize = 4;
/// G5: highlight offset beyond the geometric disparity that counts as view-dependent.
pub const G5_VIEW_DEPENDENT_PX: f64 = 0.5;

/// The G1 checker alternates its marker colour with this green (`pattern.frag`).
const CHECKER_GREEN: Rgb = [0, 255, 0];

/// One test of the Real 3D guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Test {
    G0,
    G1,
    G2,
    G3,
    G4,
    G5,
}

impl Test {
    pub const ALL: [Test; 6] = [Test::G0, Test::G1, Test::G2, Test::G3, Test::G4, Test::G5];

    pub fn name(self) -> &'static str {
        match self {
            Self::G0 => "G0",
            Self::G1 => "G1",
            Self::G2 => "G2",
            Self::G3 => "G3",
            Self::G4 => "G4",
            Self::G5 => "G5",
        }
    }

    /// The test a test-scene scene was built for, if any; G4 needs a special dump.
    pub fn for_scene(scene: &str) -> Option<Test> {
        match scene {
            "g0" => Some(Self::G0),
            "g1" => Some(Self::G1),
            "g2" => Some(Self::G2),
            "g3" => Some(Self::G3),
            "g5" => Some(Self::G5),
            _ => None,
        }
    }
}

/// How a test ended. G5 is graded rather than passed or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Pass,
    Fail,
    /// View-dependent shading present.
    ViewDependent,
    /// Shading computed from the mono camera.
    MonoShading,
}

/// One measured quantity against its expectation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    pub subject: String,
    pub measured: f64,
    pub expected: f64,
    pub tolerance: f64,
    pub ok: bool,
}

impl Measurement {
    pub(crate) fn within(subject: String, measured: f64, expected: f64, tolerance: f64) -> Self {
        let ok = (measured - expected).abs() <= tolerance;
        Self {
            subject,
            measured,
            expected,
            tolerance,
            ok,
        }
    }

    pub(crate) fn at_most(subject: String, measured: f64, limit: f64) -> Self {
        Self {
            subject,
            measured,
            expected: 0.0,
            tolerance: limit,
            ok: measured <= limit,
        }
    }

    pub(crate) fn check(subject: String, ok: bool, measured: f64, expected: f64) -> Self {
        Self {
            subject,
            measured,
            expected,
            tolerance: 0.0,
            ok,
        }
    }
}

/// The result of one test on one dump.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finding {
    pub test: Test,
    pub outcome: Outcome,
    pub summary: String,
    pub measurements: Vec<Measurement>,
}

impl Finding {
    fn from_measurements(test: Test, summary: String, measurements: Vec<Measurement>) -> Self {
        let outcome = if !measurements.is_empty() && measurements.iter().all(|m| m.ok) {
            Outcome::Pass
        } else {
            Outcome::Fail
        };
        Self {
            test,
            outcome,
            summary,
            measurements,
        }
    }

    pub fn failed(&self) -> bool {
        self.outcome == Outcome::Fail
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let outcome = match self.outcome {
            Outcome::Pass => "PASS",
            Outcome::Fail => "FAIL",
            Outcome::ViewDependent => "GRADED: view-dependent",
            Outcome::MonoShading => "GRADED: mono shading",
        };
        writeln!(f, "{} {outcome} - {}", self.test.name(), self.summary)?;
        for m in self.measurements.iter().filter(|m| !m.ok) {
            writeln!(
                f,
                "    {}: measured {:.3}, expected {:.3} (tolerance {:.3})",
                m.subject, m.measured, m.expected, m.tolerance
            )?;
        }
        Ok(())
    }
}

/// A per-eye dump: sidecar plus left and right images.
#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    pub sidecar: Sidecar,
    pub left: Image,
    pub right: Image,
}

impl Capture {
    /// Loads `<scene>-f<frame>` from `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error when a file is missing or the dump is not a per-eye one.
    pub fn load(dir: &Path, scene: &str, frame: u32) -> Result<Self> {
        let paths = CapturePaths::new(dir, scene, frame);
        let sidecar = read_sidecar(&paths.sidecar())?;
        let left = read_png(&paths.image(Some(Eye::Left)))?;
        let right = read_png(&paths.image(Some(Eye::Right)))?;
        Self::new(sidecar, left, right)
    }

    /// Checks that the parts belong together.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotStereo`] without a stereo section and [`Error::Size`] when an
    /// image does not match the sidecar's size.
    pub fn new(sidecar: Sidecar, left: Image, right: Image) -> Result<Self> {
        if sidecar.stereo.is_none() {
            return Err(Error::NotStereo(sidecar.scene));
        }
        let size = (sidecar.width, sidecar.height);
        if (left.width, left.height) != size || (right.width, right.height) != size {
            return Err(Error::Size {
                width: size.0,
                height: size.1,
            });
        }
        Ok(Self {
            sidecar,
            left,
            right,
        })
    }

    /// Expected right-minus-left disparity of a point at depth `w`, in pixels.
    fn expected_px(&self, w: f32) -> f64 {
        let (separation, convergence) = self
            .sidecar
            .stereo
            .as_ref()
            .map_or((0.0, 1.0), |s| (s.separation, s.convergence));
        f64::from(disparity(separation, convergence, w)) * f64::from(self.sidecar.width)
    }

    fn markers(&self, keep: impl Fn(&MarkerRecord) -> bool) -> Vec<&MarkerRecord> {
        self.sidecar
            .markers
            .iter()
            .filter(|m| m.kind == MarkerKind::World && keep(m))
            .collect()
    }
}

/// Runs `test` on `capture`.
///
/// # Errors
///
/// Returns [`Error::NotApplicable`] when the dump lacks what the test needs.
pub fn run(test: Test, capture: &Capture) -> Result<Finding> {
    match test {
        Test::G0 => g0(capture),
        Test::G1 => g1(capture),
        Test::G2 => g2(capture),
        Test::G3 => g3(capture),
        Test::G4 => Ok(g4(capture)),
        Test::G5 => g5(capture),
    }
}

fn not_applicable(test: Test, reason: &str) -> Error {
    Error::NotApplicable {
        test: test.name(),
        reason: reason.to_owned(),
    }
}

/// G0: each grid marker's disparity follows the formula, rows line up, and near markers
/// cross the right way.
fn g0(capture: &Capture) -> Result<Finding> {
    let markers = capture.markers(|m| m.id < 100 && m.alpha >= 1.0);
    if markers.is_empty() {
        return Err(not_applicable(Test::G0, "no grid markers (ids below 100)"));
    }
    let mut measurements = Vec::new();
    let mut worst: f64 = 0.0;
    for marker in &markers {
        let id = marker.id;
        let left = find_color(&capture.left, marker.color, COLOR_TOLERANCE);
        let right = find_color(&capture.right, marker.color, COLOR_TOLERANCE);
        let (Some(left), Some(right)) = (left, right) else {
            measurements.push(Measurement::check(
                format!("marker {id} visible"),
                false,
                0.0,
                1.0,
            ));
            continue;
        };
        let measured = right.centroid.0 - left.centroid.0;
        let expected = capture.expected_px(marker.depth);
        worst = worst.max((measured - expected).abs());
        measurements.push(Measurement::within(
            format!("marker {id} disparity px"),
            measured,
            expected,
            DISPARITY_TOLERANCE_PX,
        ));
        measurements.push(Measurement::at_most(
            format!("marker {id} vertical px"),
            (right.centroid.1 - left.centroid.1).abs(),
            VERTICAL_TOLERANCE_PX,
        ));
        if expected.abs() > DISPARITY_TOLERANCE_PX {
            measurements.push(Measurement::check(
                format!("marker {id} crossing direction"),
                measured.signum() == expected.signum(),
                measured.signum(),
                expected.signum(),
            ));
        }
    }
    let summary = format!(
        "{} markers, worst disparity error {worst:.2} px",
        markers.len()
    );
    Ok(Finding::from_measurements(Test::G0, summary, measurements))
}

/// G1: the checker hidden behind the pillar shows in exactly one eye, as real
/// high-frequency detail rather than stretched pixels.
fn g1(capture: &Capture) -> Result<Finding> {
    let [marker] = capture.markers(|m| m.id == 100)[..] else {
        return Err(not_applicable(Test::G1, "no checker marker (id 100)"));
    };
    let is_checker = |p: Rgb| {
        matches(p, marker.color, COLOR_TOLERANCE) || matches(p, CHECKER_GREEN, COLOR_TOLERANCE)
    };
    let count = |image: &Image| {
        blob_where(image, Rect::whole(image), is_checker).map_or(0, |blob| blob.count)
    };
    let (left, right) = (count(&capture.left), count(&capture.right));
    let (shown, hidden, image) = if right >= left {
        (right, left, &capture.right)
    } else {
        (left, right, &capture.left)
    };

    let mut measurements = vec![
        Measurement::check(
            "checker pixels in revealing eye".into(),
            shown >= G1_MIN_PIXELS,
            shown as f64,
            G1_MIN_PIXELS as f64,
        ),
        Measurement::at_most(
            "checker pixels in other eye".into(),
            hidden as f64,
            G1_MAX_HIDDEN_PIXELS as f64,
        ),
    ];
    let greens = blob_where(image, Rect::whole(image), |p| {
        matches(p, CHECKER_GREEN, COLOR_TOLERANCE)
    })
    .map_or(0, |b| b.count);
    let share = if shown == 0 {
        0.0
    } else {
        greens as f64 / shown as f64
    };
    measurements.push(Measurement::within(
        "green share of checker".into(),
        share,
        0.5,
        0.25,
    ));
    measurements.push(Measurement::at_most(
        "longest same-colour run px".into(),
        longest_run(image, is_checker) as f64,
        G1_MAX_RUN as f64,
    ));
    let summary = format!("checker pixels left {left}, right {right}");
    Ok(Finding::from_measurements(Test::G1, summary, measurements))
}

/// Longest horizontal run of identical colour among pixels passing `keep`.
fn longest_run(image: &Image, keep: impl Fn(Rgb) -> bool) -> usize {
    let mut longest = 0;
    for y in 0..image.height {
        let mut run = 0;
        let mut previous = None;
        for x in 0..image.width {
            let color = pixel(image, x, y);
            if !keep(color) {
                run = 0;
                previous = None;
                continue;
            }
            run = if previous == Some(color) { run + 1 } else { 1 };
            previous = Some(color);
            longest = longest.max(run);
        }
    }
    longest
}

/// G2: each half-transparent layer moves by the disparity of its own depth, not of the
/// opaque wall behind it. Measured on each layer's right edge, which no other layer
/// covers in the test scene's staggered stack.
fn g2(capture: &Capture) -> Result<Finding> {
    let layers = capture.markers(|m| (200..300).contains(&m.id) && m.alpha < 1.0);
    if layers.is_empty() {
        return Err(not_applicable(Test::G2, "no stack layers (ids 200-299)"));
    }
    let Some(backdrop) = capture.sidecar.backdrop_color else {
        return Err(not_applicable(
            Test::G2,
            "the sidecar has no backdrop_color",
        ));
    };
    let mut measurements = Vec::new();
    for layer in &layers {
        let id = layer.id;
        let color = blend(layer.color, layer.alpha, backdrop);
        let left = find_color(&capture.left, color, COLOR_TOLERANCE);
        let right = find_color(&capture.right, color, COLOR_TOLERANCE);
        let (Some(left), Some(right)) = (left, right) else {
            measurements.push(Measurement::check(
                format!("layer {id} visible"),
                false,
                0.0,
                1.0,
            ));
            continue;
        };
        measurements.push(Measurement::within(
            format!("layer {id} at depth {} disparity px", layer.depth),
            right.max_x - left.max_x,
            capture.expected_px(layer.depth),
            DISPARITY_TOLERANCE_PX,
        ));
    }
    let summary = format!("{} transparent layers", layers.len());
    Ok(Finding::from_measurements(Test::G2, summary, measurements))
}

/// G3: the mover at convergence depth sits at the same place in both eyes, and both eyes
/// show the same frame counter.
fn g3(capture: &Capture) -> Result<Finding> {
    let [mover] = capture.markers(|m| m.id == 400)[..] else {
        return Err(not_applicable(Test::G3, "no mover marker (id 400)"));
    };
    let mut measurements = Vec::new();
    let left = find_color(&capture.left, mover.color, COLOR_TOLERANCE);
    let right = find_color(&capture.right, mover.color, COLOR_TOLERANCE);
    match (left, right) {
        (Some(left), Some(right)) => measurements.push(Measurement::within(
            "mover offset px".into(),
            right.centroid.0 - left.centroid.0,
            capture.expected_px(mover.depth),
            MOVER_TOLERANCE_PX,
        )),
        _ => measurements.push(Measurement::check("mover visible".into(), false, 0.0, 1.0)),
    }
    let probes = &capture.sidecar.probes;
    let left_counter = counter::decode(&capture.left, probes)?;
    let right_counter = counter::decode(&capture.right, probes)?;
    measurements.push(Measurement::check(
        "counters equal".into(),
        left_counter == right_counter,
        f64::from(right_counter),
        f64::from(left_counter),
    ));
    let summary = format!("counters left {left_counter}, right {right_counter}");
    Ok(Finding::from_measurements(Test::G3, summary, measurements))
}

/// G4: with every right-eye draw suppressed, the right image is the clear colour (plus
/// at most an injector overlay), so it cannot have been derived from the left.
fn g4(capture: &Capture) -> Finding {
    let share = fraction_of(&capture.right, capture.sidecar.clear_color, COLOR_TOLERANCE);
    let marker_colors: Vec<Rgb> = capture.sidecar.markers.iter().map(|m| m.color).collect();
    let leaked = blob_where(&capture.right, Rect::whole(&capture.right), |p| {
        marker_colors
            .iter()
            .any(|color| matches(p, *color, COLOR_TOLERANCE))
    })
    .map_or(0, |blob| blob.count);
    // The share alone misses sparse scenes, where a copied eye is still mostly clear; the
    // scene's own marker colours must not appear at all.
    let measurements = vec![
        Measurement::check(
            "right-eye clear-colour share".into(),
            share >= CLEAR_FRACTION,
            share,
            CLEAR_FRACTION,
        ),
        Measurement::at_most(
            "scene marker pixels in right eye".into(),
            leaked as f64,
            0.0,
        ),
    ];
    let summary = format!(
        "{:.2} % of the right eye is clear colour, {leaked} scene pixels",
        share * 100.0
    );
    Finding::from_measurements(Test::G4, summary, measurements)
}

/// G5 (graded): whether the sphere's highlight moves between eyes beyond the sphere's own
/// geometric disparity, which only happens when shading uses each eye's position.
fn g5(capture: &Capture) -> Result<Finding> {
    let [sphere] = capture.markers(|m| m.id == 500)[..] else {
        return Err(not_applicable(Test::G5, "no sphere marker (id 500)"));
    };
    let sidecar = &capture.sidecar;
    let center = project(sidecar, sphere.center);
    let right_axis = [sidecar.view[0], sidecar.view[4], sidecar.view[8]];
    let edge = project(
        sidecar,
        [
            sphere.center[0] + right_axis[0] * sphere.half_extent[0],
            sphere.center[1] + right_axis[1] * sphere.half_extent[0],
            sphere.center[2] + right_axis[2] * sphere.half_extent[0],
        ],
    );
    let (width, height) = (sidecar.width, sidecar.height);
    let (cx, cy) = ndc_to_pixel([center[0], center[1]], width, height);
    let (ex, _) = ndc_to_pixel([edge[0], edge[1]], width, height);
    let radius = (ex - cx).abs();
    let geometric = capture.expected_px(sphere.depth);

    let highlight = |image: &Image, shift: f64| {
        let region = Rect::around(image, cx + shift, cy, radius);
        let brightest = region
            .pixels()
            .map(|(x, y)| luma(pixel(image, x, y)))
            .fold(0.0, f64::max);
        blob_where(image, region, |p| luma(p) >= brightest * 0.95)
    };
    let (Some(left), Some(right)) = (
        highlight(&capture.left, -geometric / 2.0),
        highlight(&capture.right, geometric / 2.0),
    ) else {
        return Err(not_applicable(Test::G5, "the sphere is not in view"));
    };
    let offset = (right.centroid.0 - left.centroid.0) - geometric;
    let outcome = if offset.abs() > G5_VIEW_DEPENDENT_PX {
        Outcome::ViewDependent
    } else {
        Outcome::MonoShading
    };
    Ok(Finding {
        test: Test::G5,
        outcome,
        summary: format!("highlight moves {offset:+.2} px beyond the sphere's disparity"),
        measurements: vec![Measurement::check(
            "highlight offset beyond geometry px".into(),
            outcome == Outcome::ViewDependent,
            offset,
            G5_VIEW_DEPENDENT_PX,
        )],
    })
}
