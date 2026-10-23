//! `s3d-verify <dir>`: runs the Real 3D guarantee tests on the per-eye dumps in `dir`.
//!
//! Each dump gets the test its scene was built for unless `--test` names others (G4 always
//! needs `--test g4`, since it applies only to dumps with the right eye suppressed). The
//! exit status is 1 when any test fails or cannot run, 2 when no dump is found.
//!
//! `s3d-verify combine` builds a per-eye dump from a left-only and a right-only run, and
//! `s3d-verify depth` compares the depth of two captures; both exit 1 on failure.

use std::{fs, path::PathBuf, process::ExitCode};

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand, ValueEnum};
use s3d_shared::capture::read_sidecar;
use s3d_verify::{
    Capture, Finding, Measurement, Test,
    combine::{Run, combine},
    depth, run,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TestArg {
    G0,
    G1,
    G2,
    G3,
    G4,
    G5,
}

impl From<TestArg> for Test {
    fn from(test: TestArg) -> Self {
        match test {
            TestArg::G0 => Test::G0,
            TestArg::G1 => Test::G1,
            TestArg::G2 => Test::G2,
            TestArg::G3 => Test::G3,
            TestArg::G4 => Test::G4,
            TestArg::G5 => Test::G5,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Checks per-eye dumps against the Real 3D guarantee",
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    check: Args,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build a per-eye dump from a left-only and a right-only run of one scene.
    Combine {
        /// The scene's own sidecar from the left-eye run.
        #[arg(long)]
        left_scene: PathBuf,
        /// The injector's sidecar from the left-eye run.
        #[arg(long)]
        left: PathBuf,
        /// The scene's own sidecar from the right-eye run.
        #[arg(long)]
        right_scene: PathBuf,
        /// The injector's sidecar from the right-eye run.
        #[arg(long)]
        right: PathBuf,
        /// Directory receiving the per-eye dump.
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare the depth of two captures of one scene, each with a `-depth.f32` dump.
    Depth {
        /// Sidecar of the reference capture, usually mono.
        reference: PathBuf,
        /// Sidecar of the capture to compare.
        other: PathBuf,
        /// Require identical pixels and depth instead of equal marker depths.
        #[arg(long)]
        identical: bool,
    },
}

#[derive(Debug, clap::Args)]
struct Args {
    /// Directory holding `<scene>-f<frame>.json` with `-left.png` and `-right.png`.
    #[arg(required = true)]
    dir: Option<PathBuf>,

    /// Only dumps of this scene.
    #[arg(long)]
    scene: Option<String>,

    /// Only dumps of this frame.
    #[arg(long)]
    frame: Option<u32>,

    /// Tests to run instead of each scene's own; repeatable.
    #[arg(long = "test", value_enum)]
    tests: Vec<TestArg>,

    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

/// Results for one dump.
#[derive(Debug, Serialize)]
struct DumpReport {
    scene: String,
    frame: u32,
    findings: Vec<Finding>,
    /// Tests that could not run, with the reason.
    errors: Vec<String>,
}

fn main() -> Result<ExitCode> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Combine {
            left_scene,
            left,
            right_scene,
            right,
            out,
        }) => {
            let left = Run {
                scene: left_scene,
                injector: left,
            };
            let right = Run {
                scene: right_scene,
                injector: right,
            };
            let sidecar = combine(&left, &right, &out)?;
            println!("wrote {}", sidecar.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Depth {
            reference,
            other,
            identical,
        }) => {
            let measurements = if identical {
                depth::identical(&reference, &other)?
            } else {
                depth::marker_depths(&reference, &other)?
            };
            Ok(report_measurements(&measurements))
        }
        None => check(cli.check),
    }
}

/// Prints every measurement and turns them into the exit status.
fn report_measurements(measurements: &[Measurement]) -> ExitCode {
    for m in measurements {
        let verdict = if m.ok { "ok  " } else { "FAIL" };
        println!(
            "{verdict} {}: measured {}, expected {} (tolerance {})",
            m.subject, m.measured, m.expected, m.tolerance
        );
    }
    if !measurements.is_empty() && measurements.iter().all(|m| m.ok) {
        println!("PASS");
        ExitCode::SUCCESS
    } else {
        println!("FAIL");
        ExitCode::FAILURE
    }
}

/// The default command: runs the guarantee tests on every per-eye dump in a directory.
fn check(args: Args) -> Result<ExitCode> {
    let dir = args.dir.context("a dump directory is required")?;
    let mut dumps = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("cannot list {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let Ok(sidecar) = read_sidecar(&path) else {
            continue;
        };
        let wanted = sidecar.stereo.is_some()
            && args.scene.as_ref().is_none_or(|s| *s == sidecar.scene)
            && args.frame.is_none_or(|f| f == sidecar.frame);
        if wanted {
            dumps.push((sidecar.scene, sidecar.frame));
        }
    }
    dumps.sort();
    if dumps.is_empty() {
        eprintln!("no per-eye dumps found in {}", dir.display());
        return Ok(ExitCode::from(2));
    }

    let mut reports = Vec::new();
    for (scene, frame) in dumps {
        let tests: Vec<Test> = if args.tests.is_empty() {
            Test::for_scene(&scene).into_iter().collect()
        } else {
            args.tests.iter().copied().map(Test::from).collect()
        };
        let mut report = DumpReport {
            scene: scene.clone(),
            frame,
            findings: Vec::new(),
            errors: Vec::new(),
        };
        match Capture::load(&dir, &scene, frame) {
            Ok(capture) => {
                for test in tests {
                    match run(test, &capture) {
                        Ok(finding) => report.findings.push(finding),
                        Err(error) => report.errors.push(error.to_string()),
                    }
                }
            }
            Err(error) => report.errors.push(error.to_string()),
        }
        reports.push(report);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        for report in &reports {
            println!("{} frame {}", report.scene, report.frame);
            for finding in &report.findings {
                print!("  {finding}");
            }
            for error in &report.errors {
                println!("  ERROR {error}");
            }
        }
    }
    let failed = reports
        .iter()
        .any(|r| !r.errors.is_empty() || r.findings.iter().any(Finding::failed));
    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}
