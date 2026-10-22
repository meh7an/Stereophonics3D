//! Patch tests over `fixtures/spirv`: every fixture validates in every variant, the patch
//! only adds code that writes `Position.x`, and disassembly snapshots pin each variant.
//!
//! Set `S3D_UPDATE_SNAPSHOTS=1` to rewrite the snapshots, then review them in the diff.

mod common;

use std::fs;

use common::{FIXTURES, fixtures_dir, module};
use s3d_shared::{Eye, StereoParams};
use s3d_spirv_patch::{
    ConstantSource, DEFAULT_SPEC_ID, Error, EyeSource, Insertion, Instruction, PatchOptions,
    ShaderModule, ShearMode, StoreTarget,
};
use s3d_spirv_val::{VULKAN_1_3, disassemble, validate};

/// Byte offset where the stereo block goes: after the 64-byte application block in
/// `glsl_push`, at 0 everywhere else.
fn push_offset(fixture: &str) -> u32 {
    if fixture == "glsl_push" { 64 } else { 0 }
}

/// Every variant under test, with a label used in messages and snapshot names.
fn variants(fixture: &str) -> Vec<(&'static str, PatchOptions)> {
    let push = ConstantSource::PushConstant {
        offset: push_offset(fixture),
    };
    let spec = ConstantSource::Specialization {
        first_id: DEFAULT_SPEC_ID,
        defaults: StereoParams::default().push_constants(Some(Eye::Left)),
    };
    let options = |constants, eye, mode| PatchOptions {
        constants,
        eye,
        mode,
    };
    vec![
        (
            "push-eye",
            options(push, EyeSource::Constant, ShearMode::Normal),
        ),
        (
            "push-view",
            options(push, EyeSource::ViewIndex, ShearMode::Normal),
        ),
        (
            "spec-eye",
            options(spec, EyeSource::Constant, ShearMode::Normal),
        ),
        (
            "spec-view",
            options(spec, EyeSource::ViewIndex, ShearMode::Normal),
        ),
        (
            "push-eye-hud",
            options(push, EyeSource::Constant, ShearMode::FixedDepth(1.5)),
        ),
        (
            "push-eye-sky",
            options(push, EyeSource::Constant, ShearMode::Infinity),
        ),
    ]
}

fn entries(module: &ShaderModule) -> Vec<String> {
    module.entry_points().into_iter().map(|e| e.name).collect()
}

/// Instructions between `OpFunction` and `OpFunctionEnd`, inclusive.
fn bodies(module: &ShaderModule) -> Vec<&Instruction> {
    let function = spirv::Op::Function as u16;
    let end = spirv::Op::FunctionEnd as u16;
    let mut inside = false;
    module
        .instructions()
        .iter()
        .filter(|instruction| {
            inside |= instruction.opcode == function;
            let keep = inside;
            inside &= instruction.opcode != end;
            keep
        })
        .collect()
}

fn is_subsequence(needle: &[&Instruction], haystack: &[&Instruction]) -> bool {
    let mut remaining = haystack.iter();
    needle
        .iter()
        .all(|wanted| remaining.any(|candidate| candidate == wanted))
}

/// `after` minus `before`, counting duplicates.
fn added_targets(before: &[StoreTarget], after: &[StoreTarget]) -> Vec<StoreTarget> {
    let mut leftover = before.to_vec();
    after
        .iter()
        .filter(|target| match leftover.iter().position(|t| t == *target) {
            Some(index) => {
                leftover.remove(index);
                false
            }
            None => true,
        })
        .copied()
        .collect()
}

#[test]
fn test_every_fixture_validates_in_every_variant() {
    for fixture in FIXTURES {
        let original = module(fixture);
        for entry in entries(&original) {
            for (label, options) in variants(fixture) {
                let context = format!("{fixture} / {entry} / {label}");
                match original.patch(&entry, &options) {
                    Ok((patched, report)) => {
                        validate(&patched.assemble(), VULKAN_1_3)
                            .unwrap_or_else(|e| panic!("{context}: {e}\n{report}"));
                        assert!(report.sites >= 1, "{context}: {report}");
                    }
                    Err(Error::NotPreRaster(_)) if fixture == "glsl_frag" => {}
                    Err(error) => panic!("{context}: {error}"),
                }
            }
        }
    }
}

#[test]
fn test_patch_keeps_original_bodies_and_only_writes_x() {
    for fixture in FIXTURES.into_iter().filter(|f| *f != "glsl_frag") {
        let original = module(fixture);
        for entry in entries(&original) {
            let before = original.analyze(&entry).expect("analyze original");
            let before_targets: Vec<_> = before.stores.iter().map(|s| s.target).collect();
            for (label, options) in variants(fixture) {
                let context = format!("{fixture} / {entry} / {label}");
                let (patched, report) = original.patch(&entry, &options).expect(&context);
                assert!(
                    is_subsequence(&bodies(&original), &bodies(&patched)),
                    "{context}: an original instruction changed"
                );
                let after = patched.analyze(&entry).expect("analyze patched");
                let after_targets: Vec<_> = after.stores.iter().map(|s| s.target).collect();
                let added = added_targets(&before_targets, &after_targets);
                assert_eq!(added.len(), report.sites, "{context}: {report}");
                assert!(
                    added.iter().all(|t| *t == StoreTarget::Component(0)),
                    "{context}: patch wrote {added:?}"
                );
            }
        }
    }
}

#[test]
fn test_insertion_follows_stage() {
    let options = variants("glsl_basic")[0].1;
    let insertion = |fixture: &str| {
        let (_, report) = module(fixture).patch("main", &options).expect(fixture);
        (report.insertion, report.sites)
    };
    assert_eq!(insertion("glsl_basic"), (Insertion::Wrapper, 1));
    assert_eq!(insertion("glsl_tese"), (Insertion::Wrapper, 1));
    assert_eq!(insertion("glsl_geom"), (Insertion::BeforeEmit, 1));
    assert_eq!(insertion("glsl_mesh"), (Insertion::AfterStore, 1));
}

#[test]
fn test_existing_push_block_is_extended_not_duplicated() {
    let options = variants("glsl_push")[0].1;
    let (patched, report) = module("glsl_push").patch("main", &options).expect("patch");
    assert!(report.extended_push_block);
    let text = disassemble(&patched.assemble()).expect("disassemble");
    assert_eq!(text.matches("PushConstant\n").count(), 1, "{text}");
    for offset in [64, 68, 72] {
        assert!(text.contains(&format!("Offset {offset}")), "{text}");
    }
}

#[test]
fn test_disassembly_snapshots() {
    let fixture = "glsl_push";
    let update = std::env::var_os("S3D_UPDATE_SNAPSHOTS").is_some();
    let dir = fixtures_dir().join("snapshots");
    for (label, options) in variants(fixture) {
        let (patched, _) = module(fixture).patch("main", &options).expect(label);
        let text = disassemble(&patched.assemble()).expect("disassemble");
        let path = dir.join(format!("{fixture}.{label}.spvasm"));
        if update {
            fs::create_dir_all(&dir).expect("snapshot dir");
            fs::write(&path, &text).expect("write snapshot");
            continue;
        }
        let expected = fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!(
                "missing {}; run with S3D_UPDATE_SNAPSHOTS=1",
                path.display()
            )
        });
        assert_eq!(text, expected, "{label} differs from {}", path.display());
    }
}
