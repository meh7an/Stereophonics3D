//! Corpus tests over `fixtures/spirv`: every module round-trips word for word and
//! validates, and the analysis finds each fixture's stage, `Position` form and stores.

mod common;

use common::{FIXTURES, module, words};
use s3d_shared::ShaderHash;
use s3d_spirv_patch::{PositionForm, Stage, StoreTarget};

#[test]
fn test_every_fixture_validates_and_round_trips_word_for_word() {
    for name in FIXTURES {
        let original = words(name);
        s3d_spirv_val::validate(&original, s3d_spirv_val::VULKAN_1_3)
            .unwrap_or_else(|e| panic!("{name} invalid: {e}"));
        let reassembled = module(name).assemble();
        assert_eq!(reassembled, original, "{name} changed in the round trip");
    }
}

#[test]
fn test_hash_is_over_original_words() {
    for name in FIXTURES {
        assert_eq!(
            module(name).hash(),
            ShaderHash::of_words(&words(name)),
            "{name}"
        );
    }
}

/// Expected analysis of one entry point.
struct Expected {
    fixture: &'static str,
    entry: &'static str,
    stage: Stage,
    form: Option<PositionForm>,
    arrayed: bool,
    stores: &'static [StoreTarget],
    /// The store sits in a function other than the entry function.
    in_helper: bool,
}

const fn expect(
    fixture: &'static str,
    entry: &'static str,
    stage: Stage,
    form: Option<PositionForm>,
    stores: &'static [StoreTarget],
) -> Expected {
    Expected {
        fixture,
        entry,
        stage,
        form,
        arrayed: false,
        stores,
        in_helper: false,
    }
}

const MEMBER_0: Option<PositionForm> = Some(PositionForm::BlockMember(0));
const VARIABLE: Option<PositionForm> = Some(PositionForm::Variable);
const WHOLE: &[StoreTarget] = &[StoreTarget::Whole];

#[test]
fn test_analysis_matches_expectations() {
    use StoreTarget::Component;
    let expectations = [
        expect("glsl_basic", "main", Stage::Vertex, MEMBER_0, WHOLE),
        expect(
            "glsl_piecewise",
            "main",
            Stage::Vertex,
            MEMBER_0,
            &[Component(0), Component(1), Component(2), Component(3)],
        ),
        expect("glsl_redeclared", "main", Stage::Vertex, MEMBER_0, WHOLE),
        Expected {
            in_helper: true,
            ..expect("glsl_function", "main", Stage::Vertex, MEMBER_0, WHOLE)
        },
        expect("glsl_tese", "main", Stage::TessEval, MEMBER_0, WHOLE),
        expect("glsl_geom", "main", Stage::Geometry, MEMBER_0, WHOLE),
        Expected {
            arrayed: true,
            ..expect("glsl_mesh", "main", Stage::Mesh, MEMBER_0, WHOLE)
        },
        expect("glsl_frag", "main", Stage::Fragment, None, &[]),
        expect("glsl_push", "main", Stage::Vertex, MEMBER_0, WHOLE),
        expect("hlsl_basic", "main", Stage::Vertex, VARIABLE, WHOLE),
        expect("hlsl_struct", "main", Stage::Vertex, VARIABLE, WHOLE),
        expect("multi_entry", "a", Stage::Vertex, VARIABLE, WHOLE),
        expect("multi_entry", "b", Stage::Vertex, VARIABLE, WHOLE),
    ];

    for expected in expectations {
        let label = format!("{} / {}", expected.fixture, expected.entry);
        let analysis = module(expected.fixture)
            .analyze(expected.entry)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(analysis.entry.stage, expected.stage, "{label}");
        assert_eq!(analysis.position.map(|p| p.form), expected.form, "{label}");
        if let Some(position) = analysis.position {
            assert_eq!(position.arrayed, expected.arrayed, "{label}");
        }
        let targets: Vec<StoreTarget> = analysis.stores.iter().map(|s| s.target).collect();
        assert_eq!(targets, expected.stores, "{label}");
        assert!(!analysis.escapes, "{label}");
        let in_entry = analysis
            .stores
            .iter()
            .all(|store| store.function == analysis.entry.function);
        assert_eq!(!in_entry, expected.in_helper, "{label}");
    }
}

#[test]
fn test_multi_entry_points_see_only_their_own_store() {
    let module = module("multi_entry");
    let a = module.analyze("a").expect("a");
    let b = module.analyze("b").expect("b");
    assert_ne!(a.entry.function, b.entry.function);
    assert_eq!(a.stores[0].function, a.entry.function);
    assert_eq!(b.stores[0].function, b.entry.function);
}

#[test]
fn test_unknown_entry_point_is_an_error() {
    assert!(module("glsl_basic").analyze("missing").is_err());
}
