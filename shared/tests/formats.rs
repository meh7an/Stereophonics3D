//! Golden-file tests: the annotated example files must parse to the expected values and
//! survive a serialize/parse round trip.

use s3d_shared::{
    AnaglyphPreset, Config, MatrixChoice, Profile, ShaderHash, ShaderOverride, StereoParams,
};

const CONFIG: &str = include_str!("fixtures/config.toml");
const PROFILE: &str = include_str!("fixtures/profile.toml");

fn hash(text: &str) -> ShaderHash {
    text.parse().expect("valid hash")
}

#[test]
fn test_config_fixture_equals_built_in_defaults() {
    let config: Config = toml::from_str(CONFIG).expect("parse");
    assert_eq!(config, Config::default());
    assert_eq!(config.stereo, StereoParams::default());
    assert_eq!(
        config.anaglyph.matrix,
        MatrixChoice::Preset(AnaglyphPreset::Dubois2001)
    );
}

#[test]
fn test_profile_fixture_parses_every_override_kind() {
    let profile: Profile = toml::from_str(PROFILE).expect("parse");
    assert_eq!(profile.executable, "ExampleGame.exe");
    assert_eq!(profile.stereo.convergence, Some(12.5));
    assert_eq!(profile.stereo.max_parallax_percent, None);
    assert_eq!(
        profile.anaglyph.matrix.map(MatrixChoice::matrix),
        Some(AnaglyphPreset::Color.matrix())
    );
    let expected = [
        ("9f2c4a1b0d3e5f67", ShaderOverride::Skip),
        ("0123456789abcdef", ShaderOverride::Mono),
        ("00000000000000bb", ShaderOverride::Stereo),
        ("fedcba9876543210", ShaderOverride::HudDepth(1.5)),
        ("00000000000000aa", ShaderOverride::Sky),
    ];
    assert_eq!(profile.shaders.len(), expected.len());
    for (key, value) in expected {
        assert_eq!(profile.shaders.get(&hash(key)), Some(&value), "{key}");
    }
}

#[test]
fn test_fixtures_round_trip() {
    let config: Config = toml::from_str(CONFIG).expect("parse");
    let text = toml::to_string(&config).expect("serialize");
    assert_eq!(toml::from_str::<Config>(&text).expect("reparse"), config);

    let profile: Profile = toml::from_str(PROFILE).expect("parse");
    let text = toml::to_string(&profile).expect("serialize");
    assert_eq!(toml::from_str::<Profile>(&text).expect("reparse"), profile);
}
