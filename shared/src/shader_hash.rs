//! Shader identity used by dumps and per-game profiles.
//!
//! The algorithm is fixed forever: XXH3 64-bit, seed 0, over the exact bytes the
//! application passes in (SPIR-V words in memory order, or a DXIL container). Changing it
//! would orphan every profile entry ever written.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::{Error, Result};

/// 64-bit shader hash, written as 16 lowercase hex digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShaderHash(u64);

impl ShaderHash {
    /// Hashes a shader's bytes.
    pub fn of(code: &[u8]) -> Self {
        Self(xxhash_rust::xxh3::xxh3_64(code))
    }

    /// Hashes SPIR-V words in little-endian byte order, which is their memory order on
    /// every platform the project targets.
    pub fn of_words(words: &[u32]) -> Self {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        Self::of(&bytes)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ShaderHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl FromStr for ShaderHash {
    type Err = Error;

    /// Accepts exactly 16 hex digits in either case.
    fn from_str(text: &str) -> Result<Self> {
        let valid = text.len() == 16 && text.bytes().all(|byte| byte.is_ascii_hexdigit());
        let parsed = valid.then(|| u64::from_str_radix(text, 16).ok()).flatten();
        parsed.map(Self).ok_or_else(|| Error::InvalidShaderHash {
            text: text.to_owned(),
        })
    }
}

impl Serialize for ShaderHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ShaderHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_hash_matches_xxh3_reference_vector() {
        assert_eq!(ShaderHash::of(b"").value(), 0x2d06_8005_38d3_94c2);
    }

    #[test]
    fn test_shader_hash_words_equal_their_bytes() {
        let words = [0x0723_0203, 0x0001_0600];
        let bytes = [0x03, 0x02, 0x23, 0x07, 0x00, 0x06, 0x01, 0x00];
        assert_eq!(ShaderHash::of_words(&words), ShaderHash::of(&bytes));
    }

    #[test]
    fn test_shader_hash_display_parse_round_trip() {
        let hash = ShaderHash::of(b"spirv");
        let text = hash.to_string();
        assert_eq!(text.len(), 16);
        assert_eq!(text.parse::<ShaderHash>().expect("parse"), hash);
        assert_eq!(
            text.to_uppercase().parse::<ShaderHash>().expect("parse"),
            hash
        );
    }

    #[test]
    fn test_shader_hash_rejects_malformed_text() {
        for text in [
            "",
            "abc",
            "0123456789abcdeg",
            "0123456789abcdef0",
            "+123456789abcdef",
        ] {
            assert!(text.parse::<ShaderHash>().is_err(), "{text}");
        }
    }
}
