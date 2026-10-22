//! A lossless, word-level representation of SPIR-V modules.
//!
//! Instructions are split by the word count in their first word, with no grammar
//! knowledge, so every opcode (including ones newer than this code) survives a round
//! trip unchanged, header and generator word included. Operands are decoded only where
//! the analysis and the patch need them, from the fixed layouts in the SPIR-V spec.

use s3d_shared::ShaderHash;
use spirv::{ExecutionModel, Op, Word};

use crate::{Error, Result};

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

/// Pipeline stage of an entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    Vertex,
    TessControl,
    TessEval,
    Geometry,
    Fragment,
    Compute,
    Task,
    Mesh,
    /// Ray tracing, kernels and anything else that never feeds the rasteriser.
    Other,
}

impl Stage {
    fn from_model(model: u32) -> Self {
        match ExecutionModel::from_u32(model) {
            Some(ExecutionModel::Vertex) => Self::Vertex,
            Some(ExecutionModel::TessellationControl) => Self::TessControl,
            Some(ExecutionModel::TessellationEvaluation) => Self::TessEval,
            Some(ExecutionModel::Geometry) => Self::Geometry,
            Some(ExecutionModel::Fragment) => Self::Fragment,
            Some(ExecutionModel::GLCompute) => Self::Compute,
            Some(ExecutionModel::TaskEXT | ExecutionModel::TaskNV) => Self::Task,
            Some(ExecutionModel::MeshEXT | ExecutionModel::MeshNV) => Self::Mesh,
            _ => Self::Other,
        }
    }
}

/// The stage whose `Position` output reaches the rasteriser: the last geometry stage
/// present. Patching an earlier one would distort the inputs of the later ones.
pub fn last_pre_raster(stages: &[Stage]) -> Option<Stage> {
    [Stage::Mesh, Stage::Geometry, Stage::TessEval, Stage::Vertex]
        .into_iter()
        .find(|stage| stages.contains(stage))
}

/// One instruction: its opcode and the words after the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: u16,
    pub operands: Vec<u32>,
}

impl Instruction {
    /// The opcode as an enum, `None` for opcodes this crate's `spirv` version predates.
    pub fn op(&self) -> Option<Op> {
        Op::from_u32(u32::from(self.opcode))
    }

    fn is(&self, op: Op) -> bool {
        u32::from(self.opcode) == op as u32
    }
}

/// One `OpEntryPoint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    pub name: String,
    pub stage: Stage,
    /// Result id of the entry function.
    pub function: Word,
    /// Global variables the entry point declares it uses.
    pub interface: Vec<Word>,
}

/// A parsed module together with the hash of its original bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderModule {
    pub(crate) header: [u32; HEADER_WORDS],
    pub(crate) instructions: Vec<Instruction>,
    hash: ShaderHash,
}

impl ShaderModule {
    /// Splits SPIR-V words into instructions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] for a missing header, a wrong magic number, a zero word
    /// count or an instruction running past the end.
    pub fn parse(words: &[u32]) -> Result<Self> {
        let (header, mut rest) = words
            .split_first_chunk::<HEADER_WORDS>()
            .ok_or_else(|| Error::Parse("shorter than the 5-word header".to_owned()))?;
        if header[0] != MAGIC {
            return Err(Error::Parse(format!(
                "bad magic number {:#010x}",
                header[0]
            )));
        }
        let mut instructions = Vec::new();
        while let Some(&first) = rest.first() {
            let count = (first >> 16) as usize;
            if count == 0 || count > rest.len() {
                let offset = words.len() - rest.len();
                return Err(Error::Parse(format!(
                    "instruction at word {offset} claims {count} words, {} remain",
                    rest.len()
                )));
            }
            instructions.push(Instruction {
                opcode: (first & 0xffff) as u16,
                operands: rest[1..count].to_vec(),
            });
            rest = &rest[count..];
        }
        Ok(Self {
            header: *header,
            instructions,
            hash: ShaderHash::of_words(words),
        })
    }

    /// Parses SPIR-V given as bytes in little-endian word order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Length`] for a partial word and [`Error::Parse`] for bad content.
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(&words_from_bytes(bytes)?)
    }

    /// Hash of the original module, the key for dumps and profiles.
    pub fn hash(&self) -> ShaderHash {
        self.hash
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// The module as words again.
    pub fn assemble(&self) -> Vec<u32> {
        let mut words = self.header.to_vec();
        for instruction in &self.instructions {
            let count = (instruction.operands.len() + 1) as u32;
            words.push(count << 16 | u32::from(instruction.opcode));
            words.extend(&instruction.operands);
        }
        words
    }

    pub fn entry_points(&self) -> Vec<EntryPoint> {
        self.instructions
            .iter()
            .filter(|instruction| instruction.is(Op::EntryPoint))
            .filter_map(|instruction| {
                let [model, function, rest @ ..] = instruction.operands.as_slice() else {
                    return None;
                };
                let (name, used) = decode_string(rest)?;
                Some(EntryPoint {
                    name,
                    stage: Stage::from_model(*model),
                    function: *function,
                    interface: rest[used..].to_vec(),
                })
            })
            .collect()
    }

    /// The entry point called `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoEntryPoint`] when there is none.
    pub fn entry_point(&self, name: &str) -> Result<EntryPoint> {
        self.entry_points()
            .into_iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| Error::NoEntryPoint(name.to_owned()))
    }
}

/// Decodes a nul-terminated literal string; returns it and the words it occupies.
fn decode_string(words: &[u32]) -> Option<(String, usize)> {
    let mut bytes = Vec::new();
    for (index, word) in words.iter().enumerate() {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return Some((String::from_utf8_lossy(&bytes).into_owned(), index + 1));
            }
            bytes.push(byte);
        }
    }
    None
}

/// Little-endian words from bytes, as Vulkan's `pCode` stores them.
///
/// # Errors
///
/// Returns [`Error::Length`] when `bytes` is not a multiple of four long.
pub fn words_from_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::Length(bytes.len()));
    }
    Ok(words.iter().map(|word| u32::from_le_bytes(*word)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_pre_raster_prefers_latest_geometry_stage() {
        use Stage::*;
        assert_eq!(last_pre_raster(&[Vertex, Fragment]), Some(Vertex));
        assert_eq!(
            last_pre_raster(&[Vertex, TessControl, TessEval, Fragment]),
            Some(TessEval)
        );
        assert_eq!(
            last_pre_raster(&[Vertex, TessEval, Geometry, Fragment]),
            Some(Geometry)
        );
        assert_eq!(last_pre_raster(&[Task, Mesh, Fragment]), Some(Mesh));
        assert_eq!(last_pre_raster(&[Compute]), None);
    }

    #[test]
    fn test_words_from_bytes_rejects_partial_words() {
        assert_eq!(words_from_bytes(&[3, 2, 35, 7]).expect("one word"), [MAGIC]);
        assert!(matches!(
            words_from_bytes(&[1, 2, 3]),
            Err(Error::Length(3))
        ));
    }

    #[test]
    fn test_unknown_opcode_survives_round_trip() {
        let words = [MAGIC, 0x0001_0600, 7, 1, 0, 0x0003_ffff, 11, 22];
        let module = ShaderModule::parse(&words).expect("parse");
        assert_eq!(module.instructions()[0].op(), None);
        assert_eq!(module.assemble(), words);
    }

    #[test]
    fn test_truncated_or_bad_modules_are_rejected() {
        assert!(ShaderModule::parse(&[MAGIC, 0, 0]).is_err());
        assert!(ShaderModule::parse(&[0xdead_beef, 0, 0, 0, 0]).is_err());
        assert!(ShaderModule::parse(&[MAGIC, 0, 0, 0, 0, 0x0005_0011, 1]).is_err());
        assert!(ShaderModule::parse(&[MAGIC, 0, 0, 0, 0, 0x0000_0011]).is_err());
    }

    #[test]
    fn test_decode_string_counts_padding_words() {
        let main = [0x6e69_616d, 0];
        assert_eq!(decode_string(&main), Some(("main".to_owned(), 2)));
        assert_eq!(decode_string(&[0x0000_0061]), Some(("a".to_owned(), 1)));
        assert_eq!(decode_string(&[0x6e69_616d]), None);
    }
}
