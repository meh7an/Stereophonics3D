//! Low-level module edits: fresh ids, and declarations inserted into the section the
//! SPIR-V logical layout requires (capabilities, extensions, annotations, globals).
//! Non-aggregate types must be unique in a module, so types are always found before
//! they are added.

use spirv::{Capability, Op, StorageClass, Word};

use crate::{ShaderModule, module::Instruction};

/// Instructions that precede annotations in the logical layout.
const PREAMBLE: [Op; 14] = [
    Op::Capability,
    Op::Extension,
    Op::ExtInstImport,
    Op::MemoryModel,
    Op::EntryPoint,
    Op::ExecutionMode,
    Op::ExecutionModeId,
    Op::String,
    Op::SourceExtension,
    Op::Source,
    Op::SourceContinued,
    Op::Name,
    Op::MemberName,
    Op::ModuleProcessed,
];

const ANNOTATIONS: [Op; 8] = [
    Op::Decorate,
    Op::MemberDecorate,
    Op::DecorationGroup,
    Op::GroupDecorate,
    Op::GroupMemberDecorate,
    Op::DecorateId,
    Op::DecorateString,
    Op::MemberDecorateString,
];

/// Builds an instruction from an opcode and its operands.
pub(crate) fn instruction(op: Op, operands: Vec<u32>) -> Instruction {
    Instruction {
        opcode: op as u16,
        operands,
    }
}

impl ShaderModule {
    /// Allocates an id by raising the header's id bound.
    pub(crate) fn new_id(&mut self) -> Word {
        let id = self.header[3];
        self.header[3] += 1;
        id
    }

    /// SPIR-V version as (major, minor).
    pub(crate) fn version(&self) -> (u32, u32) {
        ((self.header[1] >> 16) & 0xff, (self.header[1] >> 8) & 0xff)
    }

    fn last_of(&self, ops: &[Op]) -> Option<usize> {
        self.instructions
            .iter()
            .rposition(|instruction| instruction.op().is_some_and(|op| ops.contains(&op)))
    }

    /// Index of the first function, where new global declarations go.
    pub(crate) fn globals_end(&self) -> usize {
        self.instructions
            .iter()
            .position(|instruction| instruction.op() == Some(Op::Function))
            .unwrap_or(self.instructions.len())
    }

    /// Adds a type, constant or global variable at the end of the globals section.
    pub(crate) fn add_global(&mut self, op: Op, operands: Vec<u32>) {
        let at = self.globals_end();
        self.instructions.insert(at, instruction(op, operands));
    }

    /// Adds a decoration after the existing ones, or where they would start.
    pub(crate) fn add_annotation(&mut self, op: Op, operands: Vec<u32>) {
        let at = match self.last_of(&ANNOTATIONS) {
            Some(last) => last + 1,
            None => self
                .instructions
                .iter()
                .position(|i| i.op().is_none_or(|op| !PREAMBLE.contains(&op)))
                .unwrap_or(self.instructions.len()),
        };
        self.instructions.insert(at, instruction(op, operands));
    }

    /// Declares `capability` unless present; returns whether it was added.
    pub(crate) fn add_capability(&mut self, capability: Capability) -> bool {
        let value = capability as u32;
        let present = self
            .instructions
            .iter()
            .any(|i| i.op() == Some(Op::Capability) && i.operands.first() == Some(&value));
        if present {
            return false;
        }
        let at = self.last_of(&[Op::Capability]).map_or(0, |last| last + 1);
        self.instructions
            .insert(at, instruction(Op::Capability, vec![value]));
        true
    }

    /// Declares extension `name` unless present; returns whether it was added.
    pub(crate) fn add_extension(&mut self, name: &str) -> bool {
        let operands = encode_string(name);
        let present = self
            .instructions
            .iter()
            .any(|i| i.op() == Some(Op::Extension) && i.operands == operands);
        if present {
            return false;
        }
        let at = self
            .last_of(&[Op::Extension, Op::Capability])
            .map_or(0, |last| last + 1);
        self.instructions
            .insert(at, instruction(Op::Extension, operands));
        true
    }

    /// The type declared by `op` with `operands` after its result id, added if missing.
    pub(crate) fn find_or_add_type(&mut self, op: Op, rest: &[u32]) -> Word {
        let found = self.instructions[..self.globals_end()]
            .iter()
            .find_map(|i| {
                let (result, tail) = i.operands.split_first()?;
                (i.op() == Some(op) && tail == rest).then_some(*result)
            });
        found.unwrap_or_else(|| {
            let id = self.new_id();
            let mut operands = vec![id];
            operands.extend_from_slice(rest);
            self.add_global(op, operands);
            id
        })
    }

    /// The 32-bit float type.
    pub(crate) fn float_type(&mut self) -> Word {
        self.find_or_add_type(Op::TypeFloat, &[32])
    }

    /// A 32-bit integer type and whether it is signed; an unsigned one is added if the
    /// module declares neither.
    pub(crate) fn int_type(&mut self) -> (Word, bool) {
        let found = self.instructions[..self.globals_end()]
            .iter()
            .find_map(|i| match (i.op(), i.operands.as_slice()) {
                (Some(Op::TypeInt), [result, 32, signed]) => Some((*result, *signed == 1)),
                _ => None,
            });
        found.unwrap_or_else(|| (self.find_or_add_type(Op::TypeInt, &[32, 0]), false))
    }

    pub(crate) fn pointer_type(&mut self, storage: StorageClass, pointee: Word) -> Word {
        self.find_or_add_type(Op::TypePointer, &[storage as u32, pointee])
    }

    /// An `OpConstant` of type `ty` holding the bit pattern `value`, added if missing.
    pub(crate) fn constant(&mut self, ty: Word, value: u32) -> Word {
        let found = self.instructions[..self.globals_end()]
            .iter()
            .find_map(|i| match (i.op(), i.operands.as_slice()) {
                (Some(Op::Constant), [t, result, v]) if *t == ty && *v == value => Some(*result),
                _ => None,
            });
        found.unwrap_or_else(|| {
            let id = self.new_id();
            self.add_global(Op::Constant, vec![ty, id, value]);
            id
        })
    }

    pub(crate) fn float_constant(&mut self, value: f32) -> Word {
        let float = self.float_type();
        self.constant(float, value.to_bits())
    }

    /// Lists `variable` in the interface of entry point `name` unless already there.
    pub(crate) fn add_to_interface(&mut self, name: &str, variable: Word) {
        let encoded = encode_string(name);
        for instruction in &mut self.instructions {
            let matches = instruction.op() == Some(Op::EntryPoint)
                && instruction.operands.get(2..2 + encoded.len()) == Some(&encoded[..]);
            if matches && !instruction.operands[2 + encoded.len()..].contains(&variable) {
                instruction.operands.push(variable);
            }
        }
    }
}

/// A literal string as nul-terminated, zero-padded words.
pub(crate) fn encode_string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_string_pads_and_terminates() {
        assert_eq!(encode_string("main"), [0x6e69_616d, 0]);
        assert_eq!(encode_string("a"), [0x61]);
        assert_eq!(encode_string("SPV_KHR_multiview").len(), 5);
    }
}
