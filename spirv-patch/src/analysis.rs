//! Finding the `Position` output of an entry point and every store to it.
//!
//! `Position` appears in two forms: a variable decorated `BuiltIn Position` (what DXC
//! emits for `SV_Position`) or a member of a block whose member is decorated that way
//! (GLSL's `gl_PerVertex`). Either may be an array per vertex (mesh shaders). Stores are
//! found by following access chains from that variable in every function the entry point
//! can reach. Operand positions below follow the SPIR-V spec's instruction layouts.

use std::collections::{HashMap, HashSet};

use spirv::{BuiltIn, Decoration, Op, StorageClass, Word};

use crate::{EntryPoint, Result, ShaderModule, module::Instruction};

/// How `Position` is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionForm {
    /// The variable itself is decorated `BuiltIn Position`.
    Variable,
    /// Member `n` of the variable's block type is decorated `BuiltIn Position`.
    BlockMember(u32),
}

/// The output variable that carries `Position` for one entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionOutput {
    pub variable: Word,
    pub form: PositionForm,
    /// One element per vertex, as in mesh shaders.
    pub arrayed: bool,
}

/// What part of `Position` a store writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreTarget {
    /// All four components.
    Whole,
    /// One component with a constant index (0 = x).
    Component(u32),
    /// One component chosen at run time.
    DynamicComponent,
    /// The whole enclosing block (or the whole per-vertex array).
    Block,
    /// A path the analysis cannot resolve, such as a run-time block member index.
    Unknown,
}

/// One instruction writing `Position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionStore {
    /// Result id of the function containing the store.
    pub function: Word,
    /// Index into [`ShaderModule::instructions`].
    pub instruction: usize,
    pub target: StoreTarget,
}

/// Everything the patcher needs to know about one entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analysis {
    pub entry: EntryPoint,
    /// `None` for stages without a `Position` output.
    pub position: Option<PositionOutput>,
    pub stores: Vec<PositionStore>,
    /// A pointer into `Position` is passed to a function call, so stores made through it
    /// are not listed and patching must not rely on `stores` alone.
    pub escapes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Index {
    Constant(u32),
    Dynamic,
}

impl ShaderModule {
    /// Analyses the entry point called `name`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::NoEntryPoint`] when the module has no such entry point.
    pub fn analyze(&self, name: &str) -> Result<Analysis> {
        let entry = self.entry_point(name)?;
        let position = self.find_position(&entry);
        let (stores, escapes) = match position {
            Some(output) => self.find_stores(&entry, output),
            None => (Vec::new(), false),
        };
        Ok(Analysis {
            entry,
            position,
            stores,
            escapes,
        })
    }

    fn find_position(&self, entry: &EntryPoint) -> Option<PositionOutput> {
        let builtin = Decoration::BuiltIn as u32;
        let position = BuiltIn::Position as u32;
        let mut decorated = HashSet::new();
        let mut members = HashMap::new();
        let mut pointers = HashMap::new();
        let mut arrays = HashMap::new();
        for instruction in &self.instructions {
            let operands = instruction.operands.as_slice();
            match (instruction.op(), operands) {
                (Some(Op::Decorate), [target, decoration, value, ..])
                    if *decoration == builtin && *value == position =>
                {
                    decorated.insert(*target);
                }
                (Some(Op::MemberDecorate), [structure, member, decoration, value, ..])
                    if *decoration == builtin && *value == position =>
                {
                    members.insert(*structure, *member);
                }
                (Some(Op::TypePointer), [result, _storage, pointee]) => {
                    pointers.insert(*result, *pointee);
                }
                (Some(Op::TypeArray), [result, element, _])
                | (Some(Op::TypeRuntimeArray), [result, element]) => {
                    arrays.insert(*result, *element);
                }
                _ => {}
            }
        }

        let output = StorageClass::Output as u32;
        self.instructions.iter().find_map(|instruction| {
            let [pointer_type, id, storage, ..] = instruction.operands.as_slice() else {
                return None;
            };
            if instruction.op() != Some(Op::Variable)
                || *storage != output
                || !entry.interface.contains(id)
            {
                return None;
            }
            let pointee = *pointers.get(pointer_type)?;
            let (element, arrayed) = match arrays.get(&pointee) {
                Some(element) => (*element, true),
                None => (pointee, false),
            };
            let form = if decorated.contains(id) {
                PositionForm::Variable
            } else {
                PositionForm::BlockMember(*members.get(&element)?)
            };
            Some(PositionOutput {
                variable: *id,
                form,
                arrayed,
            })
        })
    }

    fn find_stores(
        &self,
        entry: &EntryPoint,
        output: PositionOutput,
    ) -> (Vec<PositionStore>, bool) {
        let constants: HashMap<Word, u32> = self
            .instructions
            .iter()
            .filter_map(
                |instruction| match (instruction.op(), &instruction.operands[..]) {
                    (Some(Op::Constant), [_ty, result, value]) => Some((*result, *value)),
                    _ => None,
                },
            )
            .collect();
        let reachable = self.reachable_functions(entry.function);

        let mut paths: HashMap<Word, Vec<Index>> = HashMap::from([(output.variable, Vec::new())]);
        let mut stores = Vec::new();
        let mut escapes = false;
        for (index, function, instruction) in self.function_instructions() {
            if !reachable.contains(&function) {
                continue;
            }
            let operands = instruction.operands.as_slice();
            match (instruction.op(), operands) {
                (
                    Some(Op::AccessChain | Op::InBoundsAccessChain),
                    [_ty, result, base, indices @ ..],
                ) => {
                    if let Some(base_path) = paths.get(base) {
                        let mut path = base_path.clone();
                        path.extend(indices.iter().map(|id| match constants.get(id) {
                            Some(value) => Index::Constant(*value),
                            None => Index::Dynamic,
                        }));
                        paths.insert(*result, path);
                    }
                }
                (Some(Op::Store | Op::CopyMemory), [pointer, ..]) => {
                    if let Some(path) = paths.get(pointer)
                        && let Some(target) = classify(output, path)
                    {
                        stores.push(PositionStore {
                            function,
                            instruction: index,
                            target,
                        });
                    }
                }
                (Some(Op::FunctionCall), [_ty, _result, _callee, arguments @ ..]) => {
                    escapes |= arguments.iter().any(|id| paths.contains_key(id));
                }
                (Some(Op::PtrAccessChain), [_ty, _result, base, ..]) => {
                    escapes |= paths.contains_key(base);
                }
                _ => {}
            }
        }
        (stores, escapes)
    }

    /// Every instruction inside a function body, with its index and the function's id.
    pub(crate) fn function_instructions(
        &self,
    ) -> impl Iterator<Item = (usize, Word, &Instruction)> {
        let mut current = None;
        self.instructions
            .iter()
            .enumerate()
            .filter_map(move |(index, instruction)| {
                match (instruction.op(), instruction.operands.as_slice()) {
                    (Some(Op::Function), [_ty, result, ..]) => current = Some(*result),
                    (Some(Op::FunctionEnd), _) => current = None,
                    _ => return current.map(|function| (index, function, instruction)),
                }
                None
            })
    }

    /// Functions the entry function can reach through `OpFunctionCall`, itself included.
    pub(crate) fn reachable_functions(&self, entry: Word) -> HashSet<Word> {
        let mut callees: HashMap<Word, Vec<Word>> = HashMap::new();
        for (_, function, instruction) in self.function_instructions() {
            if let (Some(Op::FunctionCall), [_ty, _result, callee, ..]) =
                (instruction.op(), instruction.operands.as_slice())
            {
                callees.entry(function).or_default().push(*callee);
            }
        }
        let mut reachable = HashSet::from([entry]);
        let mut pending = vec![entry];
        while let Some(function) = pending.pop() {
            for callee in callees.get(&function).into_iter().flatten() {
                if reachable.insert(*callee) {
                    pending.push(*callee);
                }
            }
        }
        reachable
    }
}

/// What a store through `path` (indices from the variable) writes, or `None` when it
/// writes a block member other than `Position`.
fn classify(output: PositionOutput, path: &[Index]) -> Option<StoreTarget> {
    let path = if output.arrayed {
        match path.split_first() {
            Some((_, rest)) => rest,
            None => return Some(StoreTarget::Block),
        }
    } else {
        path
    };
    Some(match (output.form, path) {
        (PositionForm::Variable, []) => StoreTarget::Whole,
        (PositionForm::Variable, [Index::Constant(component)]) => {
            StoreTarget::Component(*component)
        }
        (PositionForm::Variable, [Index::Dynamic]) => StoreTarget::DynamicComponent,
        (PositionForm::BlockMember(_), []) => StoreTarget::Block,
        (PositionForm::BlockMember(member), [Index::Constant(index), rest @ ..]) => {
            if *index != member {
                return None;
            }
            match rest {
                [] => StoreTarget::Whole,
                [Index::Constant(component)] => StoreTarget::Component(*component),
                [Index::Dynamic] => StoreTarget::DynamicComponent,
                _ => StoreTarget::Unknown,
            }
        }
        _ => StoreTarget::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: PositionOutput = PositionOutput {
        variable: 1,
        form: PositionForm::BlockMember(0),
        arrayed: false,
    };

    #[test]
    fn test_classify_block_member_paths() {
        use Index::*;
        assert_eq!(classify(BLOCK, &[Constant(0)]), Some(StoreTarget::Whole));
        assert_eq!(
            classify(BLOCK, &[Constant(0), Constant(2)]),
            Some(StoreTarget::Component(2))
        );
        assert_eq!(classify(BLOCK, &[Constant(1)]), None);
        assert_eq!(classify(BLOCK, &[]), Some(StoreTarget::Block));
        assert_eq!(classify(BLOCK, &[Dynamic]), Some(StoreTarget::Unknown));
    }

    #[test]
    fn test_classify_arrayed_skips_vertex_index() {
        let arrayed = PositionOutput {
            arrayed: true,
            ..BLOCK
        };
        assert_eq!(
            classify(arrayed, &[Index::Dynamic, Index::Constant(0)]),
            Some(StoreTarget::Whole)
        );
        assert_eq!(classify(arrayed, &[]), Some(StoreTarget::Block));
    }
}
