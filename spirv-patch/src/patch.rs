//! Injecting the per-eye shear into the last pre-raster stage.
//!
//! The injected code computes `x += eye_sign * separation * (w - convergence)` (or the
//! fixed-depth and infinity variants) and stores only `Position.x`; z and w are never
//! written, so depth stays identical between eyes and between a depth prepass and the main
//! pass. Where it runs depends on the stage:
//!
//! - vertex and tessellation evaluation: a new entry function calls the original one and
//!   then adjusts `Position`, covering piecewise writes and writes in helper functions;
//! - geometry: before every `OpEmitVertex`, since each emit freezes one vertex;
//! - mesh: after every whole-vector store, since invocations write each other's vertices
//!   and a post-pass would adjust them several times.

use std::fmt;

use s3d_shared::StereoPushConstants;
use spirv::{BuiltIn, Capability, Decoration, Op, StorageClass, Word};

use crate::{
    Analysis, Error, PositionForm, Result, ShaderModule, Stage, StoreTarget,
    edit::{encode_string, instruction},
    module::Instruction,
};

/// First `SpecId` used for the stereo constants unless the caller picks another; high
/// enough to stay clear of the small ids game shaders use.
pub const DEFAULT_SPEC_ID: u32 = 0x5333_0000;

/// Where separation, convergence and the eye sign come from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstantSource {
    /// The [`StereoPushConstants`] block at byte `offset` of the push-constant range; the
    /// shader's own push block is extended, or a new one declared.
    PushConstant { offset: u32 },
    /// Specialization constants `first_id`, `first_id + 1` and (for a constant eye)
    /// `first_id + 2`, defaulting to `defaults`.
    Specialization {
        first_id: u32,
        defaults: StereoPushConstants,
    },
}

/// Where the shader learns which eye it renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeSource {
    /// `eye_sign` from the constant source (single-eye rendering).
    Constant,
    /// `ViewIndex` under multiview: view 0 is the left eye.
    ViewIndex,
}

/// Depth the shear treats the geometry as being at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShearMode {
    /// Each vertex at its own depth `w`.
    Normal,
    /// Screen-space content placed at view depth `d` (profile `hud_depth = d`).
    FixedDepth(f32),
    /// Infinitely far (profile `sky`).
    Infinity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchOptions {
    pub constants: ConstantSource,
    pub eye: EyeSource,
    pub mode: ShearMode,
}

/// Where the shear code was placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insertion {
    Wrapper,
    BeforeEmit,
    AfterStore,
}

/// What a patch did, for logging.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchReport {
    pub entry: String,
    pub stage: Stage,
    pub insertion: Insertion,
    /// Places the shear code was inserted.
    pub sites: usize,
    pub options: PatchOptions,
    /// The shader already had a push block, which was extended.
    pub extended_push_block: bool,
    pub added_capabilities: Vec<Capability>,
}

impl fmt::Display for PatchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}): {:?} at {} site(s), {:?}, eye {:?}, mode {:?}",
            self.entry,
            self.stage,
            self.insertion,
            self.sites,
            self.options.constants,
            self.options.eye,
            self.options.mode,
        )?;
        if self.extended_push_block {
            write!(f, ", extended push block")?;
        }
        if !self.added_capabilities.is_empty() {
            write!(f, ", added {:?}", self.added_capabilities)?;
        }
        Ok(())
    }
}

/// How the injected code obtains one stereo value.
#[derive(Debug, Clone, Copy)]
enum Value {
    /// A constant or specialization constant id, usable directly as an operand.
    Id(Word),
    /// Member `index` (an integer constant id) of the push block `variable`.
    PushMember {
        variable: Word,
        pointer_type: Word,
        index: Word,
    },
    /// `float(ViewIndex) * 2 - 1`, read from the input `variable` of integer type `int`;
    /// `one` and `two` are the float constants the conversion needs.
    ViewIndexSign {
        variable: Word,
        int: Word,
        one: Word,
        two: Word,
    },
}

/// What `w` the shear uses.
#[derive(Debug, Clone, Copy)]
enum Depth {
    PerVertex,
    /// Float constants holding the fixed depth and 1.
    Fixed {
        depth: Word,
        one: Word,
    },
    Infinite,
}

/// Ids the shear code needs, declared once per patch.
#[derive(Debug, Clone, Copy)]
struct Context {
    float: Word,
    int_signed: bool,
    out_float_ptr: Word,
    /// Integer constants 0 and 3: the x and w components.
    x_index: Word,
    w_index: Word,
    /// Integer constant of the `Position` member when it lives in a block.
    member: Option<Word>,
    separation: Value,
    convergence: Value,
    eye: Value,
    depth: Depth,
}

impl ShaderModule {
    /// Returns a copy of this module whose entry point `entry` applies the shear.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotPreRaster`] for stages that never feed the rasteriser,
    /// [`Error::NoPosition`] when the entry writes no `Position`, and
    /// [`Error::Unsupported`] for layouts the patch cannot handle safely (mesh shaders
    /// writing components, more than one push block, an entry function shared with
    /// another entry point).
    pub fn patch(&self, entry: &str, options: &PatchOptions) -> Result<(Self, PatchReport)> {
        let before = self.analyze(entry)?;
        let insertion = match before.entry.stage {
            Stage::Vertex | Stage::TessEval => Insertion::Wrapper,
            Stage::Geometry => Insertion::BeforeEmit,
            Stage::Mesh => Insertion::AfterStore,
            stage => return Err(Error::NotPreRaster(stage)),
        };
        if before.position.is_none() {
            return Err(Error::NoPosition(entry.to_owned()));
        }

        let mut patched = self.clone();
        let mut report = PatchReport {
            entry: entry.to_owned(),
            stage: before.entry.stage,
            insertion,
            sites: 0,
            options: *options,
            extended_push_block: false,
            added_capabilities: Vec::new(),
        };
        let context = patched.declare(&before, options, &mut report)?;
        let after = patched.analyze(entry)?;
        report.sites = match insertion {
            Insertion::Wrapper => patched.insert_wrapper(&after, &context)?,
            Insertion::BeforeEmit => patched.insert_before_emits(&after, &context),
            Insertion::AfterStore => patched.insert_after_stores(&after, &context)?,
        };
        Ok((Self::parse(&patched.assemble())?, report))
    }

    /// Declares every type, constant and variable the shear code uses.
    fn declare(
        &mut self,
        analysis: &Analysis,
        options: &PatchOptions,
        report: &mut PatchReport,
    ) -> Result<Context> {
        let entry = &analysis.entry.name;
        let float = self.float_type();
        let (int, int_signed) = self.int_type();
        let out_float_ptr = self.pointer_type(StorageClass::Output, float);
        let x_index = self.constant(int, 0);
        let w_index = self.constant(int, 3);
        let member = match analysis.position.map(|p| p.form) {
            Some(PositionForm::BlockMember(member)) => Some(self.constant(int, member)),
            _ => None,
        };
        let depth = match options.mode {
            ShearMode::Normal => Depth::PerVertex,
            ShearMode::FixedDepth(depth) => Depth::Fixed {
                depth: self.float_constant(depth),
                one: self.float_constant(1.0),
            },
            ShearMode::Infinity => Depth::Infinite,
        };

        let view_index = match options.eye {
            EyeSource::Constant => None,
            EyeSource::ViewIndex => {
                if self.add_capability(Capability::MultiView) {
                    report.added_capabilities.push(Capability::MultiView);
                }
                if self.version() < (1, 3) {
                    self.add_extension("SPV_KHR_multiview");
                }
                let input_int_ptr = self.pointer_type(StorageClass::Input, int);
                let variable = self.new_id();
                self.add_global(
                    Op::Variable,
                    vec![input_int_ptr, variable, StorageClass::Input as u32],
                );
                self.add_annotation(
                    Op::Decorate,
                    vec![
                        variable,
                        Decoration::BuiltIn as u32,
                        BuiltIn::ViewIndex as u32,
                    ],
                );
                self.add_to_interface(entry, variable);
                Some(Value::ViewIndexSign {
                    variable,
                    int,
                    one: self.float_constant(1.0),
                    two: self.float_constant(2.0),
                })
            }
        };

        let (separation, convergence, constant_eye) = match options.constants {
            ConstantSource::PushConstant { offset } => {
                let (variable, first_member, extended) =
                    self.declare_push_block(entry, float, offset)?;
                report.extended_push_block = extended;
                let pointer_type = self.pointer_type(StorageClass::PushConstant, float);
                let mut member = |n: u32| Value::PushMember {
                    variable,
                    pointer_type,
                    index: self.constant(int, first_member + n),
                };
                (member(0), member(1), Some(member(2)))
            }
            ConstantSource::Specialization { first_id, defaults } => {
                let mut spec = |offset: u32, value: f32| {
                    let id = self.new_id();
                    self.add_global(Op::SpecConstant, vec![float, id, value.to_bits()]);
                    self.add_annotation(
                        Op::Decorate,
                        vec![id, Decoration::SpecId as u32, first_id + offset],
                    );
                    Value::Id(id)
                };
                let separation = spec(0, defaults.separation);
                let convergence = spec(1, defaults.convergence);
                let eye = view_index.is_none().then(|| spec(2, defaults.eye_sign));
                (separation, convergence, eye)
            }
        };
        let eye = view_index
            .or(constant_eye)
            .ok_or_else(|| Error::Unsupported("no eye source".into()))?;

        Ok(Context {
            float,
            int_signed,
            out_float_ptr,
            x_index,
            w_index,
            member,
            separation,
            convergence,
            eye,
            depth,
        })
    }

    /// Finds the shader's push block and appends the stereo members at `offset`, or
    /// declares a block of its own. Returns the variable, the index of the first stereo
    /// member and whether an existing block was extended.
    fn declare_push_block(
        &mut self,
        entry: &str,
        float: Word,
        offset: u32,
    ) -> Result<(Word, u32, bool)> {
        let push = StorageClass::PushConstant as u32;
        let variables: Vec<(Word, Word)> = self.instructions[..self.globals_end()]
            .iter()
            .filter_map(|i| match (i.op(), i.operands.as_slice()) {
                (Some(Op::Variable), [pointer, result, storage, ..]) if *storage == push => {
                    Some((*result, *pointer))
                }
                _ => None,
            })
            .collect();

        let (variable, structure, first, extended) = match variables.as_slice() {
            [] => {
                let structure = self.new_id();
                self.add_global(Op::TypeStruct, vec![structure, float, float, float]);
                self.add_annotation(Op::Decorate, vec![structure, Decoration::Block as u32]);
                let pointer = self.pointer_type(StorageClass::PushConstant, structure);
                let variable = self.new_id();
                self.add_global(Op::Variable, vec![pointer, variable, push]);
                (variable, structure, 0, false)
            }
            [(variable, pointer)] => {
                let structure = self
                    .instructions
                    .iter()
                    .find_map(|i| match (i.op(), i.operands.as_slice()) {
                        (Some(Op::TypePointer), [result, _, pointee]) if result == pointer => {
                            Some(*pointee)
                        }
                        _ => None,
                    })
                    .ok_or_else(|| Error::Unsupported("push block without pointer type".into()))?;
                let definition = self
                    .instructions
                    .iter_mut()
                    .find(|i| {
                        i.op() == Some(Op::TypeStruct) && i.operands.first() == Some(&structure)
                    })
                    .ok_or_else(|| Error::Unsupported("push block is not a struct".into()))?;
                let first = (definition.operands.len() - 1) as u32;
                definition.operands.extend([float, float, float]);
                (*variable, structure, first, true)
            }
            _ => {
                return Err(Error::Unsupported(
                    "more than one push-constant block".into(),
                ));
            }
        };

        let offsets = [
            StereoPushConstants::SEPARATION_OFFSET,
            StereoPushConstants::CONVERGENCE_OFFSET,
            StereoPushConstants::EYE_SIGN_OFFSET,
        ];
        for (n, member_offset) in (first..).zip(offsets) {
            self.add_annotation(
                Op::MemberDecorate,
                vec![
                    structure,
                    n,
                    Decoration::Offset as u32,
                    offset + member_offset,
                ],
            );
        }
        if self.version() >= (1, 4) {
            self.add_to_interface(entry, variable);
        }
        Ok((variable, first, extended))
    }

    /// Replaces the entry function with a wrapper that calls it, then shears `Position`.
    fn insert_wrapper(&mut self, analysis: &Analysis, context: &Context) -> Result<usize> {
        let original = analysis.entry.function;
        let callers = self
            .entry_points()
            .iter()
            .filter(|entry| entry.function == original)
            .count();
        if callers > 1 {
            return Err(Error::Unsupported(
                "entry function is shared by several entry points".into(),
            ));
        }
        let (void, function_type) = self
            .instructions
            .iter()
            .find_map(|i| match (i.op(), i.operands.as_slice()) {
                (Some(Op::Function), [result_type, id, _, ty]) if *id == original => {
                    Some((*result_type, *ty))
                }
                _ => None,
            })
            .ok_or_else(|| Error::Unsupported("entry function not found".into()))?;
        let position = analysis
            .position
            .ok_or_else(|| Error::NoPosition(analysis.entry.name.clone()))?;

        let wrapper = self.new_id();
        let label = self.new_id();
        let call = self.new_id();
        let mut body = vec![
            instruction(Op::Function, vec![void, wrapper, 0, function_type]),
            instruction(Op::Label, vec![label]),
            instruction(Op::FunctionCall, vec![void, call, original]),
        ];
        body.extend(self.shear_from_variable(context, position.variable));
        body.push(instruction(Op::Return, Vec::new()));
        body.push(instruction(Op::FunctionEnd, Vec::new()));
        self.instructions.extend(body);

        let name = encode_string(&analysis.entry.name);
        for instruction in &mut self.instructions {
            let op = instruction.op();
            let operands = &mut instruction.operands;
            let is_entry =
                op == Some(Op::EntryPoint) && operands.get(2..2 + name.len()) == Some(&name[..]);
            let is_mode = matches!(op, Some(Op::ExecutionMode | Op::ExecutionModeId))
                && operands.first() == Some(&original);
            if is_entry {
                operands[1] = wrapper;
            } else if is_mode {
                operands[0] = wrapper;
            }
        }
        Ok(1)
    }

    /// Shears `Position` before every vertex emit in a function the entry can reach.
    fn insert_before_emits(&mut self, analysis: &Analysis, context: &Context) -> usize {
        let Some(position) = analysis.position else {
            return 0;
        };
        let reachable = self.reachable_functions(analysis.entry.function);
        let sites: Vec<usize> = self
            .function_instructions()
            .filter(|(_, function, i)| {
                reachable.contains(function)
                    && matches!(i.op(), Some(Op::EmitVertex | Op::EmitStreamVertex))
            })
            .map(|(index, _, _)| index)
            .collect();
        for &index in sites.iter().rev() {
            let code = self.shear_from_variable(context, position.variable);
            self.instructions.splice(index..index, code);
        }
        sites.len()
    }

    /// Shears each whole-vector `Position` store right after it.
    fn insert_after_stores(&mut self, analysis: &Analysis, context: &Context) -> Result<usize> {
        if analysis.escapes
            || analysis
                .stores
                .iter()
                .any(|store| store.target != StoreTarget::Whole)
        {
            return Err(Error::Unsupported(
                "mesh shader writes Position other than as a whole vector".into(),
            ));
        }
        for store in analysis.stores.iter().rev() {
            let [pointer, value, ..] = self.instructions[store.instruction].operands[..] else {
                return Err(Error::Unsupported("malformed OpStore".into()));
            };
            let code = self.shear_from_value(context, pointer, value);
            let at = store.instruction + 1;
            self.instructions.splice(at..at, code);
        }
        Ok(analysis.stores.len())
    }

    /// Code reading x and w through `variable` and storing the sheared x.
    fn shear_from_variable(&mut self, context: &Context, variable: Word) -> Vec<Instruction> {
        let mut code = Vec::new();
        let (x_pointer, x) = self.load_component(context, variable, context.x_index, &mut code);
        let (_, w) = self.load_component(context, variable, context.w_index, &mut code);
        self.shear_x(context, x_pointer, x, w, &mut code);
        code
    }

    /// Code loading one component of `Position` through `variable`; returns the
    /// component's pointer and value.
    fn load_component(
        &mut self,
        context: &Context,
        variable: Word,
        component: Word,
        code: &mut Vec<Instruction>,
    ) -> (Word, Word) {
        let pointer = self.new_id();
        let value = self.new_id();
        let mut chain = vec![context.out_float_ptr, pointer, variable];
        chain.extend(context.member);
        chain.push(component);
        code.push(instruction(Op::AccessChain, chain));
        code.push(instruction(Op::Load, vec![context.float, value, pointer]));
        (pointer, value)
    }

    /// Code taking x and w from the just-stored `value` and storing the sheared x through
    /// `pointer`.
    fn shear_from_value(
        &mut self,
        context: &Context,
        pointer: Word,
        value: Word,
    ) -> Vec<Instruction> {
        let x = self.new_id();
        let w = self.new_id();
        let x_pointer = self.new_id();
        let mut code = vec![
            instruction(Op::CompositeExtract, vec![context.float, x, value, 0]),
            instruction(Op::CompositeExtract, vec![context.float, w, value, 3]),
            instruction(
                Op::AccessChain,
                vec![context.out_float_ptr, x_pointer, pointer, context.x_index],
            ),
        ];
        self.shear_x(context, x_pointer, x, w, &mut code);
        code
    }

    /// Appends `*x_pointer = x + eye * separation * depth_term` and returns nothing; the
    /// depth term is `w - convergence`, `w * (1 - convergence / d)` or `w`.
    fn shear_x(
        &mut self,
        context: &Context,
        x_pointer: Word,
        x: Word,
        w: Word,
        code: &mut Vec<Instruction>,
    ) {
        let separation = self.read(context, context.separation, code);
        let convergence = self.read(context, context.convergence, code);
        let eye = self.read(context, context.eye, code);
        let depth_term = match context.depth {
            Depth::PerVertex => self.binary(context, Op::FSub, w, convergence, code),
            Depth::Fixed { depth, one } => {
                let ratio = self.binary(context, Op::FDiv, convergence, depth, code);
                let keep = self.binary(context, Op::FSub, one, ratio, code);
                self.binary(context, Op::FMul, w, keep, code)
            }
            Depth::Infinite => w,
        };
        let scaled = self.binary(context, Op::FMul, separation, depth_term, code);
        let delta = self.binary(context, Op::FMul, eye, scaled, code);
        let sheared = self.binary(context, Op::FAdd, x, delta, code);
        code.push(instruction(Op::Store, vec![x_pointer, sheared]));
    }

    /// Appends a float binary operation; returns its result id.
    fn binary(
        &mut self,
        context: &Context,
        op: Op,
        a: Word,
        b: Word,
        code: &mut Vec<Instruction>,
    ) -> Word {
        let result = self.new_id();
        code.push(instruction(op, vec![context.float, result, a, b]));
        result
    }

    /// Appends code producing one stereo value; returns its id.
    fn read(&mut self, context: &Context, value: Value, code: &mut Vec<Instruction>) -> Word {
        match value {
            Value::Id(id) => id,
            Value::PushMember {
                variable,
                pointer_type,
                index,
            } => {
                let pointer = self.new_id();
                let result = self.new_id();
                code.push(instruction(
                    Op::AccessChain,
                    vec![pointer_type, pointer, variable, index],
                ));
                code.push(instruction(Op::Load, vec![context.float, result, pointer]));
                result
            }
            Value::ViewIndexSign {
                variable,
                int,
                one,
                two,
            } => {
                let index = self.new_id();
                let as_float = self.new_id();
                let convert = if context.int_signed {
                    Op::ConvertSToF
                } else {
                    Op::ConvertUToF
                };
                code.push(instruction(Op::Load, vec![int, index, variable]));
                code.push(instruction(convert, vec![context.float, as_float, index]));
                let doubled = self.binary(context, Op::FMul, as_float, two, code);
                self.binary(context, Op::FSub, doubled, one, code)
            }
        }
    }
}
