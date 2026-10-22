//! Descriptor set layouts, pipeline layouts and every graphics pipeline of the test scene.

use std::io::Cursor;

use anyhow::Result;
use ash::{Device, vk};

use super::passes::Pass;
use crate::scene::{Material, TONEMAP_EXPOSURE};

macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

/// Position, normal, RGBA colour and UV: 12 `f32`.
pub const VERTEX_STRIDE: u32 = 48;

/// One `mat4` model matrix for the vertex stage. The layer appends its own range after
/// this, so it must stay the first and only application range.
pub const PUSH_CONSTANT_SIZE: u32 = 64;

/// Which vertex attributes a pipeline reads from the shared vertex buffer.
#[derive(Clone, Copy)]
enum Input {
    Mesh,
    Position,
    PositionColor,
    None,
}

#[derive(Clone, Copy)]
enum Blend {
    Opaque,
    Alpha,
}

#[derive(Clone, Copy)]
enum Depth {
    Off,
    Write,
    ReadOnly(vk::CompareOp),
}

struct Desc<'a> {
    vertex: &'a [u8],
    fragment: Option<&'a [u8]>,
    /// Fragment specialization constants; entry `i` sets `constant_id = i`.
    fragment_constants: &'a [f32],
    layout: vk::PipelineLayout,
    input: Input,
    blend: Blend,
    depth: Depth,
    depth_bias: bool,
    pass: &'a Pass,
}

pub struct Pipelines {
    pub frame_set_layout: vk::DescriptorSetLayout,
    pub post_set_layout: vk::DescriptorSetLayout,
    /// Frame set plus the model push constant; used by every mesh, sky and HUD pipeline.
    pub mesh_layout: vk::PipelineLayout,
    pub post_layout: vk::PipelineLayout,
    pub unlit: vk::Pipeline,
    pub pattern: vk::Pipeline,
    pub lit: vk::Pipeline,
    pub reflective: vk::Pipeline,
    pub transparent: vk::Pipeline,
    pub sky: vk::Pipeline,
    pub shadow: vk::Pipeline,
    pub hud: vk::Pipeline,
    pub tonemap: Option<vk::Pipeline>,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        shadow_pass: &Pass,
        main_pass: &Pass,
        post_pass: Option<&Pass>,
        hud_pass: &Pass,
    ) -> Result<Self> {
        let frame_set_layout = create_set_layout(
            device,
            &[
                (
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        )?;
        let post_set_layout = create_set_layout(
            device,
            &[(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )],
        )?;

        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(PUSH_CONSTANT_SIZE)];
        let frame_sets = [frame_set_layout];
        let mesh_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&frame_sets)
                    .push_constant_ranges(&push_constants),
                None,
            )?
        };
        let post_sets = [post_set_layout];
        let post_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&post_sets),
                None,
            )?
        };

        let mesh = |fragment: &'static [u8], blend, depth| Desc {
            vertex: spv!("mesh.vert"),
            fragment: Some(fragment),
            fragment_constants: &[],
            layout: mesh_layout,
            input: Input::Mesh,
            blend,
            depth,
            depth_bias: false,
            pass: main_pass,
        };
        let unlit = create(
            device,
            &mesh(spv!("unlit.frag"), Blend::Opaque, Depth::Write),
        )?;
        let pattern = create(
            device,
            &mesh(spv!("pattern.frag"), Blend::Opaque, Depth::Write),
        )?;
        let lit = create(device, &mesh(spv!("lit.frag"), Blend::Opaque, Depth::Write))?;
        let reflective = create(
            device,
            &mesh(spv!("reflect.frag"), Blend::Opaque, Depth::Write),
        )?;
        let transparent = create(
            device,
            &mesh(
                spv!("unlit.frag"),
                Blend::Alpha,
                Depth::ReadOnly(vk::CompareOp::LESS),
            ),
        )?;
        let sky = create(
            device,
            &Desc {
                vertex: spv!("sky.vert"),
                fragment: Some(spv!("sky.frag")),
                input: Input::Position,
                depth: Depth::ReadOnly(vk::CompareOp::LESS_OR_EQUAL),
                ..mesh(spv!("unlit.frag"), Blend::Opaque, Depth::Off)
            },
        )?;
        let shadow = create(
            device,
            &Desc {
                vertex: spv!("shadow.vert"),
                fragment: None,
                fragment_constants: &[],
                layout: mesh_layout,
                input: Input::Position,
                blend: Blend::Opaque,
                depth: Depth::Write,
                depth_bias: true,
                pass: shadow_pass,
            },
        )?;
        let hud = create(
            device,
            &Desc {
                vertex: spv!("hud.vert"),
                fragment: Some(spv!("hud.frag")),
                fragment_constants: &[],
                layout: mesh_layout,
                input: Input::PositionColor,
                blend: Blend::Alpha,
                depth: Depth::Off,
                depth_bias: false,
                pass: hud_pass,
            },
        )?;
        let tonemap = post_pass
            .map(|pass| {
                create(
                    device,
                    &Desc {
                        vertex: spv!("fullscreen.vert"),
                        fragment: Some(spv!("tonemap.frag")),
                        fragment_constants: &[TONEMAP_EXPOSURE],
                        layout: post_layout,
                        input: Input::None,
                        blend: Blend::Opaque,
                        depth: Depth::Off,
                        depth_bias: false,
                        pass,
                    },
                )
            })
            .transpose()?;

        Ok(Self {
            frame_set_layout,
            post_set_layout,
            mesh_layout,
            post_layout,
            unlit,
            pattern,
            lit,
            reflective,
            transparent,
            sky,
            shadow,
            hud,
            tonemap,
        })
    }

    pub fn for_material(&self, material: Material) -> vk::Pipeline {
        match material {
            Material::Unlit => self.unlit,
            Material::Pattern => self.pattern,
            Material::Lit => self.lit,
            Material::Reflective => self.reflective,
            Material::Transparent => self.transparent,
        }
    }

    pub fn destroy(&self, device: &Device) {
        let pipelines = [
            self.unlit,
            self.pattern,
            self.lit,
            self.reflective,
            self.transparent,
            self.sky,
            self.shadow,
            self.hud,
        ];
        unsafe {
            for pipeline in pipelines.into_iter().chain(self.tonemap) {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.mesh_layout, None);
            device.destroy_pipeline_layout(self.post_layout, None);
            device.destroy_descriptor_set_layout(self.frame_set_layout, None);
            device.destroy_descriptor_set_layout(self.post_set_layout, None);
        }
    }
}

/// One descriptor per binding, numbered from 0 in the order given.
fn create_set_layout(
    device: &Device,
    bindings: &[(vk::DescriptorType, vk::ShaderStageFlags)],
) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(index, &(kind, stages))| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(index as u32)
                .descriptor_type(kind)
                .descriptor_count(1)
                .stage_flags(stages)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
}

fn create(device: &Device, desc: &Desc<'_>) -> Result<vk::Pipeline> {
    let vertex_module = create_shader_module(device, desc.vertex)?;
    let fragment_module = desc
        .fragment
        .map(|spv| create_shader_module(device, spv))
        .transpose()?;
    let mut stages = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(c"main"),
    ];
    let constant_entries: Vec<_> = (0..desc.fragment_constants.len() as u32)
        .map(|id| {
            vk::SpecializationMapEntry::default()
                .constant_id(id)
                .offset(id * 4)
                .size(4)
        })
        .collect();
    let constant_data: Vec<u8> = desc
        .fragment_constants
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let specialization = vk::SpecializationInfo::default()
        .map_entries(&constant_entries)
        .data(&constant_data);
    if let Some(module) = fragment_module {
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(c"main");
        if !constant_entries.is_empty() {
            stage = stage.specialization_info(&specialization);
        }
        stages.push(stage);
    }

    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(VERTEX_STRIDE)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attribute = |location: u32, format: vk::Format, offset: u32| {
        vk::VertexInputAttributeDescription::default()
            .location(location)
            .format(format)
            .offset(offset)
    };
    let position = attribute(0, vk::Format::R32G32B32_SFLOAT, 0);
    let normal = attribute(1, vk::Format::R32G32B32_SFLOAT, 12);
    let color = attribute(2, vk::Format::R32G32B32A32_SFLOAT, 24);
    let uv = attribute(3, vk::Format::R32G32_SFLOAT, 40);
    let attributes: Vec<_> = match desc.input {
        Input::Mesh => vec![position, normal, color, uv],
        Input::Position => vec![position],
        Input::PositionColor => vec![position, color],
        Input::None => Vec::new(),
    };
    let mut vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attributes);
    if !attributes.is_empty() {
        vertex_input = vertex_input.vertex_binding_descriptions(&bindings);
    }

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .depth_bias_enable(desc.depth_bias)
        .depth_bias_constant_factor(1.25)
        .depth_bias_slope_factor(1.75)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let (test, write, compare) = match desc.depth {
        Depth::Off => (false, false, vk::CompareOp::ALWAYS),
        Depth::Write => (true, true, vk::CompareOp::LESS),
        Depth::ReadOnly(compare) => (true, false, compare),
    };
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(test)
        .depth_write_enable(write)
        .depth_compare_op(compare);

    let blend_attachment = match desc.blend {
        Blend::Opaque => vk::PipelineColorBlendAttachmentState::default(),
        Blend::Alpha => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD),
    }
    .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments: Vec<_> = desc
        .pass
        .color_format()
        .map(|_| blend_attachment)
        .into_iter()
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats: Vec<_> = desc.pass.color_format().into_iter().collect();
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(desc.pass.depth_format().unwrap_or(vk::Format::UNDEFINED));
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(desc.layout);
    info = match desc.pass.render_pass() {
        Some(render_pass) => info.render_pass(render_pass).subpass(0),
        None => info.push_next(&mut rendering),
    };

    let result =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe {
        device.destroy_shader_module(vertex_module, None);
        if let Some(module) = fragment_module {
            device.destroy_shader_module(module, None);
        }
    }
    let pipelines = result.map_err(|(_, error)| error)?;
    Ok(pipelines[0])
}

fn create_shader_module(device: &Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut Cursor::new(spv))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&info, None)? })
}
