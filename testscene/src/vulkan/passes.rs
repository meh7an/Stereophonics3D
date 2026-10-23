//! Render passes shared by both rendering modes.
//!
//! A `Pass` describes its attachments once: load and store behaviour plus the layouts
//! before and after. Pass mode bakes that into a `VkRenderPass`; dynamic mode issues the
//! same transitions as barriers around `vkCmdBeginRendering`, so both paths leave every
//! image in the same state. Synchronisation is deliberately broad (all commands, all
//! memory): the test scene needs correctness, not throughput.

use anyhow::Result;
use ash::{Device, vk};

use super::{RenderingMode, memory::subresource_range};

const COLOR_LAYOUT: vk::ImageLayout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
const DEPTH_LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

/// What happens to an attachment's contents when the pass begins.
#[derive(Clone, Copy)]
pub enum Load {
    Clear(vk::ClearValue),
    Keep,
    DontCare,
}

#[derive(Clone, Copy)]
pub struct AttachmentSpec {
    pub format: vk::Format,
    pub load: Load,
    pub store: bool,
    pub initial: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

/// The images one framebuffer of a pass renders into.
#[derive(Clone, Copy)]
pub struct Target {
    pub color: Option<(vk::Image, vk::ImageView)>,
    pub depth: Option<(vk::Image, vk::ImageView)>,
}

pub struct Pass {
    color: Option<AttachmentSpec>,
    depth: Option<AttachmentSpec>,
    extent: vk::Extent2D,
    render_pass: Option<vk::RenderPass>,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Pass {
    /// Creates the pass; in pass mode also one framebuffer per entry of `targets`.
    pub fn new(
        device: &Device,
        mode: RenderingMode,
        color: Option<AttachmentSpec>,
        depth: Option<AttachmentSpec>,
        extent: vk::Extent2D,
        targets: &[Target],
    ) -> Result<Self> {
        let render_pass = match mode {
            RenderingMode::Pass => Some(create_render_pass(device, color, depth)?),
            RenderingMode::Dynamic => None,
        };
        let framebuffers = match render_pass {
            Some(render_pass) => targets
                .iter()
                .map(|target| create_framebuffer(device, render_pass, target, extent))
                .collect::<Result<_>>()?,
            None => Vec::new(),
        };
        Ok(Self {
            color,
            depth,
            extent,
            render_pass,
            framebuffers,
        })
    }

    pub fn render_pass(&self) -> Option<vk::RenderPass> {
        self.render_pass
    }

    pub fn color_format(&self) -> Option<vk::Format> {
        self.color.map(|spec| spec.format)
    }

    pub fn depth_format(&self) -> Option<vk::Format> {
        self.depth.map(|spec| spec.format)
    }

    /// Opens the pass on `target` (framebuffer `index` in pass mode) and sets a viewport
    /// and scissor covering the whole extent.
    pub fn begin(&self, device: &Device, cmd: vk::CommandBuffer, index: usize, target: &Target) {
        let area = vk::Rect2D::default().extent(self.extent);
        match self.render_pass {
            Some(render_pass) => {
                let clears: Vec<vk::ClearValue> = [self.color, self.depth]
                    .into_iter()
                    .flatten()
                    .map(|spec| match spec.load {
                        Load::Clear(value) => value,
                        Load::Keep | Load::DontCare => vk::ClearValue::default(),
                    })
                    .collect();
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(self.framebuffers[index])
                    .render_area(area)
                    .clear_values(&clears);
                unsafe { device.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE) };
            }
            None => {
                self.transition(device, cmd, target, |spec, attachment| {
                    (spec.initial, attachment)
                });
                let color: Vec<_> = attachment_info(self.color, target.color, COLOR_LAYOUT)
                    .into_iter()
                    .collect();
                let depth = attachment_info(self.depth, target.depth, DEPTH_LAYOUT);
                let mut rendering = vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&color);
                if let Some(depth) = depth.as_ref() {
                    rendering = rendering.depth_attachment(depth);
                }
                unsafe { device.cmd_begin_rendering(cmd, &rendering) };
            }
        }

        let viewport = vk::Viewport::default()
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .max_depth(1.0);
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[area]);
        }
    }

    /// Closes the pass, leaving every attachment in its final layout.
    pub fn end(&self, device: &Device, cmd: vk::CommandBuffer, target: &Target) {
        match self.render_pass {
            Some(_) => unsafe { device.cmd_end_render_pass(cmd) },
            None => {
                unsafe { device.cmd_end_rendering(cmd) };
                self.transition(device, cmd, target, |spec, attachment| {
                    (attachment, spec.final_layout)
                });
            }
        }
    }

    /// Points the pass at new targets of a new extent, recreating the framebuffers in pass
    /// mode. The render pass stays, so pipelines built against it remain valid.
    pub fn retarget(
        &mut self,
        device: &Device,
        extent: vk::Extent2D,
        targets: &[Target],
    ) -> Result<()> {
        self.destroy_framebuffers(device);
        self.extent = extent;
        if let Some(render_pass) = self.render_pass {
            for target in targets {
                let framebuffer = create_framebuffer(device, render_pass, target, extent)?;
                self.framebuffers.push(framebuffer);
            }
        }
        Ok(())
    }

    fn destroy_framebuffers(&mut self, device: &Device) {
        for framebuffer in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &framebuffer in &self.framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }
            if let Some(render_pass) = self.render_pass {
                device.destroy_render_pass(render_pass, None);
            }
        }
    }

    /// Issues one barrier per attachment; `layouts` maps the spec and the attachment
    /// layout to the (old, new) pair.
    fn transition(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        target: &Target,
        layouts: impl Fn(&AttachmentSpec, vk::ImageLayout) -> (vk::ImageLayout, vk::ImageLayout),
    ) {
        let attachments = [
            (
                self.color,
                target.color,
                COLOR_LAYOUT,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                self.depth,
                target.depth,
                DEPTH_LAYOUT,
                vk::ImageAspectFlags::DEPTH,
            ),
        ];
        for (spec, image, attachment_layout, aspect) in attachments {
            if let (Some(spec), Some((image, _))) = (spec, image) {
                let (old, new) = layouts(&spec, attachment_layout);
                layout_barrier(device, cmd, image, aspect, old, new);
            }
        }
    }
}

/// Full barrier with a layout transition, covering every stage and memory access.
pub fn layout_barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(access)
        .dst_access_mask(access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range(aspect));
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn attachment_info(
    spec: Option<AttachmentSpec>,
    image: Option<(vk::Image, vk::ImageView)>,
    layout: vk::ImageLayout,
) -> Option<vk::RenderingAttachmentInfo<'static>> {
    let (spec, (_, view)) = (spec?, image?);
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(load_op(spec.load))
        .store_op(store_op(spec.store));
    if let Load::Clear(value) = spec.load {
        info = info.clear_value(value);
    }
    Some(info)
}

fn load_op(load: Load) -> vk::AttachmentLoadOp {
    match load {
        Load::Clear(_) => vk::AttachmentLoadOp::CLEAR,
        Load::Keep => vk::AttachmentLoadOp::LOAD,
        Load::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

fn store_op(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}

fn create_render_pass(
    device: &Device,
    color: Option<AttachmentSpec>,
    depth: Option<AttachmentSpec>,
) -> Result<vk::RenderPass> {
    let description = |spec: AttachmentSpec| {
        vk::AttachmentDescription::default()
            .format(spec.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(load_op(spec.load))
            .store_op(store_op(spec.store))
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(spec.initial)
            .final_layout(spec.final_layout)
    };
    let attachments: Vec<_> = [color, depth]
        .into_iter()
        .flatten()
        .map(description)
        .collect();
    let color_refs: Vec<_> = color
        .map(|_| {
            vk::AttachmentReference::default()
                .attachment(0)
                .layout(COLOR_LAYOUT)
        })
        .into_iter()
        .collect();
    let depth_ref = depth.map(|_| {
        vk::AttachmentReference::default()
            .attachment(color_refs.len() as u32)
            .layout(DEPTH_LAYOUT)
    });
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    if let Some(depth_ref) = depth_ref.as_ref() {
        subpass = subpass.depth_stencil_attachment(depth_ref);
    }

    let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
            .src_access_mask(access)
            .dst_access_mask(access),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
            .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .src_access_mask(access)
            .dst_access_mask(access),
    ];
    let subpasses = [subpass];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    Ok(unsafe { device.create_render_pass(&info, None)? })
}

fn create_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    target: &Target,
    extent: vk::Extent2D,
) -> Result<vk::Framebuffer> {
    let views: Vec<_> = [target.color, target.depth]
        .into_iter()
        .flatten()
        .map(|(_, view)| view)
        .collect();
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&views)
        .width(extent.width)
        .height(extent.height)
        .layers(1);
    Ok(unsafe { device.create_framebuffer(&info, None)? })
}
