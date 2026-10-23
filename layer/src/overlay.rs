//! The readout's text pass, drawn inside the compositor's render pass after the image.
//!
//! The composite command buffers are recorded once, so the pass draws indirectly: each
//! image has a host-visible buffer holding a `VkDrawIndirectCommand` followed by the
//! cells, and the CPU rewrites it (instance count 0 when hidden) before a composite of
//! that image is submitted. The font sits in a uniform buffer written once.

use std::sync::{Mutex, PoisonError};

use ash::vk;

use crate::{
    compositor::{CompositorError, PipelineSpec, create_pipeline, memory_type},
    readout::{CELL_BYTES, Cell, FONT_BYTES, MAX_CELLS, font_bytes, scale},
    swapchain::DeviceContext,
};

const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text.frag.spv"));
const COMMAND_BYTES: usize = size_of::<vk::DrawIndirectCommand>();
const OVERLAY_BYTES: usize = COMMAND_BYTES + MAX_CELLS * CELL_BYTES;
/// `vec2 extent; float cell;` in `text.vert`.
const PUSH_BYTES: u32 = 12;

/// A persistently mapped host-visible, host-coherent buffer.
#[derive(Debug)]
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Address of the mapping, kept as an integer so the compositor stays `Send`.
    mapped: usize,
}

impl HostBuffer {
    /// # Safety
    ///
    /// `device` must be live and `context` its physical-device facts.
    unsafe fn new(
        device: &ash::Device,
        context: &DeviceContext,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, CompositorError> {
        let mut host = Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: 0,
        };
        // SAFETY: forwarded from the caller; on failure the partial buffer is destroyed.
        let result = unsafe {
            (|| -> Result<(), CompositorError> {
                host.buffer = device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size as vk::DeviceSize)
                        .usage(usage),
                    None,
                )?;
                let requirements = device.get_buffer_memory_requirements(host.buffer);
                let wanted =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let memory_type =
                    memory_type(&context.memory, requirements.memory_type_bits, wanted)
                        .ok_or(CompositorError::NoMemory)?;
                host.memory = device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type),
                    None,
                )?;
                device.bind_buffer_memory(host.buffer, host.memory, 0)?;
                let mapped = device.map_memory(
                    host.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?;
                host.mapped = mapped as usize;
                Ok(())
            })()
        };
        match result {
            Ok(()) => Ok(host),
            Err(error) => {
                // SAFETY: never used by the GPU.
                unsafe { host.destroy(device) };
                Err(error)
            }
        }
    }

    /// Copies `bytes` to `offset`.
    ///
    /// # Safety
    ///
    /// The range must lie inside the buffer, and the GPU must not be reading it.
    unsafe fn write(&self, offset: usize, bytes: &[u8]) {
        // SAFETY: forwarded from the caller; the mapping covers the whole buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.mapped as *mut u8).add(offset),
                bytes.len(),
            );
        }
    }

    /// # Safety
    ///
    /// The GPU must be done with the buffer. Null handles are skipped.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from the caller; freeing memory also unmaps it.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub struct TextPass {
    extent: vk::Extent2D,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    font: Option<HostBuffer>,
    overlays: Vec<HostBuffer>,
    /// Per image: the cells its buffer holds now.
    written: Mutex<Vec<Vec<Cell>>>,
}

impl TextPass {
    /// Creates the pass for `images` images of `extent`, drawing in subpass 0 of
    /// `render_pass`.
    ///
    /// # Errors
    ///
    /// Any failed Vulkan call; everything created so far is destroyed again.
    ///
    /// # Safety
    ///
    /// `render_pass` must be a live render pass of `device`.
    pub unsafe fn new(
        device: &ash::Device,
        context: &DeviceContext,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        images: usize,
    ) -> Result<Self, CompositorError> {
        let mut pass = Self {
            extent,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            font: None,
            overlays: Vec::new(),
            written: Mutex::new(vec![Vec::new(); images]),
        };
        // SAFETY: forwarded from the caller.
        match unsafe { pass.build(device, context, render_pass, images) } {
            Ok(()) => Ok(pass),
            Err(error) => {
                // SAFETY: nothing was submitted.
                unsafe { pass.destroy(device) };
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// As [`TextPass::new`].
    unsafe fn build(
        &mut self,
        device: &ash::Device,
        context: &DeviceContext,
        render_pass: vk::RenderPass,
        images: usize,
    ) -> Result<(), CompositorError> {
        // SAFETY: every object is created on `device` and stored in `self` at once, so
        // `destroy` releases it whatever fails later.
        unsafe {
            let font = HostBuffer::new(
                device,
                context,
                FONT_BYTES,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            font.write(0, &font_bytes());
            let font_buffer = font.buffer;
            self.font = Some(font);
            for _ in 0..images {
                let overlay = HostBuffer::new(
                    device,
                    context,
                    OVERLAY_BYTES,
                    vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                )?;
                overlay.write(0, &command_bytes(0));
                self.overlays.push(overlay);
            }

            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            self.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let set_layouts = [self.set_layout];
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .size(PUSH_BYTES)];
            self.pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
            let bindings = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(CELL_BYTES as u32)
                .input_rate(vk::VertexInputRate::INSTANCE)];
            let attributes = [
                vk::VertexInputAttributeDescription::default()
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(0),
                vk::VertexInputAttributeDescription::default()
                    .location(1)
                    .format(vk::Format::R32_UINT)
                    .offset(8),
                vk::VertexInputAttributeDescription::default()
                    .location(2)
                    .format(vk::Format::R32_UINT)
                    .offset(12),
            ];
            // Alpha stays as the image had it: a translucent panel must not make the
            // window itself translucent under a non-opaque composite alpha.
            let blend = vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B,
                );
            self.pipeline = create_pipeline(
                device,
                self.pipeline_layout,
                render_pass,
                self.extent,
                &PipelineSpec {
                    vertex: VERTEX_SHADER,
                    fragment: FRAGMENT_SHADER,
                    vertex_input: vk::PipelineVertexInputStateCreateInfo::default()
                        .vertex_binding_descriptions(&bindings)
                        .vertex_attribute_descriptions(&attributes),
                    blend,
                },
            )?;

            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)];
            self.descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            self.set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&set_layouts),
            )?[0];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(font_buffer)
                .range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info);
            device.update_descriptor_sets(&[write], &[]);
        }
        Ok(())
    }

    /// Records the text draw of image `index` inside the open render pass.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording inside subpass 0 of the pass this text pass
    /// was created for.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        index: usize,
    ) {
        let cell = (8 * scale(self.extent)) as f32;
        let push: Vec<u8> = [self.extent.width as f32, self.extent.height as f32, cell]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let buffer = self.overlays[index].buffer;
        // SAFETY: forwarded from the caller; every object is owned by `self`.
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push,
            );
            device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[buffer],
                &[COMMAND_BYTES as vk::DeviceSize],
            );
            device.cmd_draw_indirect(command_buffer, buffer, 0, 1, COMMAND_BYTES as u32);
        }
    }

    /// Makes image `index` draw `cells` (at most [`MAX_CELLS`]) from its next composite
    /// on. Unchanged content is not rewritten; otherwise `wait` runs first and must
    /// return only once no submitted composite of that image is still executing.
    ///
    /// # Safety
    ///
    /// `index` must be an image index of this swapchain, and `wait` must keep its promise.
    pub unsafe fn update(&self, index: usize, cells: &[Cell], wait: impl FnOnce()) {
        let cells = &cells[..cells.len().min(MAX_CELLS)];
        let mut written = self.written.lock().unwrap_or_else(PoisonError::into_inner);
        if written[index] == cells {
            return;
        }
        wait();
        let overlay = &self.overlays[index];
        let bytes: Vec<u8> = cells.iter().flat_map(cell_bytes).collect();
        // SAFETY: both ranges fit `OVERLAY_BYTES`, and `wait` ensured the GPU is not
        // reading this image's buffer.
        unsafe {
            overlay.write(COMMAND_BYTES, &bytes);
            overlay.write(0, &command_bytes(cells.len() as u32));
        }
        written[index] = cells.to_vec();
    }

    /// # Safety
    ///
    /// No submitted composite may still be executing. Null handles are skipped.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from the caller.
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for overlay in &self.overlays {
                overlay.destroy(device);
            }
            if let Some(font) = &self.font {
                font.destroy(device);
            }
        }
    }
}

/// A draw of one six-vertex quad per cell.
fn command_bytes(cells: u32) -> [u8; COMMAND_BYTES] {
    words_bytes([6, cells, 0, 0])
}

/// A cell as the vertex input reads it.
fn cell_bytes(cell: &Cell) -> [u8; CELL_BYTES] {
    words_bytes([cell.x.to_bits(), cell.y.to_bits(), cell.glyph, cell.color])
}

/// Four 32-bit words in memory order.
fn words_bytes(words: [u32; 4]) -> [u8; 16] {
    let mut bytes = [0; 16];
    let (chunks, _) = bytes.as_chunks_mut::<4>();
    for (target, word) in chunks.iter_mut().zip(words) {
        *target = word.to_ne_bytes();
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_bytes_match_draw_indirect_layout() {
        let bytes = command_bytes(7);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_ne_bytes(*word))
            .collect();
        assert_eq!(words, [6, 7, 0, 0]);
        assert_eq!(COMMAND_BYTES, 16);
    }

    #[test]
    fn test_cell_bytes_follow_the_vertex_attributes() {
        let cell = Cell {
            x: 16.0,
            y: 32.0,
            glyph: 33,
            color: 0x1122_3344,
        };
        let bytes = cell_bytes(&cell);
        assert_eq!(&bytes[0..4], &16.0_f32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &32.0_f32.to_ne_bytes());
        assert_eq!(&bytes[8..12], &33_u32.to_ne_bytes());
        assert_eq!(&bytes[12..16], &0x1122_3344_u32.to_ne_bytes());
    }
}
