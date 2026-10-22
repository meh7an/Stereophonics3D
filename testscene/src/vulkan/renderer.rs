//! Frame recording, submission, presentation and swapchain readback.

use anyhow::{Result, bail};
use ash::vk;
use winit::window::Window;

use super::{
    RenderingMode,
    context::Context,
    memory::{HostBuffer, subresource_range},
    pipeline::{Pipeline, VERTEX_STRIDE},
    swapchain::Swapchain,
};
use crate::scene::{CLEAR_COLOR, Scene};

/// Draws one scene, one frame in flight at a time.
///
/// The CPU waits for each frame before recording the next, which keeps synchronisation
/// trivial and makes every frame's GPU work identical across runs.
pub struct Renderer {
    mode: RenderingMode,
    view_proj: [f32; 16],
    vertex_count: u32,
    vertex_buffer: HostBuffer,
    capture_buffer: HostBuffer,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    pipeline: Pipeline,
    swapchain: Swapchain,
    // Last field: dropped after everything created from its device.
    ctx: Context,
}

impl Renderer {
    pub fn new(
        window: &Window,
        scene: &Scene,
        extent: vk::Extent2D,
        mode: RenderingMode,
        validate: bool,
    ) -> Result<Self> {
        let ctx = Context::new(window, validate)?;
        let swapchain = Swapchain::new(&ctx, extent)?;
        let pipeline = Pipeline::new(&ctx.device, &swapchain, mode)?;

        let vertices = scene.vertices();
        let bytes: Vec<u8> = vertices
            .iter()
            .flat_map(|vertex| vertex.position.to_array().into_iter().chain(vertex.color))
            .flat_map(f32::to_ne_bytes)
            .collect();
        let vertex_buffer = HostBuffer::new(
            &ctx,
            bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        vertex_buffer.write(&ctx.device, &bytes)?;
        let vertex_count = u32::try_from(bytes.len())? / VERTEX_STRIDE;

        let capture_size = u64::from(extent.width) * u64::from(extent.height) * 4;
        let capture_buffer =
            HostBuffer::new(&ctx, capture_size, vk::BufferUsageFlags::TRANSFER_DST)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { ctx.device.allocate_command_buffers(&allocate_info)?[0] };
        let image_available = unsafe {
            ctx.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { ctx.device.create_fence(&fence_info, None)? };

        Ok(Self {
            mode,
            view_proj: scene.view_proj().to_cols_array(),
            vertex_count,
            vertex_buffer,
            capture_buffer,
            command_pool,
            command_buffer,
            image_available,
            in_flight,
            pipeline,
            swapchain,
            ctx,
        })
    }

    /// Renders and presents one frame. With `capture`, also returns the presented image
    /// as tightly packed RGBA8 rows, top to bottom.
    pub fn render(&mut self, capture: bool) -> Result<Option<Vec<u8>>> {
        let device = &self.ctx.device;
        unsafe {
            device.wait_for_fences(&[self.in_flight], true, u64::MAX)?;
            device.reset_fences(&[self.in_flight])?;
        }
        let (image_index, _suboptimal) = unsafe {
            self.ctx.swapchain_loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )?
        };
        self.record(image_index as usize, capture)?;

        let render_finished = self.swapchain.render_finished[image_index as usize];
        let wait_semaphores = [self.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffer];
        let signal_semaphores = [render_finished];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe { device.queue_submit(self.ctx.queue, &[submit], self.in_flight)? };

        let pixels = if capture {
            unsafe { device.wait_for_fences(&[self.in_flight], true, u64::MAX)? };
            let mut pixels = self.capture_buffer.read(device)?;
            if self.swapchain.format == vk::Format::B8G8R8A8_UNORM {
                let (bgra, _) = pixels.as_chunks_mut::<4>();
                bgra.iter_mut().for_each(|pixel| pixel.swap(0, 2));
            }
            Some(pixels)
        } else {
            None
        };

        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe {
            self.ctx
                .swapchain_loader
                .queue_present(self.ctx.queue, &present)
        } {
            Ok(_) => Ok(pixels),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                bail!("swapchain went out of date; the test scene needs a fixed-size window")
            }
            Err(error) => Err(error.into()),
        }
    }

    fn record(&self, image_index: usize, capture: bool) -> Result<()> {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        let image = self.swapchain.images[image_index];
        let extent = self.swapchain.extent;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [
                    f32::from(CLEAR_COLOR[0]) / 255.0,
                    f32::from(CLEAR_COLOR[1]) / 255.0,
                    f32::from(CLEAR_COLOR[2]) / 255.0,
                    1.0,
                ],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let render_area = vk::Rect2D::default().extent(extent);

        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            match self.mode {
                RenderingMode::Pass => {
                    let clears = [clear_color, clear_depth];
                    // `Pipeline::new` always creates a render pass in pass mode.
                    let render_pass = self.pipeline.render_pass.expect("pass mode has one");
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass)
                        .framebuffer(self.pipeline.framebuffers[image_index])
                        .render_area(render_area)
                        .clear_values(&clears);
                    device.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE);
                }
                RenderingMode::Dynamic => {
                    self.begin_dynamic_rendering(image_index, clear_color, clear_depth);
                }
            }

            let viewport = vk::Viewport::default()
                .width(extent.width as f32)
                .height(extent.height as f32)
                .max_depth(1.0);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[render_area]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            let push: Vec<u8> = self
                .view_proj
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push,
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_draw(cmd, self.vertex_count, 1, 0, 0);

            // Both paths leave the image in PRESENT_SRC_KHR with colour writes made
            // available to transfer, which is what `copy_to_capture` expects.
            match self.mode {
                RenderingMode::Pass => device.cmd_end_render_pass(cmd),
                RenderingMode::Dynamic => {
                    device.cmd_end_rendering(cmd);
                    image_barrier(
                        device,
                        cmd,
                        image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::PRESENT_SRC_KHR,
                        (
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ),
                        (
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::TRANSFER_READ,
                        ),
                    );
                }
            }
            if capture {
                self.copy_to_capture(image);
            }
            device.end_command_buffer(cmd)?;
        }
        Ok(())
    }

    /// Transitions the colour and depth targets and opens a dynamic rendering scope.
    fn begin_dynamic_rendering(
        &self,
        image_index: usize,
        clear_color: vk::ClearValue,
        clear_depth: vk::ClearValue,
    ) {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        image_barrier(
            device,
            cmd,
            self.swapchain.images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::NONE,
            ),
            (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
        );
        let depth_stages = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        let depth_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.swapchain.depth.image)
            .subresource_range(subresource_range(vk::ImageAspectFlags::DEPTH));
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                depth_stages,
                depth_stages,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[depth_barrier],
            );
        }

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color)];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(self.swapchain.extent))
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        unsafe { device.cmd_begin_rendering(cmd, &rendering) };
    }

    /// Copies the presentable image into `capture_buffer` and returns it to
    /// `PRESENT_SRC_KHR`. Expects prior colour writes to be visible to transfer.
    fn copy_to_capture(&self, image: vk::Image) {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        image_barrier(
            device,
            cmd,
            image,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::NONE),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(self.swapchain.extent.into());
        let buffer_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.capture_buffer.buffer)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.capture_buffer.buffer,
                &[region],
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );
        }
        image_barrier(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
            (
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::NONE,
            ),
        );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        unsafe {
            // Nothing can be released while the GPU may still use it; on failure the
            // device is lost and destruction is the only remaining option anyway.
            let _ = device.device_wait_idle();
            device.destroy_fence(self.in_flight, None);
            device.destroy_semaphore(self.image_available, None);
            device.destroy_command_pool(self.command_pool, None);
        }
        self.capture_buffer.destroy(device);
        self.vertex_buffer.destroy(device);
        self.pipeline.destroy(device);
        self.swapchain.destroy(&self.ctx);
    }
}

/// Records a single colour-image layout transition.
fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    (src_stage, src_access): (vk::PipelineStageFlags, vk::AccessFlags),
    (dst_stage, dst_access): (vk::PipelineStageFlags, vk::AccessFlags),
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR));
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
