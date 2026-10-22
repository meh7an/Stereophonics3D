//! Frame recording, submission, presentation and swapchain readback.
//!
//! Each frame runs up to four passes: shadow (depth only, when the light casts shadows),
//! main (to the swapchain, or to an HDR target when the scene post-processes), post
//! (tone-map to the swapchain) and HUD (always, possibly empty, ending in present layout).

use anyhow::{Result, bail};
use ash::vk;
use glam::{Mat4, Vec3};
use winit::window::Window;

use super::{
    RenderingMode,
    context::Context,
    memory::{DeviceImage, HostBuffer, subresource_range},
    passes::{AttachmentSpec, Load, Pass, Target, layout_barrier},
    pipeline::{Pipelines, VERTEX_STRIDE},
    swapchain::{DEPTH_FORMAT, Swapchain},
};
use crate::scene::{CLEAR_COLOR, Material, Scene, Vertex};

const SHADOW_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 1024,
    height: 1024,
};
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const SHADOW_READ_LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
/// Three `mat4` and three `vec4`, matching `Frame` in `frame.glsl` (std140).
const UNIFORM_FLOATS: usize = 3 * 16 + 3 * 4;

/// One mesh's slice of the shared vertex buffer.
struct Draw {
    material: Material,
    model: Mat4,
    first: u32,
    count: u32,
}

/// Vertex ranges for one frame, in draw order.
struct FrameDraws {
    meshes: Vec<Draw>,
    sky: Option<(u32, u32)>,
    hud: (u32, u32),
}

/// Draws one scene, one frame in flight at a time.
///
/// The CPU waits for each frame before recording the next, which keeps synchronisation
/// trivial and makes every frame's GPU work identical across runs.
pub struct Renderer {
    mode: RenderingMode,
    uniform_buffer: HostBuffer,
    vertex_buffer: HostBuffer,
    capture_buffer: HostBuffer,
    shadow_map: DeviceImage,
    scene_color: Option<DeviceImage>,
    shadow_sampler: vk::Sampler,
    color_sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    frame_set: vk::DescriptorSet,
    post_set: Option<vk::DescriptorSet>,
    shadow_pass: Pass,
    main_pass: Pass,
    post_pass: Option<Pass>,
    hud_pass: Pass,
    pipelines: Pipelines,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    swapchain: Swapchain,
    // Last field: dropped after everything created from its device.
    ctx: Context,
}

impl Renderer {
    /// Creates every resource `scene` needs. Later frames of the same scene must keep the
    /// same post-processing setting and at most twice the vertex count.
    pub fn new(
        window: &Window,
        scene: &Scene,
        extent: vk::Extent2D,
        mode: RenderingMode,
        validate: bool,
    ) -> Result<Self> {
        let ctx = Context::new(window, validate)?;
        let swapchain = Swapchain::new(&ctx, extent)?;
        let device = &ctx.device;

        let shadow_map = DeviceImage::new(
            &ctx,
            SHADOW_EXTENT,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let scene_color = scene
            .post
            .then(|| {
                DeviceImage::new(
                    &ctx,
                    extent,
                    HDR_FORMAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .transpose()?;

        let swapchain_targets: Vec<Target> = swapchain
            .images
            .iter()
            .zip(&swapchain.views)
            .map(|(&image, &view)| Target {
                color: Some((image, view)),
                depth: None,
            })
            .collect();
        let depth_target = Some((swapchain.depth.image, swapchain.depth.view));

        let shadow_pass = Pass::new(
            device,
            mode,
            None,
            Some(AttachmentSpec {
                format: DEPTH_FORMAT,
                load: Load::Clear(clear_depth()),
                store: true,
                initial: vk::ImageLayout::UNDEFINED,
                final_layout: SHADOW_READ_LAYOUT,
            }),
            SHADOW_EXTENT,
            &[shadow_target(&shadow_map)],
        )?;
        let main_targets: Vec<Target> = match &scene_color {
            Some(color) => vec![Target {
                color: Some((color.image, color.view)),
                depth: depth_target,
            }],
            None => swapchain_targets
                .iter()
                .map(|target| Target {
                    depth: depth_target,
                    ..*target
                })
                .collect(),
        };
        let main_pass = Pass::new(
            device,
            mode,
            Some(AttachmentSpec {
                format: if scene.post {
                    HDR_FORMAT
                } else {
                    swapchain.format
                },
                load: Load::Clear(clear_color()),
                store: true,
                initial: vk::ImageLayout::UNDEFINED,
                final_layout: if scene.post {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                },
            }),
            Some(AttachmentSpec {
                format: DEPTH_FORMAT,
                load: Load::Clear(clear_depth()),
                store: false,
                initial: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
            extent,
            &main_targets,
        )?;
        let post_pass = scene
            .post
            .then(|| {
                Pass::new(
                    device,
                    mode,
                    Some(AttachmentSpec {
                        format: swapchain.format,
                        load: Load::DontCare,
                        store: true,
                        initial: vk::ImageLayout::UNDEFINED,
                        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    }),
                    None,
                    extent,
                    &swapchain_targets,
                )
            })
            .transpose()?;
        let hud_pass = Pass::new(
            device,
            mode,
            Some(AttachmentSpec {
                format: swapchain.format,
                load: Load::Keep,
                store: true,
                initial: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            }),
            None,
            extent,
            &swapchain_targets,
        )?;
        let pipelines = Pipelines::new(
            device,
            &shadow_pass,
            &main_pass,
            post_pass.as_ref(),
            &hud_pass,
        )?;

        let shadow_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                None,
            )?
        };
        let color_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let uniform_buffer = HostBuffer::new(
            &ctx,
            (UNIFORM_FLOATS * 4) as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let vertex_capacity = 2 * vertex_count(scene) as u64 * u64::from(VERTEX_STRIDE);
        let vertex_buffer =
            HostBuffer::new(&ctx, vertex_capacity, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let capture_size = u64::from(extent.width) * u64::from(extent.height) * 4;
        let capture_buffer =
            HostBuffer::new(&ctx, capture_size, vk::BufferUsageFlags::TRANSFER_DST)?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let layouts = [pipelines.frame_set_layout, pipelines.post_set_layout];
        let set_count = if scene.post { 2 } else { 1 };
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts[..set_count]),
            )?
        };
        let frame_set = sets[0];
        let post_set = sets.get(1).copied();

        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer)
            .range(vk::WHOLE_SIZE)];
        let shadow_info = [vk::DescriptorImageInfo::default()
            .sampler(shadow_sampler)
            .image_view(shadow_map.view)
            .image_layout(SHADOW_READ_LAYOUT)];
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(frame_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(frame_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_info),
        ];
        let color_info: Vec<_> = scene_color
            .iter()
            .map(|color| {
                vk::DescriptorImageInfo::default()
                    .sampler(color_sampler)
                    .image_view(color.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        if let Some(post_set) = post_set {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(post_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_info),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)?[0] };
        let image_available =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { device.create_fence(&fence_info, None)? };

        let renderer = Self {
            mode,
            uniform_buffer,
            vertex_buffer,
            capture_buffer,
            shadow_map,
            scene_color,
            shadow_sampler,
            color_sampler,
            descriptor_pool,
            frame_set,
            post_set,
            shadow_pass,
            main_pass,
            post_pass,
            hud_pass,
            pipelines,
            command_pool,
            command_buffer,
            image_available,
            in_flight,
            swapchain,
            ctx,
        };
        renderer.initialize_shadow_map()?;
        tracing::info!(mode = ?renderer.mode, post = scene.post, "renderer ready");
        Ok(renderer)
    }

    /// Renders and presents one frame of `scene`. With `capture`, also returns the
    /// presented image as tightly packed RGBA8 rows, top to bottom.
    pub fn render(&mut self, scene: &Scene, capture: bool) -> Result<Option<Vec<u8>>> {
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

        self.uniform_buffer
            .write(device, &floats_to_bytes(&uniforms(scene)))?;
        let draws = self.upload_vertices(scene)?;
        self.record(scene, &draws, image_index as usize, capture)?;

        let render_finished = self.swapchain.render_finished[image_index as usize];
        let wait_semaphores = [self.image_available];
        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
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

    /// Writes meshes, then the sky cube, then HUD triangles into the vertex buffer.
    fn upload_vertices(&self, scene: &Scene) -> Result<FrameDraws> {
        let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count(scene));
        let mut range = |new: &[Vertex]| -> Result<(u32, u32)> {
            let first = u32::try_from(vertices.len())?;
            vertices.extend_from_slice(new);
            Ok((first, u32::try_from(new.len())?))
        };
        let mut meshes = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let (first, count) = range(&mesh.vertices)?;
            meshes.push(Draw {
                material: mesh.material,
                model: mesh.model,
                first,
                count,
            });
        }
        let sky = if scene.sky {
            Some(range(&scene.sky_vertices())?)
        } else {
            None
        };
        let hud = range(&scene.hud)?;

        let floats: Vec<f32> = vertices
            .iter()
            .flat_map(|v| {
                v.position
                    .to_array()
                    .into_iter()
                    .chain(v.normal.to_array())
                    .chain(v.color)
                    .chain(v.uv)
            })
            .collect();
        let bytes = floats_to_bytes(&floats);
        if bytes.len() as u64 > self.vertex_buffer.size {
            bail!(
                "frame needs {} vertex bytes, buffer holds {}",
                bytes.len(),
                self.vertex_buffer.size
            );
        }
        self.vertex_buffer.write(&self.ctx.device, &bytes)?;
        Ok(FrameDraws { meshes, sky, hud })
    }

    fn record(
        &self,
        scene: &Scene,
        draws: &FrameDraws,
        image_index: usize,
        capture: bool,
    ) -> Result<()> {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        let pipelines = &self.pipelines;
        let swapchain_image = self.swapchain.images[image_index];
        let swapchain_target = Target {
            color: Some((swapchain_image, self.swapchain.views[image_index])),
            depth: None,
        };
        let depth = Some((self.swapchain.depth.image, self.swapchain.depth.view));
        let opaque = || {
            draws
                .meshes
                .iter()
                .filter(|d| d.material != Material::Transparent)
        };
        let transparent = || {
            draws
                .meshes
                .iter()
                .filter(|d| d.material == Material::Transparent)
        };

        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.mesh_layout,
                0,
                &[self.frame_set],
                &[],
            );
        }

        if scene.light_view_proj().is_some() {
            let target = shadow_target(&self.shadow_map);
            self.shadow_pass.begin(device, cmd, 0, &target);
            self.bind(pipelines.shadow);
            opaque().for_each(|draw| self.draw_mesh(draw));
            self.shadow_pass.end(device, cmd, &target);
        }

        let (main_index, main_target) = match &self.scene_color {
            Some(color) => (
                0,
                Target {
                    color: Some((color.image, color.view)),
                    depth,
                },
            ),
            None => (
                image_index,
                Target {
                    depth,
                    ..swapchain_target
                },
            ),
        };
        self.main_pass.begin(device, cmd, main_index, &main_target);
        for draw in opaque() {
            self.bind(pipelines.for_material(draw.material));
            self.draw_mesh(draw);
        }
        if let Some((first, count)) = draws.sky {
            self.bind(pipelines.sky);
            unsafe { device.cmd_draw(cmd, count, 1, first, 0) };
        }
        for draw in transparent() {
            self.bind(pipelines.transparent);
            self.draw_mesh(draw);
        }
        self.main_pass.end(device, cmd, &main_target);

        if let (Some(post_pass), Some(tonemap), Some(post_set)) =
            (&self.post_pass, pipelines.tonemap, self.post_set)
        {
            post_pass.begin(device, cmd, image_index, &swapchain_target);
            self.bind(tonemap);
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.post_layout,
                    0,
                    &[post_set],
                    &[],
                );
                device.cmd_draw(cmd, 3, 1, 0, 0);
            }
            post_pass.end(device, cmd, &swapchain_target);
        }

        self.hud_pass
            .begin(device, cmd, image_index, &swapchain_target);
        let (first, count) = draws.hud;
        if count > 0 {
            self.bind(pipelines.hud);
            unsafe { device.cmd_draw(cmd, count, 1, first, 0) };
        }
        self.hud_pass.end(device, cmd, &swapchain_target);

        if capture {
            self.copy_to_capture(swapchain_image);
        }
        unsafe { device.end_command_buffer(cmd)? };
        Ok(())
    }

    fn bind(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.ctx.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }
    }

    fn draw_mesh(&self, draw: &Draw) {
        let push = floats_to_bytes(&draw.model.to_cols_array());
        unsafe {
            self.ctx.device.cmd_push_constants(
                self.command_buffer,
                self.pipelines.mesh_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push,
            );
            self.ctx
                .device
                .cmd_draw(self.command_buffer, draw.count, 1, draw.first, 0);
        }
    }

    /// Copies the presentable image into `capture_buffer` and returns it to
    /// `PRESENT_SRC_KHR`.
    fn copy_to_capture(&self, image: vk::Image) {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        layout_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
        layout_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
    }

    /// Clears the shadow map to the far plane and leaves it in its sampled layout, so lit
    /// scenes without a shadow pass still bind a valid image.
    fn initialize_shadow_map(&self) -> Result<()> {
        let device = &self.ctx.device;
        let cmd = self.command_buffer;
        let image = self.shadow_map.image;
        let range = subresource_range(vk::ImageAspectFlags::DEPTH);
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        layout_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        unsafe {
            device.cmd_clear_depth_stencil_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
                &[range],
            );
        }
        layout_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            SHADOW_READ_LAYOUT,
        );
        unsafe {
            device.end_command_buffer(cmd)?;
            let command_buffers = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device.queue_submit(self.ctx.queue, &[submit], vk::Fence::null())?;
            device.queue_wait_idle(self.ctx.queue)?;
        }
        Ok(())
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
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_sampler(self.color_sampler, None);
        }
        self.pipelines.destroy(device);
        self.hud_pass.destroy(device);
        if let Some(pass) = &self.post_pass {
            pass.destroy(device);
        }
        self.main_pass.destroy(device);
        self.shadow_pass.destroy(device);
        if let Some(color) = &self.scene_color {
            color.destroy(device);
        }
        self.shadow_map.destroy(device);
        self.capture_buffer.destroy(device);
        self.vertex_buffer.destroy(device);
        self.uniform_buffer.destroy(device);
        self.swapchain.destroy(&self.ctx);
    }
}

fn shadow_target(shadow_map: &DeviceImage) -> Target {
    Target {
        color: None,
        depth: Some((shadow_map.image, shadow_map.view)),
    }
}

fn clear_color() -> vk::ClearValue {
    let [r, g, b] = CLEAR_COLOR.map(|channel| f32::from(channel) / 255.0);
    vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [r, g, b, 1.0],
        },
    }
}

fn clear_depth() -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    }
}

fn vertex_count(scene: &Scene) -> usize {
    let sky = if scene.sky {
        scene.sky_vertices().len()
    } else {
        0
    };
    scene.meshes.iter().map(|m| m.vertices.len()).sum::<usize>() + sky + scene.hud.len()
}

/// The `Frame` uniform block for `scene`, in std140 order.
fn uniforms(scene: &Scene) -> Vec<f32> {
    let light_view_proj = scene.light_view_proj();
    let (direction, color, ambient) = match scene.light {
        Some(light) => (light.direction, light.color, light.ambient),
        None => (Vec3::Y, Vec3::ZERO, 1.0),
    };
    let shadowed = if light_view_proj.is_some() { 1.0 } else { 0.0 };
    let mut floats = Vec::with_capacity(UNIFORM_FLOATS);
    floats.extend(scene.view_proj().to_cols_array());
    floats.extend(scene.sky_view_proj().to_cols_array());
    floats.extend(light_view_proj.unwrap_or(Mat4::IDENTITY).to_cols_array());
    floats.extend(scene.camera.position.extend(1.0).to_array());
    floats.extend(direction.extend(shadowed).to_array());
    floats.extend(color.extend(ambient).to_array());
    floats
}

fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}
