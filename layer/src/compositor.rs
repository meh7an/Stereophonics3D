//! The compositor of one virtual swapchain: layer-owned images the application renders
//! into, and the pass that draws them into the real swapchain images at present.
//!
//! Virtual image `i` pairs with real image `i`, so the application's acquire goes to the
//! driver unchanged. Re-acquiring real image `i` proves the previous composite of `i` has
//! finished (its present waited on the composite), so the composite command buffers are
//! recorded once per image and queue family and resubmitted without a CPU wait; they are
//! marked simultaneous-use because the CPU cannot observe that completion itself.

use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    io::Cursor,
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use ash::vk::{self, Handle};

use crate::{chain, loader::SetDeviceLoaderData, swapchain::DeviceContext};

const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/identity.frag.spv"));

/// Why a compositor could not be created or used.
#[derive(Debug, thiserror::Error)]
pub enum CompositorError {
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("embedded shader is not SPIR-V: {0}")]
    Spirv(#[from] std::io::Error),
    #[error("no memory type for the virtual swapchain images")]
    NoMemory,
}

/// One layer-owned image the application sees as a swapchain image.
#[derive(Debug, Clone, Copy)]
struct VirtualImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// Composite command buffers for presents from one queue family.
struct Recorded {
    pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    /// The family writes timestamps around each composite.
    timed: bool,
}

/// One composite ready for submission.
#[derive(Debug, Clone, Copy)]
pub struct Composite {
    pub command_buffer: vk::CommandBuffer,
    /// Signalled when the composite finishes; the present waits on it.
    pub finished: vk::Semaphore,
    pub timed: bool,
}

pub struct Compositor {
    extent: vk::Extent2D,
    images: Vec<VirtualImage>,
    /// Views of the real swapchain images and the framebuffers around them.
    targets: Vec<(vk::ImageView, vk::Framebuffer)>,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    finished: Vec<vk::Semaphore>,
    /// Two timestamps per image.
    queries: vk::QueryPool,
    /// Per image: a timed composite was submitted, so its timestamps can be read.
    measured: Vec<AtomicBool>,
    recorded: Mutex<HashMap<u32, Recorded>>,
}

impl Compositor {
    /// Creates virtual images matching `info` and the pass that draws them into
    /// `real_images`.
    ///
    /// # Errors
    ///
    /// Any failed Vulkan call; everything created so far is destroyed again.
    ///
    /// # Safety
    ///
    /// `real_images` must be the images of a live swapchain created on `device` from
    /// `info`, with `COLOR_ATTACHMENT` usage.
    pub unsafe fn new(
        device: &ash::Device,
        context: &DeviceContext,
        info: &vk::SwapchainCreateInfoKHR<'_>,
        real_images: &[vk::Image],
    ) -> Result<Self, CompositorError> {
        let mut compositor = Self {
            extent: info.image_extent,
            images: Vec::new(),
            targets: Vec::new(),
            sampler: vk::Sampler::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            finished: Vec::new(),
            queries: vk::QueryPool::null(),
            measured: real_images.iter().map(|_| AtomicBool::new(false)).collect(),
            recorded: Mutex::new(HashMap::new()),
        };
        // SAFETY: forwarded from the caller.
        match unsafe { compositor.build(device, context, info, real_images) } {
            Ok(()) => Ok(compositor),
            Err(error) => {
                // SAFETY: nothing was submitted, and every handle is null or ours.
                unsafe { compositor.destroy(device) };
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// As [`Compositor::new`].
    unsafe fn build(
        &mut self,
        device: &ash::Device,
        context: &DeviceContext,
        info: &vk::SwapchainCreateInfoKHR<'_>,
        real_images: &[vk::Image],
    ) -> Result<(), CompositorError> {
        let format = info.image_format;
        // SAFETY: every object is created on `device` and stored in `self` at once, so
        // `destroy` releases it whatever fails later.
        unsafe {
            for _ in real_images {
                let image = create_virtual_image(device, context, info)?;
                self.images.push(image);
            }
            for &image in real_images {
                let view = create_view(device, image, format)?;
                self.targets.push((view, vk::Framebuffer::null()));
            }
            self.render_pass = create_render_pass(device, format)?;
            for (view, framebuffer) in &mut self.targets {
                let attachments = [*view];
                *framebuffer = device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&attachments)
                        .width(self.extent.width)
                        .height(self.extent.height)
                        .layers(1),
                    None,
                )?;
            }
            self.sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let samplers = [self.sampler];
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&samplers)];
            self.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let set_layouts = [self.set_layout];
            self.pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )?;
            self.pipeline =
                create_pipeline(device, self.pipeline_layout, self.render_pass, self.extent)?;
            self.create_sets(device)?;
            for _ in real_images {
                let semaphore =
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                self.finished.push(semaphore);
            }
            self.queries = device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2 * real_images.len() as u32),
                None,
            )?;
        }
        Ok(())
    }

    /// One descriptor set per virtual image, sampling it in `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// # Safety
    ///
    /// The images and set layout must be created.
    unsafe fn create_sets(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        let count = self.images.len() as u32;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(count)];
        let layouts = vec![self.set_layout; self.images.len()];
        // SAFETY: forwarded from the caller; each write names a set allocated here and a
        // live view.
        unsafe {
            self.descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count)
                    .pool_sizes(&sizes),
                None,
            )?;
            self.sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )?;
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = self
                .images
                .iter()
                .map(|image| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(image.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet<'_>> = self
                .sets
                .iter()
                .zip(&infos)
                .map(|(&set, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    /// The images handed to the application in place of the real ones.
    pub fn images(&self) -> Vec<vk::Image> {
        self.images.iter().map(|image| image.image).collect()
    }

    /// The composite of image `index` for a present on a queue of `family`, recording
    /// the family's command buffers on first use.
    ///
    /// # Errors
    ///
    /// A failed allocation or recording.
    ///
    /// # Safety
    ///
    /// `index` must be an image index of this swapchain and `family` a queue family of
    /// `device` that supports graphics.
    pub unsafe fn composite(
        &self,
        device: &ash::Device,
        context: &DeviceContext,
        set_loader_data: Option<SetDeviceLoaderData>,
        family: u32,
        index: usize,
    ) -> Result<Composite, vk::Result> {
        let mut guard = self.recorded.lock().unwrap_or_else(PoisonError::into_inner);
        let recorded = match guard.entry(family) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let timed = context
                    .families
                    .get(family as usize)
                    .is_some_and(|properties| properties.timestamp_valid_bits > 0);
                // SAFETY: forwarded from the caller.
                entry.insert(unsafe { self.record(device, set_loader_data, family, timed)? })
            }
        };
        Ok(Composite {
            command_buffer: recorded.command_buffers[index],
            finished: self.finished[index],
            timed: recorded.timed,
        })
    }

    /// Records one composite command buffer per image for presents on `family`.
    ///
    /// # Safety
    ///
    /// As [`Compositor::composite`].
    unsafe fn record(
        &self,
        device: &ash::Device,
        set_loader_data: Option<SetDeviceLoaderData>,
        family: u32,
        timed: bool,
    ) -> Result<Recorded, vk::Result> {
        // SAFETY: forwarded from the caller; the pool is destroyed again if any step fails.
        unsafe {
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )?;
            let recorded = Recorded {
                pool,
                command_buffers: Vec::new(),
                timed,
            };
            match self.record_into(device, set_loader_data, recorded) {
                Ok(recorded) => Ok(recorded),
                Err(error) => {
                    device.destroy_command_pool(pool, None);
                    Err(error)
                }
            }
        }
    }

    /// # Safety
    ///
    /// As [`Compositor::composite`], with `recorded.pool` a fresh pool of that family.
    unsafe fn record_into(
        &self,
        device: &ash::Device,
        set_loader_data: Option<SetDeviceLoaderData>,
        mut recorded: Recorded,
    ) -> Result<Recorded, vk::Result> {
        // SAFETY: forwarded from the caller.
        unsafe {
            recorded.command_buffers = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(recorded.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.images.len() as u32),
            )?;
            for (index, &command_buffer) in recorded.command_buffers.iter().enumerate() {
                // Layer-created dispatchable objects need the loader's dispatch pointer.
                if let Some(set_loader_data) = set_loader_data {
                    set_loader_data(device.handle(), command_buffer.as_raw() as *mut c_void)
                        .result()?;
                }
                self.record_one(device, command_buffer, index, recorded.timed)?;
            }
        }
        Ok(recorded)
    }

    /// Samples virtual image `index` into real image `index`. The virtual image arrives
    /// in `PRESENT_SRC_KHR`, where the application left it for presentation, and returns
    /// to it, so the application's next frame finds the layout it expects.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be freshly allocated on this compositor's device.
    unsafe fn record_one(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        index: usize,
        timed: bool,
    ) -> Result<(), vk::Result> {
        let image = self.images[index].image;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        let to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        let first_query = 2 * index as u32;
        // SAFETY: forwarded from the caller; every object recorded is owned by `self`.
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
            )?;
            if timed {
                device.cmd_reset_query_pool(command_buffer, self.queries, first_query, 2);
                device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    first_query,
                );
            }
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.targets[index].1)
                    .render_area(vk::Rect2D::default().extent(self.extent)),
                vk::SubpassContents::INLINE,
            );
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
                &[self.sets[index]],
                &[],
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
            if timed {
                device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    first_query + 1,
                );
            }
            device.end_command_buffer(command_buffer)
        }
    }

    /// Marks image `index`'s composite as submitted with timestamps.
    pub fn submitted_timed(&self, index: usize) {
        self.measured[index].store(true, Ordering::Relaxed);
    }

    /// GPU nanoseconds of image `index`'s previous timed composite, if it is known.
    ///
    /// # Safety
    ///
    /// `index` must be an image index of this swapchain and `device` its device.
    pub unsafe fn gpu_nanoseconds(
        &self,
        device: &ash::Device,
        context: &DeviceContext,
        family: u32,
        index: usize,
    ) -> Option<f64> {
        if !self.measured[index].load(Ordering::Relaxed) {
            return None;
        }
        let bits = context.families.get(family as usize)?.timestamp_valid_bits;
        let mut stamps = [0_u64; 2];
        // SAFETY: both queries were written by a submitted composite; without WAIT a
        // composite still running answers NOT_READY instead of blocking.
        unsafe {
            device
                .get_query_pool_results(
                    self.queries,
                    2 * index as u32,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64,
                )
                .ok()?;
        }
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        let ticks = stamps[1].wrapping_sub(stamps[0]) & mask;
        Some(ticks as f64 * f64::from(context.timestamp_period))
    }

    /// Destroys every object; null handles are skipped by Vulkan.
    ///
    /// # Safety
    ///
    /// No submitted composite may still be executing, and the real swapchain must still
    /// exist (its images are viewed here).
    pub unsafe fn destroy(&self, device: &ash::Device) {
        let recorded =
            std::mem::take(&mut *self.recorded.lock().unwrap_or_else(PoisonError::into_inner));
        // SAFETY: forwarded from the caller.
        unsafe {
            for recorded in recorded.into_values() {
                device.destroy_command_pool(recorded.pool, None);
            }
            device.destroy_query_pool(self.queries, None);
            for &semaphore in &self.finished {
                device.destroy_semaphore(semaphore, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            for &(view, framebuffer) in &self.targets {
                device.destroy_framebuffer(framebuffer, None);
                device.destroy_image_view(view, None);
            }
            device.destroy_render_pass(self.render_pass, None);
            for image in &self.images {
                device.destroy_image_view(image.view, None);
                device.destroy_image(image.image, None);
                device.free_memory(image.memory, None);
            }
        }
    }
}

/// An image with the swapchain's format, size, usage and sharing, plus `SAMPLED` so the
/// compositor can read it.
///
/// # Safety
///
/// `info` must be a valid swapchain create info for `device`.
unsafe fn create_virtual_image(
    device: &ash::Device,
    context: &DeviceContext,
    info: &vk::SwapchainCreateInfoKHR<'_>,
) -> Result<VirtualImage, CompositorError> {
    let mut image_info = virtual_image_info(info);
    let families = if info.p_queue_family_indices.is_null() {
        &[][..]
    } else {
        // SAFETY: a valid create info holds this many family indices.
        unsafe {
            std::slice::from_raw_parts(
                info.p_queue_family_indices,
                info.queue_family_index_count as usize,
            )
        }
    };
    image_info = image_info.queue_family_indices(families);
    // A mutable-format swapchain may be viewed in every format of its list; the virtual
    // image must allow the same views.
    // SAFETY: a valid create info has a valid chain.
    let list = unsafe {
        chain::find(
            info.p_next,
            vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO,
        )
    };
    let mut format_list = list.map(|list| {
        // SAFETY: the structure found has this type.
        let list = unsafe { *list.cast::<vk::ImageFormatListCreateInfo<'_>>() };
        vk::ImageFormatListCreateInfo {
            p_next: std::ptr::null(),
            ..list
        }
    });
    if let Some(format_list) = &mut format_list {
        image_info = image_info.push_next(format_list);
    }
    let mut image = VirtualImage {
        image: vk::Image::null(),
        memory: vk::DeviceMemory::null(),
        view: vk::ImageView::null(),
    };
    // SAFETY: objects created on `device`; on failure the partial image is destroyed.
    unsafe {
        let result = (|| -> Result<(), CompositorError> {
            image.image = device.create_image(&image_info, None)?;
            let requirements = device.get_image_memory_requirements(image.image);
            let memory_type = memory_type(
                &context.memory,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                memory_type(
                    &context.memory,
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or(CompositorError::NoMemory)?;
            image.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )?;
            device.bind_image_memory(image.image, image.memory, 0)?;
            image.view = create_view(device, image.image, info.image_format)?;
            Ok(())
        })();
        if let Err(error) = result {
            device.destroy_image_view(image.view, None);
            device.destroy_image(image.image, None);
            device.free_memory(image.memory, None);
            return Err(error);
        }
    }
    Ok(image)
}

/// The create info of a virtual image for swapchain `info`, without queue families or
/// chained structures.
pub fn virtual_image_info<'a>(info: &vk::SwapchainCreateInfoKHR<'_>) -> vk::ImageCreateInfo<'a> {
    let flags = if info
        .flags
        .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
    {
        vk::ImageCreateFlags::MUTABLE_FORMAT
    } else {
        vk::ImageCreateFlags::empty()
    };
    vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(info.image_format)
        .extent(info.image_extent.into())
        .mip_levels(1)
        .array_layers(info.image_array_layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(info.image_usage | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(info.image_sharing_mode)
        .initial_layout(vk::ImageLayout::UNDEFINED)
}

/// # Safety
///
/// `image` must be a live 2D colour image of `device` with format `format`.
unsafe fn create_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, vk::Result> {
    // SAFETY: forwarded from the caller.
    unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
}

/// A single-subpass pass that overwrites the real image and leaves it ready to present,
/// with its writes visible to a capture copy recorded after it.
///
/// # Safety
///
/// `device` must be live.
unsafe fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let color = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];
    // SAFETY: forwarded from the caller.
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
}

/// The fullscreen-triangle pipeline with a fixed viewport over the whole swapchain.
///
/// # Safety
///
/// `layout` and `render_pass` must be live objects of `device`.
unsafe fn create_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<vk::Pipeline, CompositorError> {
    let vertex_code = ash::util::read_spv(&mut Cursor::new(VERTEX_SHADER))?;
    let fragment_code = ash::util::read_spv(&mut Cursor::new(FRAGMENT_SHADER))?;
    // SAFETY: forwarded from the caller; the modules are destroyed once the pipeline
    // holds its own copy of the code.
    unsafe {
        let vertex = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vertex_code),
            None,
        )?;
        let fragment = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&fragment_code),
            None,
        );
        let fragment = match fragment {
            Ok(fragment) => fragment,
            Err(error) => {
                device.destroy_shader_module(vertex, None);
                return Err(error.into());
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0)];
        let scissors = [vk::Rect2D::default().extent(extent)];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        device.destroy_shader_module(fragment, None);
        device.destroy_shader_module(vertex, None);
        match pipelines {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((_, error)) => Err(error.into()),
        }
    }
}

/// The first memory type allowed by `bits` with all of `wanted`.
pub fn memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory.memory_type_count).find(|&index| {
        bits & (1 << index) != 0
            && memory.memory_types[index as usize]
                .property_flags
                .contains(wanted)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_needs_every_wanted_flag() {
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..Default::default()
        };
        memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        memory.memory_types[1].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        memory.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        assert_eq!(memory_type(&memory, 0b111, wanted), Some(2));
        assert_eq!(memory_type(&memory, 0b011, wanted), None);
        assert_eq!(
            memory_type(&memory, 0b110, vk::MemoryPropertyFlags::empty()),
            Some(1)
        );
    }

    #[test]
    fn test_virtual_image_info_mirrors_swapchain_and_adds_sampled() {
        let swapchain = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_extent(vk::Extent2D {
                width: 640,
                height: 360,
            })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::CONCURRENT);
        let image = virtual_image_info(&swapchain);
        assert_eq!(image.flags, vk::ImageCreateFlags::MUTABLE_FORMAT);
        assert_eq!(image.format, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(
            (image.extent.width, image.extent.height, image.extent.depth),
            (640, 360, 1)
        );
        assert_eq!(
            image.usage,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
        );
        assert_eq!(image.sharing_mode, vk::SharingMode::CONCURRENT);
        assert_eq!(image.initial_layout, vk::ImageLayout::UNDEFINED);
    }
}
