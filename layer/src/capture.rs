//! Per-eye frame capture: copies a presented swapchain image to the host and writes it,
//! with a sidecar, in the capture format of `s3d_shared::capture`.
//!
//! The copy is recorded into the layer's present-time submission after any composite, so
//! it reads exactly what reaches the screen. Real swapchains get `TRANSFER_SRC` usage
//! added at creation so their images can be copied at all.

use std::{ffi::c_void, path::PathBuf, sync::Arc};

use ash::vk::{self, Handle};
use s3d_shared::{
    Settings,
    capture::{CapturePaths, Sidecar, StereoMethod, StereoRecord, write_png, write_sidecar},
    settings::{Snapshot, default_capture_dir},
};

use crate::{activation, compositor::memory_type, dispatch::DeviceData, swapchain::DeviceContext};

/// Why a capture failed.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("swapchain format {0:?} cannot be written as RGBA8")]
    Format(vk::Format),
    #[error("the swapchain was created without TRANSFER_SRC usage")]
    Usage,
    #[error("no host-visible memory for the readback buffer")]
    NoMemory,
    #[error(transparent)]
    File(#[from] s3d_shared::Error),
}

/// The presented image a capture copies.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub image: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub usage: vk::ImageUsageFlags,
}

/// The current snapshot when its settings ask for present `frame` to be captured.
pub fn wanted(frame: u64) -> Option<Arc<Snapshot>> {
    let snapshot = activation::get().snapshot()?;
    let settings = snapshot.settings.as_ref()?;
    (settings.capture.frame.map(u64::from) == Some(frame)).then_some(snapshot)
}

/// A recorded capture copy waiting for its submission to finish.
pub struct Pending<'a> {
    resources: Resources<'a>,
    pub command_buffer: vk::CommandBuffer,
    swizzle: bool,
    extent: vk::Extent2D,
}

/// Records a copy of `target` into a host buffer, for submission on a queue of `family`.
///
/// # Errors
///
/// An unsupported format or usage, or a failed Vulkan call.
///
/// # Safety
///
/// `target` must be a live presentable image of `data`'s device that will be in
/// `PRESENT_SRC_KHR` when the copy executes, and `family` a queue family of that device.
pub unsafe fn record<'a>(
    data: &'a DeviceData,
    context: &DeviceContext,
    family: u32,
    target: &Target,
) -> Result<Pending<'a>, CaptureError> {
    let swizzle = match target.format {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => true,
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => false,
        other => return Err(CaptureError::Format(other)),
    };
    if !target.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        return Err(CaptureError::Usage);
    }
    let vk::Extent2D { width, height } = target.extent;
    let size = u64::from(width) * u64::from(height) * 4;

    let mut resources = Resources::new(&data.next);
    // SAFETY: every object is created on this device and destroyed by `resources` when it
    // drops, which `Pending::finish` does only after the submission has finished.
    unsafe {
        let device = &data.next;
        let pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?;
        resources.pool = pool;
        let command_buffer = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        if let Some(set_loader_data) = data.swapchains.set_loader_data {
            set_loader_data(data.handle, command_buffer.as_raw() as *mut c_void).result()?;
        }
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        )?;
        resources.buffer = buffer;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = memory_type(&context.memory, requirements.memory_type_bits, wanted)
            .ok_or(CaptureError::NoMemory)?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )?;
        resources.memory = memory;
        resources.size = size;
        device.bind_buffer_memory(buffer, memory, 0)?;
        record_copy(device, command_buffer, target.image, buffer, target.extent)?;
        Ok(Pending {
            resources,
            command_buffer,
            swizzle,
            extent: target.extent,
        })
    }
}

impl Pending<'_> {
    /// Reads the copy back and writes the capture; returns the image path.
    ///
    /// # Errors
    ///
    /// A failed map or file write.
    ///
    /// # Safety
    ///
    /// The submission holding `command_buffer` must have finished.
    pub unsafe fn finish(self, settings: &Settings, frame: u64) -> Result<PathBuf, CaptureError> {
        let Resources {
            device,
            memory,
            size,
            ..
        } = &self.resources;
        let (memory, size) = (*memory, *size);
        // SAFETY: the copy finished and the memory is host-visible and coherent.
        let mut pixels = unsafe {
            let mapped = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
            let pixels = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize).to_vec();
            device.unmap_memory(memory);
            pixels
        };
        if self.swizzle {
            let (bgra, _) = pixels.as_chunks_mut::<4>();
            bgra.iter_mut().for_each(|pixel| pixel.swap(0, 2));
        }
        save(
            settings,
            &activation::get().executable,
            frame,
            Rgba {
                width: self.extent.width,
                height: self.extent.height,
                pixels,
            },
        )
    }
}

/// Pixels and size of a capture.
struct Rgba {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Records the image-to-buffer copy, leaving the image in `PRESENT_SRC_KHR` again.
///
/// # Safety
///
/// `command_buffer` must be freshly allocated; `image` a presentable image in
/// `PRESENT_SRC_KHR` and `buffer` large enough for it.
unsafe fn record_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    buffer: vk::Buffer,
    extent: vk::Extent2D,
) -> Result<(), vk::Result> {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let to_transfer = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let to_present = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let to_host = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .size(vk::WHOLE_SIZE);
    let region = vk::BufferImageCopy::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(extent.into());
    // SAFETY: forwarded from the caller.
    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer],
        );
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[region],
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[to_host],
            &[to_present],
        );
        device.end_command_buffer(command_buffer)
    }
}

/// Objects a capture creates, destroyed in reverse order when it ends.
struct Resources<'a> {
    device: &'a ash::Device,
    pool: vk::CommandPool,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl<'a> Resources<'a> {
    fn new(device: &'a ash::Device) -> Self {
        Self {
            device,
            pool: vk::CommandPool::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
        }
    }
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        // SAFETY: each handle is null or was created by the capture on this device, and
        // the GPU is done with them: either the submission finished or it never happened.
        // Destroying null handles is a no-op.
        unsafe {
            self.device.free_memory(self.memory, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Writes the capture and its sidecar; returns the image path.
fn save(
    settings: &Settings,
    executable: &str,
    frame: u64,
    rgba: Rgba,
) -> Result<PathBuf, CaptureError> {
    let stem = executable
        .rsplit_once('.')
        .map_or(executable, |(stem, _)| stem);
    let dir = default_capture_dir()?.join(executable);
    let frame = u32::try_from(frame).unwrap_or(u32::MAX);
    let paths = CapturePaths::new(&dir, stem, frame);
    let eye = settings.stereo.eye;
    let stereo = eye.map(|eye| StereoRecord {
        method: StereoMethod::SingleEye,
        separation: settings.stereo.separation(),
        convergence: settings.stereo.convergence,
        eyes: vec![eye],
    });
    let image = paths.image(eye);
    write_png(&image, rgba.width, rgba.height, &rgba.pixels)?;
    let sidecar = Sidecar::injector(stem, frame, (rgba.width, rgba.height), stereo);
    write_sidecar(&paths.sidecar(), &sidecar)?;
    Ok(image)
}
