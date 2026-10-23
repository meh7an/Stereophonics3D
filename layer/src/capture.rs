//! Per-eye frame capture: copies a presented swapchain image to the host and writes it,
//! with a sidecar, in the capture format of `s3d_shared::capture`.
//!
//! The copy runs on the presenting queue, waits on the present's semaphores itself and
//! completes before the present, so the present no longer waits on them. Swapchains get
//! `TRANSFER_SRC` usage added at creation so their images can be copied at all.

use std::{
    collections::HashMap,
    ffi::c_void,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use ash::vk::{self, Handle};
use s3d_shared::{
    Settings,
    capture::{CapturePaths, Sidecar, StereoMethod, StereoRecord, write_png, write_sidecar},
    ffi::ffi_guard,
    settings::default_capture_dir,
};

use crate::{
    activation,
    dispatch::{DeviceData, read, write},
    hooks::next_device_fn,
    loader::SetDeviceLoaderData,
};

/// Why a capture failed.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("swapchain format {0:?} cannot be written as RGBA8")]
    Format(vk::Format),
    #[error("no host-visible memory for the readback buffer")]
    NoMemory,
    #[error("the present uses an unknown queue or swapchain")]
    Unknown,
    #[error(transparent)]
    File(#[from] s3d_shared::Error),
}

#[derive(Debug)]
struct SwapchainInfo {
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
}

/// Per-device capture state.
pub struct CaptureState {
    /// Queue family of every queue the application retrieved.
    queues: RwLock<HashMap<vk::Queue, u32>>,
    swapchains: RwLock<HashMap<vk::SwapchainKHR, Arc<SwapchainInfo>>>,
    memory: vk::PhysicalDeviceMemoryProperties,
    set_loader_data: Option<SetDeviceLoaderData>,
    swapchain_fn: ash::khr::swapchain::DeviceFn,
}

impl CaptureState {
    pub fn new(
        memory: vk::PhysicalDeviceMemoryProperties,
        set_loader_data: Option<SetDeviceLoaderData>,
        swapchain_fn: ash::khr::swapchain::DeviceFn,
    ) -> Self {
        Self {
            queues: RwLock::new(HashMap::new()),
            swapchains: RwLock::new(HashMap::new()),
            memory,
            set_loader_data,
            swapchain_fn,
        }
    }

    fn record_queue(&self, queue: vk::Queue, family: u32) {
        if queue != vk::Queue::null() {
            write(&self.queues).insert(queue, family);
        }
    }
}

pub unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    family: u32,
    index: u32,
    queue: *mut vk::Queue,
) {
    // SAFETY: a live device from the application.
    let found = unsafe { next_device_fn("vkGetDeviceQueue", device, |f| f.get_device_queue) };
    let Some((data, next)) = found else {
        return;
    };
    // SAFETY: the next layer's query with the application's arguments.
    unsafe { next(device, family, index, queue) };
    // SAFETY: the query just wrote the handle.
    ffi_guard("vkGetDeviceQueue", (), || unsafe {
        data.capture.record_queue(*queue, family)
    });
}

pub unsafe extern "system" fn get_device_queue2(
    device: vk::Device,
    info: *const vk::DeviceQueueInfo2<'_>,
    queue: *mut vk::Queue,
) {
    // SAFETY: a live device from the application.
    let found = unsafe { next_device_fn("vkGetDeviceQueue2", device, |f| f.get_device_queue2) };
    let Some((data, next)) = found else {
        return;
    };
    // SAFETY: the next layer's query with the application's arguments.
    unsafe { next(device, info, queue) };
    // SAFETY: a valid queue info, and the query just wrote the handle.
    ffi_guard("vkGetDeviceQueue2", (), || unsafe {
        data.capture
            .record_queue(*queue, (*info).queue_family_index)
    });
}

pub unsafe extern "system" fn create_swapchain(
    device: vk::Device,
    info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found = unsafe { next_device_fn("vkCreateSwapchainKHR", device, |f| f.create_swapchain) };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: a copy of the application's create info that may also be copied from.
    let info =
        unsafe { (*info).image_usage((*info).image_usage | vk::ImageUsageFlags::TRANSFER_SRC) };
    // SAFETY: the next layer's create with the extended copy.
    let result = unsafe { next(device, &info, allocator, swapchain) };
    if result == vk::Result::SUCCESS {
        ffi_guard("vkCreateSwapchainKHR", (), || {
            // SAFETY: the swapchain was just created on this device.
            let handle = unsafe { *swapchain };
            match unsafe { swapchain_images(&data, device, handle) } {
                Ok(images) => {
                    let swapchain = Arc::new(SwapchainInfo {
                        format: info.image_format,
                        extent: info.image_extent,
                        images,
                    });
                    write(&data.capture.swapchains).insert(handle, swapchain);
                }
                Err(error) => tracing::warn!(%error, "cannot list swapchain images"),
            }
        });
    }
    result
}

/// The images of a swapchain, from the next layer.
///
/// # Safety
///
/// `swapchain` must be a live swapchain of `device`.
unsafe fn swapchain_images(
    data: &DeviceData,
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> Result<Vec<vk::Image>, vk::Result> {
    let get = data.capture.swapchain_fn.get_swapchain_images_khr;
    let mut count = 0;
    // SAFETY: a count query on a live swapchain.
    unsafe { get(device, swapchain, &mut count, std::ptr::null_mut()) }.result()?;
    let mut images = vec![vk::Image::null(); count as usize];
    // SAFETY: `images` holds `count` handles.
    unsafe { get(device, swapchain, &mut count, images.as_mut_ptr()) }.result()?;
    images.truncate(count as usize);
    Ok(images)
}

pub unsafe extern "system" fn destroy_swapchain(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a live device from the application.
    let found = unsafe { next_device_fn("vkDestroySwapchainKHR", device, |f| f.destroy_swapchain) };
    let Some((data, next)) = found else {
        return;
    };
    ffi_guard("vkDestroySwapchainKHR", (), || {
        write(&data.capture.swapchains).remove(&swapchain);
    });
    // SAFETY: the next layer's destroy with the application's arguments.
    unsafe { next(device, swapchain, allocator) };
}

/// Captures the present's first image when the settings ask for frame `frame`. Returns
/// whether the capture consumed the present's wait semaphores, in which case the
/// present must be issued without them.
///
/// # Safety
///
/// `queue` and `info` must be the application's present arguments.
pub unsafe fn on_present(
    data: &DeviceData,
    queue: vk::Queue,
    info: &vk::PresentInfoKHR<'_>,
    frame: u64,
) -> bool {
    let activation = activation::get();
    let Some(snapshot) = activation.snapshot() else {
        return false;
    };
    let Some(settings) = &snapshot.settings else {
        return false;
    };
    if settings.capture.frame.map(u64::from) != Some(frame) || info.swapchain_count == 0 {
        return false;
    }
    let mut consumed = false;
    // SAFETY: forwarded from the caller.
    let result = unsafe { capture(data, queue, info, &mut consumed) };
    match result.and_then(|rgba| save(settings, &activation.executable, frame, rgba)) {
        Ok(path) => tracing::info!(frame, path = %path.display(), "captured frame"),
        Err(error) => tracing::warn!(frame, %error, "capture failed"),
    }
    consumed
}

/// Pixels and size of a capture.
struct Rgba {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Copies the first presented image to the host as RGBA8 rows.
///
/// # Safety
///
/// As [`on_present`].
unsafe fn capture(
    data: &DeviceData,
    queue: vk::Queue,
    info: &vk::PresentInfoKHR<'_>,
    consumed: &mut bool,
) -> Result<Rgba, CaptureError> {
    let state = &data.capture;
    let family = *read(&state.queues)
        .get(&queue)
        .ok_or(CaptureError::Unknown)?;
    // SAFETY: the present names at least one swapchain and image index.
    let (swapchain, index) = unsafe { (*info.p_swapchains, *info.p_image_indices) };
    let swapchain = read(&state.swapchains)
        .get(&swapchain)
        .cloned()
        .ok_or(CaptureError::Unknown)?;
    let swizzle = match swapchain.format {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => true,
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => false,
        other => return Err(CaptureError::Format(other)),
    };
    let image = *swapchain
        .images
        .get(index as usize)
        .ok_or(CaptureError::Unknown)?;
    let vk::Extent2D { width, height } = swapchain.extent;
    let size = u64::from(width) * u64::from(height) * 4;

    let mut resources = Resources::new(&data.next);
    // SAFETY: every object is created on this device, used once and destroyed by
    // `resources` when it drops; the submission waits on the application's semaphores and
    // finishes (fence wait) before the image is presented.
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
        if let Some(set_loader_data) = state.set_loader_data {
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
        let memory_type = memory_type(&state.memory, requirements.memory_type_bits, wanted)
            .ok_or(CaptureError::NoMemory)?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )?;
        resources.memory = memory;
        device.bind_buffer_memory(buffer, memory, 0)?;
        record_copy(device, command_buffer, image, buffer, swapchain.extent)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        resources.fence = fence;

        let waits = if info.p_wait_semaphores.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
        };
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let command_buffers = [command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&command_buffers);
        device.queue_submit(queue, &[submit], fence)?;
        *consumed = true;
        device.wait_for_fences(&[fence], true, u64::MAX)?;

        let mapped = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
        let mut pixels = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize).to_vec();
        device.unmap_memory(memory);
        if swizzle {
            let (bgra, _) = pixels.as_chunks_mut::<4>();
            bgra.iter_mut().for_each(|pixel| pixel.swap(0, 2));
        }
        Ok(Rgba {
            width,
            height,
            pixels,
        })
    }
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

/// The first memory type allowed by `bits` with all of `wanted`.
fn memory_type(
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

/// Objects a capture creates, destroyed in reverse order when it ends.
struct Resources<'a> {
    device: &'a ash::Device,
    pool: vk::CommandPool,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    fence: vk::Fence,
}

impl<'a> Resources<'a> {
    fn new(device: &'a ash::Device) -> Self {
        Self {
            device,
            pool: vk::CommandPool::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            fence: vk::Fence::null(),
        }
    }
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        // SAFETY: each handle is null or was created by the capture on this device, and
        // the GPU is done with them: either the fence wait returned or nothing was
        // submitted. Destroying null handles is a no-op.
        unsafe {
            self.device.destroy_fence(self.fence, None);
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
    }
}
