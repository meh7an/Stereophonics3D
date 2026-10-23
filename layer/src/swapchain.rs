//! Swapchain interception: the application's queues and swapchains, and the virtual
//! swapchain that hands the application layer-owned images.
//!
//! A swapchain is virtualised when the layer can composite into it: the real swapchain
//! gets `COLOR_ATTACHMENT` (and `TRANSFER_SRC` for captures) added, the application gets
//! one virtual image per real image, and the present hook draws each presented virtual
//! image into its real image first. A swapchain that cannot be virtualised keeps its real
//! images and is presented directly.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use ash::vk;
use s3d_shared::ffi::ffi_guard;

use crate::{
    compositor::{Compositor, virtual_image_info},
    dispatch::{DeviceData, read, write},
    hooks::next_device_fn,
    loader::SetDeviceLoaderData,
    present::{FenceRing, Timing},
};

/// Usage the layer adds to real swapchains: the compositor draws into them and captures
/// copy from them.
const LAYER_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);

/// Physical-device facts the swapchain and present hooks need, taken at device creation.
pub struct DeviceContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub families: Vec<vk::QueueFamilyProperties>,
    /// Queue families the device was created with.
    pub requested_families: Vec<u32>,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    pub surface_fn: ash::khr::surface::InstanceFn,
}

impl DeviceContext {
    fn has_graphics(&self, family: u32) -> bool {
        self.families
            .get(family as usize)
            .is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
    }
}

/// One application swapchain.
pub struct Swapchain {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    /// Usage the real images were created with.
    pub usage: vk::ImageUsageFlags,
    /// The real images, owned by the driver.
    pub images: Vec<vk::Image>,
    /// Present when the swapchain is virtualised.
    pub compositor: Option<Compositor>,
}

/// Per-device swapchain and present state.
pub struct SwapchainState {
    /// `None` when the device's instance is unknown; nothing is virtualised then.
    pub context: Option<DeviceContext>,
    pub set_loader_data: Option<SetDeviceLoaderData>,
    swapchain_fn: ash::khr::swapchain::DeviceFn,
    /// Queue family of every queue the application retrieved.
    queues: RwLock<HashMap<vk::Queue, u32>>,
    swapchains: RwLock<HashMap<vk::SwapchainKHR, Arc<Swapchain>>>,
    /// Fences of the layer's present-time submissions.
    pub ring: Mutex<FenceRing>,
    pub timing: Mutex<Timing>,
}

impl SwapchainState {
    pub fn new(
        context: Option<DeviceContext>,
        set_loader_data: Option<SetDeviceLoaderData>,
        swapchain_fn: ash::khr::swapchain::DeviceFn,
    ) -> Self {
        Self {
            context,
            set_loader_data,
            swapchain_fn,
            queues: RwLock::new(HashMap::new()),
            swapchains: RwLock::new(HashMap::new()),
            ring: Mutex::new(FenceRing::default()),
            timing: Mutex::new(Timing::default()),
        }
    }

    fn record_queue(&self, queue: vk::Queue, family: u32) {
        if queue != vk::Queue::null() {
            write(&self.queues).insert(queue, family);
        }
    }

    /// The family of a queue the application retrieved.
    pub fn queue_family(&self, queue: vk::Queue) -> Option<u32> {
        read(&self.queues).get(&queue).copied()
    }

    pub fn swapchain(&self, handle: vk::SwapchainKHR) -> Option<Arc<Swapchain>> {
        read(&self.swapchains).get(&handle).cloned()
    }

    /// Waits for the layer's submissions and destroys everything it still owns.
    ///
    /// # Safety
    ///
    /// `device` must be this state's device, about to be destroyed.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        let mut ring = self.ring.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: forwarded from the caller; after the wait no composite is executing.
        unsafe {
            ring.wait_all(device);
            for swapchain in std::mem::take(&mut *write(&self.swapchains)).into_values() {
                if let Some(compositor) = &swapchain.compositor {
                    compositor.destroy(device);
                }
            }
            ring.destroy(device);
        }
    }

    /// Why a swapchain created from `info` keeps its real images, or `None` when it can
    /// be virtualised.
    ///
    /// # Safety
    ///
    /// `info` must be a valid swapchain create info for this state's device.
    unsafe fn refusal(&self, info: &vk::SwapchainCreateInfoKHR<'_>) -> Option<String> {
        if let Some(reason) = unsupported(info) {
            return Some(reason.to_owned());
        }
        let Some(context) = &self.context else {
            return Some("the device's instance is unknown".to_owned());
        };
        let image = virtual_image_info(info);
        // SAFETY: a live physical device of the instance the table was loaded from.
        let supported = unsafe {
            context
                .instance
                .get_physical_device_image_format_properties(
                    context.physical_device,
                    image.format,
                    image.image_type,
                    image.tiling,
                    image.usage,
                    image.flags,
                )
        };
        match supported {
            Ok(limits)
                if limits.max_extent.width >= image.extent.width
                    && limits.max_extent.height >= image.extent.height
                    && limits.max_array_layers >= image.array_layers => {}
            Ok(_) => return Some("the format's image limits are too small".to_owned()),
            Err(error) => return Some(format!("{:?} cannot be sampled ({error})", image.format)),
        }
        // The composite runs on the presenting queue, so every family that can present to
        // this surface must be able to draw.
        let support = context.surface_fn.get_physical_device_surface_support_khr;
        for &family in &context.requested_families {
            let mut presents = vk::FALSE;
            // SAFETY: a live surface and a family the device was created with.
            let result =
                unsafe { support(context.physical_device, family, info.surface, &mut presents) };
            if result != vk::Result::SUCCESS {
                return Some(format!("surface support query failed ({result})"));
            }
            if presents == vk::TRUE && !context.has_graphics(family) {
                return Some(format!("queue family {family} presents without graphics"));
            }
        }
        None
    }
}

/// Swapchain kinds the compositor does not handle, judged from the create info alone.
fn unsupported(info: &vk::SwapchainCreateInfoKHR<'_>) -> Option<&'static str> {
    let flags = info.flags & !vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT;
    if !flags.is_empty() {
        return Some("protected or split-instance swapchain");
    }
    if info.image_array_layers != 1 {
        return Some("layered swapchain");
    }
    None
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
        data.swapchains.record_queue(*queue, family)
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
        data.swapchains
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
    // SAFETY: the application's valid create info.
    let original = unsafe { *info };
    let refusal = ffi_guard(
        "vkCreateSwapchainKHR",
        Some("internal error".to_owned()),
        || {
            // SAFETY: as above.
            unsafe { data.swapchains.refusal(&original) }
        },
    );
    let extended = original.image_usage(original.image_usage | LAYER_USAGE);
    // SAFETY: the next layer's create with a copy of the application's info.
    let mut result = unsafe { next(device, &extended, allocator, swapchain) };
    let mut created = extended;
    if result != vk::Result::SUCCESS && extended.image_usage != original.image_usage {
        tracing::warn!(%result, "driver refused the layer's swapchain usage, retrying as is");
        // SAFETY: the next layer's create with the application's arguments.
        result = unsafe { next(device, info, allocator, swapchain) };
        created = original;
    }
    if result == vk::Result::SUCCESS {
        let refusal = if created.image_usage.contains(LAYER_USAGE) {
            refusal
        } else {
            Some("the driver refused the layer's usage".to_owned())
        };
        ffi_guard("vkCreateSwapchainKHR", (), || {
            // SAFETY: the swapchain was just created on this device from `created`.
            unsafe { register(&data, *swapchain, &created, refusal) }
        });
    }
    result
}

/// Records a new swapchain and, unless `refusal` says why not, virtualises it.
///
/// # Safety
///
/// `handle` must be a live swapchain of `data`'s device created from `info`.
unsafe fn register(
    data: &DeviceData,
    handle: vk::SwapchainKHR,
    info: &vk::SwapchainCreateInfoKHR<'_>,
    refusal: Option<String>,
) {
    let state = &data.swapchains;
    // SAFETY: forwarded from the caller.
    let images = match unsafe { real_images(state, data.handle, handle) } {
        Ok(images) => images,
        Err(error) => {
            tracing::warn!(%error, "cannot list swapchain images");
            return;
        }
    };
    let format = info.image_format;
    let vk::Extent2D { width, height } = info.image_extent;
    let compositor = match (refusal, &state.context) {
        (Some(reason), _) => {
            tracing::info!(reason, "swapchain keeps its real images");
            None
        }
        (None, None) => None,
        // SAFETY: forwarded from the caller; `refusal` checked the info is supported.
        (None, Some(context)) => {
            match unsafe { Compositor::new(&data.next, context, info, &images) } {
                Ok(compositor) => {
                    tracing::info!(
                        images = images.len(),
                        ?format,
                        width,
                        height,
                        "virtual swapchain"
                    );
                    Some(compositor)
                }
                Err(error) => {
                    tracing::warn!(%error, "virtual swapchain unavailable, presenting directly");
                    None
                }
            }
        }
    };
    let swapchain = Arc::new(Swapchain {
        format,
        extent: info.image_extent,
        usage: info.image_usage,
        images,
        compositor,
    });
    write(&state.swapchains).insert(handle, swapchain);
}

/// The real images of a swapchain, from the next layer.
///
/// # Safety
///
/// `swapchain` must be a live swapchain of `device`.
unsafe fn real_images(
    state: &SwapchainState,
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> Result<Vec<vk::Image>, vk::Result> {
    let get = state.swapchain_fn.get_swapchain_images_khr;
    let mut count = 0;
    // SAFETY: a count query on a live swapchain.
    unsafe { get(device, swapchain, &mut count, std::ptr::null_mut()) }.result()?;
    let mut images = vec![vk::Image::null(); count as usize];
    // SAFETY: `images` holds `count` handles.
    unsafe { get(device, swapchain, &mut count, images.as_mut_ptr()) }.result()?;
    images.truncate(count as usize);
    Ok(images)
}

pub unsafe extern "system" fn get_swapchain_images(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found = unsafe {
        next_device_fn("vkGetSwapchainImagesKHR", device, |f| {
            f.get_swapchain_images
        })
    };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let own = ffi_guard("vkGetSwapchainImagesKHR", None, || {
        let swapchain = data.swapchains.swapchain(swapchain)?;
        Some(swapchain.compositor.as_ref()?.images())
    });
    let Some(own) = own else {
        // SAFETY: the next layer's query with the application's arguments.
        return unsafe { next(device, swapchain, count, images) };
    };
    // SAFETY: the application passes a writable count and, if not null, room for `*count`
    // handles, as the query's contract requires.
    unsafe {
        if images.is_null() {
            *count = own.len() as u32;
            return vk::Result::SUCCESS;
        }
        let written = (*count as usize).min(own.len());
        std::ptr::copy_nonoverlapping(own.as_ptr(), images, written);
        *count = written as u32;
        if written < own.len() {
            vk::Result::INCOMPLETE
        } else {
            vk::Result::SUCCESS
        }
    }
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
        let state = &data.swapchains;
        let Some(removed) = write(&state.swapchains).remove(&swapchain) else {
            return;
        };
        if let Some(compositor) = &removed.compositor {
            // The application only waits for its own work; composites it never saw may
            // still read the virtual images or write the real ones.
            // SAFETY: after the wait no composite of this swapchain is executing, and the
            // real swapchain is destroyed only below.
            unsafe {
                state
                    .ring
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .wait_all(&data.next);
                compositor.destroy(&data.next);
            }
        }
    });
    // SAFETY: the next layer's destroy with the application's arguments.
    unsafe { next(device, swapchain, allocator) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_accepts_plain_and_mutable_format_swapchains() {
        let plain = vk::SwapchainCreateInfoKHR::default().image_array_layers(1);
        assert_eq!(unsupported(&plain), None);
        let mutable = plain.flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT);
        assert_eq!(unsupported(&mutable), None);
    }

    #[test]
    fn test_unsupported_refuses_protected_and_layered_swapchains() {
        let protected = vk::SwapchainCreateInfoKHR::default()
            .image_array_layers(1)
            .flags(vk::SwapchainCreateFlagsKHR::PROTECTED);
        assert!(unsupported(&protected).is_some());
        let layered = vk::SwapchainCreateInfoKHR::default().image_array_layers(2);
        assert!(unsupported(&layered).is_some());
    }

    #[test]
    fn test_layer_usage_is_color_attachment_and_transfer_source() {
        assert_eq!(
            LAYER_USAGE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
        );
    }
}
