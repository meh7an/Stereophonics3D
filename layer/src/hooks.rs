//! Intercepted Vulkan entry points.
//!
//! Every hook runs its own logic inside `ffi_guard`, so a panic becomes an error result
//! instead of unwinding into the game. Calls down the chain stay outside any Rust logic
//! that could panic.

use std::{
    ffi::{CStr, c_char},
    mem::transmute,
    sync::atomic::Ordering,
};

use ash::vk;
use s3d_shared::ffi::ffi_guard;

use crate::{
    activation,
    dispatch::{self, DeviceData, InstanceData, key},
    loader,
};

/// Frames between progress lines in the log.
const FRAME_LOG_INTERVAL: u64 = 600;

/// Casts a function returned by a `GetProcAddr` to its real signature; only used inside
/// unsafe blocks, where the loader guarantees the named entry point has signature `$ty`.
macro_rules! cast {
    ($function:expr, $ty:ty) => {
        $function.map(|f| transmute::<unsafe extern "system" fn(), $ty>(f))
    };
}

/// Exposes one of this layer's hooks as a `PFN_vkVoidFunction`.
macro_rules! expose {
    ($function:expr, $ty:ty) => {
        // SAFETY: function pointers of every signature have the same representation; the
        // loader casts it back to `$ty` before calling.
        Some(unsafe { transmute::<$ty, unsafe extern "system" fn()>($function) })
    };
}

pub unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    ffi_guard("vkGetInstanceProcAddr", None, || {
        // SAFETY: the loader passes a nul-terminated entry-point name.
        let wanted = unsafe { CStr::from_ptr(name) };
        if let Some(own) = own_instance_function(wanted) {
            return own;
        }
        if instance == vk::Instance::null() {
            return None;
        }
        // SAFETY: a non-null instance handle passed by the loader is live.
        let data = dispatch::instance(unsafe { key(instance) })?;
        if let Some(own) = own_device_function(wanted, data.active) {
            return own;
        }
        // SAFETY: the next layer's GetInstanceProcAddr with its own instance.
        unsafe { (data.next_get_instance_proc_addr)(instance, name) }
    })
}

pub unsafe extern "system" fn get_device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    ffi_guard("vkGetDeviceProcAddr", None, || {
        // SAFETY: the loader passes a nul-terminated entry-point name.
        let wanted = unsafe { CStr::from_ptr(name) };
        // SAFETY: the loader passes a live device handle.
        let data = dispatch::device(unsafe { key(device) })?;
        if let Some(own) = own_device_function(wanted, data.active) {
            return own;
        }
        // SAFETY: the next layer's GetDeviceProcAddr with its own device.
        unsafe { (data.next_get_device_proc_addr)(device, name) }
    })
}

/// Instance-level entry points this layer always implements.
fn own_instance_function(name: &CStr) -> Option<vk::PFN_vkVoidFunction> {
    Some(match name.to_bytes() {
        b"vkGetInstanceProcAddr" => {
            expose!(get_instance_proc_addr, vk::PFN_vkGetInstanceProcAddr)
        }
        b"vkCreateInstance" => expose!(create_instance, vk::PFN_vkCreateInstance),
        b"vkDestroyInstance" => expose!(destroy_instance, vk::PFN_vkDestroyInstance),
        b"vkCreateDevice" => expose!(create_device, vk::PFN_vkCreateDevice),
        _ => return None,
    })
}

/// Device-level entry points; the present hook only exists when the gate is open, so an
/// inactive process calls the driver directly.
fn own_device_function(name: &CStr, active: bool) -> Option<vk::PFN_vkVoidFunction> {
    Some(match name.to_bytes() {
        b"vkGetDeviceProcAddr" => expose!(get_device_proc_addr, vk::PFN_vkGetDeviceProcAddr),
        b"vkDestroyDevice" => expose!(destroy_device, vk::PFN_vkDestroyDevice),
        b"vkQueuePresentKHR" if active => {
            expose!(queue_present, vk::PFN_vkQueuePresentKHR)
        }
        _ => return None,
    })
}

unsafe extern "system" fn create_instance(
    create_info: *const vk::InstanceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    instance: *mut vk::Instance,
) -> vk::Result {
    ffi_guard(
        "vkCreateInstance",
        vk::Result::ERROR_INITIALIZATION_FAILED,
        || {
            // SAFETY: the loader passes a valid create info whose chain holds our link,
            // and expects each layer to advance it before calling down.
            unsafe {
                let Some(link) = loader::instance_link((*create_info).p_next) else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                let layer = (*link).u.layer_info;
                let Some(next_gipa) = (*layer).pfn_next_get_instance_proc_addr else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                (*link).u.layer_info = (*layer).p_next;

                let next_create = cast!(
                    next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()),
                    vk::PFN_vkCreateInstance
                );
                let Some(next_create) = next_create else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                let result = next_create(create_info, allocator, instance);
                if result != vk::Result::SUCCESS {
                    return result;
                }
                let handle = *instance;
                let destroy = cast!(
                    next_gipa(handle, c"vkDestroyInstance".as_ptr()),
                    vk::PFN_vkDestroyInstance
                );
                let Some(destroy) = destroy else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                dispatch::insert_instance(
                    key(handle),
                    InstanceData {
                        handle,
                        next_get_instance_proc_addr: next_gipa,
                        destroy,
                        active: activation::get().active,
                    },
                );
                result
            }
        },
    )
}

unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    if instance == vk::Instance::null() {
        return;
    }
    // SAFETY: a live instance handle from the application.
    let data = ffi_guard("vkDestroyInstance", None, || {
        dispatch::remove_instance(unsafe { key(instance) })
    });
    if let Some(data) = data {
        // SAFETY: the next layer's destroy with the handle it created.
        unsafe { (data.destroy)(data.handle, allocator) };
    }
}

unsafe extern "system" fn create_device(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    ffi_guard(
        "vkCreateDevice",
        vk::Result::ERROR_INITIALIZATION_FAILED,
        || {
            // SAFETY: as in `create_instance`, for the device chain.
            unsafe {
                let Some(link) = loader::device_link((*create_info).p_next) else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                let layer = (*link).u.layer_info;
                let (Some(next_gipa), Some(next_gdpa)) = (
                    (*layer).pfn_next_get_instance_proc_addr,
                    (*layer).pfn_next_get_device_proc_addr,
                ) else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                (*link).u.layer_info = (*layer).p_next;

                let instance = dispatch::instance(key(physical_device));
                let owner = instance.as_ref().map_or(vk::Instance::null(), |i| i.handle);
                let next_create = cast!(
                    next_gipa(owner, c"vkCreateDevice".as_ptr()),
                    vk::PFN_vkCreateDevice
                );
                let Some(next_create) = next_create else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                let result = next_create(physical_device, create_info, allocator, device);
                if result != vk::Result::SUCCESS {
                    return result;
                }
                let handle = *device;
                let destroy = cast!(
                    next_gdpa(handle, c"vkDestroyDevice".as_ptr()),
                    vk::PFN_vkDestroyDevice
                );
                let Some(destroy) = destroy else {
                    return vk::Result::ERROR_INITIALIZATION_FAILED;
                };
                let queue_present = cast!(
                    next_gdpa(handle, c"vkQueuePresentKHR".as_ptr()),
                    vk::PFN_vkQueuePresentKHR
                );
                let active = instance.is_some_and(|i| i.active);
                tracing::info!(active, "device created");
                dispatch::insert_device(
                    key(handle),
                    DeviceData {
                        handle,
                        next_get_device_proc_addr: next_gdpa,
                        destroy,
                        queue_present,
                        active,
                        frames: Default::default(),
                    },
                );
                result
            }
        },
    )
}

unsafe extern "system" fn destroy_device(
    device: vk::Device,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    if device == vk::Device::null() {
        return;
    }
    // SAFETY: a live device handle from the application.
    let data = ffi_guard("vkDestroyDevice", None, || {
        dispatch::remove_device(unsafe { key(device) })
    });
    if let Some(data) = data {
        let frames = data.frames.load(Ordering::Relaxed);
        tracing::info!(frames, "device destroyed");
        // SAFETY: the next layer's destroy with the handle it created.
        unsafe { (data.destroy)(data.handle, allocator) };
    }
}

unsafe extern "system" fn queue_present(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    let next = ffi_guard("vkQueuePresentKHR", None, || {
        // SAFETY: queues share their device's dispatch key.
        let data = dispatch::device(unsafe { key(queue) })?;
        let frame = data.count_frame();
        if frame == 1 || frame.is_multiple_of(FRAME_LOG_INTERVAL) {
            tracing::info!(frame, "presented");
        }
        data.queue_present
    });
    match next {
        // SAFETY: the next layer's present with the application's arguments.
        Some(present) => unsafe { present(queue, present_info) },
        None => vk::Result::ERROR_DEVICE_LOST,
    }
}
