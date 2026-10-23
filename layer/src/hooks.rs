//! Loader-facing entry points: `GetProcAddr`, instance and device lifetime, present.
//!
//! Every hook runs its own logic inside `ffi_guard`, so a panic becomes an error result
//! instead of unwinding into the game. Calls down the chain stay outside any Rust logic
//! that could panic.

use std::{
    ffi::{CStr, c_char, c_void},
    sync::{Arc, atomic::Ordering},
};

use ash::vk::{self, Handle};
use s3d_shared::ffi::ffi_guard;

use crate::{
    activation, commands, controls,
    dispatch::{self, DeviceData, InstanceData, key},
    extensions::{self, InstanceFns},
    loader, pipelines, present,
    stereo::{self, StereoState},
    swapchain::{self, DeviceContext, SwapchainState},
};

/// Frames between progress lines in the log.
pub const FRAME_LOG_INTERVAL: u64 = 600;

function_table! {
    /// Device-level functions the layer replaces for active processes.
    DeviceFns, vk::Device, vk::PFN_vkGetDeviceProcAddr {
        queue_present: c"vkQueuePresentKHR" => vk::PFN_vkQueuePresentKHR = queue_present,
        get_device_queue: c"vkGetDeviceQueue"
            => vk::PFN_vkGetDeviceQueue = swapchain::get_device_queue,
        get_device_queue2: c"vkGetDeviceQueue2"
            => vk::PFN_vkGetDeviceQueue2 = swapchain::get_device_queue2,
        create_swapchain: c"vkCreateSwapchainKHR"
            => vk::PFN_vkCreateSwapchainKHR = swapchain::create_swapchain,
        destroy_swapchain: c"vkDestroySwapchainKHR"
            => vk::PFN_vkDestroySwapchainKHR = swapchain::destroy_swapchain,
        get_swapchain_images: c"vkGetSwapchainImagesKHR"
            => vk::PFN_vkGetSwapchainImagesKHR = swapchain::get_swapchain_images,
        create_shader_module: c"vkCreateShaderModule"
            => vk::PFN_vkCreateShaderModule = pipelines::create_shader_module,
        destroy_shader_module: c"vkDestroyShaderModule"
            => vk::PFN_vkDestroyShaderModule = pipelines::destroy_shader_module,
        create_pipeline_layout: c"vkCreatePipelineLayout"
            => vk::PFN_vkCreatePipelineLayout = pipelines::create_pipeline_layout,
        destroy_pipeline_layout: c"vkDestroyPipelineLayout"
            => vk::PFN_vkDestroyPipelineLayout = pipelines::destroy_pipeline_layout,
        create_graphics_pipelines: c"vkCreateGraphicsPipelines"
            => vk::PFN_vkCreateGraphicsPipelines = pipelines::create_graphics_pipelines,
        destroy_pipeline: c"vkDestroyPipeline"
            => vk::PFN_vkDestroyPipeline = pipelines::destroy_pipeline,
        create_pipeline_cache: c"vkCreatePipelineCache"
            => vk::PFN_vkCreatePipelineCache = pipelines::create_pipeline_cache,
        begin_command_buffer: c"vkBeginCommandBuffer"
            => vk::PFN_vkBeginCommandBuffer = commands::begin_command_buffer,
        end_command_buffer: c"vkEndCommandBuffer"
            => vk::PFN_vkEndCommandBuffer = commands::end_command_buffer,
        free_command_buffers: c"vkFreeCommandBuffers"
            => vk::PFN_vkFreeCommandBuffers = commands::free_command_buffers,
        cmd_bind_pipeline: c"vkCmdBindPipeline"
            => vk::PFN_vkCmdBindPipeline = commands::cmd_bind_pipeline,
        cmd_push_constants: c"vkCmdPushConstants"
            => vk::PFN_vkCmdPushConstants = commands::cmd_push_constants,
        cmd_draw: c"vkCmdDraw" => vk::PFN_vkCmdDraw = commands::cmd_draw,
        cmd_draw_indexed: c"vkCmdDrawIndexed"
            => vk::PFN_vkCmdDrawIndexed = commands::cmd_draw_indexed,
        cmd_draw_indirect: c"vkCmdDrawIndirect"
            => vk::PFN_vkCmdDrawIndirect = commands::cmd_draw_indirect,
        cmd_draw_indexed_indirect: c"vkCmdDrawIndexedIndirect"
            => vk::PFN_vkCmdDrawIndexedIndirect = commands::cmd_draw_indexed_indirect,
        cmd_draw_indirect_count: c"vkCmdDrawIndirectCount"
            => vk::PFN_vkCmdDrawIndirectCount = commands::cmd_draw_indirect_count,
        cmd_draw_indirect_count_khr: c"vkCmdDrawIndirectCountKHR"
            => vk::PFN_vkCmdDrawIndirectCount = commands::cmd_draw_indirect_count_khr,
        cmd_draw_indirect_count_amd: c"vkCmdDrawIndirectCountAMD"
            => vk::PFN_vkCmdDrawIndirectCount = commands::cmd_draw_indirect_count_amd,
        cmd_draw_indexed_indirect_count: c"vkCmdDrawIndexedIndirectCount"
            => vk::PFN_vkCmdDrawIndexedIndirectCount
            = commands::cmd_draw_indexed_indirect_count,
        cmd_draw_indexed_indirect_count_khr: c"vkCmdDrawIndexedIndirectCountKHR"
            => vk::PFN_vkCmdDrawIndexedIndirectCount
            = commands::cmd_draw_indexed_indirect_count_khr,
        cmd_draw_indexed_indirect_count_amd: c"vkCmdDrawIndexedIndirectCountAMD"
            => vk::PFN_vkCmdDrawIndexedIndirectCount
            = commands::cmd_draw_indexed_indirect_count_amd,
        cmd_draw_mesh_tasks_ext: c"vkCmdDrawMeshTasksEXT"
            => vk::PFN_vkCmdDrawMeshTasksEXT = commands::cmd_draw_mesh_tasks_ext,
        cmd_draw_mesh_tasks_indirect_ext: c"vkCmdDrawMeshTasksIndirectEXT"
            => vk::PFN_vkCmdDrawMeshTasksIndirectEXT
            = commands::cmd_draw_mesh_tasks_indirect_ext,
        cmd_draw_mesh_tasks_indirect_count_ext: c"vkCmdDrawMeshTasksIndirectCountEXT"
            => vk::PFN_vkCmdDrawMeshTasksIndirectCountEXT
            = commands::cmd_draw_mesh_tasks_indirect_count_ext,
        cmd_draw_mesh_tasks_nv: c"vkCmdDrawMeshTasksNV"
            => vk::PFN_vkCmdDrawMeshTasksNV = commands::cmd_draw_mesh_tasks_nv,
        cmd_draw_mesh_tasks_indirect_nv: c"vkCmdDrawMeshTasksIndirectNV"
            => vk::PFN_vkCmdDrawMeshTasksIndirectNV
            = commands::cmd_draw_mesh_tasks_indirect_nv,
        cmd_draw_mesh_tasks_indirect_count_nv: c"vkCmdDrawMeshTasksIndirectCountNV"
            => vk::PFN_vkCmdDrawMeshTasksIndirectCountNV
            = commands::cmd_draw_mesh_tasks_indirect_count_nv,
        cmd_draw_multi_ext: c"vkCmdDrawMultiEXT"
            => vk::PFN_vkCmdDrawMultiEXT = commands::cmd_draw_multi_ext,
        cmd_draw_multi_indexed_ext: c"vkCmdDrawMultiIndexedEXT"
            => vk::PFN_vkCmdDrawMultiIndexedEXT = commands::cmd_draw_multi_indexed_ext,
        cmd_draw_indirect_byte_count_ext: c"vkCmdDrawIndirectByteCountEXT"
            => vk::PFN_vkCmdDrawIndirectByteCountEXT
            = commands::cmd_draw_indirect_byte_count_ext,
    }
}

/// The device data behind a dispatchable `handle` and the next layer's function that
/// `select` picks, or `None` when either is missing.
///
/// # Safety
///
/// `handle` must be a live device, queue or command buffer.
pub unsafe fn next_device_fn<T>(
    entry_point: &'static str,
    handle: impl Handle,
    select: impl FnOnce(&DeviceFns) -> Option<T>,
) -> Option<(Arc<DeviceData>, T)> {
    ffi_guard(entry_point, None, || {
        // SAFETY: forwarded from the caller.
        let data = dispatch::device(unsafe { key(handle) })?;
        let next = select(&data.fns)?;
        Some((data, next))
    })
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
        if data.active
            && let Some(own) = data.fns.hook(wanted)
        {
            return own;
        }
        if let Some(own) = own_device_function(wanted) {
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
        if let Some(own) = own_device_function(wanted) {
            return own;
        }
        // An inactive process calls the driver directly for everything else.
        if data.active
            && let Some(own) = data.fns.hook(wanted)
        {
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

/// Device-level entry points the layer needs whether or not it is active.
fn own_device_function(name: &CStr) -> Option<vk::PFN_vkVoidFunction> {
    Some(match name.to_bytes() {
        b"vkGetDeviceProcAddr" => expose!(get_device_proc_addr, vk::PFN_vkGetDeviceProcAddr),
        b"vkDestroyDevice" => expose!(destroy_device, vk::PFN_vkDestroyDevice),
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
                        fns: InstanceFns::load(next_gipa, handle),
                        next: ash::Instance::load_with(
                            |name| as_void(next_gipa(handle, name.as_ptr())),
                            handle,
                        ),
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
                let set_loader_data = loader::device_loader_data((*create_info).p_next);
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
                let active = instance.as_ref().is_some_and(|i| i.active);
                if active {
                    let hidden = extensions::requested_hidden(&*create_info);
                    if !hidden.is_empty() {
                        tracing::warn!(?hidden, "application enables hidden extensions");
                    }
                }
                let properties = instance.as_ref().map_or_else(Default::default, |i| {
                    i.next.get_physical_device_properties(physical_device)
                });
                let context = instance.as_ref().map(|i| DeviceContext {
                    instance: i.next.clone(),
                    physical_device,
                    memory: i
                        .next
                        .get_physical_device_memory_properties(physical_device),
                    families: i
                        .next
                        .get_physical_device_queue_family_properties(physical_device),
                    requested_families: requested_families(&*create_info),
                    timestamp_period: properties.limits.timestamp_period,
                    surface_fn: ash::khr::surface::InstanceFn::load(|name| {
                        as_void(next_gipa(owner, name.as_ptr()))
                    }),
                });
                let enabled = extensions::enabled(&*create_info);
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
                tracing::info!(active, "device created");
                dispatch::insert_device(
                    key(handle),
                    DeviceData {
                        handle,
                        next_get_device_proc_addr: next_gdpa,
                        destroy,
                        fns: DeviceFns::load(next_gdpa, handle),
                        next: ash::Device::load_with(
                            |name| as_void(next_gdpa(handle, name.as_ptr())),
                            handle,
                        ),
                        active,
                        frames: Default::default(),
                        tracker: Default::default(),
                        commands: Default::default(),
                        stereo: StereoState::new(
                            &properties,
                            &enabled,
                            stereo::layout_rules(&*create_info, properties.api_version, &enabled),
                        ),
                        swapchains: SwapchainState::new(
                            context,
                            set_loader_data,
                            ash::khr::swapchain::DeviceFn::load(|name| {
                                as_void(next_gdpa(handle, name.as_ptr()))
                            }),
                        ),
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
        let (modules, layouts, pipelines) = data.tracker.counts();
        tracing::info!(frames, modules, layouts, pipelines, "device destroyed");
        ffi_guard("vkDestroyDevice", (), || data.stereo.destroy(&data.next));
        // SAFETY: the device is being destroyed; the swapchain state owns its objects.
        ffi_guard("vkDestroyDevice", (), || unsafe {
            data.swapchains.destroy(&data.next)
        });
        // SAFETY: the next layer's destroy with the handle it created.
        unsafe { (data.destroy)(data.handle, allocator) };
    }
}

/// The queue families a device create info asks for.
///
/// # Safety
///
/// `info` must be a valid device create info.
unsafe fn requested_families(info: &vk::DeviceCreateInfo<'_>) -> Vec<u32> {
    if info.p_queue_create_infos.is_null() {
        return Vec::new();
    }
    // SAFETY: a valid create info holds this many queue create infos.
    let queues = unsafe {
        std::slice::from_raw_parts(
            info.p_queue_create_infos,
            info.queue_create_info_count as usize,
        )
    };
    queues
        .iter()
        .map(|queue| queue.queue_family_index)
        .collect()
}

/// A `GetProcAddr` result as the untyped pointer `ash` loads tables from.
fn as_void(function: vk::PFN_vkVoidFunction) -> *const c_void {
    function.map_or(std::ptr::null(), |f| f as *const c_void)
}

unsafe extern "system" fn queue_present(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    let found = ffi_guard("vkQueuePresentKHR", None, || {
        // SAFETY: queues share their device's dispatch key.
        let data = dispatch::device(unsafe { key(queue) })?;
        let frame = data.count_frame();
        if frame == 1 || frame.is_multiple_of(FRAME_LOG_INTERVAL) {
            tracing::info!(frame, "presented");
        }
        let next = data.fns.queue_present?;
        Some((data, next, frame - 1))
    });
    let Some((data, next, index)) = found else {
        return vk::Result::ERROR_DEVICE_LOST;
    };
    ffi_guard("vkQueuePresentKHR", (), || controls::poll(index));
    // SAFETY: the application's present arguments, on the queue it synchronises.
    let waits = ffi_guard("vkQueuePresentKHR", None, || unsafe {
        present::prepare(&data, queue, &*present_info, index)
    });
    match waits {
        Some(waits) => {
            // The layer's submission already waited on the application's semaphores.
            // SAFETY: a copy of the application's present info waiting on the layer's.
            let info = unsafe { *present_info }.wait_semaphores(&waits);
            // SAFETY: the next layer's present with that copy.
            unsafe { next(queue, &info) }
        }
        // SAFETY: the next layer's present with the application's arguments.
        None => unsafe { next(queue, present_info) },
    }
}
