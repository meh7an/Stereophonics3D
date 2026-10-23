//! Vulkan implicit layer that injects same-frame geometric stereo.
//!
//! Negotiates loader interface version 2, keeps per-instance and per-device dispatch
//! data, and stays pure pass-through unless a profile opts the process in (and no
//! anti-cheat module is loaded). Active processes get their shader modules, pipelines and
//! command buffers tracked, interception-defeating extensions hidden, shaders optionally
//! dumped, and draws of profile-skipped shaders dropped. Their swapchains are virtualised:
//! the application renders into layer-owned images that a compositor draws into the real
//! swapchain at present, with an on-screen readout of the stereo values hotkeys adjust.

#[macro_use]
mod macros;

mod activation;
mod capture;
mod chain;
mod commands;
mod compositor;
mod controls;
mod dispatch;
mod dump;
mod extensions;
mod hooks;
mod loader;
mod overlay;
mod pipelines;
mod present;
mod readout;
mod stereo;
mod swapchain;

use std::ffi::c_char;

use ash::vk;
use s3d_shared::ffi::ffi_guard;

use crate::loader::{INTERFACE_VERSION, NEGOTIATE_INTERFACE_STRUCT, NegotiateLayerInterface};

/// Loader entry point: agrees on interface version 2 and hands over the layer's
/// `GetProcAddr` functions.
///
/// # Safety
///
/// Called by the Vulkan loader with a valid, writable negotiation structure.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    negotiation: *mut NegotiateLayerInterface,
) -> vk::Result {
    ffi_guard(
        "vkNegotiateLoaderLayerInterfaceVersion",
        vk::Result::ERROR_INITIALIZATION_FAILED,
        || {
            // SAFETY: guaranteed by the loader.
            let Some(negotiation) = (unsafe { negotiation.as_mut() }) else {
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            };
            if negotiation.s_type != NEGOTIATE_INTERFACE_STRUCT
                || negotiation.loader_layer_interface_version < INTERFACE_VERSION
            {
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            }
            negotiation.loader_layer_interface_version = INTERFACE_VERSION;
            negotiation.pfn_get_instance_proc_addr = Some(hooks::get_instance_proc_addr);
            negotiation.pfn_get_device_proc_addr = Some(hooks::get_device_proc_addr);
            negotiation.pfn_get_physical_device_proc_addr = None;
            vk::Result::SUCCESS
        },
    )
}

/// Exported for tools that resolve layer functions by name rather than negotiation.
///
/// # Safety
///
/// As `vkGetInstanceProcAddr`: a valid or null instance and a nul-terminated name.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: forwarded unchanged.
    unsafe { hooks::get_instance_proc_addr(instance, name) }
}

/// Exported for tools that resolve layer functions by name rather than negotiation.
///
/// # Safety
///
/// As `vkGetDeviceProcAddr`: a valid device and a nul-terminated name.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: forwarded unchanged.
    unsafe { hooks::get_device_proc_addr(device, name) }
}
