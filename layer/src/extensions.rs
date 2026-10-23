//! Hides from active processes the extensions whose shaders would bypass interception.
//!
//! Shader module identifiers and shader objects hand the driver shaders without a
//! `VkShaderModule` or pipeline the layer can patch, and graphics pipeline libraries move
//! pipeline assembly into pieces compiled before the layer sees the whole pipeline. Hiding
//! them steers applications onto the plain module-plus-pipeline path; both the extension
//! list and the matching feature bits are filtered, since either can be queried.

use std::{
    ffi::{CStr, c_char},
    sync::atomic::{AtomicBool, Ordering},
};

use ash::vk;
use s3d_shared::{ffi::ffi_guard, layer::LAYER_NAME};

use crate::{
    chain,
    dispatch::{self, key},
};

/// Extensions active processes never see.
pub const HIDDEN_EXTENSIONS: [&CStr; 3] = [
    ash::ext::shader_module_identifier::NAME,
    ash::ext::shader_object::NAME,
    ash::ext::graphics_pipeline_library::NAME,
];

/// Set once the hidden extensions have been logged, so the log names them once per
/// process rather than on every query.
static LOGGED: AtomicBool = AtomicBool::new(false);

function_table! {
    /// Instance-level functions the layer replaces for active processes.
    InstanceFns, vk::Instance, vk::PFN_vkGetInstanceProcAddr {
        enumerate_device_extension_properties: c"vkEnumerateDeviceExtensionProperties"
            => vk::PFN_vkEnumerateDeviceExtensionProperties
            = enumerate_device_extension_properties,
        get_physical_device_features2: c"vkGetPhysicalDeviceFeatures2"
            => vk::PFN_vkGetPhysicalDeviceFeatures2 = get_physical_device_features2,
        get_physical_device_features2_khr: c"vkGetPhysicalDeviceFeatures2KHR"
            => vk::PFN_vkGetPhysicalDeviceFeatures2 = get_physical_device_features2_khr,
    }
}

/// Whether `name` is one of [`HIDDEN_EXTENSIONS`].
pub fn is_hidden(name: &CStr) -> bool {
    HIDDEN_EXTENSIONS.contains(&name)
}

/// Removes the hidden extensions from `properties`, returning the names removed.
pub fn filter(properties: &mut Vec<vk::ExtensionProperties>) -> Vec<String> {
    let mut removed = Vec::new();
    properties.retain(|extension| match extension.extension_name_as_c_str() {
        Ok(name) if is_hidden(name) => {
            removed.push(name.to_string_lossy().into_owned());
            false
        }
        _ => true,
    });
    removed
}

/// The extensions a device create info enables.
///
/// # Safety
///
/// `info` must be a valid device create info; the names live as long as it does.
pub unsafe fn enabled<'a>(info: &vk::DeviceCreateInfo<'a>) -> Vec<&'a CStr> {
    if info.pp_enabled_extension_names.is_null() {
        return Vec::new();
    }
    // SAFETY: the application passes `enabled_extension_count` valid string pointers.
    let names = unsafe {
        std::slice::from_raw_parts(
            info.pp_enabled_extension_names,
            info.enabled_extension_count as usize,
        )
    };
    names
        .iter()
        // SAFETY: each entry is a nul-terminated extension name.
        .map(|&name| unsafe { CStr::from_ptr(name) })
        .collect()
}

/// Hidden extensions named in a device's enabled-extension list.
///
/// # Safety
///
/// `info` must be a valid device create info.
pub unsafe fn requested_hidden(info: &vk::DeviceCreateInfo<'_>) -> Vec<String> {
    // SAFETY: forwarded from the caller.
    unsafe { enabled(info) }
        .into_iter()
        .filter(|name| is_hidden(name))
        .map(|name| name.to_string_lossy().into_owned())
        .collect()
}

unsafe extern "system" fn enumerate_device_extension_properties(
    physical_device: vk::PhysicalDevice,
    layer_name: *const c_char,
    count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    ffi_guard(
        "vkEnumerateDeviceExtensionProperties",
        vk::Result::ERROR_INITIALIZATION_FAILED,
        || {
            // SAFETY: the application passes a live physical device of this instance.
            let data = dispatch::instance(unsafe { key(physical_device) });
            let Some(next) = data.and_then(|d| d.fns.enumerate_device_extension_properties) else {
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            };
            // SAFETY: arguments as received; `layer_name` is null or nul-terminated, and
            // `count`/`properties` follow the usual two-call contract.
            unsafe {
                if !layer_name.is_null() {
                    if CStr::from_ptr(layer_name).to_bytes() == LAYER_NAME.as_bytes() {
                        *count = 0;
                        return vk::Result::SUCCESS;
                    }
                    return next(physical_device, layer_name, count, properties);
                }
                let mut all = match read_all(next, physical_device) {
                    Ok(all) => all,
                    Err(error) => return error,
                };
                let hidden = filter(&mut all);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::info!(?hidden, "hiding device extensions");
                }
                copy_out(&all, count, properties)
            }
        },
    )
}

/// Every device extension the next layer reports.
///
/// # Safety
///
/// `next` must be the next layer's function and `physical_device` live.
unsafe fn read_all(
    next: vk::PFN_vkEnumerateDeviceExtensionProperties,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<vk::ExtensionProperties>, vk::Result> {
    loop {
        let mut count = 0;
        // SAFETY: a count query with valid out-pointers.
        let result = unsafe {
            next(
                physical_device,
                std::ptr::null(),
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(result);
        }
        let mut all = vec![vk::ExtensionProperties::default(); count as usize];
        // SAFETY: `all` holds `count` elements.
        let result = unsafe {
            next(
                physical_device,
                std::ptr::null(),
                &mut count,
                all.as_mut_ptr(),
            )
        };
        match result {
            vk::Result::SUCCESS => {
                all.truncate(count as usize);
                return Ok(all);
            }
            // The list grew between the calls; ask again.
            vk::Result::INCOMPLETE => {}
            error => return Err(error),
        }
    }
}

/// Answers a two-call enumeration from `all`: the count when `out` is null, otherwise as
/// many elements as fit, with `INCOMPLETE` when some did not.
///
/// # Safety
///
/// `count` must be valid; `out` null or valid for `*count` elements.
unsafe fn copy_out<T: Copy>(all: &[T], count: *mut u32, out: *mut T) -> vk::Result {
    // SAFETY: forwarded from the caller.
    let written = unsafe {
        if out.is_null() {
            *count = all.len() as u32;
            return vk::Result::SUCCESS;
        }
        let written = (*count as usize).min(all.len());
        std::ptr::copy_nonoverlapping(all.as_ptr(), out, written);
        *count = written as u32;
        written
    };
    if written < all.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}

unsafe extern "system" fn get_physical_device_features2(
    physical_device: vk::PhysicalDevice,
    features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    // SAFETY: arguments as received.
    unsafe {
        features2(physical_device, features, |fns| {
            fns.get_physical_device_features2
        })
    }
}

unsafe extern "system" fn get_physical_device_features2_khr(
    physical_device: vk::PhysicalDevice,
    features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    // SAFETY: arguments as received.
    unsafe {
        features2(physical_device, features, |fns| {
            fns.get_physical_device_features2_khr
        })
    }
}

/// Queries the next layer's features, then clears the hidden extensions' feature bits.
///
/// # Safety
///
/// As `vkGetPhysicalDeviceFeatures2`.
unsafe fn features2(
    physical_device: vk::PhysicalDevice,
    features: *mut vk::PhysicalDeviceFeatures2<'_>,
    select: fn(&InstanceFns) -> Option<vk::PFN_vkGetPhysicalDeviceFeatures2>,
) {
    let next = ffi_guard("vkGetPhysicalDeviceFeatures2", None, || {
        // SAFETY: the application passes a live physical device of this instance.
        dispatch::instance(unsafe { key(physical_device) }).and_then(|data| select(&data.fns))
    });
    let Some(next) = next else {
        return;
    };
    // SAFETY: the next layer's query with the application's arguments.
    unsafe { next(physical_device, features) };
    // SAFETY: `features` heads a valid, writable output chain.
    ffi_guard("vkGetPhysicalDeviceFeatures2", (), || unsafe {
        clear_hidden_features(features)
    });
}

/// Reports every hidden extension's feature as unsupported.
///
/// # Safety
///
/// `features` must head a valid, writable output chain.
unsafe fn clear_hidden_features(features: *mut vk::PhysicalDeviceFeatures2<'_>) {
    // SAFETY: each structure found is the type its sType names, and writable.
    unsafe {
        let next = (*features).p_next.cast_const();
        if let Some(found) = chain::find(
            next,
            vk::StructureType::PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        ) {
            let found = found
                .cast_mut()
                .cast::<vk::PhysicalDeviceShaderObjectFeaturesEXT<'_>>();
            (*found).shader_object = vk::FALSE;
        }
        if let Some(found) = chain::find(
            next,
            vk::StructureType::PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
        ) {
            let found = found
                .cast_mut()
                .cast::<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT<'_>>();
            (*found).graphics_pipeline_library = vk::FALSE;
        }
        if let Some(found) = chain::find(
            next,
            vk::StructureType::PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES_EXT,
        ) {
            let found = found
                .cast_mut()
                .cast::<vk::PhysicalDeviceShaderModuleIdentifierFeaturesEXT<'_>>();
            (*found).shader_module_identifier = vk::FALSE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extension(name: &CStr) -> vk::ExtensionProperties {
        let mut properties = vk::ExtensionProperties::default();
        for (slot, byte) in properties.extension_name.iter_mut().zip(name.to_bytes()) {
            *slot = *byte as c_char;
        }
        properties
    }

    #[test]
    fn test_filter_removes_exactly_the_hidden_extensions() {
        let mut list = vec![
            extension(ash::khr::swapchain::NAME),
            extension(ash::ext::shader_object::NAME),
            extension(ash::ext::graphics_pipeline_library::NAME),
            extension(ash::khr::pipeline_library::NAME),
            extension(ash::ext::shader_module_identifier::NAME),
        ];
        let removed = filter(&mut list);
        assert_eq!(
            removed,
            [
                "VK_EXT_shader_object",
                "VK_EXT_graphics_pipeline_library",
                "VK_EXT_shader_module_identifier"
            ]
        );
        let kept: Vec<_> = list
            .iter()
            .map(|e| e.extension_name_as_c_str().expect("terminated"))
            .collect();
        assert_eq!(
            kept,
            [ash::khr::swapchain::NAME, ash::khr::pipeline_library::NAME]
        );
    }

    #[test]
    fn test_copy_out_follows_the_two_call_contract() {
        let all = [1u32, 2, 3];
        let mut count = 0;
        // SAFETY: valid count pointer; null output asks for the count.
        let result = unsafe { copy_out(&all, &mut count, std::ptr::null_mut()) };
        assert_eq!((result, count), (vk::Result::SUCCESS, 3));

        let mut short = [0u32; 2];
        count = 2;
        // SAFETY: `short` holds `count` elements.
        let result = unsafe { copy_out(&all, &mut count, short.as_mut_ptr()) };
        assert_eq!((result, count, short), (vk::Result::INCOMPLETE, 2, [1, 2]));

        let mut full = [0u32; 4];
        count = 4;
        // SAFETY: `full` holds `count` elements.
        let result = unsafe { copy_out(&all, &mut count, full.as_mut_ptr()) };
        assert_eq!(
            (result, count, full),
            (vk::Result::SUCCESS, 3, [1, 2, 3, 0])
        );
    }

    #[test]
    fn test_hidden_feature_bits_are_cleared_and_others_kept() {
        let mut library = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default()
            .graphics_pipeline_library(true);
        let mut object = vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut library)
            .push_next(&mut object)
            .push_next(&mut vulkan13);
        // SAFETY: a valid chain of writable structures built above.
        unsafe { clear_hidden_features(&mut features) };
        assert_eq!(library.graphics_pipeline_library, vk::FALSE);
        assert_eq!(object.shader_object, vk::FALSE);
        assert_eq!(vulkan13.dynamic_rendering, vk::TRUE);
    }
}
