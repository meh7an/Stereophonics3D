//! Mirrors of the loader-layer interface in the SDK's `vk_layer.h`, which `ash` does not
//! provide. Layouts must match the header exactly.

use std::ffi::{c_char, c_void};

use ash::vk;

/// `LAYER_NEGOTIATE_INTERFACE_STRUCT`.
pub const NEGOTIATE_INTERFACE_STRUCT: u32 = 1;
/// The loader-layer interface version this layer implements.
pub const INTERFACE_VERSION: u32 = 2;
/// `VK_LAYER_LINK_INFO` in `VkLayerFunction`.
const LAYER_LINK_INFO: u32 = 0;
/// `VK_LOADER_DATA_CALLBACK` in `VkLayerFunction`.
const LOADER_DATA_CALLBACK: u32 = 1;

/// `PFN_vkSetDeviceLoaderData`: initialises the dispatch pointer of a dispatchable object
/// the layer creates itself, such as a command buffer.
pub type SetDeviceLoaderData =
    unsafe extern "system" fn(device: vk::Device, object: *mut c_void) -> vk::Result;

/// `PFN_GetPhysicalDeviceProcAddr`.
pub type GetPhysicalDeviceProcAddr =
    Option<unsafe extern "system" fn(vk::Instance, *const c_char) -> vk::PFN_vkVoidFunction>;

/// `VkNegotiateLayerInterface`.
#[repr(C)]
pub struct NegotiateLayerInterface {
    pub s_type: u32,
    pub p_next: *mut c_void,
    pub loader_layer_interface_version: u32,
    pub pfn_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
    pub pfn_get_physical_device_proc_addr: GetPhysicalDeviceProcAddr,
}

/// `VkLayerInstanceLink`.
#[repr(C)]
pub struct LayerInstanceLink {
    pub p_next: *mut LayerInstanceLink,
    pub pfn_next_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_next_get_physical_device_proc_addr: GetPhysicalDeviceProcAddr,
}

/// The union in `VkLayerInstanceCreateInfo`; only `layer_info` is used here, the other
/// members give it the header's size.
#[repr(C)]
pub union LayerInstanceUnion {
    pub layer_info: *mut LayerInstanceLink,
    layer_device: [*const c_void; 2],
    loader_features: u32,
}

/// `VkLayerInstanceCreateInfo`.
#[repr(C)]
pub struct LayerInstanceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: u32,
    pub u: LayerInstanceUnion,
}

/// `VkLayerDeviceLink`.
#[repr(C)]
pub struct LayerDeviceLink {
    pub p_next: *mut LayerDeviceLink,
    pub pfn_next_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_next_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
}

/// The union in `VkLayerDeviceCreateInfo`.
#[repr(C)]
pub union LayerDeviceUnion {
    pub layer_info: *mut LayerDeviceLink,
    pub set_device_loader_data: Option<SetDeviceLoaderData>,
}

/// `VkLayerDeviceCreateInfo`.
#[repr(C)]
pub struct LayerDeviceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: u32,
    pub u: LayerDeviceUnion,
}

/// The loader's link-info structure in an instance create info's `pNext` chain.
///
/// # Safety
///
/// `next` must be a valid `pNext` chain of Vulkan structures.
pub unsafe fn instance_link(next: *const c_void) -> Option<*mut LayerInstanceCreateInfo> {
    // SAFETY: forwarded from the caller.
    unsafe {
        find_link(
            next,
            vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            LAYER_LINK_INFO,
        )
    }
    .map(|base| base.cast::<LayerInstanceCreateInfo>())
}

/// The loader's link-info structure in a device create info's `pNext` chain.
///
/// # Safety
///
/// `next` must be a valid `pNext` chain of Vulkan structures.
pub unsafe fn device_link(next: *const c_void) -> Option<*mut LayerDeviceCreateInfo> {
    // SAFETY: forwarded from the caller.
    unsafe {
        find_link(
            next,
            vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            LAYER_LINK_INFO,
        )
    }
    .map(|base| base.cast::<LayerDeviceCreateInfo>())
}

/// The loader's `vkSetDeviceLoaderData` from a device create info's `pNext` chain.
///
/// # Safety
///
/// `next` must be a valid `pNext` chain of Vulkan structures.
pub unsafe fn device_loader_data(next: *const c_void) -> Option<SetDeviceLoaderData> {
    // SAFETY: forwarded from the caller; the structure found is a device create info.
    unsafe {
        let found = find_link(
            next,
            vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            LOADER_DATA_CALLBACK,
        )?;
        (*found.cast::<LayerDeviceCreateInfo>())
            .u
            .set_device_loader_data
    }
}

/// Walks a `pNext` chain for a loader structure of `s_type` whose `VkLayerFunction` is
/// `function`. Both loader structures share the instance one's prefix.
///
/// The loader hands the chain over as const but expects each layer to advance the link
/// in place, hence the mutable pointer.
unsafe fn find_link(
    mut next: *const c_void,
    s_type: vk::StructureType,
    wanted: u32,
) -> Option<*mut LayerInstanceCreateInfo> {
    while !next.is_null() {
        let candidate = next.cast::<LayerInstanceCreateInfo>();
        // SAFETY: every structure in a valid chain starts with sType and pNext; the
        // `function` field is read only once sType identifies a loader structure.
        let (found_type, function, following) = unsafe {
            (
                (*candidate).s_type,
                (*candidate).function,
                (*candidate).p_next,
            )
        };
        if found_type == s_type && function == wanted {
            return Some(candidate.cast_mut());
        }
        next = following;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;

    #[test]
    fn test_layouts_match_vk_layer_h_on_64_bit() {
        // sType + padding, pNext, version + padding, then three function pointers.
        assert_eq!(size_of::<NegotiateLayerInterface>(), 48);
        assert_eq!(
            offset_of!(NegotiateLayerInterface, pfn_get_instance_proc_addr),
            24
        );
        assert_eq!(size_of::<LayerInstanceUnion>(), 16);
        assert_eq!(offset_of!(LayerInstanceCreateInfo, u), 24);
        assert_eq!(size_of::<LayerInstanceCreateInfo>(), 40);
        assert_eq!(offset_of!(LayerDeviceCreateInfo, u), 24);
        assert_eq!(size_of::<LayerDeviceLink>(), 24);
    }

    #[test]
    fn test_find_link_skips_other_structures() {
        let mut link = LayerInstanceLink {
            p_next: std::ptr::null_mut(),
            pfn_next_get_instance_proc_addr: None,
            pfn_next_get_physical_device_proc_addr: None,
        };
        let loader_data = LayerInstanceCreateInfo {
            s_type: vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            function: 1,
            u: LayerInstanceUnion { loader_features: 0 },
        };
        let link_info = LayerInstanceCreateInfo {
            s_type: vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            p_next: (&raw const loader_data).cast(),
            function: LAYER_LINK_INFO,
            u: LayerInstanceUnion {
                layer_info: &raw mut link,
            },
        };
        let other = vk::ValidationFeaturesEXT {
            p_next: (&raw const link_info).cast(),
            ..Default::default()
        };
        // SAFETY: a chain of three valid structures built above.
        let found = unsafe { instance_link((&raw const other).cast()) };
        assert_eq!(found, Some((&raw const link_info).cast_mut()));
        // SAFETY: the same chain, minus the link.
        assert_eq!(
            unsafe { instance_link((&raw const loader_data).cast()) },
            None
        );
    }
}
