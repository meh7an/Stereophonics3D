//! Walking `pNext` chains of Vulkan structures.

use std::ffi::c_void;

use ash::vk;

/// The first structure of type `s_type` in the chain starting at `next`.
///
/// # Safety
///
/// `next` must be null or a valid `pNext` chain of Vulkan structures.
pub unsafe fn find(mut next: *const c_void, s_type: vk::StructureType) -> Option<*const c_void> {
    while !next.is_null() {
        let base = next.cast::<vk::BaseInStructure<'_>>();
        // SAFETY: every structure in a valid chain starts with sType and pNext.
        let (found, following) = unsafe { ((*base).s_type, (*base).p_next) };
        if found == s_type {
            return Some(next);
        }
        next = following.cast();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_returns_first_match_and_none_when_absent() {
        let identifier = vk::PipelineShaderStageModuleIdentifierCreateInfoEXT::default();
        let module = vk::ShaderModuleCreateInfo {
            p_next: (&raw const identifier).cast(),
            ..Default::default()
        };
        let head = (&raw const module).cast::<c_void>();
        // SAFETY: a chain of two valid structures built above.
        unsafe {
            assert_eq!(
                find(head, vk::StructureType::SHADER_MODULE_CREATE_INFO),
                Some(head)
            );
            assert_eq!(
                find(
                    head,
                    vk::StructureType::PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT
                ),
                Some((&raw const identifier).cast())
            );
            assert_eq!(
                find(head, vk::StructureType::PIPELINE_LIBRARY_CREATE_INFO_KHR),
                None
            );
            assert_eq!(
                find(
                    std::ptr::null(),
                    vk::StructureType::SHADER_MODULE_CREATE_INFO
                ),
                None
            );
        }
    }
}
