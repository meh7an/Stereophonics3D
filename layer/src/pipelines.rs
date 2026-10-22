//! Shader modules, pipeline layouts, graphics pipelines and pipeline caches, tracked per
//! device so every draw can be traced back to the shaders that produce it.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, PoisonError, RwLock},
};

use ash::vk;
use s3d_shared::ShaderHash;

use crate::{
    chain,
    dispatch::{DeviceData, read, write},
    dump,
    hooks::next_device_fn,
};

/// Pre-raster stages from last to first: the first one a pipeline has is the stage whose
/// Position reaches the rasterizer.
const PRE_RASTER_LAST_FIRST: [vk::ShaderStageFlags; 4] = [
    vk::ShaderStageFlags::MESH_EXT,
    vk::ShaderStageFlags::GEOMETRY,
    vk::ShaderStageFlags::TESSELLATION_EVALUATION,
    vk::ShaderStageFlags::VERTEX,
];

#[derive(Debug)]
pub struct ShaderModuleInfo {
    pub hash: ShaderHash,
}

#[derive(Debug)]
pub struct LayoutInfo {
    /// The application's ranges; the layer's block is appended after them in M3.3.
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl LayoutInfo {
    /// First byte after every application push constant.
    pub fn push_constant_end(&self) -> u32 {
        self.push_constant_ranges
            .iter()
            .map(|range| range.offset + range.size)
            .max()
            .unwrap_or(0)
    }
}

/// One shader stage of a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageShader {
    pub stage: vk::ShaderStageFlags,
    /// `None` when the code is unknown: a module identifier, or a module the layer never
    /// saw created.
    pub hash: Option<ShaderHash>,
}

#[derive(Debug, Default)]
pub struct PipelineInfo {
    /// Own stages plus those of linked pipeline libraries.
    pub stages: Vec<StageShader>,
}

impl PipelineInfo {
    /// The stage whose Position output is rasterized.
    pub fn last_pre_raster(&self) -> Option<StageShader> {
        PRE_RASTER_LAST_FIRST
            .iter()
            .find_map(|&stage| self.stages.iter().find(|s| s.stage == stage).copied())
    }

    /// Hashes of every stage whose code is known.
    pub fn hashes(&self) -> impl Iterator<Item = ShaderHash> + '_ {
        self.stages.iter().filter_map(|stage| stage.hash)
    }
}

/// Everything known about one device's shaders and pipelines.
#[derive(Debug, Default)]
pub struct Tracker {
    modules: RwLock<HashMap<vk::ShaderModule, Arc<ShaderModuleInfo>>>,
    layouts: RwLock<HashMap<vk::PipelineLayout, Arc<LayoutInfo>>>,
    pipelines: RwLock<HashMap<vk::Pipeline, Arc<PipelineInfo>>>,
    /// Last pre-raster hashes already logged, so each is named once.
    seen: Mutex<HashSet<ShaderHash>>,
}

impl Tracker {
    pub fn module(&self, module: vk::ShaderModule) -> Option<Arc<ShaderModuleInfo>> {
        read(&self.modules).get(&module).cloned()
    }

    pub fn pipeline(&self, pipeline: vk::Pipeline) -> Option<Arc<PipelineInfo>> {
        read(&self.pipelines).get(&pipeline).cloned()
    }

    pub fn layout(&self, layout: vk::PipelineLayout) -> Option<Arc<LayoutInfo>> {
        read(&self.layouts).get(&layout).cloned()
    }

    /// Live (modules, layouts, pipelines).
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            read(&self.modules).len(),
            read(&self.layouts).len(),
            read(&self.pipelines).len(),
        )
    }

    /// Whether `hash` is logged here for the first time.
    fn first_sighting(&self, hash: ShaderHash) -> bool {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(hash)
    }
}

/// The SPIR-V words of a create info, which the application guarantees to be valid.
///
/// # Safety
///
/// `info.p_code` must point at `info.code_size` bytes.
unsafe fn code<'a>(info: &vk::ShaderModuleCreateInfo<'a>) -> &'a [u32] {
    if info.p_code.is_null() {
        return &[];
    }
    // SAFETY: forwarded from the caller; code_size is a multiple of 4 by the spec.
    unsafe { std::slice::from_raw_parts(info.p_code, info.code_size / 4) }
}

/// Hash and code of a module create info; the hash covers the exact bytes passed in.
///
/// # Safety
///
/// As [`code`].
unsafe fn identify<'a>(info: &vk::ShaderModuleCreateInfo<'a>) -> (ShaderHash, &'a [u32]) {
    // SAFETY: forwarded from the caller.
    let words = unsafe { code(info) };
    // SAFETY: the same memory viewed as the bytes the application passed.
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
    (ShaderHash::of(bytes), words)
}

pub unsafe extern "system" fn create_shader_module(
    device: vk::Device,
    info: *const vk::ShaderModuleCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    module: *mut vk::ShaderModule,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found =
        unsafe { next_device_fn("vkCreateShaderModule", device, |f| f.create_shader_module) };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: the next layer's create with the application's arguments.
    let result = unsafe { next(device, info, allocator, module) };
    if result == vk::Result::SUCCESS {
        s3d_shared::ffi::ffi_guard("vkCreateShaderModule", (), || {
            // SAFETY: a valid create info and the module just created from it.
            let ((hash, words), handle) = unsafe { (identify(&*info), *module) };
            dump::shader(hash, words);
            let info = Arc::new(ShaderModuleInfo { hash });
            write(&data.tracker.modules).insert(handle, info);
        });
    }
    result
}

pub unsafe extern "system" fn destroy_shader_module(
    device: vk::Device,
    module: vk::ShaderModule,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a live device from the application.
    let found =
        unsafe { next_device_fn("vkDestroyShaderModule", device, |f| f.destroy_shader_module) };
    let Some((data, next)) = found else {
        return;
    };
    s3d_shared::ffi::ffi_guard("vkDestroyShaderModule", (), || {
        write(&data.tracker.modules).remove(&module);
    });
    // SAFETY: the next layer's destroy with the application's arguments.
    unsafe { next(device, module, allocator) };
}

pub unsafe extern "system" fn create_pipeline_layout(
    device: vk::Device,
    info: *const vk::PipelineLayoutCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    layout: *mut vk::PipelineLayout,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found = unsafe {
        next_device_fn("vkCreatePipelineLayout", device, |f| {
            f.create_pipeline_layout
        })
    };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: the next layer's create with the application's arguments.
    let result = unsafe { next(device, info, allocator, layout) };
    if result == vk::Result::SUCCESS {
        s3d_shared::ffi::ffi_guard("vkCreatePipelineLayout", (), || {
            // SAFETY: a valid create info and the layout just created from it.
            let (ranges, handle) = unsafe {
                let info = &*info;
                let ranges = if info.p_push_constant_ranges.is_null() {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(
                        info.p_push_constant_ranges,
                        info.push_constant_range_count as usize,
                    )
                    .to_vec()
                };
                (ranges, *layout)
            };
            let info = Arc::new(LayoutInfo {
                push_constant_ranges: ranges,
            });
            write(&data.tracker.layouts).insert(handle, info);
        });
    }
    result
}

pub unsafe extern "system" fn destroy_pipeline_layout(
    device: vk::Device,
    layout: vk::PipelineLayout,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a live device from the application.
    let found = unsafe {
        next_device_fn("vkDestroyPipelineLayout", device, |f| {
            f.destroy_pipeline_layout
        })
    };
    let Some((data, next)) = found else {
        return;
    };
    s3d_shared::ffi::ffi_guard("vkDestroyPipelineLayout", (), || {
        write(&data.tracker.layouts).remove(&layout);
    });
    // SAFETY: the next layer's destroy with the application's arguments.
    unsafe { next(device, layout, allocator) };
}

pub unsafe extern "system" fn create_graphics_pipelines(
    device: vk::Device,
    cache: vk::PipelineCache,
    count: u32,
    infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    pipelines: *mut vk::Pipeline,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found = unsafe {
        next_device_fn("vkCreateGraphicsPipelines", device, |f| {
            f.create_graphics_pipelines
        })
    };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: the next layer's create with the application's arguments.
    let result = unsafe { next(device, cache, count, infos, allocator, pipelines) };
    // Failures other than these leave every handle null; PIPELINE_COMPILE_REQUIRED leaves
    // some created, so record whatever is non-null.
    s3d_shared::ffi::ffi_guard("vkCreateGraphicsPipelines", (), || {
        // SAFETY: `count` valid create infos and as many output handles.
        let (infos, handles) = unsafe {
            (
                std::slice::from_raw_parts(infos, count as usize),
                std::slice::from_raw_parts(pipelines, count as usize),
            )
        };
        for (info, &handle) in infos.iter().zip(handles) {
            if handle != vk::Pipeline::null() {
                // SAFETY: a valid create info whose pipeline was created.
                unsafe { record_pipeline(&data, info, handle) };
            }
        }
    });
    result
}

/// Collects a new pipeline's stages, dumps inline code and names new pre-raster shaders.
///
/// # Safety
///
/// `info` must be the valid create info `handle` was created from.
unsafe fn record_pipeline(
    data: &DeviceData,
    info: &vk::GraphicsPipelineCreateInfo<'_>,
    handle: vk::Pipeline,
) {
    let tracker = &data.tracker;
    let mut pipeline = PipelineInfo::default();
    if !info.p_stages.is_null() {
        // SAFETY: `stage_count` valid stage infos.
        let stages =
            unsafe { std::slice::from_raw_parts(info.p_stages, info.stage_count as usize) };
        for stage in stages {
            // SAFETY: a valid stage info.
            pipeline
                .stages
                .push(unsafe { stage_shader(tracker, stage) });
        }
    }
    // SAFETY: the create info's chain is valid.
    if let Some(found) = unsafe {
        chain::find(
            info.p_next,
            vk::StructureType::PIPELINE_LIBRARY_CREATE_INFO_KHR,
        )
    } {
        // SAFETY: the structure its sType names, with `library_count` handles.
        let libraries = unsafe {
            let found = &*found.cast::<vk::PipelineLibraryCreateInfoKHR<'_>>();
            std::slice::from_raw_parts(found.p_libraries, found.library_count as usize)
        };
        for library in libraries {
            if let Some(linked) = tracker.pipeline(*library) {
                pipeline.stages.extend_from_slice(&linked.stages);
            }
        }
    }
    if let Some(StageShader {
        stage,
        hash: Some(hash),
    }) = pipeline.last_pre_raster()
        && tracker.first_sighting(hash)
    {
        tracing::info!(%hash, ?stage, "pre-raster shader");
    }
    let push_constant_end = tracker
        .layout(info.layout)
        .map(|layout| layout.push_constant_end());
    tracing::debug!(?handle, stages = ?pipeline.stages, ?push_constant_end, "graphics pipeline");
    write(&tracker.pipelines).insert(handle, Arc::new(pipeline));
}

/// The shader behind one stage: a tracked module, inline module code, or unknown.
///
/// # Safety
///
/// `stage` must be a valid stage create info.
unsafe fn stage_shader(
    tracker: &Tracker,
    stage: &vk::PipelineShaderStageCreateInfo<'_>,
) -> StageShader {
    let shader = |hash| StageShader {
        stage: stage.stage,
        hash,
    };
    if stage.module != vk::ShaderModule::null() {
        let hash = tracker.module(stage.module).map(|module| module.hash);
        if hash.is_none() {
            tracing::warn!(module = ?stage.module, "pipeline uses a module created before the layer");
        }
        return shader(hash);
    }
    // SAFETY: the stage's chain is valid.
    let inline = unsafe { chain::find(stage.p_next, vk::StructureType::SHADER_MODULE_CREATE_INFO) };
    if let Some(found) = inline {
        // SAFETY: the structure its sType names, with valid code.
        let (hash, words) = unsafe { identify(&*found.cast::<vk::ShaderModuleCreateInfo<'_>>()) };
        dump::shader(hash, words);
        return shader(Some(hash));
    }
    tracing::warn!(stage = ?stage.stage, "stage given by module identifier, code unknown");
    shader(None)
}

pub unsafe extern "system" fn destroy_pipeline(
    device: vk::Device,
    pipeline: vk::Pipeline,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a live device from the application.
    let found = unsafe { next_device_fn("vkDestroyPipeline", device, |f| f.destroy_pipeline) };
    let Some((data, next)) = found else {
        return;
    };
    s3d_shared::ffi::ffi_guard("vkDestroyPipeline", (), || {
        write(&data.tracker.pipelines).remove(&pipeline);
    });
    // SAFETY: the next layer's destroy with the application's arguments.
    unsafe { next(device, pipeline, allocator) };
}

pub unsafe extern "system" fn create_pipeline_cache(
    device: vk::Device,
    info: *const vk::PipelineCacheCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    cache: *mut vk::PipelineCache,
) -> vk::Result {
    // SAFETY: a live device from the application.
    let found =
        unsafe { next_device_fn("vkCreatePipelineCache", device, |f| f.create_pipeline_cache) };
    let Some((_, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // Prebuilt cache contents only hold pipelines compiled from unpatched shaders, so
    // their size tells how much the game will compile again once patching starts.
    s3d_shared::ffi::ffi_guard("vkCreatePipelineCache", (), || {
        // SAFETY: a valid create info.
        let initial_bytes = unsafe { (*info).initial_data_size };
        tracing::info!(initial_bytes, "pipeline cache created");
    });
    // SAFETY: the next layer's create with the application's arguments.
    unsafe { next(device, info, allocator, cache) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(stage: vk::ShaderStageFlags, value: u64) -> StageShader {
        StageShader {
            stage,
            hash: Some(format!("{value:016x}").parse().expect("hash")),
        }
    }

    #[test]
    fn test_last_pre_raster_prefers_the_latest_stage() {
        let vertex = stage(vk::ShaderStageFlags::VERTEX, 1);
        let tese = stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION, 2);
        let geometry = stage(vk::ShaderStageFlags::GEOMETRY, 3);
        let fragment = stage(vk::ShaderStageFlags::FRAGMENT, 4);
        let pipeline = |stages: &[StageShader]| PipelineInfo {
            stages: stages.to_vec(),
        };
        assert_eq!(
            pipeline(&[vertex, fragment]).last_pre_raster(),
            Some(vertex)
        );
        assert_eq!(
            pipeline(&[vertex, tese, geometry, fragment]).last_pre_raster(),
            Some(geometry)
        );
        assert_eq!(pipeline(&[fragment]).last_pre_raster(), None);
    }

    #[test]
    fn test_identify_hashes_the_exact_bytes() {
        let bytes = include_bytes!("../../fixtures/spirv/glsl_basic.spv");
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `info` points at `words`.
        let (hash, code) = unsafe { identify(&info) };
        assert_eq!(hash, ShaderHash::of(bytes));
        assert_eq!(code, words.as_slice());
    }
}
