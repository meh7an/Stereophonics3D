//! Same-frame stereo on the GPU: the stereo push-constant block appended to every
//! pipeline layout, the patched last pre-raster shaders, and the constants pushed after
//! each pipeline bind.
//!
//! Push constants are the primary source so separation, convergence and the eye follow
//! hot reload on the next bind. Vulkan lets a stage appear in one push-constant range
//! only, so application ranges sharing a stage with the block grow to cover it, and the
//! application's own pushes are widened to the stages those grown ranges require. A layout whose push budget has no room for the block
//! falls back to specialization constants, whose values are baked in when the pipeline is
//! created.

use std::{
    collections::HashMap,
    ffi::CStr,
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use ash::vk;
use s3d_shared::{ShaderHash, ShaderOverride, StereoPushConstants};
use s3d_spirv_patch::{
    ConstantSource, DEFAULT_SPEC_ID, EyeSource, PatchOptions, ShaderModule, ShearMode,
};
use s3d_spirv_val::LayoutRules;

use crate::chain;

/// Why a shader stays unpatched.
#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error(transparent)]
    Patch(#[from] s3d_spirv_patch::Error),
    #[error("patched module fails validation: {0}")]
    Invalid(#[from] s3d_spirv_val::Error),
    #[error("cannot create the patched module: {0}")]
    Create(vk::Result),
}

/// A layout's push-constant ranges once the stereo block is added.
#[derive(Debug, Clone)]
pub struct Extension {
    pub offset: u32,
    pub ranges: Vec<vk::PushConstantRange>,
    /// Every stage of a range overlapping the block, which a push of the block must name.
    pub push_stages: vk::ShaderStageFlags,
}

/// The appended block of one pipeline layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoRange {
    pub offset: u32,
    /// Stages a push of the block names: those of every range overlapping it.
    pub stages: vk::ShaderStageFlags,
    /// A layer-owned layout with the same push-constant ranges, used to push the block:
    /// layouts with identical ranges are push-compatible, and this one outlives whatever
    /// the application destroys.
    pub push_layout: vk::PipelineLayout,
}

/// Cache key of one patched module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatchKey {
    hash: ShaderHash,
    entry: String,
    /// Push offset, or the specialization defaults as bits.
    constants: ConstantKey,
    mode: ModeKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantKey {
    Push(u32),
    Specialization([u32; 3]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ModeKey {
    Normal,
    FixedDepth(u32),
    Infinity,
}

impl PatchKey {
    fn new(hash: ShaderHash, entry: &str, options: &PatchOptions) -> Self {
        let constants = match options.constants {
            ConstantSource::PushConstant { offset } => ConstantKey::Push(offset),
            ConstantSource::Specialization { defaults, .. } => ConstantKey::Specialization([
                defaults.separation.to_bits(),
                defaults.convergence.to_bits(),
                defaults.eye_sign.to_bits(),
            ]),
        };
        let mode = match options.mode {
            ShearMode::Normal => ModeKey::Normal,
            ShearMode::FixedDepth(depth) => ModeKey::FixedDepth(depth.to_bits()),
            ShearMode::Infinity => ModeKey::Infinity,
        };
        Self {
            hash,
            entry: entry.to_owned(),
            constants,
            mode,
        }
    }
}

/// Per-device patching state.
#[derive(Debug)]
pub struct StereoState {
    /// Stages of the appended range: every pre-raster stage the device can run.
    stages: vk::ShaderStageFlags,
    max_push_bytes: u32,
    /// SPIRV-Tools environment matching the device's Vulkan version.
    environment: &'static str,
    /// Block layouts the device enabled; game shaders may rely on them.
    layout_rules: LayoutRules,
    /// Patched modules by source; `None` records a failed patch so it is not retried.
    patched: Mutex<HashMap<PatchKey, Option<vk::ShaderModule>>>,
    /// Layer-owned layouts keyed by their ranges as (offset, size, stage bits).
    push_layouts: Mutex<HashMap<Vec<RangeKey>, vk::PipelineLayout>>,
    budget_reported: AtomicBool,
}

/// One push-constant range as (offset, size, stage bits), hashable.
type RangeKey = (u32, u32, u32);

fn range_key(range: &vk::PushConstantRange) -> RangeKey {
    (range.offset, range.size, range.stage_flags.as_raw())
}

impl StereoState {
    pub fn new(
        properties: &vk::PhysicalDeviceProperties,
        enabled_extensions: &[&CStr],
        layout_rules: LayoutRules,
    ) -> Self {
        let mut stages = vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::TESSELLATION_EVALUATION
            | vk::ShaderStageFlags::GEOMETRY;
        let mesh = [ash::ext::mesh_shader::NAME, ash::nv::mesh_shader::NAME];
        if enabled_extensions.iter().any(|name| mesh.contains(name)) {
            stages |= vk::ShaderStageFlags::MESH_EXT;
        }
        let version = properties.api_version;
        let environment = match (
            vk::api_version_major(version),
            vk::api_version_minor(version),
        ) {
            (1, 0) => "vulkan1.0",
            (1, 1) => "vulkan1.1",
            (1, 2) => "vulkan1.2",
            _ => "vulkan1.3",
        };
        Self {
            stages,
            max_push_bytes: properties.limits.max_push_constants_size,
            environment,
            layout_rules,
            patched: Mutex::new(HashMap::new()),
            push_layouts: Mutex::new(HashMap::new()),
            budget_reported: AtomicBool::new(false),
        }
    }

    /// The application's ranges extended by the stereo block; `None` when the device's
    /// push budget has no room for it.
    pub fn extend(&self, ranges: &[vk::PushConstantRange]) -> Option<Extension> {
        let extended = extended_ranges(ranges, self.max_push_bytes, self.stages);
        if extended.is_none() && !self.budget_reported.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                max_push_bytes = self.max_push_bytes,
                "push budget exhausted, falling back to specialization constants"
            );
        }
        extended
    }

    /// A layer-owned layout with exactly `ranges`, created once per distinct set.
    ///
    /// # Errors
    ///
    /// Returns the driver's error when the layout cannot be created.
    pub fn push_layout(
        &self,
        next: &ash::Device,
        ranges: &[vk::PushConstantRange],
    ) -> Result<vk::PipelineLayout, vk::Result> {
        let key: Vec<_> = ranges.iter().map(range_key).collect();
        let mut layouts = self
            .push_layouts
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(&layout) = layouts.get(&key) {
            return Ok(layout);
        }
        let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(ranges);
        // SAFETY: a valid create info for the next layer's live device.
        let layout = unsafe { next.create_pipeline_layout(&info, None)? };
        layouts.insert(key, layout);
        Ok(layout)
    }

    /// The patched module for `code`'s `entry`, created once per distinct source and
    /// options; `None` when patching failed (logged once).
    pub fn patched_module(
        &self,
        next: &ash::Device,
        hash: ShaderHash,
        code: &[u32],
        entry: &str,
        options: &PatchOptions,
    ) -> Option<vk::ShaderModule> {
        let key = PatchKey::new(hash, entry, options);
        let mut patched = self.patched.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&module) = patched.get(&key) {
            return module;
        }
        let module = match self.create_patched(next, code, entry, options) {
            Ok((module, report)) => {
                tracing::info!(%hash, %report, "patched shader");
                Some(module)
            }
            Err(error) => {
                tracing::warn!(%hash, entry, %error, "shader left unpatched");
                None
            }
        };
        patched.insert(key, module);
        module
    }

    fn create_patched(
        &self,
        next: &ash::Device,
        code: &[u32],
        entry: &str,
        options: &PatchOptions,
    ) -> Result<(vk::ShaderModule, s3d_spirv_patch::PatchReport), PatchError> {
        let (module, report) = ShaderModule::parse(code)?.patch(entry, options)?;
        let words = module.assemble();
        s3d_spirv_val::validate_with(&words, self.environment, self.layout_rules)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: validated SPIR-V for the next layer's live device.
        let module =
            unsafe { next.create_shader_module(&info, None) }.map_err(PatchError::Create)?;
        Ok((module, report))
    }

    /// Destroys every module and layout this state created.
    pub fn destroy(&self, next: &ash::Device) {
        let patched =
            std::mem::take(&mut *self.patched.lock().unwrap_or_else(PoisonError::into_inner));
        let layouts = std::mem::take(
            &mut *self
                .push_layouts
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        // SAFETY: the layer created these on this device and the application can no longer
        // use them: pipelines built from the modules keep their own copy of the code, and
        // the layouts were only ever used by the layer's own pushes.
        unsafe {
            for module in patched.into_values().flatten() {
                next.destroy_shader_module(module, None);
            }
            for layout in layouts.into_values() {
                next.destroy_pipeline_layout(layout, None);
            }
        }
    }
}

/// The block-layout rules a device create info enables.
///
/// # Safety
///
/// `info` must be a valid device create info.
pub unsafe fn layout_rules(
    info: &vk::DeviceCreateInfo<'_>,
    api_version: u32,
    enabled_extensions: &[&CStr],
) -> LayoutRules {
    let feature = |s_type| {
        // SAFETY: the create info's chain is valid.
        unsafe { chain::find(info.p_next, s_type) }
    };
    let (mut scalar, mut uniform_buffer_standard) = (false, false);
    if let Some(found) = feature(vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
        // SAFETY: the structure its sType names.
        let found = unsafe { &*found.cast::<vk::PhysicalDeviceVulkan12Features<'_>>() };
        scalar |= found.scalar_block_layout == vk::TRUE;
        uniform_buffer_standard |= found.uniform_buffer_standard_layout == vk::TRUE;
    }
    if let Some(found) = feature(vk::StructureType::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES) {
        // SAFETY: the structure its sType names.
        let found = unsafe { &*found.cast::<vk::PhysicalDeviceScalarBlockLayoutFeatures<'_>>() };
        scalar |= found.scalar_block_layout == vk::TRUE;
    }
    if let Some(found) =
        feature(vk::StructureType::PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES)
    {
        // SAFETY: the structure its sType names.
        let found =
            unsafe { &*found.cast::<vk::PhysicalDeviceUniformBufferStandardLayoutFeatures<'_>>() };
        uniform_buffer_standard |= found.uniform_buffer_standard_layout == vk::TRUE;
    }
    LayoutRules {
        relaxed: api_version >= vk::API_VERSION_1_1
            || enabled_extensions.contains(&ash::khr::relaxed_block_layout::NAME),
        uniform_buffer_standard,
        scalar,
    }
}

/// `ranges` plus the stereo block at the next aligned offset, when it fits in
/// `max_push_bytes`.
fn extended_ranges(
    ranges: &[vk::PushConstantRange],
    max_push_bytes: u32,
    stages: vk::ShaderStageFlags,
) -> Option<Extension> {
    let end = ranges.iter().map(|r| r.offset + r.size).max().unwrap_or(0);
    let offset = StereoPushConstants::append_offset(end);
    let block_end = offset + StereoPushConstants::SIZE;
    if block_end > max_push_bytes {
        return None;
    }
    let mut covered = vk::ShaderStageFlags::empty();
    let mut extended: Vec<_> = ranges
        .iter()
        .map(|range| {
            if range.stage_flags.intersects(stages) {
                covered |= range.stage_flags;
                vk::PushConstantRange {
                    size: block_end - range.offset,
                    ..*range
                }
            } else {
                *range
            }
        })
        .collect();
    let rest = stages & !covered;
    if !rest.is_empty() {
        extended.push(vk::PushConstantRange {
            stage_flags: rest,
            offset,
            size: StereoPushConstants::SIZE,
        });
    }
    Some(Extension {
        offset,
        ranges: extended,
        push_stages: covered | rest,
    })
}

/// One `vkCmdPushConstants` call: stages, offset and size.
pub type PushSegment = (vk::ShaderStageFlags, u32, u32);

/// Splits a push of `size` bytes at `offset` naming `stages` into pieces that each name
/// every stage of the ranges overlapping them, as an extended layout requires. Bytes
/// pushed to a stage that does not read them are ignored.
pub fn push_segments(
    ranges: &[vk::PushConstantRange],
    stages: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
) -> Vec<PushSegment> {
    let end = offset + size;
    let mut points = vec![offset, end];
    for range in ranges {
        for point in [range.offset, range.offset + range.size] {
            if point > offset && point < end {
                points.push(point);
            }
        }
    }
    points.sort_unstable();
    points.dedup();
    let mut segments: Vec<PushSegment> = Vec::new();
    for pair in points.windows(2) {
        let (start, stop) = (pair[0], pair[1]);
        let flags = ranges
            .iter()
            .filter(|r| r.offset < stop && r.offset + r.size > start)
            .fold(stages, |flags, r| flags | r.stage_flags);
        match segments.last_mut() {
            Some(last) if last.0 == flags => last.2 += stop - start,
            _ => segments.push((flags, start, stop - start)),
        }
    }
    segments
}

/// How a profile override shears a shader; `None` leaves it unpatched.
pub fn shear_mode(entry: Option<&ShaderOverride>) -> Option<ShearMode> {
    match entry {
        None | Some(ShaderOverride::Stereo) => Some(ShearMode::Normal),
        Some(ShaderOverride::HudDepth(depth)) => Some(ShearMode::FixedDepth(*depth)),
        Some(ShaderOverride::Sky) => Some(ShearMode::Infinity),
        Some(_) => None,
    }
}

/// Patch options for a layout with `range`, or specialization constants holding
/// `current` when the layout has no room for the block.
pub fn patch_options(
    range: Option<&StereoRange>,
    current: StereoPushConstants,
    mode: ShearMode,
) -> PatchOptions {
    let constants = match range {
        Some(range) => ConstantSource::PushConstant {
            offset: range.offset,
        },
        None => ConstantSource::Specialization {
            first_id: DEFAULT_SPEC_ID,
            defaults: current,
        },
    };
    PatchOptions {
        constants,
        eye: EyeSource::Constant,
        mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::VERTEX;

    fn range(offset: u32, size: u32) -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset,
            size,
        }
    }

    const PRE_RASTER: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
        vk::ShaderStageFlags::VERTEX.as_raw() | vk::ShaderStageFlags::GEOMETRY.as_raw(),
    );

    fn keys(ranges: &[vk::PushConstantRange]) -> Vec<RangeKey> {
        ranges.iter().map(range_key).collect()
    }

    fn staged(stage_flags: vk::ShaderStageFlags, offset: u32, size: u32) -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags,
            offset,
            size,
        }
    }

    #[test]
    fn test_block_appends_at_the_next_aligned_offset() {
        let extension = extended_ranges(&[], 128, STAGES).expect("fits");
        assert_eq!(extension.offset, 0);
        assert_eq!(keys(&extension.ranges), keys(&[staged(STAGES, 0, 16)]));

        let fragment = staged(vk::ShaderStageFlags::FRAGMENT, 64, 4);
        let extension = extended_ranges(&[fragment], 128, STAGES).expect("fits");
        assert_eq!(extension.offset, 80);
        assert_eq!(
            keys(&extension.ranges),
            keys(&[fragment, staged(STAGES, 80, 16)])
        );
    }

    #[test]
    fn test_ranges_sharing_a_stage_grow_instead_of_duplicating_it() {
        let vertex = staged(vk::ShaderStageFlags::VERTEX, 0, 64);
        let fragment = staged(vk::ShaderStageFlags::FRAGMENT, 64, 16);
        let extension = extended_ranges(&[vertex, fragment], 128, PRE_RASTER).expect("fits");
        assert_eq!(extension.offset, 80);
        assert_eq!(
            keys(&extension.ranges),
            keys(&[
                staged(vk::ShaderStageFlags::VERTEX, 0, 96),
                fragment,
                staged(vk::ShaderStageFlags::GEOMETRY, 80, 16),
            ])
        );
        assert_eq!(extension.push_stages, PRE_RASTER);
        for stage in [vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::GEOMETRY] {
            let holding = extension
                .ranges
                .iter()
                .filter(|r| r.stage_flags.contains(stage));
            assert_eq!(holding.count(), 1, "{stage:?}");
        }
    }

    #[test]
    fn test_push_segments_name_every_overlapping_range() {
        let ranges = [
            staged(vk::ShaderStageFlags::VERTEX, 0, 96),
            staged(vk::ShaderStageFlags::FRAGMENT, 64, 16),
            staged(vk::ShaderStageFlags::GEOMETRY, 80, 16),
        ];
        let vertex = vk::ShaderStageFlags::VERTEX;
        let fragment = vk::ShaderStageFlags::FRAGMENT;
        assert_eq!(push_segments(&ranges, vertex, 0, 64), [(vertex, 0, 64)]);
        assert_eq!(
            push_segments(&ranges, fragment, 64, 16),
            [(vertex | fragment, 64, 16)]
        );
        assert_eq!(
            push_segments(&ranges, vertex, 32, 48),
            [(vertex, 32, 32), (vertex | fragment, 64, 16)]
        );
        assert_eq!(
            push_segments(&ranges, vertex, 80, 16),
            [(vertex | vk::ShaderStageFlags::GEOMETRY, 80, 16)]
        );
    }

    #[test]
    fn test_full_budget_falls_back() {
        assert!(extended_ranges(&[range(0, 128)], 128, STAGES).is_none());
        assert!(extended_ranges(&[range(0, 116)], 128, STAGES).is_none());
        let extension = extended_ranges(&[range(0, 112)], 128, STAGES).expect("fits");
        assert_eq!(keys(&extension.ranges), keys(&[range(0, 128)]));
    }

    #[test]
    fn test_layout_rules_follow_enabled_features() {
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default().scalar_block_layout(true);
        let info = vk::DeviceCreateInfo::default().push_next(&mut vulkan12);
        // SAFETY: a valid create info built above.
        let rules = unsafe { layout_rules(&info, vk::API_VERSION_1_3, &[]) };
        assert_eq!(
            rules,
            LayoutRules {
                relaxed: true,
                uniform_buffer_standard: false,
                scalar: true,
            }
        );
        let plain = vk::DeviceCreateInfo::default();
        // SAFETY: a valid create info.
        assert_eq!(
            unsafe { layout_rules(&plain, vk::API_VERSION_1_0, &[]) },
            LayoutRules::default()
        );
    }

    #[test]
    fn test_overrides_map_to_shear_modes() {
        assert_eq!(shear_mode(None), Some(ShearMode::Normal));
        assert_eq!(
            shear_mode(Some(&ShaderOverride::Stereo)),
            Some(ShearMode::Normal)
        );
        assert_eq!(
            shear_mode(Some(&ShaderOverride::HudDepth(2.0))),
            Some(ShearMode::FixedDepth(2.0))
        );
        assert_eq!(
            shear_mode(Some(&ShaderOverride::Sky)),
            Some(ShearMode::Infinity)
        );
        assert_eq!(shear_mode(Some(&ShaderOverride::Mono)), None);
        assert_eq!(shear_mode(Some(&ShaderOverride::Skip)), None);
    }

    #[test]
    fn test_options_use_push_or_specialization() {
        let current = StereoPushConstants {
            separation: 0.03,
            convergence: 4.0,
            eye_sign: -1.0,
            reserved: 0.0,
        };
        let range = StereoRange {
            offset: 64,
            stages: STAGES,
            push_layout: vk::PipelineLayout::null(),
        };
        let push = patch_options(Some(&range), current, ShearMode::Normal);
        assert_eq!(push.constants, ConstantSource::PushConstant { offset: 64 });
        let spec = patch_options(None, current, ShearMode::Normal);
        assert_eq!(
            spec.constants,
            ConstantSource::Specialization {
                first_id: DEFAULT_SPEC_ID,
                defaults: current
            }
        );
        assert_ne!(
            PatchKey::new(ShaderHash::of(b"a"), "main", &push),
            PatchKey::new(ShaderHash::of(b"a"), "main", &spec)
        );
    }
}
