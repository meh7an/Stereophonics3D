//! Command-buffer tracking, draw skipping and the stereo constants.
//!
//! Binding a graphics pipeline decides, against the current profile, whether the draws
//! that follow are dropped: a profile `skip` entry matching any of the pipeline's shaders
//! hides those draws. A pipeline with a patched stage also gets the stereo block pushed
//! right after the bind. Deciding at bind time means a hot-reloaded profile applies to the
//! next recording without recreating pipelines.

use std::{
    collections::HashSet,
    sync::{
        Mutex, PoisonError, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use s3d_shared::{
    Settings, ShaderHash, ShaderOverride, StereoParams, StereoPushConstants, ffi::ffi_guard,
};

use crate::{
    activation,
    dispatch::{self, DeviceData, key, read, write},
    hooks::{DeviceFns, next_device_fn},
    pipelines::PipelineInfo,
    stereo::{StereoRange, push_segments},
};

/// Command buffers whose bound graphics pipeline is skipped.
#[derive(Debug, Default)]
pub struct CommandTracker {
    skipping: RwLock<HashSet<vk::CommandBuffer>>,
    /// Size of `skipping`, so draws need no lock while nothing is skipped.
    count: AtomicUsize,
    /// Shaders whose draws were already reported as skipped.
    reported: Mutex<HashSet<ShaderHash>>,
}

impl CommandTracker {
    /// Records whether `command_buffer`'s draws are dropped from now on.
    pub fn set(&self, command_buffer: vk::CommandBuffer, skip: bool) {
        let mut skipping = write(&self.skipping);
        let changed = if skip {
            skipping.insert(command_buffer)
        } else {
            skipping.remove(&command_buffer)
        };
        if changed {
            self.count.store(skipping.len(), Ordering::Release);
        }
    }

    /// Whether the next draw recorded into `command_buffer` is dropped.
    pub fn skips(&self, command_buffer: vk::CommandBuffer) -> bool {
        self.count.load(Ordering::Acquire) > 0 && read(&self.skipping).contains(&command_buffer)
    }

    /// Whether `hash` is reported as skipped for the first time.
    fn first_report(&self, hash: ShaderHash) -> bool {
        self.reported
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(hash)
    }
}

/// The first of `hashes` the profile marks `skip`.
fn skipped_by<'a>(
    hashes: impl IntoIterator<Item = ShaderHash>,
    overrides: impl Fn(&ShaderHash) -> Option<&'a ShaderOverride>,
) -> Option<ShaderHash> {
    hashes
        .into_iter()
        .find(|hash| matches!(overrides(hash), Some(ShaderOverride::Skip)))
}

/// Whether draws with `info`'s pipeline bound are dropped under `settings`.
fn skips_pipeline(data: &DeviceData, info: &PipelineInfo, settings: &Settings) -> bool {
    if !settings
        .shaders
        .values()
        .any(|o| matches!(o, ShaderOverride::Skip))
    {
        return false;
    }
    let Some(hash) = skipped_by(info.hashes(), |hash| settings.shaders.get(hash)) else {
        return false;
    };
    if data.commands.first_report(hash) {
        tracing::info!(%hash, "skipping draws");
    }
    true
}

pub unsafe extern "system" fn cmd_bind_pipeline(
    command_buffer: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    // SAFETY: a live command buffer from the application.
    let found =
        unsafe { next_device_fn("vkCmdBindPipeline", command_buffer, |f| f.cmd_bind_pipeline) };
    let Some((data, next)) = found else {
        return;
    };
    let push = if bind_point == vk::PipelineBindPoint::GRAPHICS {
        ffi_guard("vkCmdBindPipeline", None, || {
            bind_graphics(&data, command_buffer, pipeline)
        })
    } else {
        None
    };
    // SAFETY: the next layer's bind with the application's arguments.
    unsafe { next(command_buffer, bind_point, pipeline) };
    if let Some((range, block)) = push {
        // SAFETY: the layer's own push layout matches the range, and the command buffer
        // is recording since the application just bound a pipeline into it.
        unsafe {
            data.next.cmd_push_constants(
                command_buffer,
                range.push_layout,
                range.stages,
                range.offset,
                &block.to_bytes(),
            );
        }
    }
}

/// Records the skip decision for a graphics bind and returns the stereo block to push
/// after it, if the pipeline reads one.
fn bind_graphics(
    data: &DeviceData,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
) -> Option<(StereoRange, StereoPushConstants)> {
    let snapshot = activation::get().snapshot();
    let settings = snapshot.as_ref().and_then(|s| s.settings.as_ref());
    let Some(info) = data.tracker.pipeline(pipeline) else {
        data.commands.set(command_buffer, false);
        return None;
    };
    let skip = settings.is_some_and(|settings| skips_pipeline(data, &info, settings));
    data.commands.set(command_buffer, skip);
    let range = info.stereo_push?;
    // Without a matching profile any more, the shear degenerates to mono.
    let block = settings.map_or_else(
        || StereoParams::default().push_constants(None),
        |settings| settings.stereo.push_constants(settings.stereo.eye),
    );
    Some((range, block))
}

/// Forwards the application's push, split and widened where the layout's ranges were
/// extended to hold the stereo block.
pub unsafe extern "system" fn cmd_push_constants(
    command_buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    values: *const std::ffi::c_void,
) {
    // SAFETY: a live command buffer from the application.
    let found = unsafe {
        next_device_fn("vkCmdPushConstants", command_buffer, |f| {
            f.cmd_push_constants
        })
    };
    let Some((data, next)) = found else {
        return;
    };
    let segments = ffi_guard("vkCmdPushConstants", None, || {
        let info = data.tracker.layout(layout)?;
        let ranges = info.extended.as_ref()?;
        Some(push_segments(ranges, stages, offset, size))
    });
    match segments.as_deref() {
        Some([(widened, ..)]) if *widened == stages => {}
        Some(segments) if !segments.is_empty() => {
            for &(segment_stages, segment_offset, segment_size) in segments {
                // SAFETY: the application's bytes for this segment of its push.
                let bytes = unsafe { values.cast::<u8>().add((segment_offset - offset) as usize) };
                // SAFETY: the next layer's push of part of the application's values.
                unsafe {
                    next(
                        command_buffer,
                        layout,
                        segment_stages,
                        segment_offset,
                        segment_size,
                        bytes.cast(),
                    );
                }
            }
            return;
        }
        _ => {}
    }
    // SAFETY: the next layer's push with the application's arguments.
    unsafe { next(command_buffer, layout, stages, offset, size, values) };
}

pub unsafe extern "system" fn begin_command_buffer(
    command_buffer: vk::CommandBuffer,
    info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    // SAFETY: a live command buffer from the application.
    let found = unsafe {
        next_device_fn("vkBeginCommandBuffer", command_buffer, |f| {
            f.begin_command_buffer
        })
    };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // Beginning resets all recorded state, including a pool reset's leftovers.
    ffi_guard("vkBeginCommandBuffer", (), || {
        data.commands.set(command_buffer, false);
    });
    // SAFETY: the next layer's begin with the application's arguments.
    unsafe { next(command_buffer, info) }
}

pub unsafe extern "system" fn end_command_buffer(command_buffer: vk::CommandBuffer) -> vk::Result {
    // SAFETY: a live command buffer from the application.
    let found = unsafe {
        next_device_fn("vkEndCommandBuffer", command_buffer, |f| {
            f.end_command_buffer
        })
    };
    let Some((data, next)) = found else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    ffi_guard("vkEndCommandBuffer", (), || {
        data.commands.set(command_buffer, false);
    });
    // SAFETY: the next layer's end with the application's arguments.
    unsafe { next(command_buffer) }
}

pub unsafe extern "system" fn free_command_buffers(
    device: vk::Device,
    pool: vk::CommandPool,
    count: u32,
    command_buffers: *const vk::CommandBuffer,
) {
    // SAFETY: a live device from the application.
    let found =
        unsafe { next_device_fn("vkFreeCommandBuffers", device, |f| f.free_command_buffers) };
    let Some((data, next)) = found else {
        return;
    };
    ffi_guard("vkFreeCommandBuffers", (), || {
        if command_buffers.is_null() {
            return;
        }
        // SAFETY: `count` handles, some of which may be null.
        let freed = unsafe { std::slice::from_raw_parts(command_buffers, count as usize) };
        for &command_buffer in freed {
            data.commands.set(command_buffer, false);
        }
    });
    // SAFETY: the next layer's free with the application's arguments.
    unsafe { next(device, pool, count, command_buffers) };
}

/// The next layer's draw function, or `None` when the draw is dropped.
///
/// # Safety
///
/// `command_buffer` must be a live command buffer.
unsafe fn draw_next<T>(
    name: &'static str,
    command_buffer: vk::CommandBuffer,
    select: impl FnOnce(&DeviceFns) -> Option<T>,
) -> Option<T> {
    ffi_guard(name, None, || {
        // SAFETY: forwarded from the caller.
        let data = dispatch::device(unsafe { key(command_buffer) })?;
        if data.commands.skips(command_buffer) {
            return None;
        }
        select(&data.fns)
    })
}

/// Defines a draw hook per entry that forwards unless the bound pipeline is skipped.
macro_rules! draw_hooks {
    ($($hook:ident: $name:literal ($($arg:ident: $ty:ty),*);)*) => {$(
        pub unsafe extern "system" fn $hook(command_buffer: vk::CommandBuffer, $($arg: $ty),*) {
            // SAFETY: a live command buffer from the application.
            if let Some(next) = unsafe { draw_next($name, command_buffer, |f| f.$hook) } {
                // SAFETY: the next layer's draw with the application's arguments.
                unsafe { next(command_buffer, $($arg),*) };
            }
        }
    )*};
}

draw_hooks! {
    cmd_draw: "vkCmdDraw"(
        vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32
    );
    cmd_draw_indexed: "vkCmdDrawIndexed"(
        index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32,
        first_instance: u32
    );
    cmd_draw_indirect: "vkCmdDrawIndirect"(
        buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32
    );
    cmd_draw_indexed_indirect: "vkCmdDrawIndexedIndirect"(
        buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32
    );
    cmd_draw_indirect_count: "vkCmdDrawIndirectCount"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_indirect_count_khr: "vkCmdDrawIndirectCountKHR"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_indirect_count_amd: "vkCmdDrawIndirectCountAMD"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_indexed_indirect_count: "vkCmdDrawIndexedIndirectCount"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_indexed_indirect_count_khr: "vkCmdDrawIndexedIndirectCountKHR"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_indexed_indirect_count_amd: "vkCmdDrawIndexedIndirectCountAMD"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_mesh_tasks_ext: "vkCmdDrawMeshTasksEXT"(
        group_count_x: u32, group_count_y: u32, group_count_z: u32
    );
    cmd_draw_mesh_tasks_indirect_ext: "vkCmdDrawMeshTasksIndirectEXT"(
        buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32
    );
    cmd_draw_mesh_tasks_indirect_count_ext: "vkCmdDrawMeshTasksIndirectCountEXT"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_mesh_tasks_nv: "vkCmdDrawMeshTasksNV"(task_count: u32, first_task: u32);
    cmd_draw_mesh_tasks_indirect_nv: "vkCmdDrawMeshTasksIndirectNV"(
        buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32
    );
    cmd_draw_mesh_tasks_indirect_count_nv: "vkCmdDrawMeshTasksIndirectCountNV"(
        buffer: vk::Buffer, offset: vk::DeviceSize, count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize, max_draw_count: u32, stride: u32
    );
    cmd_draw_multi_ext: "vkCmdDrawMultiEXT"(
        draw_count: u32, vertex_info: *const vk::MultiDrawInfoEXT, instance_count: u32,
        first_instance: u32, stride: u32
    );
    cmd_draw_multi_indexed_ext: "vkCmdDrawMultiIndexedEXT"(
        draw_count: u32, index_info: *const vk::MultiDrawIndexedInfoEXT, instance_count: u32,
        first_instance: u32, stride: u32, vertex_offset: *const i32
    );
    cmd_draw_indirect_byte_count_ext: "vkCmdDrawIndirectByteCountEXT"(
        instance_count: u32, first_instance: u32, counter_buffer: vk::Buffer,
        counter_buffer_offset: vk::DeviceSize, counter_offset: u32, vertex_stride: u32
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ash::vk::Handle;

    use super::*;

    fn hash(value: u64) -> ShaderHash {
        format!("{value:016x}").parse().expect("hash")
    }

    #[test]
    fn test_skipped_by_finds_any_stage_marked_skip() {
        let overrides = BTreeMap::from([
            (hash(1), ShaderOverride::Mono),
            (hash(2), ShaderOverride::Skip),
        ]);
        let lookup = |h: &ShaderHash| overrides.get(h);
        assert_eq!(skipped_by([hash(1), hash(2)], lookup), Some(hash(2)));
        assert_eq!(skipped_by([hash(1), hash(3)], lookup), None);
    }

    #[test]
    fn test_command_tracker_counts_skipping_buffers() {
        let tracker = CommandTracker::default();
        let (a, b) = (
            vk::CommandBuffer::from_raw(0x10),
            vk::CommandBuffer::from_raw(0x20),
        );
        assert!(!tracker.skips(a));
        tracker.set(a, true);
        tracker.set(a, true);
        assert!(tracker.skips(a) && !tracker.skips(b));
        assert_eq!(tracker.count.load(Ordering::Acquire), 1);
        tracker.set(a, false);
        assert!(!tracker.skips(a));
        assert_eq!(tracker.count.load(Ordering::Acquire), 0);
    }
}
