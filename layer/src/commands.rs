//! Command-buffer tracking and draw skipping.
//!
//! Binding a graphics pipeline decides, against the current profile, whether the draws
//! that follow are dropped: a profile `skip` entry matching any of the pipeline's shaders
//! hides those draws. Deciding at bind time means a hot-reloaded profile applies to the
//! next recording without recreating pipelines.

use std::{
    collections::HashSet,
    sync::{
        Mutex, PoisonError, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use s3d_shared::{ShaderHash, ShaderOverride, ffi::ffi_guard};

use crate::{
    activation,
    dispatch::{self, DeviceData, key, read, write},
    hooks::{DeviceFns, next_device_fn},
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

/// Whether draws with `pipeline` bound are dropped under the current profile.
fn skips_pipeline(data: &DeviceData, pipeline: vk::Pipeline) -> bool {
    let Some(snapshot) = activation::get().snapshot() else {
        return false;
    };
    let Some(settings) = &snapshot.settings else {
        return false;
    };
    if !settings
        .shaders
        .values()
        .any(|o| matches!(o, ShaderOverride::Skip))
    {
        return false;
    }
    let Some(info) = data.tracker.pipeline(pipeline) else {
        return false;
    };
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
    if bind_point == vk::PipelineBindPoint::GRAPHICS {
        ffi_guard("vkCmdBindPipeline", (), || {
            data.commands
                .set(command_buffer, skips_pipeline(&data, pipeline));
        });
    }
    // SAFETY: the next layer's bind with the application's arguments.
    unsafe { next(command_buffer, bind_point, pipeline) };
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
