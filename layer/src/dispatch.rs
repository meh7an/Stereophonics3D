//! Per-instance and per-device state, keyed by the loader's dispatch pointer.
//!
//! Every dispatchable handle starts with a pointer to the loader's dispatch table; all
//! handles derived from one instance (physical devices) or one device (queues, command
//! buffers) share it, so it identifies the owning object without wrapping handles.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

use ash::vk::{self, Handle};

pub type Key = usize;

/// The dispatch key of a dispatchable handle.
///
/// # Safety
///
/// `handle` must be a live dispatchable Vulkan handle created through the loader.
pub unsafe fn key(handle: impl Handle) -> Key {
    let object = handle.as_raw() as *const usize;
    // SAFETY: dispatchable handles point at an object whose first word is the loader's
    // dispatch table pointer.
    unsafe { *object }
}

pub struct InstanceData {
    pub handle: vk::Instance,
    pub next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub destroy: vk::PFN_vkDestroyInstance,
    /// The process passed the opt-in gate.
    pub active: bool,
}

pub struct DeviceData {
    pub handle: vk::Device,
    pub next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub destroy: vk::PFN_vkDestroyDevice,
    pub queue_present: Option<vk::PFN_vkQueuePresentKHR>,
    pub active: bool,
    pub frames: AtomicU64,
}

impl DeviceData {
    /// Counts one presented frame and returns the new total.
    pub fn count_frame(&self) -> u64 {
        self.frames.fetch_add(1, Ordering::Relaxed) + 1
    }
}

static INSTANCES: Mutex<Option<HashMap<Key, Arc<InstanceData>>>> = Mutex::new(None);
static DEVICES: Mutex<Option<HashMap<Key, Arc<DeviceData>>>> = Mutex::new(None);

fn with<T, R>(
    map: &Mutex<Option<HashMap<Key, Arc<T>>>>,
    f: impl FnOnce(&mut HashMap<Key, Arc<T>>) -> R,
) -> R {
    // A panic inside a hook must not disable the layer for the rest of the process.
    let mut guard = map.lock().unwrap_or_else(PoisonError::into_inner);
    f(guard.get_or_insert_with(HashMap::new))
}

pub fn insert_instance(key: Key, data: InstanceData) {
    with(&INSTANCES, |map| map.insert(key, Arc::new(data)));
}

pub fn instance(key: Key) -> Option<Arc<InstanceData>> {
    with(&INSTANCES, |map| map.get(&key).cloned())
}

pub fn remove_instance(key: Key) -> Option<Arc<InstanceData>> {
    with(&INSTANCES, |map| map.remove(&key))
}

pub fn insert_device(key: Key, data: DeviceData) {
    with(&DEVICES, |map| map.insert(key, Arc::new(data)));
}

pub fn device(key: Key) -> Option<Arc<DeviceData>> {
    with(&DEVICES, |map| map.get(&key).cloned())
}

pub fn remove_device(key: Key) -> Option<Arc<DeviceData>> {
    with(&DEVICES, |map| map.remove(&key))
}
