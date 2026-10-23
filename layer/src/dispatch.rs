//! Per-instance and per-device state, keyed by the loader's dispatch pointer.
//!
//! Every dispatchable handle starts with a pointer to the loader's dispatch table; all
//! handles derived from one instance (physical devices) or one device (queues, command
//! buffers) share it, so it identifies the owning object without wrapping handles.

use std::{
    collections::HashMap,
    sync::{
        Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use ash::vk::{self, Handle};

use crate::{
    commands::CommandTracker, extensions::InstanceFns, hooks::DeviceFns, pipelines::Tracker,
    stereo::StereoState, swapchain::SwapchainState,
};

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
    pub fns: InstanceFns,
    /// The next layer's instance functions, for the layer's own calls.
    pub next: ash::Instance,
    /// The process passed the opt-in gate.
    pub active: bool,
}

pub struct DeviceData {
    pub handle: vk::Device,
    pub next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub destroy: vk::PFN_vkDestroyDevice,
    pub fns: DeviceFns,
    /// The next layer's device functions, for the layer's own calls.
    pub next: ash::Device,
    pub active: bool,
    pub frames: AtomicU64,
    pub tracker: Tracker,
    pub commands: CommandTracker,
    pub stereo: StereoState,
    pub swapchains: SwapchainState,
}

impl DeviceData {
    /// Counts one presented frame and returns the new total.
    pub fn count_frame(&self) -> u64 {
        self.frames.fetch_add(1, Ordering::Relaxed) + 1
    }
}

// Read on every hooked call, including every draw; written only at create and destroy.
static INSTANCES: RwLock<Option<HashMap<Key, Arc<InstanceData>>>> = RwLock::new(None);
static DEVICES: RwLock<Option<HashMap<Key, Arc<DeviceData>>>> = RwLock::new(None);

/// Read access that survives a panic in another hook: a poisoned lock must not disable
/// the layer for the rest of the process.
pub fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Write access, recovering from poisoning like [`read`].
pub fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

type Map<T> = RwLock<Option<HashMap<Key, Arc<T>>>>;

fn get<T>(map: &Map<T>, key: Key) -> Option<Arc<T>> {
    read(map).as_ref()?.get(&key).cloned()
}

fn insert<T>(map: &Map<T>, key: Key, data: T) {
    write(map)
        .get_or_insert_with(HashMap::new)
        .insert(key, Arc::new(data));
}

fn remove<T>(map: &Map<T>, key: Key) -> Option<Arc<T>> {
    write(map).as_mut()?.remove(&key)
}

pub fn insert_instance(key: Key, data: InstanceData) {
    insert(&INSTANCES, key, data);
}

pub fn instance(key: Key) -> Option<Arc<InstanceData>> {
    get(&INSTANCES, key)
}

pub fn remove_instance(key: Key) -> Option<Arc<InstanceData>> {
    remove(&INSTANCES, key)
}

pub fn insert_device(key: Key, data: DeviceData) {
    insert(&DEVICES, key, data);
}

pub fn device(key: Key) -> Option<Arc<DeviceData>> {
    get(&DEVICES, key)
}

pub fn remove_device(key: Key) -> Option<Arc<DeviceData>> {
    remove(&DEVICES, key)
}
