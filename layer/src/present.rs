//! Present-time work: composites of virtual swapchain images and the configured capture,
//! in one submission chained between the application's semaphores and the real present.
//!
//! The submission waits on the application's present semaphores and signals each
//! composite's own semaphore, which the present then waits on instead. Nothing waits on
//! the CPU except a capture, which must read its copy back before the present returns.

use std::{
    sync::{
        Arc, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use ash::vk;
use s3d_shared::settings::Snapshot;

use crate::{
    capture::{self, CaptureError, Pending, Target},
    compositor::Composite,
    dispatch::DeviceData,
    hooks::FRAME_LOG_INTERVAL,
    swapchain::Swapchain,
};

/// Fences in the ring. A slot is reused this many submissions later, long after its
/// work finished, so waiting on it before reuse does not block in practice.
const RING_SIZE: usize = 8;

/// Fences of the layer's present-time submissions, reused round-robin.
#[derive(Debug, Default)]
pub struct FenceRing {
    fences: Vec<vk::Fence>,
    /// Per fence: a submission was made with it and has not been waited on.
    pending: Vec<bool>,
    next: usize,
}

impl FenceRing {
    /// Submits `submit` on `queue` with the next fence of the ring and returns that fence.
    ///
    /// # Errors
    ///
    /// A failed fence creation, wait, reset or submission.
    ///
    /// # Safety
    ///
    /// `submit` must be a valid submission for `queue`, a queue of `device`, and the
    /// caller must hold the queue's external synchronisation.
    pub unsafe fn submit(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        submit: &vk::SubmitInfo<'_>,
    ) -> Result<vk::Fence, vk::Result> {
        // SAFETY: forwarded from the caller; every fence is ours.
        unsafe {
            while self.fences.len() < RING_SIZE {
                let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
                self.fences.push(fence);
                self.pending.push(false);
            }
            let slot = self.next;
            let fence = self.fences[slot];
            if self.pending[slot] {
                device.wait_for_fences(&[fence], true, u64::MAX)?;
                device.reset_fences(&[fence])?;
                self.pending[slot] = false;
            }
            device.queue_submit(queue, &[*submit], fence)?;
            self.pending[slot] = true;
            self.next = (slot + 1) % RING_SIZE;
            Ok(fence)
        }
    }

    /// Waits until every submission made through the ring has finished.
    ///
    /// # Safety
    ///
    /// `device` must be the ring's device.
    pub unsafe fn wait_all(&self, device: &ash::Device) {
        let pending: Vec<vk::Fence> = self
            .fences
            .iter()
            .zip(&self.pending)
            .filter(|&(_, &pending)| pending)
            .map(|(&fence, _)| fence)
            .collect();
        if pending.is_empty() {
            return;
        }
        // SAFETY: forwarded from the caller.
        if let Err(error) = unsafe { device.wait_for_fences(&pending, true, u64::MAX) } {
            tracing::warn!(%error, "waiting for present-time submissions failed");
        }
    }

    /// Waits for and destroys every fence.
    ///
    /// # Safety
    ///
    /// `device` must be the ring's device.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: forwarded from the caller; after the wait no fence is in use.
        unsafe {
            self.wait_all(device);
            for fence in self.fences.drain(..) {
                device.destroy_fence(fence, None);
            }
        }
        self.pending.clear();
        self.next = 0;
    }
}

/// Averages of the compositor's cost over one logging interval.
#[derive(Debug, Default)]
pub struct Timing {
    presents: u64,
    gpu_nanoseconds: f64,
    gpu_samples: u64,
    cpu_nanoseconds: f64,
    cpu_samples: u64,
}

/// Mean microseconds per composite (GPU) and per present (CPU) over one interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Averages {
    pub gpu_microseconds: Option<f64>,
    pub cpu_microseconds: Option<f64>,
}

impl Timing {
    /// Adds one present's measurements; every `interval` presents returns the averages
    /// and starts a new interval.
    pub fn add(&mut self, gpu: &[f64], cpu: Option<Duration>, interval: u64) -> Option<Averages> {
        self.presents += 1;
        self.gpu_nanoseconds += gpu.iter().sum::<f64>();
        self.gpu_samples += gpu.len() as u64;
        if let Some(cpu) = cpu {
            self.cpu_nanoseconds += cpu.as_nanos() as f64;
            self.cpu_samples += 1;
        }
        if self.presents < interval {
            return None;
        }
        let mean = |total: f64, samples: u64| (samples > 0).then(|| total / samples as f64 / 1e3);
        let averages = Averages {
            gpu_microseconds: mean(self.gpu_nanoseconds, self.gpu_samples),
            cpu_microseconds: mean(self.cpu_nanoseconds, self.cpu_samples),
        };
        *self = Self::default();
        Some(averages)
    }
}

/// Logged once: presents from a queue the application never retrieved through the loader.
static UNKNOWN_QUEUE: AtomicBool = AtomicBool::new(false);

/// Submits the layer's present-time work for `info`, which is present number `frame`
/// (0-based). Returns the semaphores the present must wait on instead of the
/// application's, or `None` when it goes through unchanged.
///
/// # Safety
///
/// `queue` and `info` must be the application's present arguments, with the queue's
/// external synchronisation held.
pub unsafe fn prepare(
    data: &DeviceData,
    queue: vk::Queue,
    info: &vk::PresentInfoKHR<'_>,
    frame: u64,
) -> Option<Vec<vk::Semaphore>> {
    let started = Instant::now();
    let state = &data.swapchains;
    let context = state.context.as_ref()?;
    let count = info.swapchain_count as usize;
    if count == 0 {
        return None;
    }
    let Some(family) = state.queue_family(queue) else {
        if !UNKNOWN_QUEUE.swap(true, Ordering::Relaxed) {
            tracing::warn!("present on an unknown queue goes through uncomposited");
        }
        return None;
    };
    // SAFETY: a valid present names `count` swapchains and image indices.
    let (handles, indices) = unsafe {
        (
            std::slice::from_raw_parts(info.p_swapchains, count),
            std::slice::from_raw_parts(info.p_image_indices, count),
        )
    };
    let device = &data.next;

    let mut composites = Vec::with_capacity(count);
    let mut gpu = Vec::with_capacity(count);
    for (&handle, &index) in handles.iter().zip(indices) {
        let Some(swapchain) = state.swapchain(handle) else {
            continue;
        };
        let Some(compositor) = &swapchain.compositor else {
            continue;
        };
        let index = index as usize;
        // SAFETY: the application presents an index of this swapchain, on this device.
        unsafe {
            if let Some(nanoseconds) = compositor.gpu_nanoseconds(device, context, family, index) {
                gpu.push(nanoseconds);
            }
            match compositor.composite(device, context, state.set_loader_data, family, index) {
                Ok(composite) => composites.push((swapchain.clone(), index, composite)),
                Err(error) => tracing::warn!(%error, "cannot record the composite"),
            }
        }
    }

    let snapshot = capture::wanted(frame);
    let capture = snapshot.as_ref().and_then(|_| {
        let swapchain = state.swapchain(handles[0])?;
        let index = indices[0] as usize;
        let target = Target {
            image: *swapchain.images.get(index)?,
            format: swapchain.format,
            extent: swapchain.extent,
            usage: swapchain.usage,
        };
        // SAFETY: the real image of the present, which the composite (if any) leaves in
        // PRESENT_SRC_KHR before the copy runs.
        match unsafe { capture::record(data, context, family, &target) } {
            Ok(pending) => Some(pending),
            Err(error) => {
                tracing::warn!(frame, %error, "capture failed");
                None
            }
        }
    });
    if composites.is_empty() && capture.is_none() {
        return None;
    }

    // SAFETY: forwarded from the caller.
    let fence = unsafe { submit(data, queue, info, &composites, capture.as_ref()) }?;
    for (swapchain, index, composite) in &composites {
        if composite.timed
            && let Some(compositor) = &swapchain.compositor
        {
            compositor.submitted_timed(*index);
        }
    }
    let cpu = match (capture, snapshot) {
        (Some(pending), Some(snapshot)) => {
            // SAFETY: the fence covers the submission holding the copy.
            unsafe { finish_capture(data, fence, pending, &snapshot, frame) };
            None
        }
        _ => Some(started.elapsed()),
    };
    let averages = state
        .timing
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .add(&gpu, cpu, FRAME_LOG_INTERVAL);
    if let Some(averages) = averages {
        tracing::info!(
            gpu_us = averages.gpu_microseconds,
            cpu_us = averages.cpu_microseconds,
            "compositor timing"
        );
    }
    Some(
        composites
            .iter()
            .map(|(_, _, composite)| composite.finished)
            .collect(),
    )
}

/// Submits the composites and the capture copy, waiting on the application's present
/// semaphores. Returns the submission's fence, or `None` (logged) when it failed.
///
/// # Safety
///
/// As [`prepare`], with every command buffer recorded for `queue`'s family.
unsafe fn submit(
    data: &DeviceData,
    queue: vk::Queue,
    info: &vk::PresentInfoKHR<'_>,
    composites: &[(Arc<Swapchain>, usize, Composite)],
    capture: Option<&Pending<'_>>,
) -> Option<vk::Fence> {
    let waits = if info.p_wait_semaphores.is_null() {
        &[][..]
    } else {
        // SAFETY: a valid present holds this many wait semaphores.
        unsafe {
            std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
        }
    };
    let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
    let command_buffers: Vec<vk::CommandBuffer> = composites
        .iter()
        .map(|(_, _, composite)| composite.command_buffer)
        .chain(capture.map(|pending| pending.command_buffer))
        .collect();
    let signals: Vec<vk::Semaphore> = composites
        .iter()
        .map(|(_, _, composite)| composite.finished)
        .collect();
    let submit = vk::SubmitInfo::default()
        .wait_semaphores(waits)
        .wait_dst_stage_mask(&stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signals);
    let mut ring = data
        .swapchains
        .ring
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    // SAFETY: forwarded from the caller.
    match unsafe { ring.submit(&data.next, queue, &submit) } {
        Ok(fence) => Some(fence),
        Err(error) => {
            tracing::warn!(%error, "present-time submission failed, presenting directly");
            None
        }
    }
}

/// Waits for the capture's submission and writes the capture.
///
/// # Safety
///
/// `fence` must be the fence of the submission holding `pending`'s copy.
unsafe fn finish_capture(
    data: &DeviceData,
    fence: vk::Fence,
    pending: Pending<'_>,
    snapshot: &Snapshot,
    frame: u64,
) {
    let Some(settings) = &snapshot.settings else {
        return;
    };
    // SAFETY: forwarded from the caller.
    let result = unsafe { data.next.wait_for_fences(&[fence], true, u64::MAX) }
        .map_err(CaptureError::from)
        // SAFETY: the wait above returned, so the copy finished.
        .and_then(|()| unsafe { pending.finish(settings, frame) });
    match result {
        Ok(path) => tracing::info!(frame, path = %path.display(), "captured frame"),
        Err(error) => tracing::warn!(frame, %error, "capture failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_reports_means_once_per_interval() {
        let mut timing = Timing::default();
        assert_eq!(
            timing.add(&[1_000.0], Some(Duration::from_micros(10)), 3),
            None
        );
        assert_eq!(timing.add(&[3_000.0, 2_000.0], None, 3), None);
        let averages = timing
            .add(&[], Some(Duration::from_micros(30)), 3)
            .expect("third present closes the interval");
        assert_eq!(averages.gpu_microseconds, Some(2.0));
        assert_eq!(averages.cpu_microseconds, Some(20.0));
        assert_eq!(
            timing.add(&[], None, 1),
            Some(Averages {
                gpu_microseconds: None,
                cpu_microseconds: None,
            })
        );
    }
}
