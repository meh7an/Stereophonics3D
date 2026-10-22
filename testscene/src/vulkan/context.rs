//! Instance, surface, device and queue setup.

use std::{
    ffi::{CStr, c_void},
    sync::atomic::{AtomicU32, Ordering},
};

use anyhow::{Context as _, Result, bail};
use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use s3d_shared::ffi::ffi_guard;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

/// Number of error-severity messages reported by the validation layer so far.
pub fn validation_error_count() -> u32 {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// Everything below the swapchain: loader, instance, surface, device and one queue.
pub struct Context {
    pub instance: Instance,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device: Device,
    pub swapchain_loader: swapchain::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    // Keeps the loader DLL mapped for as long as any handle above exists.
    _entry: Entry,
}

impl Context {
    pub fn new(window: &Window, validate: bool) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("failed to load the Vulkan loader")?;
        let display = window.display_handle()?.as_raw();
        let handle = window.window_handle()?.as_raw();

        let mut extensions = ash_window::enumerate_required_extensions(display)?.to_vec();
        let mut layers = Vec::new();
        if validate {
            let available = unsafe { entry.enumerate_instance_layer_properties()? };
            let found = available
                .iter()
                .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
            if !found {
                bail!("--validate needs VK_LAYER_KHRONOS_validation from the Vulkan SDK");
            }
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(debug_utils::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"s3d-testscene")
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debug = if validate {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
            Some((loader, messenger))
        } else {
            None
        };

        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, display, handle, None)? };

        let (physical_device, queue_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        tracing::info!(
            device = %properties.device_name_as_c_str().unwrap_or(c"?").to_string_lossy(),
            "selected physical device"
        );

        let priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_extensions = [swapchain::NAME.as_ptr()];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features13);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        Ok(Self {
            instance,
            surface_loader,
            surface,
            physical_device,
            memory_properties,
            device,
            swapchain_loader,
            queue_family,
            queue,
            debug,
            _entry: entry,
        })
    }
}

impl Context {
    /// Prints the selected device's extensions as `extension <name>` lines, then the
    /// feature bits the Stereophonics3D layer clears as `feature <name> <0|1>` lines, so
    /// a run with the layer active shows what the application can see.
    pub fn print_device_report(&self) -> Result<()> {
        let extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(self.physical_device)?
        };
        for extension in &extensions {
            let name = extension.extension_name_as_c_str()?;
            println!("extension {}", name.to_string_lossy());
        }
        let mut object = vk::PhysicalDeviceShaderObjectFeaturesEXT::default();
        let mut library = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
        let mut identifier = vk::PhysicalDeviceShaderModuleIdentifierFeaturesEXT::default();
        {
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut object)
                .push_next(&mut library)
                .push_next(&mut identifier);
            unsafe {
                self.instance
                    .get_physical_device_features2(self.physical_device, &mut features)
            };
        }
        println!("feature shaderObject {}", object.shader_object);
        println!(
            "feature graphicsPipelineLibrary {}",
            library.graphics_pipeline_library
        );
        println!(
            "feature shaderModuleIdentifier {}",
            identifier.shader_module_identifier
        );
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Picks a Vulkan 1.3 device with dynamic rendering, swapchain support and a queue
/// family that can both draw and present, preferring discrete GPUs.
fn pick_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let mut best: Option<(bool, vk::PhysicalDevice, u32)> = None;
    for device in unsafe { instance.enumerate_physical_devices()? } {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        if properties.api_version < vk::API_VERSION_1_3
            || !supports_dynamic_rendering(instance, device)
        {
            continue;
        }
        let has_swapchain = unsafe { instance.enumerate_device_extension_properties(device)? }
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(swapchain::NAME));
        if !has_swapchain {
            continue;
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        let mut family = None;
        for (index, properties) in families.iter().enumerate() {
            let index = index as u32;
            let presents = unsafe {
                surface_loader.get_physical_device_surface_support(device, index, surface)?
            };
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) && presents {
                family = Some(index);
                break;
            }
        }
        let Some(family) = family else { continue };
        let discrete = properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        if best.is_none_or(|(best_discrete, ..)| discrete && !best_discrete) {
            best = Some((discrete, device, family));
        }
    }
    best.map(|(_, device, family)| (device, family))
        .context("no Vulkan 1.3 GPU with dynamic rendering and presentation support")
}

fn supports_dynamic_rendering(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
        unsafe { instance.get_physical_device_features2(device, &mut features) };
    }
    features13.dynamic_rendering == vk::TRUE
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    ffi_guard("debug_callback", vk::FALSE, || {
        // The loader guarantees `data` is valid for the duration of the callback.
        let message = unsafe { data.as_ref().and_then(|data| data.message_as_c_str()) }
            .map(|message| message.to_string_lossy().into_owned())
            .unwrap_or_default();
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
            tracing::error!(target: "vulkan", "{message}");
            eprintln!("[vulkan error] {message}");
        } else {
            tracing::warn!(target: "vulkan", "{message}");
            eprintln!("[vulkan warning] {message}");
        }
        vk::FALSE
    })
}
