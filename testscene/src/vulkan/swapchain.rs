//! Swapchain, its image views, the shared depth buffer and per-image semaphores.

use anyhow::{Context as _, Result, bail};
use ash::vk;

use super::{
    context::Context,
    memory::{DeviceImage, create_view},
};

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// UNORM formats only, so colours written by the shader reach the capture bit-exact.
const COLOR_FORMATS: [vk::Format; 2] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM];

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub depth: DeviceImage,
    /// One per image: presentation of an image may still be waiting on its semaphore
    /// when the next frame starts.
    pub render_finished: Vec<vk::Semaphore>,
}

impl Swapchain {
    pub fn new(ctx: &Context, requested: vk::Extent2D) -> Result<Self> {
        let caps = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)?
        };
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            bail!("swapchain images cannot be transfer sources, so captures are impossible");
        }
        let extent = if caps.current_extent.width == u32::MAX {
            requested
        } else {
            caps.current_extent
        };
        if extent != requested {
            bail!(
                "surface is {}x{}, expected {}x{}; captures need the fixed size",
                extent.width,
                extent.height,
                requested.width,
                requested.height
            );
        }

        let formats = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)?
        };
        let surface_format = COLOR_FORMATS
            .iter()
            .find_map(|&wanted| formats.iter().find(|f| f.format == wanted))
            .copied()
            .context("surface supports neither B8G8R8A8_UNORM nor R8G8B8A8_UNORM")?;

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|&mode| caps.supported_composite_alpha.contains(mode))
        .context("surface supports neither opaque nor inherited alpha")?;

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(ctx.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        let handle = unsafe { ctx.swapchain_loader.create_swapchain(&info, None)? };
        let images = unsafe { ctx.swapchain_loader.get_swapchain_images(handle)? };

        let views = images
            .iter()
            .map(|&image| {
                create_view(
                    &ctx.device,
                    image,
                    surface_format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let depth = DeviceImage::new(
            ctx,
            extent,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        let render_finished = images
            .iter()
            .map(|_| unsafe {
                ctx.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            handle,
            format: surface_format.format,
            extent,
            images,
            views,
            depth,
            render_finished,
        })
    }

    pub fn destroy(&self, ctx: &Context) {
        unsafe {
            for &semaphore in &self.render_finished {
                ctx.device.destroy_semaphore(semaphore, None);
            }
            self.depth.destroy(&ctx.device);
            for &view in &self.views {
                ctx.device.destroy_image_view(view, None);
            }
            ctx.swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }
}
