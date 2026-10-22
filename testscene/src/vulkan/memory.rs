//! Buffer and image allocation with one dedicated allocation per resource.
//!
//! The test scene owns a handful of resources, so a sub-allocator would add code
//! without benefit.

use anyhow::{Context as _, Result};
use ash::{Device, vk};

use super::context::Context;

/// A buffer bound to its own host-visible, host-coherent allocation.
pub struct HostBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl HostBuffer {
    pub fn new(ctx: &Context, size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Result<Self> {
        let device = &ctx.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = allocate(
            ctx,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
        Ok(Self {
            buffer,
            memory,
            size,
        })
    }

    /// Copies `bytes` to the start of the buffer.
    pub fn write(&self, device: &Device, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(bytes.len() as u64 <= self.size, "write exceeds buffer size");
        unsafe {
            let mapped =
                device.map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            device.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Returns a copy of the whole buffer. The caller must have waited for GPU writes.
    pub fn read(&self, device: &Device) -> Result<Vec<u8>> {
        let len = usize::try_from(self.size)?;
        let mut bytes = vec![0; len];
        unsafe {
            let mapped =
                device.map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), len);
            device.unmap_memory(self.memory);
        }
        Ok(bytes)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A device-local 2D image with one view.
pub struct DeviceImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

impl DeviceImage {
    pub fn new(
        ctx: &Context,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self> {
        let device = &ctx.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory = allocate(ctx, requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        unsafe { device.bind_image_memory(image, memory, 0)? };
        let view = create_view(device, image, format, aspect)?;
        Ok(Self {
            image,
            memory,
            view,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Creates a single-mip, single-layer 2D view of `image`.
pub fn create_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range(aspect));
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// The only subresource range the test scene uses: mip 0, layer 0.
pub fn subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .level_count(1)
        .layer_count(1)
}

fn allocate(
    ctx: &Context,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory> {
    let properties = &ctx.memory_properties;
    let type_index = (0..properties.memory_type_count)
        .find(|&index| {
            let supported = requirements.memory_type_bits & (1 << index) != 0;
            supported
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(flags)
        })
        .with_context(|| format!("no memory type with {flags:?}"))?;
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    Ok(unsafe { ctx.device.allocate_memory(&info, None)? })
}
