//! Vulkan 1.3 backend for the test scene.
//!
//! Types here hold raw `ash` handles, which do not implement `Debug`, so they derive
//! nothing and release their handles in `Drop`.

mod context;
mod memory;
mod passes;
mod pipeline;
mod renderer;
mod swapchain;

use clap::ValueEnum;

pub use context::validation_error_count;
pub use renderer::{Readback, Renderer};

/// How draws are wrapped, so the layer is exercised on both paths games use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RenderingMode {
    /// `VkRenderPass` and framebuffers.
    Pass,
    /// Core 1.3 dynamic rendering (`vkCmdBeginRendering`).
    Dynamic,
}

impl RenderingMode {
    /// Label recorded in capture sidecars.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Dynamic => "dynamic",
        }
    }
}
