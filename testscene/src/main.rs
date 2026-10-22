//! Deterministic Vulkan test scene for the Real 3D guarantee (G0 to G5).
//!
//! Renders a fixed scene for a fixed number of frames and optionally captures the last
//! frame with a ground-truth sidecar, so injector output can be judged by tools instead
//! of by eye.

mod capture;
mod scene;
mod vulkan;

use std::path::PathBuf;

use anyhow::{Result, bail};
use ash::vk;
use clap::Parser;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::{
    scene::{Scene, SceneId},
    vulkan::{Renderer, RenderingMode},
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

#[derive(Debug, Parser)]
#[command(version, about = "Stereophonics3D deterministic test scene")]
struct Args {
    /// Scene to render.
    #[arg(long, value_enum, default_value_t = SceneId::G0)]
    scene: SceneId,

    /// Frames to render before exiting; the last one is captured.
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..))]
    frames: u32,

    /// Directory receiving the last frame as PNG plus JSON sidecar.
    #[arg(long)]
    capture: Option<PathBuf>,

    /// Render path to exercise.
    #[arg(long, value_enum, default_value_t = RenderingMode::Pass)]
    rendering: RenderingMode,

    /// Enable the Khronos validation layer; any validation error fails the run.
    #[arg(long)]
    validate: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _log_guard = s3d_shared::logging::init("s3d-testscene")?;
    tracing::info!(?args, "starting");

    let event_loop = EventLoop::new()?;
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?;
    app.finish()
}

/// Window, renderer and frame counter, driven by winit's event loop.
struct App {
    args: Args,
    scene: Scene,
    frame: u32,
    // Declared before `window`: the surface must be destroyed before its window.
    renderer: Option<Renderer>,
    window: Option<Window>,
    failure: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args) -> Self {
        let scene = Scene::build(args.scene, WIDTH as f32 / HEIGHT as f32);
        Self {
            args,
            scene,
            frame: 0,
            renderer: None,
            window: None,
            failure: None,
        }
    }

    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attributes = Window::default_attributes()
            .with_title("Stereophonics3D test scene")
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
            .with_resizable(false);
        let window = event_loop.create_window(attributes)?;
        let extent = vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        };
        let renderer = Renderer::new(
            &window,
            &self.scene,
            extent,
            self.args.rendering,
            self.args.validate,
        )?;
        window.request_redraw();
        self.renderer = Some(renderer);
        self.window = Some(window);
        Ok(())
    }

    /// Renders one frame and reports whether the run is complete.
    fn step(&mut self) -> Result<bool> {
        let Some(renderer) = self.renderer.as_mut() else {
            return Ok(false);
        };
        let last = self.frame + 1 == self.args.frames;
        let capture_dir = self.args.capture.as_deref().filter(|_| last);
        if let Some(rgba) = renderer.render(capture_dir.is_some())?
            && let Some(dir) = capture_dir
        {
            let sidecar = capture::Sidecar::new(
                &self.scene,
                self.frame,
                (WIDTH, HEIGHT),
                self.args.rendering,
            );
            let path = capture::write(dir, &rgba, &sidecar)?;
            tracing::info!(path = %path.display(), "captured frame");
            println!("captured {}", path.display());
        }
        self.frame += 1;
        Ok(last)
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.failure = Some(error);
        event_loop.exit();
    }

    /// Releases GPU resources, then turns the run's outcome into the exit status.
    fn finish(mut self) -> Result<()> {
        self.renderer = None;
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.frame != self.args.frames {
            bail!(
                "window closed after {} of {} frames",
                self.frame,
                self.args.frames
            );
        }
        let errors = vulkan::validation_error_count();
        if errors > 0 {
            bail!("validation layer reported {errors} error(s)");
        }
        tracing::info!(frames = self.frame, "finished");
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none()
            && let Err(error) = self.start(event_loop)
        {
            self.fail(event_loop, error);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => match self.step() {
                Ok(true) => event_loop.exit(),
                Ok(false) => {}
                Err(error) => self.fail(event_loop, error),
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
