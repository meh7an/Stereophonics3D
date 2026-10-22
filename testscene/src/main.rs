//! Deterministic Vulkan test scene for the Real 3D guarantee (G0 to G5).
//!
//! Renders a fixed scene for a fixed number of frames and optionally captures the last
//! frame with a ground-truth sidecar, so injector output can be judged by tools instead
//! of by eye.

mod capture;
mod scene;
mod vulkan;

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use ash::vk;
use clap::Parser;
use s3d_shared::capture::Image;
use windows::{Win32::System::LibraryLoader::LoadLibraryW, core::HSTRING};
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

    /// Read the frame counter back from the last frame's pixels and fail on a mismatch.
    #[arg(long)]
    self_check: bool,

    /// Load these DLLs before Vulkan starts, e.g. a renamed harmless DLL to test the
    /// layer's anti-cheat refusal; repeatable.
    #[arg(long)]
    preload: Vec<PathBuf>,

    /// Print the device's extensions and the layer-hidden feature bits after start-up.
    #[arg(long)]
    list_device_extensions: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _log_guard = s3d_shared::logging::init("s3d-testscene")?;
    tracing::info!(?args, "starting");
    for path in &args.preload {
        // SAFETY: loading a DLL the user named; it stays loaded for the whole run.
        unsafe { LoadLibraryW(&HSTRING::from(path.as_os_str())) }
            .with_context(|| format!("cannot load {}", path.display()))?;
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(args);
    event_loop.run_app(&mut app)?;
    app.finish()
}

/// Window, renderer and frame counter, driven by winit's event loop.
struct App {
    args: Args,
    frame: u32,
    // Declared before `window`: the surface must be destroyed before its window.
    renderer: Option<Renderer>,
    window: Option<Window>,
    failure: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            args,
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
            &self.scene(0),
            extent,
            self.args.rendering,
            self.args.validate,
        )?;
        if self.args.list_device_extensions {
            renderer.print_device_report()?;
        }
        window.request_redraw();
        self.renderer = Some(renderer);
        self.window = Some(window);
        Ok(())
    }

    fn scene(&self, frame: u32) -> Scene {
        Scene::build(self.args.scene, WIDTH as f32 / HEIGHT as f32, frame)
    }

    /// Renders one frame and reports whether the run is complete.
    fn step(&mut self) -> Result<bool> {
        let scene = self.scene(self.frame);
        let Some(renderer) = self.renderer.as_mut() else {
            return Ok(false);
        };
        let last = self.frame + 1 == self.args.frames;
        let needs_pixels = last && (self.args.capture.is_some() || self.args.self_check);
        if let Some(rgba) = renderer.render(&scene, needs_pixels)? {
            if let Some(dir) = &self.args.capture {
                let sidecar = capture::sidecar(&scene, (WIDTH, HEIGHT), self.args.rendering.name());
                let path = capture::write(dir, &rgba, &sidecar)?;
                tracing::info!(path = %path.display(), "captured frame");
                println!("captured {}", path.display());
            }
            if self.args.self_check {
                self_check(&scene, &rgba)?;
            }
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

/// Fails unless the counter decoded from `rgba` equals the scene's counter.
fn self_check(scene: &Scene, rgba: &[u8]) -> Result<()> {
    let Some(expected) = scene.counter else {
        println!("self-check: scene {} has no counter", scene.id.name());
        return Ok(());
    };
    let probes = capture::sidecar(scene, (WIDTH, HEIGHT), "").probes;
    let image = Image {
        width: WIDTH,
        height: HEIGHT,
        rgba: rgba.to_vec(),
    };
    let decoded = s3d_verify::counter::decode(&image, &probes)?;
    if decoded != expected {
        bail!("self-check: counter reads {decoded}, frame encodes {expected}");
    }
    println!(
        "self-check: counter {decoded} matches frame {}",
        scene.frame
    );
    Ok(())
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
