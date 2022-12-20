use std::cell::RefCell;

use nalgebra_glm::{IVec2, Vec3};
use raw_window_handle::HasRawWindowHandle;
use winit::{
    event::{Event, WindowEvent},
    event_loop,
    window::{Fullscreen, WindowBuilder},
};

use crate::{
    app_config::{self, AppConfig},
    arcball_camera::ArcballCamera,
    draw_context::DrawContext,
    game_world::GameWorld,
    ui_backend::UiBackend,
    vk_renderer::{UniqueImage, VulkanRenderer},
};

use nalgebra_glm as glm;

pub struct MainWindow {}

impl MainWindow {
    pub fn run() {
        let logger = flexi_logger::Logger::with(
            flexi_logger::LogSpecification::builder()
                .default(flexi_logger::LevelFilter::Trace)
                .build(),
        )
        .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
        .start()
        .unwrap_or_else(|e| {
            panic!("Failed to start the logger {}", e);
        });

        let app_config = crate::app_config::AppConfig::load();

        log::info!("uraaa this be info!");
        log::warn!("urraa! this be warn cyka!");
        log::error!("urrra! this be error pierdole!");
        log::trace!("urrraa ! this be trace blyat!");
        log::debug!("urraa! this be debug, kurwa jebane !");

        let event_loop = event_loop::EventLoop::new();
        let pmon = event_loop
            .primary_monitor()
            .expect("Failed to get the primary monitor!");
        let vidmode = pmon.video_modes().next().expect("Failed to get video mode");

        log::info!("Primary monitor video mode: {}", vidmode);

        let window = winit::window::Window::new(&event_loop).expect("Failed to create window");

        window.set_decorations(false);
        window.set_visible(true);
        window.set_inner_size(vidmode.size());
        window.set_fullscreen(Some(Fullscreen::Exclusive(vidmode)));

        let mut game_main = GameMain::new(&window);

        event_loop.run(move |event, _, control_flow| {
            control_flow.set_poll();

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    log::info!("Shutting down ...");
                    control_flow.set_exit();
                }

                Event::WindowEvent { window_id, event } => {
                    game_main.handle_event(&event);
                }

                Event::MainEventsCleared => {
                    game_main.main();
                }

                _ => (),
            }
        });
    }
}

struct GameMain {
    ui: UiBackend,
    game_world: RefCell<GameWorld>,
    app_config: AppConfig,
    renderer: VulkanRenderer,
    camera: ArcballCamera,
    framebuffer_size: IVec2,
}

impl std::ops::Drop for GameMain {
    fn drop(&mut self) {
        log::info!("Waiting for all GPU submits to finish ...");
        self.renderer.wait_idle();
    }
}

impl GameMain {
    fn new(window: &winit::window::Window) -> GameMain {
        let app_config = AppConfig::load();

        let renderer = VulkanRenderer::create(&window).expect("Failed to create renderer!");
        renderer.begin_resource_loading();

        let ui = UiBackend::new(&renderer, &window).expect("Failed to create ui backend");
        let game_world =
            GameWorld::new(&renderer, &app_config).expect("Failed to create game world");

        renderer.wait_all_work_packages();
        renderer.wait_resources_loaded();
        log::info!("Resource loaded ...");

        let client_size = window.inner_size();
        let framebuffer_size = IVec2::new(client_size.width as i32, client_size.height as i32);

        GameMain {
            ui,
            game_world: RefCell::new(game_world),
            app_config,
            renderer,
            camera: ArcballCamera::new(Vec3::new(0f32, 0f32, 0f32), 0.1f32, framebuffer_size),
            framebuffer_size,
        }
    }

    fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        match event {
            WindowEvent::Resized(new_size) => {
                self.framebuffer_size = IVec2::new(new_size.width as i32, new_size.height as i32);
            }

            _ => (),
        }

        self.camera.input_event(event);
        self.ui.input_event(event);
    }

    fn do_ui(&self) {
        let mut ui = self.ui.new_frame();
        self.game_world.borrow().ui(&mut ui);
    }

    fn draw_frame(&self) {
        self.renderer.begin_frame();

        {
            let draw_context = DrawContext::create(
                &self.renderer,
                self.framebuffer_size.x,
                self.framebuffer_size.y,
                &self.camera,
                perspective(
                    75f32,
                    self.framebuffer_size.x as f32 / self.framebuffer_size.y as f32,
                    0.1f32,
                    5000f32,
                ),
            );

            self.game_world.borrow().draw(&draw_context);
            self.ui.draw_frame(&draw_context);
        }

        self.renderer.end_frame();
    }

    fn main(&mut self) {
        self.do_ui();
        self.draw_frame();
    }
}

/// Symmetric perspective projection with reverse depth (1.0 -> 0.0) and
/// Vulkan coordinate space.
pub fn perspective(vertical_fov: f32, aspect_ratio: f32, n: f32, f: f32) -> glm::Mat4 {
    let fov_rad = vertical_fov * 2.0f32 * std::f32::consts::PI / 360.0f32;
    let focal_length = 1.0f32 / (fov_rad / 2.0f32).tan();

    let x = focal_length / aspect_ratio;
    let y = -focal_length;
    let a: f32 = n / (f - n);
    let b: f32 = f * a;

    // clang-format off
    glm::Mat4::from_column_slice(&[
        x, 0.0f32, 0.0f32, 0.0f32, 0.0f32, y, 0.0f32, 0.0f32, 0.0f32, 0.0f32, a, -1.0f32, 0.0f32,
        0.0f32, b, 0.0f32,
    ])

    //   if (inverse)
    //   {
    //       *inverse = glm::mat4{
    //           1/x,  0.0f, 0.0f,  0.0f,
    //           0.0f,  1/y, 0.0f,  0.0f,
    //           0.0f, 0.0f, 0.0f, -1.0f,
    //           0.0f, 0.0f,  1/B,   A/B,
    //       };
    //   }
    //
    // // clang-format on
    // return projection;
}
