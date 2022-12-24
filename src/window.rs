use std::{
    cell::{Cell, RefCell},
    time::Instant,
};

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
    camera::Camera,
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::DrawContext,
    game_world::GameWorld,
    math,
    ui_backend::UiBackend,
    vk_renderer::{UniqueImage, VulkanRenderer},
};

use nalgebra_glm as glm;

#[derive(Clone, Copy, Debug)]
pub struct GamepadStick {
    pub code: gilrs::ev::Code,
    pub deadzone: f32,
    pub axis_data: Option<gilrs::ev::state::AxisData>,
}

#[derive(Clone, Copy, Debug)]
pub struct GamepadButton {
    pub code: gilrs::ev::Code,
    pub deadzone: f32,
    pub data: Option<gilrs::ev::state::ButtonData>,
}

#[derive(Clone, Debug)]
pub struct GamepadInputState {
    pub id: gilrs::GamepadId,
    pub right_stick_x: GamepadStick,
    pub right_stick_y: GamepadStick,
    pub right_z: GamepadButton,
    pub left_z: GamepadButton,
    pub counter: u64,
}

#[derive(Clone, Debug)]
pub struct InputState {
    pub gamepad: GamepadInputState,
}

pub struct MainWindow {}

impl MainWindow {
    pub fn run() {
        let logger = flexi_logger::Logger::with(
            flexi_logger::LogSpecification::builder()
                .default(flexi_logger::LevelFilter::Debug)
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
        let mut gilrs = gilrs::Gilrs::new().expect("Failed to initialize input library");
        let mut gamepad_input_state = None;

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
                    while let Some(event) = gilrs.next_event() {
                        // log::info!("gamepad {:?}", event);

                        if gamepad_input_state.is_none() {
                            let gamepad = gilrs.gamepad(event.id);
                            let code_right_x = gamepad.axis_code(gilrs::Axis::RightStickX).unwrap();
                            let code_right_y = gamepad.axis_code(gilrs::Axis::RightStickY).unwrap();
                            let code_z_right =
                                gamepad.button_code(gilrs::Button::LeftTrigger2).unwrap();
                            let code_z_left =
                                gamepad.button_code(gilrs::Button::RightTrigger2).unwrap();

                            gamepad_input_state = Some(InputState {
                                gamepad: GamepadInputState {
                                    id: event.id,
                                    right_stick_x: GamepadStick {
                                        code: code_right_x,
                                        deadzone: gamepad.deadzone(code_right_x).unwrap_or(0.1f32),
                                        axis_data: None,
                                    },
                                    right_stick_y: GamepadStick {
                                        code: code_right_y,
                                        deadzone: gamepad.deadzone(code_right_y).unwrap_or(0.1f32),
                                        axis_data: None,
                                    },
                                    left_z: GamepadButton {
                                        code: code_z_left,
                                        deadzone: gamepad.deadzone(code_z_left).unwrap_or(0.1f32),
                                        data: None,
                                    },
                                    right_z: GamepadButton {
                                        code: code_z_right,
                                        deadzone: gamepad.deadzone(code_z_right).unwrap_or(0.1f32),
                                        data: None,
                                    },

                                    counter: gilrs.counter(),
                                },
                            })
                        }
                    }

                    gamepad_input_state.as_mut().map(|in_st| {
                        in_st.gamepad.counter = gilrs.counter();
                        let gamepad = gilrs.gamepad(in_st.gamepad.id);

                        in_st.gamepad.right_stick_x.axis_data = gamepad
                            .state()
                            .axis_data(in_st.gamepad.right_stick_x.code)
                            .copied();
                        in_st.gamepad.right_stick_y.axis_data = gamepad
                            .state()
                            .axis_data(in_st.gamepad.right_stick_y.code)
                            .copied();
                        in_st.gamepad.left_z.data = gamepad
                            .state()
                            .button_data(in_st.gamepad.left_z.code)
                            .copied();
                        in_st.gamepad.right_z.data = gamepad
                            .state()
                            .button_data(in_st.gamepad.right_z.code)
                            .copied();

                        game_main.gamepad_input(in_st);
                    });

                    gilrs.inc();
                    game_main.main();
                }

                _ => (),
            }
        });
    }
}

struct GameMain {
    ui: UiBackend,
    debug_draw_overlay: std::rc::Rc<RefCell<DebugDrawOverlay>>,
    game_world: RefCell<GameWorld>,
    timestamp: Cell<Instant>,
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
            debug_draw_overlay: std::rc::Rc::new(RefCell::new(
                DebugDrawOverlay::create(&renderer).expect("Failed to create debug draw overlay"),
            )),
            game_world: RefCell::new(game_world),
            timestamp: Cell::new(Instant::now()),
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

            _ => {}
        }

        self.game_world.borrow().input_event(event);
        self.camera.input_event(event);
        self.ui.input_event(event);
    }

    fn gamepad_input(&mut self, input_state: &InputState) {
        self.game_world.borrow().gamepad_input(input_state);
    }

    fn do_ui(&self) {
        let mut ui = self.ui.new_frame();
        self.game_world.borrow().ui(&mut ui);
    }

    fn draw_frame(&self) {
        self.renderer.begin_frame();

        let projection = math::perspective(
            75f32,
            self.framebuffer_size.x as f32 / self.framebuffer_size.y as f32,
            0.1f32,
            5000f32,
        );

        {
            self.debug_draw_overlay.borrow_mut().clear();

            let draw_context = DrawContext::create(
                &self.renderer,
                self.framebuffer_size.x,
                self.framebuffer_size.y,
                &self.camera,
                projection,
                self.debug_draw_overlay.clone(),
            );

            self.game_world.borrow().draw(&draw_context);
            self.ui.draw_frame(&draw_context);
        }

        self.debug_draw_overlay
            .borrow_mut()
            .draw(&self.renderer, &(projection * self.camera.view_transform()));

        self.renderer.end_frame();
    }

    fn main(&mut self) {
        let current_time = Instant::now();

        let frame_time = (current_time - self.timestamp.get())
            .as_secs_f64()
            .clamp(0f64, 0.25f64);
        self.timestamp.set(current_time);

        self.game_world.borrow().update(frame_time);
        self.do_ui();
        self.draw_frame();
    }
}
