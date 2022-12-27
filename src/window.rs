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
    draw_context::FrameRenderContext,
    game_world::GameWorld,
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
    pub left_stick_x: GamepadStick,
    pub left_stick_y: GamepadStick,
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

                            let code_left_x = gamepad.axis_code(gilrs::Axis::LeftStickX).unwrap();
                            let code_left_y = gamepad.axis_code(gilrs::Axis::LeftStickY).unwrap();

                            gamepad_input_state = Some(InputState {
                                gamepad: GamepadInputState {
                                    id: event.id,
                                    left_stick_x: GamepadStick {
                                        code: code_left_x,
                                        deadzone: gamepad.deadzone(code_left_x).unwrap_or(0.1f32),
                                        axis_data: None,
                                    },
                                    left_stick_y: GamepadStick {
                                        code: code_left_y,
                                        deadzone: gamepad.deadzone(code_left_y).unwrap_or(0.1f32),
                                        axis_data: None,
                                    },
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

                        in_st.gamepad.left_stick_x.axis_data = gamepad
                            .state()
                            .axis_data(in_st.gamepad.left_stick_x.code)
                            .copied();

                        in_st.gamepad.left_stick_y.axis_data = gamepad
                            .state()
                            .axis_data(in_st.gamepad.left_stick_y.code)
                            .copied();

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
    game_world: RefCell<GameWorld>,
    timestamp: Cell<Instant>,
    app_config: AppConfig,
    renderer: VulkanRenderer,
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
            timestamp: Cell::new(Instant::now()),
            app_config,
            renderer,
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

        let frame_context = FrameRenderContext {
            renderer: &self.renderer,
            cmd_buff: self.renderer.current_command_buffer(),
            frame_id: self.renderer.current_frame_id(),
            viewport: self.renderer.viewport(),
            scissor: self.renderer.scissor(),
            framebuffer_size: self.framebuffer_size,
        };

        self.game_world.borrow().draw(&frame_context);
        self.ui.draw_frame(&frame_context);

        self.renderer.end_frame();
    }

    fn main(&mut self) {
        let elapsed = self.timestamp.get().elapsed();
        self.timestamp.set(Instant::now());


        let frame_time = elapsed.as_secs_f64().clamp(0f64, 0.25f64);

        self.game_world.borrow().update(frame_time);
        self.do_ui();
        self.draw_frame();
    }
}
