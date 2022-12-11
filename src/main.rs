use glfw::{Action, Context, Key};
use imgui::Condition;
use log::{debug, error, info, trace, warn};
use std::{
    cell::{Cell, RefCell},
    sync::mpsc::Receiver,
};

mod ui_backend;
mod vk_renderer;
use crate::vk_renderer::{DrawContext, VulkanRenderer};

struct OKurwaJebaneObject {
    value: Cell<i32>,
}

impl OKurwaJebaneObject {
    pub fn new() -> OKurwaJebaneObject {
        OKurwaJebaneObject {
            value: Cell::new(0),
        }
    }

    pub fn ui(&self, ui: &mut imgui::Ui) {
        let choices = ["test test this is 1", "test test this is 2"];
        ui.window("Hello world")
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(|| {
                ui.text_wrapped("Hello world!");
                ui.text_wrapped("O kurwa! Jebane pierdole bober!");
                if ui.button(choices[self.value.get() as usize]) {
                    self.value.set(self.value.get() + 1);
                    self.value.set(self.value.get() % 2);
                }

                ui.button("This...is...imgui-rs!");
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
    }
}

struct BasicWindow {
    glfw: glfw::Glfw,
    window: glfw::Window,
    kurwa: OKurwaJebaneObject,
    ui: ui_backend::UiBackend,
    renderer: std::cell::RefCell<VulkanRenderer>,
}

impl BasicWindow {
    pub fn new(
        glfw: glfw::Glfw,
        window: glfw::Window,
        renderer: VulkanRenderer,
    ) -> Option<BasicWindow> {
        renderer.begin_resource_loading();
        let ui = ui_backend::UiBackend::new(&renderer, &window)?;
        renderer.wait_resources_loaded();
        info!("Resource loaded ...");

        Some(BasicWindow {
            glfw,
            window,
            kurwa: OKurwaJebaneObject::new(),
            ui,
            renderer: RefCell::new(renderer),
        })
    }

    fn main_loop(&mut self, events: &Receiver<(f64, glfw::WindowEvent)>) {
        self.window.set_all_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            let queued_events = glfw::flush_messages(events);

            for (_, event) in queued_events {
                self.handle_window_event(&event);
                self.ui.handle_event(&event);
            }

            self.do_ui();
        }

        self.renderer.borrow().wait_idle();
    }

    fn do_ui(&self) {
        {
            let mut ui = self.ui.new_frame();
            self.kurwa.ui(&mut ui);
        }

        self.renderer.borrow().begin_frame();
        self.ui.draw_frame(self.renderer.borrow().draw_context());
        self.renderer.borrow().end_frame();
    }

    fn handle_window_event(&mut self, event: &glfw::WindowEvent) {
        if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = *event {
            self.window.set_should_close(true)
        }
    }
}

fn main() {
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

    info!("uraaa this be info!");
    warn!("urraa! this be warn cyka!");
    error!("urrra! this be error pierdole!");
    trace!("urrraa ! this be trace blyat!");
    debug!("urraa! this be debug, kurwa jebane !");

    glfw::init(glfw::FAIL_ON_ERRORS)
        .and_then(|mut glfw| {
            glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
            glfw.window_hint(glfw::WindowHint::Decorated(false));
            let mut vidmode = glfw.with_primary_monitor(|_, pmon| pmon.map(|p| p.get_video_mode()));

            vidmode
                .take()
                .map(move |vidmode| vidmode.map(|v| (glfw, v)))
                .ok_or(glfw::InitError::Internal)
        })
        .ok()
        .flatten()
        .and_then(|(mut glfw, vidmode)| {
            glfw.create_window(
                vidmode.width,
                vidmode.height,
                "Vulkan + Rust + Babylon5",
                glfw::WindowMode::Windowed,
            )
            .map(move |(window, events)| (glfw, window, events))
        })
        .and_then(|(mut glfw, mut window, events)| {
            let renderer = VulkanRenderer::create(&mut window)?;
            let mut wnd = BasicWindow::new(glfw, window, renderer)?;

            wnd.main_loop(&events);

            Some(())
        })
        .expect("Failed ...");
}
