use winit::{event_loop, window::{WindowBuilder, Fullscreen}, event::{WindowEvent, Event}};
use raw_window_handle::HasRawWindowHandle;


pub struct MainWindow {

}

impl MainWindow {
    pub fn run() {
        let event_loop = event_loop::EventLoop::new();
        let window = WindowBuilder::new().with_title("Rust + Vulkan + B5")
            .with_fullscreen(Some(Fullscreen::Borderless(event_loop.primary_monitor())))
            .build(&event_loop).expect("Failed to create window!");

        let wh = window.raw_window_handle();
        // winit::platform::unix::WindowExtUnix::

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
                ,

                Event::MainEventsCleared => {
                    window.request_redraw();
                },

                _ => ()
            }
        });

    }

}
