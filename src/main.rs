mod app_config;
mod arcball_camera;
mod color_palettes;
mod debug_draw_overlay;
mod draw_context;
mod flight_cam;
mod fps_camera;
mod frustrum;
mod game_object;
mod game_world;
mod imported_geometry;
mod math;
mod missile_sys;
mod particles;
// mod pbr;
mod physics_engine;
mod plane;
mod projectile_system;
mod resource_system;
mod shadow_swarm;
mod skybox;
mod sprite_batch;
mod starfury;
mod ui_backend;
mod vk_renderer;
mod window;
mod drawing_system;
mod bindless;

#[derive(Copy, Clone, Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("Graphics api (Vulkan) error")]
    GraphicsSystemError(#[from] ash::vk::Result)
}

pub fn main() {
    self::window::MainWindow::run();
}
