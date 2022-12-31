mod app_config;
mod arcball_camera;
mod camera;
mod debug_draw_overlay;
mod draw_context;
mod flight_cam;
mod game_object;
mod game_world;
mod imported_geometry;
mod math;
mod particles;
mod pbr;
mod physics_engine;
mod projectile_system;
mod resource_cache;
mod shadow_swarm;
mod skybox;
mod starfury;
mod ui_backend;
mod vk_renderer;
mod window;

pub fn main() {
    self::window::MainWindow::run();
}
