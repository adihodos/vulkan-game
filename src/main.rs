mod app_config;
mod arcball_camera;
mod camera;
mod color_palettes;
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
mod sprite_batch;
mod starfury;
mod ui_backend;
mod vk_renderer;
mod window;
mod plane;
mod test_world;
mod fps_camera;
mod frustrum;

pub fn main() {
    self::window::MainWindow::run();
}
