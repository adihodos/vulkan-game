use ash::vk::{CommandBuffer, Rect2D, Viewport};

use crate::{
    app_config::AppConfig, debug_draw_overlay::DebugDrawOverlay, game_world::QueuedCommand,
    physics_engine::PhysicsEngine, resource_system::ResourceSystem, vk_renderer::VulkanRenderer,
};

pub struct InitContext<'a> {
    pub window: &'a winit::window::Window,
    pub renderer: &'a VulkanRenderer,
    pub cfg: &'a AppConfig,
    pub rsys: &'a mut ResourceSystem,
}

pub struct FrameRenderContext<'a> {
    pub window: &'a winit::window::Window,
    pub renderer: &'a VulkanRenderer,
    pub cmd_buff: CommandBuffer,
    pub frame_id: u32,
    pub viewport: Viewport,
    pub scissor: Rect2D,
    pub framebuffer_size: nalgebra_glm::IVec2,
}

pub struct DrawContext<'a> {
    pub physics: &'a PhysicsEngine,
    pub renderer: &'a VulkanRenderer,
    pub rsys: &'a ResourceSystem,
    pub cmd_buff: CommandBuffer,
    pub frame_id: u32,
    pub viewport: Viewport,
    pub scissor: Rect2D,
    pub global_ubo_handle: u32,
    pub skybox_handle: u32,
    pub cam_position: nalgebra_glm::Vec3,
    pub view_matrix: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub inverse_projection: nalgebra_glm::Mat4,
    pub projection_view: nalgebra_glm::Mat4,
    pub debug_draw: std::rc::Rc<std::cell::RefCell<DebugDrawOverlay>>,
}

pub struct UpdateContext<'a> {
    pub cfg: &'a AppConfig,
    pub elapsed_time: std::time::Duration,
    pub frame_time: f64,
    pub physics_engine: &'a mut PhysicsEngine,
    pub queued_commands: Vec<QueuedCommand>,
    pub camera_pos: nalgebra_glm::Vec3,
}

impl<'a> DrawContext<'a> {
    // pub fn create(
    //     renderer: &'a VulkanRenderer,
    //     width: i32,
    //     height: i32,
    //     camera: &'a dyn Camera,
    //     projection: nalgebra_glm::Mat4,
    //     debug_draw: std::rc::Rc<std::cell::RefCell<DebugDrawOverlay>>,
    // ) -> Self {
    //     Self {
    //         renderer,
    //         cmd_buff: renderer.current_command_buffer(),
    //         frame_id: renderer.current_frame_id(),
    //         viewport: Viewport {
    //             x: 0f32,
    //             y: 0f32,
    //             width: width as f32,
    //             height: height as f32,
    //             min_depth: 1f32,
    //             max_depth: 0f32,
    //         },
    //         scissor: Rect2D {
    //             offset: Offset2D { x: 0, y: 0 },
    //             extent: Extent2D {
    //                 width: width as u32,
    //                 height: height as u32,
    //             },
    //         },
    //         camera,
    //         projection,
    //         projection_view: projection * camera.view_transform(),
    //         debug_draw,
    //     }
    // }
}
