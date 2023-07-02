use ash::vk::{CommandBuffer, Rect2D, Viewport};

use crate::{
    camera::Camera, debug_draw_overlay::DebugDrawOverlay, game_world::QueuedCommand,
    physics_engine::PhysicsEngine, vk_renderer::VulkanRenderer, resource_cache::ResourceHolder,
};

pub struct FrameRenderContext<'a> {
    pub renderer: &'a VulkanRenderer,
    pub cmd_buff: CommandBuffer,
    pub frame_id: u32,
    pub viewport: Viewport,
    pub scissor: Rect2D,
    pub framebuffer_size: nalgebra_glm::IVec2,
}

pub struct DrawContext<'a> {
    pub renderer: &'a VulkanRenderer,
    pub rcache: &'a ResourceHolder,
    pub cmd_buff: CommandBuffer,
    pub frame_id: u32,
    pub viewport: Viewport,
    pub scissor: Rect2D,

    pub cam_position: nalgebra_glm::Vec3,
    pub view_matrix: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub inverse_projection: nalgebra_glm::Mat4,
    pub projection_view: nalgebra_glm::Mat4,
    pub debug_draw: std::rc::Rc<std::cell::RefCell<DebugDrawOverlay>>,
}

pub struct UpdateContext<'a> {
    pub frame_time: f64,
    pub physics_engine: &'a mut PhysicsEngine,
    pub queued_commands: Vec<QueuedCommand>,
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
