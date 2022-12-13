use ash::vk::{CommandBuffer, Extent2D, Offset2D, Rect2D, Viewport};

use crate::{camera::Camera, vk_renderer::VulkanRenderer};

pub struct DrawContext<'a> {
    pub renderer: &'a VulkanRenderer,
    pub cmd_buff: CommandBuffer,
    pub frame_id: u32,
    pub viewport: Viewport,
    pub scissor: Rect2D,
    pub camera: &'a dyn Camera, // pub view_matrix: Mat4,
                                // pub eye_pos: Vec3,
}

impl<'a> DrawContext<'a> {
    pub fn create(
        renderer: &'a VulkanRenderer,
        width: i32,
        height: i32,
        camera: &'a dyn Camera,
    ) -> Self {
        Self {
            renderer,
            cmd_buff: renderer.current_command_buffer(),
            frame_id: renderer.current_frame_id(),
            viewport: Viewport {
                x: 0f32,
                y: 0f32,
                width: width as f32,
                height: height as f32,
                min_depth: 1f32,
                max_depth: 0f32,
            },
            scissor: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
            camera,
        }
    }
}
