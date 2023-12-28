use std::mem::size_of;

use ash::vk::{
    BufferUsageFlags, DeviceSize, DynamicState, Format, MemoryPropertyFlags, PipelineBindPoint,
    PrimitiveTopology, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate,
};
use memoffset::offset_of;
use nalgebra_glm as glm;

use crate::{
    color_palettes::StdColors,
    draw_context::{DrawContext, InitContext},
    frustrum::{Frustrum, FrustrumPlane},
    plane::Plane,
    vk_renderer::{
        BindlessPipeline, GraphicsPipelineBuilder, ShaderModuleDescription, ShaderModuleSource,
        UniqueBuffer,
    },
    ProgramError,
};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Line {
    start_pos: glm::Vec3,
    end_pos: glm::Vec3,
    start_color: u32,
    end_color: u32,
}

pub struct DebugDrawOverlay {
    lines_gpu: UniqueBuffer,
    pipeline: BindlessPipeline,
    lines_cpu: Vec<Line>,
}

impl DebugDrawOverlay {
    const MAX_LINES: u64 = 4096;

    pub fn new(init_ctx: &mut InitContext) -> Result<Self, ProgramError> {
        let lines_gpu = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            Self::MAX_LINES as usize,
            std::mem::size_of::<Line>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        let pipeline = GraphicsPipelineBuilder::new()
            .set_input_assembly_state(PrimitiveTopology::POINT_LIST, false)
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription::builder()
                    .location(0)
                    .binding(0)
                    .format(Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(Line, start_pos) as u32)
                    .build(),
                VertexInputAttributeDescription::builder()
                    .location(1)
                    .binding(0)
                    .format(Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(Line, end_pos) as u32)
                    .build(),
                VertexInputAttributeDescription::builder()
                    .location(2)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(Line, start_color) as u32)
                    .build(),
                VertexInputAttributeDescription::builder()
                    .location(3)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(Line, end_color) as u32)
                    .build(),
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<Line>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("dbg.draw.vert.spv"),
                    ),

                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::GEOMETRY,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("dbg.draw.geom.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("dbg.draw.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .add_dynamic_state(DynamicState::SCISSOR)
            .add_dynamic_state(DynamicState::VIEWPORT)
            .build_bindless(
                init_ctx.renderer.graphics_device(),
                init_ctx.renderer.pipeline_cache(),
                init_ctx.rsys.bindless_setup().pipeline_layout,
                init_ctx.renderer.renderpass(),
                0,
            )?;

        Ok(Self {
            lines_gpu,
            pipeline,
            lines_cpu: vec![],
        })
    }

    pub fn add_point(&mut self, p: glm::Vec3, orientation: &glm::Mat3, ext: f32, color: u32) {
        if self.lines_cpu.len() + 6 >= Self::MAX_LINES as usize {
            return;
        }

        let [ax, ay, az] = [
            orientation.column(0),
            orientation.column(1),
            orientation.column(2),
        ];

        let lines = [
            p + ax * ext,
            p - ax * ext,
            p + ay * ext,
            p - ay * ext,
            p + az * ext,
            p - az * ext,
        ];

        self.lines_cpu.extend(lines.chunks_exact(2).map(|pt| Line {
            start_pos: pt[0],
            end_pos: pt[1],
            start_color: color,
            end_color: color,
        }));
    }

    pub fn add_line(
        &mut self,
        start_pos: glm::Vec3,
        end_pos: glm::Vec3,
        start_color: u32,
        end_color: u32,
    ) {
        if self.lines_cpu.len() < Self::MAX_LINES as usize {
            self.lines_cpu.push(Line {
                start_pos,
                end_pos,
                start_color,
                end_color,
            });
        }
    }

    pub fn add_aabb_oriented(
        &mut self,
        origin: glm::Vec3,
        orientation: &glm::Mat3,
        extents: glm::Vec3,
        color: Option<u32>,
    ) {
        if self.lines_cpu.len() + 24 >= Self::MAX_LINES as usize {
            return;
        }

        let [ax, ay, az] = [
            glm::normalize(&glm::column(orientation, 0)),
            glm::normalize(&glm::column(orientation, 1)),
            glm::normalize(&glm::column(orientation, 2)),
        ];

        let o = origin;
        let [ex, ey, ez] = [extents.x, extents.y, extents.z];

        let points = [
            // 1st face
            o - az * ez + ay * ey + ax * ex,
            o - az * ez + ay * ey - ax * ex,
            o - az * ez - ay * ey - ax * ex,
            o - az * ez - ay * ey + ax * ex,
            // 2nd face
            o + az * ez + ay * ey + ax * ex,
            o + az * ez + ay * ey - ax * ex,
            o + az * ez - ay * ey - ax * ex,
            o + az * ez - ay * ey + ax * ex,
        ];

        let indices = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        let c = color.unwrap_or(0xFF0000FF);
        indices.iter().for_each(|&(i, j)| {
            self.add_line(points[i as usize], points[j as usize], c, c);
        });
    }

    pub fn add_aabb(&mut self, min: &glm::Vec3, max: &glm::Vec3, color: u32) {
        let points = [
            //
            // 1st face
            glm::vec3(max.x, max.y, max.z),
            glm::vec3(min.x, max.y, max.z),
            glm::vec3(min.x, min.y, max.z),
            glm::vec3(max.x, min.y, max.z),
            //
            // 2nd face
            glm::vec3(max.x, max.y, min.z),
            glm::vec3(min.x, max.y, min.z),
            glm::vec3(min.x, min.y, min.z),
            glm::vec3(max.x, min.y, min.z),
        ];

        let indices = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        indices.iter().for_each(|&(i, j)| {
            self.add_line(points[i as usize], points[j as usize], color, color);
        });
    }

    pub fn world_space_coord_sys(&mut self, extent: f32) {
        let axis_vectors = [
            glm::Vec3::x_axis().xyz(),
            glm::Vec3::y_axis().xyz(),
            glm::Vec3::z_axis().xyz(),
        ];

        let colors = [StdColors::RED, StdColors::GREEN, StdColors::BLUE];

        self.add_axes(
            glm::Vec3::zeros(),
            extent,
            &glm::Mat3::from_columns(&axis_vectors),
            Some(&colors),
        );
    }

    pub fn add_axes(
        &mut self,
        origin: glm::Vec3,
        extents: f32,
        orientation: &glm::Mat3,
        axis_colors: Option<&[u32]>,
    ) {
        if self.lines_cpu.len() + 6 >= Self::MAX_LINES as usize {
            return;
        }

        let axis_vectors = [
            glm::normalize(&glm::column(orientation, 0)),
            glm::normalize(&glm::column(orientation, 1)),
            glm::normalize(&glm::column(orientation, 2)),
        ];

        let axis_colors = if let Some(colors) = axis_colors {
            colors
        } else {
            &[0xFF0000FF, 0xFF00FF00, 0xFFFF0000]
        };

        axis_vectors
            .iter()
            .zip(axis_colors.iter())
            .for_each(|(axis, color)| {
                self.add_line(origin, origin + axis * extents, *color, *color);
            });
    }

    pub fn add_plane(&mut self, p: &Plane, o: &glm::Vec3, size: f32, color: u32) {
        let (u, v, w) = p.coord_sys();

        let half_size = size * 0.5f32;

        let vertices = [
            *o - (u + v) * half_size,
            *o + (u - v) * half_size,
            *o + (u + v) * half_size,
            *o + (v - u) * half_size,
        ];

        let indices = [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3];

        indices.windows(2).for_each(|idx| {
            self.add_line(vertices[idx[0]], vertices[idx[1]], color, color);
        });

        self.add_line(*o, *o + w * half_size, color, color);
    }

    pub fn add_half_plane(&mut self, p: &Plane, o: &glm::Vec3, size: f32, color: u32) {
        let (u, v, w) = p.coord_sys();

        let vertices = [
            *o,
            *o + v * size,
            *o + (u + v) * size,
            *o + u,
            *o + w * size,
        ];
        let indices = [0, 1, 1, 2, 2, 3, 3, 0, 0, 4];

        indices.windows(2).for_each(|idx| {
            self.add_line(vertices[idx[0]], vertices[idx[1]], color, color);
        });
    }

    pub fn add_frustrum(
        &mut self,
        f: &Frustrum,
        origin: &glm::Vec3,
        faces: enumflags2::BitFlags<FrustrumPlane>,
    ) {
        const PLANE_SIZE: f32 = 32f32;

        if faces.intersects(FrustrumPlane::Near) {
            self.add_plane(&f.near_face, &origin, PLANE_SIZE, StdColors::GREEN);
        }

        if faces.intersects(FrustrumPlane::Far) {
            self.add_plane(&f.far_face, &origin, PLANE_SIZE, StdColors::CORNFLOWER_BLUE);
        }

        if faces.intersects(FrustrumPlane::Bottom) {
            self.add_plane(&f.bottom_face, &origin, PLANE_SIZE, StdColors::DARK_ORANGE);
        }

        if faces.intersects(FrustrumPlane::Left) {
            self.add_plane(&f.left_face, &origin, PLANE_SIZE, StdColors::BLUE_VIOLET);
        }

        if faces.intersects(FrustrumPlane::Right) {
            self.add_plane(&f.right_face, &origin, PLANE_SIZE, StdColors::INDIAN_RED);
        }

        if faces.intersects(FrustrumPlane::Top) {
            self.add_plane(&f.top_face, &origin, PLANE_SIZE, StdColors::DARK_ORANGE);
        }
    }

    pub fn add_frustrum_pyramid(
        &mut self,
        fovy: f32,
        near: f32,
        far: f32,
        aspect: f32,
        cam_frame: (glm::Vec3, glm::Vec3, glm::Vec3),
        origin: glm::Vec3,
        color: u32,
    ) {
        let (right, up, dir) = cam_frame;

        let hnear = 2f32 * (fovy * 0.5f32).tan() * near;
        let wnear = hnear * aspect;

        let hfar = 2f32 * (fovy * 0.5f32).tan() * far;
        let wfar = hfar * aspect;

        let plane_idx = [0, 1, 1, 2, 2, 3, 3, 0];

        let points_near = [
            origin + near * dir + right * wnear * 0.5f32 - up * hnear * 0.5f32,
            origin + near * dir + right * wnear * 0.5f32 + up * hnear * 0.5f32,
            origin + near * dir - right * wnear * 0.5f32 + up * hnear * 0.5f32,
            origin + near * dir - right * wnear * 0.5f32 - up * hnear * 0.5f32,
        ];

        let points_far = [
            origin + far * dir + right * wfar * 0.5f32 - up * hfar * 0.5f32,
            origin + far * dir + right * wfar * 0.5f32 + up * hfar * 0.5f32,
            origin + far * dir - right * wfar * 0.5f32 + up * hfar * 0.5f32,
            origin + far * dir - right * wfar * 0.5f32 - up * hfar * 0.5f32,
        ];

        plane_idx.windows(2).for_each(|idx| {
            self.add_line(points_near[idx[0]], points_near[idx[1]], color, color);
        });

        plane_idx.windows(2).for_each(|idx| {
            self.add_line(points_far[idx[0]], points_far[idx[1]], color, color);
        });

        [0, 1, 2, 3].iter().for_each(|i| {
            self.add_line(origin, points_far[*i], color, color);
        });
    }

    pub fn draw(&mut self, draw_ctx: &DrawContext) {
        if self.lines_cpu.is_empty() {
            return;
        }

        let _ = self
            .lines_gpu
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id)
            .map(|mut gpu_buff| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.lines_cpu.as_ptr(),
                    gpu_buff.as_mut_ptr() as *mut Line,
                    self.lines_cpu.len(),
                );
            });

        unsafe {
            draw_ctx.renderer.graphics_device().cmd_bind_pipeline(
                draw_ctx.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );

            let viewports = [draw_ctx.renderer.viewport()];
            let scissors = [draw_ctx.renderer.scissor()];

            draw_ctx
                .renderer
                .graphics_device()
                .cmd_set_viewport(draw_ctx.cmd_buff, 0, &viewports);

            draw_ctx
                .renderer
                .graphics_device()
                .cmd_set_scissor(draw_ctx.cmd_buff, 0, &scissors);

            let vertex_buffers = [self.lines_gpu.buffer];
            let vertex_offsets = [
                self.lines_gpu.aligned_slab_size as DeviceSize * draw_ctx.frame_id as DeviceSize
            ];

            draw_ctx.renderer.graphics_device().cmd_bind_vertex_buffers(
                draw_ctx.cmd_buff,
                0,
                &vertex_buffers,
                &vertex_offsets,
            );

            draw_ctx.renderer.graphics_device().cmd_draw(
                draw_ctx.cmd_buff,
                self.lines_cpu.len() as u32,
                1,
                0,
                0,
            );
        }

        self.lines_cpu.clear();
    }
}

impl rapier3d::pipeline::DebugRenderBackend for DebugDrawOverlay {
    fn draw_line(
        &mut self,
        _object: rapier3d::prelude::DebugRenderObject,
        a: rapier3d::prelude::Point<rapier3d::prelude::Real>,
        b: rapier3d::prelude::Point<rapier3d::prelude::Real>,
        color: [f32; 4],
    ) {
        // let color = [0f32, 1f32, 0f32, 1f32];

        let (color_u32, _) =
            color
                .iter()
                .map(|x| (x * 255f32) as u8)
                .fold((0u32, 0u32), |(acc, i), c| {
                    let acc = acc | (c as u32) << (i * 8);
                    (acc, i + 1)
                });

        self.add_line(
            a.to_homogeneous().xyz(),
            b.to_homogeneous().xyz(),
            color_u32,
            color_u32,
        );
    }
}
