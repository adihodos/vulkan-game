use std::{mem::size_of, path::Path};

use ash::vk::{
    BufferUsageFlags, DescriptorBufferInfo, DescriptorSet, DescriptorSetAllocateInfo,
    DescriptorSetLayoutBinding, DescriptorType, DeviceSize, DynamicState, Format,
    MemoryPropertyFlags, PipelineBindPoint, PrimitiveTopology, ShaderStageFlags,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    WriteDescriptorSet,
};
use memoffset::offset_of;
use nalgebra_glm as glm;

use crate::{
    draw_context::DrawContext,
    vk_renderer::{
        GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, ScopedBufferMapping,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        VulkanRenderer,
    },
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
    aligned_lines_size: DeviceSize,
    aligned_ubo_size: DeviceSize,
    descriptor_set: Vec<DescriptorSet>,
    lines_gpu: UniqueBuffer,
    uniforms: UniqueBuffer,
    pipeline: UniqueGraphicsPipeline,
    lines_cpu: Vec<Line>,
}

impl DebugDrawOverlay {
    const MAX_LINES: u64 = 4096;

    pub fn create(renderer: &VulkanRenderer) -> Option<DebugDrawOverlay> {
        let aligned_lines_size = VulkanRenderer::aligned_size_of_type::<Line>(
            renderer.device_properties().limits.non_coherent_atom_size,
        );

        let lines_gpu = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            aligned_lines_size * Self::MAX_LINES * renderer.max_inflight_frames() as u64,
        )?;

        let aligned_ubo_size = VulkanRenderer::aligned_size_of_type::<glm::Mat4>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );

        let uniforms = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            aligned_ubo_size * renderer.max_inflight_frames() as DeviceSize,
        )?;

        let pipeline = GraphicsPipelineBuilder::new()
            .set_input_assembly_state(PrimitiveTopology::POINT_LIST, false)
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(0)
                    .binding(0)
                    .format(Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(Line, start_pos) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(1)
                    .binding(0)
                    .format(Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(Line, end_pos) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(2)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(Line, start_color) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_description(
                VertexInputAttributeDescription::builder()
                    .location(3)
                    .binding(0)
                    .format(Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(Line, end_color) as u32)
                    .build(),
            )
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<Line>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::VERTEX,
                source: ShaderModuleSource::File(Path::new("data/shaders/dbg.draw.vert.spv")),
                entry_point: "main",
            })
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::GEOMETRY,
                source: ShaderModuleSource::File(Path::new("data/shaders/dbg.draw.geom.spv")),
                entry_point: "main",
            })
            .add_shader_stage(ShaderModuleDescription {
                stage: ShaderStageFlags::FRAGMENT,
                source: ShaderModuleSource::File(Path::new("data/shaders/dbg.draw.frag.spv")),
                entry_point: "main",
            })
            .add_dynamic_state(DynamicState::SCISSOR)
            .add_dynamic_state(DynamicState::VIEWPORT)
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .add_binding(
                        DescriptorSetLayoutBinding::builder()
                            .stage_flags(ShaderStageFlags::GEOMETRY)
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .descriptor_count(1)
                            .binding(0)
                            .build(),
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let descriptor_set = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(pipeline.descriptor_layouts())
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .map_err(|e| log::error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        let buff_info = [DescriptorBufferInfo::builder()
            .buffer(uniforms.buffer)
            .offset(0)
            .range(size_of::<glm::Mat4>() as DeviceSize)
            .build()];

        let write_desc_set = [WriteDescriptorSet::builder()
            .dst_set(descriptor_set[0])
            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .dst_binding(0)
            .buffer_info(&buff_info)
            .build()];

        unsafe {
            renderer
                .graphics_device()
                .update_descriptor_sets(&write_desc_set, &[]);
        }

        Some(DebugDrawOverlay {
            aligned_lines_size,
            aligned_ubo_size,
            descriptor_set,
            lines_gpu,
            uniforms,
            pipeline,
            lines_cpu: Vec::with_capacity(Self::MAX_LINES as usize),
        })
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

    pub fn clear(&mut self) {
        self.lines_cpu.clear();
    }

    pub fn draw(&mut self, renderer: &VulkanRenderer, view_projection: &glm::Mat4) {
        ScopedBufferMapping::create(
            renderer,
            &self.lines_gpu,
            Self::MAX_LINES * size_of::<Line>() as DeviceSize,
            self.aligned_lines_size * Self::MAX_LINES * renderer.current_frame_id() as DeviceSize,
        )
        .map(|buffer_mapping| unsafe {
            std::ptr::copy_nonoverlapping(
                self.lines_cpu.as_ptr(),
                buffer_mapping.memptr() as *mut Line,
                self.lines_cpu.len(),
            );
        });

        ScopedBufferMapping::create(
            renderer,
            &self.uniforms,
            size_of::<glm::Mat4>() as DeviceSize,
            self.aligned_ubo_size * renderer.current_frame_id() as DeviceSize,
        )
        .map(|mapping| unsafe {
            let mtx_slice = view_projection.as_slice();
            std::ptr::copy_nonoverlapping(
                mtx_slice.as_ptr(),
                mapping.memptr() as *mut f32,
                mtx_slice.len(),
            );
        });

        unsafe {
            renderer.graphics_device().cmd_bind_pipeline(
                renderer.current_command_buffer(),
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            let viewports = [renderer.viewport()];
            let scissors = [renderer.scissor()];

            renderer.graphics_device().cmd_set_viewport(
                renderer.current_command_buffer(),
                0,
                &viewports,
            );
            renderer.graphics_device().cmd_set_scissor(
                renderer.current_command_buffer(),
                0,
                &scissors,
            );

            let vertex_buffers = [self.lines_gpu.buffer];
            let vertex_offsets = [self.aligned_lines_size
                * Self::MAX_LINES
                * renderer.current_frame_id() as DeviceSize];

            renderer.graphics_device().cmd_bind_vertex_buffers(
                renderer.current_command_buffer(),
                0,
                &vertex_buffers,
                &vertex_offsets,
            );

            let descriptor_offsets = [self.aligned_ubo_size as u32 * renderer.current_frame_id()];
            renderer.graphics_device().cmd_bind_descriptor_sets(
                renderer.current_command_buffer(),
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &self.descriptor_set,
                &descriptor_offsets,
            );

            renderer.graphics_device().cmd_draw(
                renderer.current_command_buffer(),
                self.lines_cpu.len() as u32,
                1,
                0,
                0,
            );
        }
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
        let color: palette::Packed = palette::Srgba::new(color[0], color[1], color[2], color[3])
            .into_format()
            .into();
        self.add_line(
            a.to_homogeneous().xyz(),
            b.to_homogeneous().xyz(),
            color.color,
            color.color,
        );
    }
}
