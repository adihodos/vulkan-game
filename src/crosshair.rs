use crate::{
    app_config::AppConfig,
    draw_context::DrawContext,
    math,
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        UniqueImageWithView, UniqueSampler, VulkanRenderer,
    },
};
use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ColorComponentFlags, DescriptorBufferInfo,
    DescriptorImageInfo, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding,
    DescriptorType, DeviceSize, DynamicState, Filter, Format, ImageLayout, IndexType,
    MemoryPropertyFlags, PipelineBindPoint, PipelineColorBlendAttachmentState, PrimitiveTopology,
    SamplerAddressMode, SamplerMipmapMode, ShaderStageFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate, WriteDescriptorSet,
};
use memoffset::offset_of;
use nalgebra_glm as glm;

pub struct CrosshairSystem {
    ubo_transforms: Cpu2GpuBuffer<glm::Mat4>,
    vertex_buffer: Cpu2GpuBuffer<VertexPT>,
    vertices_cpu: Vec<VertexPT>,
    descriptor_sets: Vec<DescriptorSet>,
    pipeline: UniqueGraphicsPipeline,
    sampler: UniqueSampler,
    texture: UniqueImageWithView,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct VertexPT {
    pos: glm::Vec2,
    uv: glm::Vec2,
    texid: u32,
}

impl CrosshairSystem {
    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Option<CrosshairSystem> {
        let ubo_transforms = Cpu2GpuBuffer::<glm::Mat4>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let vertex_buffer = Cpu2GpuBuffer::<VertexPT>::create(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            1024,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let tex_load_work_pkg = renderer.create_work_package()?;
        let texture = UniqueImageWithView::from_ktx(
            renderer,
            &tex_load_work_pkg,
            app_config.engine.texture_path("ui/reticles/reticle.ktx2"),
        )?;

        renderer.push_work_package(tex_load_work_pkg);

        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &ash::vk::SamplerCreateInfo::builder()
                .min_lod(0f32)
                .max_lod(1f32)
                .min_filter(Filter::LINEAR)
                .mag_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK)
                .max_anisotropy(1f32)
                .build(),
        )?;

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(VertexPT, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(VertexPT, uv) as u32,
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32_UINT,
                    offset: offset_of!(VertexPT, texid) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<VertexPT>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .set_input_assembly_state(PrimitiveTopology::TRIANGLE_LIST, false)
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("crosshair.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("crosshair.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .set_depth_test(false)
            .set_colorblend_attachment(
                0,
                PipelineColorBlendAttachmentState::builder()
                    .blend_enable(true)
                    .color_blend_op(BlendOp::ADD)
                    .alpha_blend_op(BlendOp::ADD)
                    .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .src_alpha_blend_factor(BlendFactor::ONE)
                    .dst_alpha_blend_factor(BlendFactor::ZERO)
                    .color_write_mask(ColorComponentFlags::RGBA)
                    .build(),
            )
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .set(
                        0,
                        &[DescriptorSetLayoutBinding::builder()
                            .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .stage_flags(ShaderStageFlags::VERTEX)
                            .descriptor_count(1)
                            .binding(0)
                            .build()],
                    )
                    .set(
                        1,
                        &[DescriptorSetLayoutBinding::builder()
                            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .stage_flags(ShaderStageFlags::FRAGMENT)
                            .descriptor_count(1)
                            .binding(0)
                            .build()],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let layouts = [
            pipeline.descriptor_layouts()[0],
            pipeline.descriptor_layouts()[1],
        ];

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(&layouts)
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .map_err(|e| log::error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        unsafe {
            renderer.graphics_device().update_descriptor_sets(
                &[
                    WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&[DescriptorBufferInfo::builder()
                            .buffer(ubo_transforms.buffer.buffer)
                            .range(ubo_transforms.bytes_one_frame)
                            .offset(0)
                            .build()])
                        .build(),
                    WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[1])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[DescriptorImageInfo::builder()
                            .sampler(sampler.sampler)
                            .image_view(texture.image_view())
                            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .build()])
                        .build(),
                    // WriteDescriptorSet::builder()
                    //     .dst_set(descriptor_sets[2])
                    //     .dst_binding(0)
                    //     .dst_array_element(0)
                    //     .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    //     .image_info(&[DescriptorImageInfo::builder()
                    //         .sampler(sampler.sampler)
                    //         .image_view(xtex.image_view())
                    //         .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    //         .build()])
                    //     .build(),
                ],
                &[],
            );
        }

        Some(CrosshairSystem {
            ubo_transforms,
            vertex_buffer,
            vertices_cpu: Vec::new(),
            descriptor_sets,
            pipeline,
            sampler,
            texture,
        })
    }

    pub fn render(&mut self, draw_context: &DrawContext) {
        if self.vertices_cpu.is_empty() {
            return;
        }

        assert!(self.vertices_cpu.len() % 3 == 0);

        self.ubo_transforms
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|ubo| {
                let ortho = math::orthographic(
                    0f32,
                    draw_context.viewport.width,
                    0f32,
                    draw_context.viewport.height,
                    1f32,
                    0f32,
                );

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &ortho as *const _,
                        ubo.memptr() as *mut glm::Mat4,
                        1,
                    );
                }
            });

        self.vertex_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|vb| unsafe {
                std::ptr::copy_nonoverlapping(
                    self.vertices_cpu.as_ptr(),
                    vb.memptr() as *mut VertexPT,
                    self.vertices_cpu.len(),
                );
            });

        unsafe {
            draw_context.renderer.graphics_device().cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            draw_context.renderer.graphics_device().cmd_set_viewport(
                draw_context.cmd_buff,
                0,
                &[draw_context.viewport],
            );
            draw_context.renderer.graphics_device().cmd_set_scissor(
                draw_context.cmd_buff,
                0,
                &[draw_context.scissor],
            );

            draw_context
                .renderer
                .graphics_device()
                .cmd_bind_vertex_buffers(
                    draw_context.cmd_buff,
                    0,
                    &[self.vertex_buffer.buffer.buffer],
                    &[self
                        .vertex_buffer
                        .offset_for_frame(draw_context.frame_id as DeviceSize)],
                );

            //
            // draw circle reticle
            draw_context
                .renderer
                .graphics_device()
                .cmd_bind_descriptor_sets(
                    draw_context.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[self.descriptor_sets[0], self.descriptor_sets[1]],
                    &[self
                        .ubo_transforms
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32],
                );

            draw_context.renderer.graphics_device().cmd_draw(
                draw_context.cmd_buff,
                self.vertices_cpu.len() as u32,
                1,
                0,
                0,
            );
        }

        self.vertices_cpu.clear();
    }

    pub fn draw_rect(&mut self, left: f32, top: f32, width: f32, height: f32, texid: u32) {
        let vertices = [
            VertexPT {
                pos: glm::vec2(left, top + height),
                uv: glm::vec2(0f32, 0f32),
                texid,
            },
            VertexPT {
                pos: glm::vec2(left, top),
                uv: glm::vec2(0f32, 1f32),
                texid,
            },
            VertexPT {
                pos: glm::vec2(left + width, top),
                uv: glm::vec2(1f32, 1f32),
                texid,
            },
            VertexPT {
                pos: glm::vec2(left + width, top + height),
                uv: glm::vec2(1f32, 0f32),
                texid,
            },
        ];

        self.vertices_cpu
            .extend([0, 2, 1, 0, 3, 2].iter().map(|&idx| vertices[idx as usize]));
    }

    pub fn draw_rect_with_origin(
        &mut self,
        orgx: f32,
        orgy: f32,
        width: f32,
        height: f32,
        texid: u32,
    ) {
        self.draw_rect(
            orgx - width * 0.5f32,
            orgy - height * 0.5f32,
            width,
            height,
            texid,
        );
    }

    pub fn draw_crosshair(&mut self) {
        let crosshair_scale = 32f32;
        self.draw_rect_with_origin(
            1920f32 / 2f32,
            1200f32 / 2f32,
            crosshair_scale,
            crosshair_scale,
            0,
        );
        self.draw_rect_with_origin(
            1920f32 / 2f32,
            1200f32 / 2f32,
            crosshair_scale,
            crosshair_scale,
            1,
        );
    }
}
