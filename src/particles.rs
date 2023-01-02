use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ColorComponentFlags, DescriptorBufferInfo,
    DescriptorImageInfo, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding,
    DescriptorType, DeviceSize, DynamicState, Filter, Format, ImageLayout, PipelineBindPoint,
    PipelineColorBlendAttachmentState, PrimitiveTopology, SamplerAddressMode, SamplerMipmapMode,
    ShaderStageFlags, VertexInputAttributeDescription, VertexInputBindingDescription,
    VertexInputRate, WriteDescriptorSet,
};
use memoffset::offset_of;
use nalgebra::{Isometry3, Translation};
use nalgebra_glm as glm;
use rand::Rng;

use crate::{
    app_config::AppConfig,
    draw_context::{DrawContext, UpdateContext},
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ShaderModuleDescription, ShaderModuleSource, UniqueGraphicsPipeline, UniqueImageWithView,
        UniqueSampler, VulkanRenderer,
    },
};

#[derive(Copy, Clone, Debug)]
pub struct ImpactSpark {
    pub pos: nalgebra::Point3<f32>,
    pub dir: glm::Vec3,
    pub color: glm::Vec3,
    pub speed: f32,
    pub life: f32,
}

#[repr(C)]
struct GpuSparkData {
    pos: nalgebra::Point3<f32>,
    color: glm::Vec3,
    intensity: f32,
}

#[repr(C, align(16))]
struct SparksUniform {
    view_projection: glm::Mat4,
}

pub struct SparksSystem {
    uniform_buffer: Cpu2GpuBuffer<SparksUniform>,
    instances_buf: Cpu2GpuBuffer<GpuSparkData>,
    pipeline: UniqueGraphicsPipeline,
    sparks_tex: UniqueImageWithView,
    descriptor_sets: Vec<DescriptorSet>,
    sampler: UniqueSampler,
    sparks_cpu: Vec<ImpactSpark>,
    sparks_dir_vecs: Vec<glm::Vec3>,
}

/// TODO: number of spawned particles should be based on the distance between the shooter and the
/// imact point. The further away the camera is, the fewer particles get spawned
impl SparksSystem {
    const MAX_SPARKS: usize = 1024;
    const MAX_LIFE: f32 = 2f32;

    fn make_sparks_dir_vecs(radius: f32, num_sparks: i32) -> Vec<glm::Vec3> {
        let angle = (2f32 * std::f32::consts::PI) / num_sparks as f32;
        (0..num_sparks)
            .map(|i| {
                let i = i as f32;
                let (sin_theta, cos_theta) = (i * angle).sin_cos();
                glm::normalize(&glm::vec3(radius * cos_theta, radius * sin_theta, radius))
            })
            .collect()
    }

    pub fn create(renderer: &VulkanRenderer, app_config: &AppConfig) -> Option<SparksSystem> {
        let uniform_buffer = Cpu2GpuBuffer::<SparksUniform>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let instances_buf = Cpu2GpuBuffer::<GpuSparkData>::create(
            renderer,
            BufferUsageFlags::VERTEX_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            Self::MAX_SPARKS as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GpuSparkData, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GpuSparkData, color) as u32,
                },
                VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: Format::R32_SFLOAT,
                    offset: offset_of!(GpuSparkData, intensity) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<GpuSparkData>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .set_input_assembly_state(PrimitiveTopology::POINT_LIST, false)
            .shader_stages(&[
                ShaderModuleDescription {
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("sparks.vert.spv"),
                    ),
                    stage: ShaderStageFlags::VERTEX,
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("sparks.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .set_colorblend_attachment(
                0,
                PipelineColorBlendAttachmentState::builder()
                    .blend_enable(true)
                    .color_blend_op(BlendOp::ADD)
                    .alpha_blend_op(BlendOp::ADD)
                    .src_color_blend_factor(BlendFactor::ONE)
                    .dst_color_blend_factor(BlendFactor::ONE)
                    .src_alpha_blend_factor(BlendFactor::ONE)
                    .dst_alpha_blend_factor(BlendFactor::ONE)
                    .color_write_mask(ColorComponentFlags::RGBA)
                    .build(),
            )
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .set(
                        0,
                        &[
                            DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .descriptor_count(1)
                                .build(),
                            DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .stage_flags(ShaderStageFlags::FRAGMENT)
                                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .build(),
                        ],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let work_pkg = renderer.create_work_package()?;
        let sparks_tex = UniqueImageWithView::from_ktx(
            renderer,
            &work_pkg,
            app_config.engine.texture_path("particles/spark.ktx2"),
        )?;
        renderer.push_work_package(work_pkg);

        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &ash::vk::SamplerCreateInfo::builder()
                .min_lod(0f32)
                .max_lod(sparks_tex.info().num_levels as f32)
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

        let descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(pipeline.descriptor_layouts())
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
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .dst_set(descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .buffer_info(&[DescriptorBufferInfo::builder()
                            .buffer(uniform_buffer.buffer.buffer)
                            .range(uniform_buffer.bytes_one_frame)
                            .offset(0)
                            .build()])
                        .build(),
                    WriteDescriptorSet::builder()
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_set(descriptor_sets[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .image_info(&[DescriptorImageInfo::builder()
                            .sampler(sampler.sampler)
                            .image_view(sparks_tex.image_view())
                            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .build()])
                        .build(),
                ],
                &[],
            );
        }

        Some(SparksSystem {
            uniform_buffer,
            instances_buf,
            pipeline,
            sparks_tex,
            descriptor_sets,
            sampler,
            sparks_cpu: Vec::new(),
            sparks_dir_vecs: Self::make_sparks_dir_vecs(1f32, 8),
        })
    }

    pub fn spawn_sparks(&mut self, s: ImpactSpark) {
        use rand_distr::{Distribution, UnitCircle, UnitSphere};
        let mut rng = rand::thread_rng();

        self.sparks_cpu.extend((0..8).map(|_| {
            let dir = loop {
                let rng_dir: glm::Vec3 = UnitSphere.sample(&mut rng).into();
                if glm::dot(&rng_dir, &s.dir) < 0f32 {
                    break rng_dir;
                }
            };

            ImpactSpark { dir, ..s }
        }));
    }

    pub fn update(&mut self, update_context: &mut UpdateContext) {
        self.sparks_cpu.retain_mut(|s| {
            s.life -= update_context.frame_time as f32;
            if s.life > 0f32 {
                s.pos += s.dir * s.speed * update_context.frame_time as f32;
                true
            } else {
                false
            }
        });
    }

    pub fn render(&self, draw_context: &DrawContext) {
        if self.sparks_cpu.len() == 0 {
            return;
        }

        self.uniform_buffer
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|ubo| {
                let sparks_uniform = SparksUniform {
                    view_projection: draw_context.projection_view,
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &sparks_uniform as *const _,
                        ubo.memptr() as *mut SparksUniform,
                        1,
                    );
                }
            });

        self.instances_buf
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|instances| unsafe {
                let gpu_sparks = std::slice::from_raw_parts_mut(
                    instances.memptr() as *mut GpuSparkData,
                    self.sparks_cpu.len(),
                );
                self.sparks_cpu.iter().zip(gpu_sparks.iter_mut()).for_each(
                    |(cpu_spark, gpu_spark)| {
                        gpu_spark.pos = cpu_spark.pos;
                        gpu_spark.color = cpu_spark.color;
                        gpu_spark.intensity = cpu_spark.life / Self::MAX_LIFE; // TODO : hardcoded value
                    },
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
                    &[self.instances_buf.buffer.buffer],
                    &[self
                        .instances_buf
                        .offset_for_frame(draw_context.frame_id as DeviceSize)],
                );

            draw_context
                .renderer
                .graphics_device()
                .cmd_bind_descriptor_sets(
                    draw_context.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &self.descriptor_sets,
                    &[self
                        .uniform_buffer
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32],
                );
            draw_context.renderer.graphics_device().cmd_draw(
                draw_context.cmd_buff,
                self.sparks_cpu.len() as u32,
                1,
                0,
                0,
            );
        }
    }
}
