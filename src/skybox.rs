#[allow(dead_code)]
use std::{mem::size_of, path::Path, slice::from_raw_parts};

use ash::vk::{
    BorderColor, BufferUsageFlags, CompareOp, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType, DeviceSize,
    DynamicState, Filter, ImageLayout, ImageTiling, ImageUsageFlags, ImageViewCreateInfo,
    IndexType, MemoryPropertyFlags, PipelineBindPoint, PushConstantRange, SamplerAddressMode,
    SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags, WriteDescriptorSet,
};
use log::{error, info};
use nalgebra_glm::Mat4;

use crate::{
    app_config::{EngineConfig, SceneDescription},
    draw_context::DrawContext,
    vk_renderer::{
        GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, ScopedBufferMapping,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        UniqueImage, UniqueImageView, UniqueImageWithView, UniqueSampler, VulkanRenderer,
    },
};

pub struct SkyboxIBL {
    pub specular: UniqueImageWithView,
    pub irradiance: UniqueImageWithView,
    pub brdf_lut: UniqueImageWithView,
}

pub struct Skybox {
    ibl: Vec<SkyboxIBL>,
    index_buffer: UniqueBuffer,
    pipeline: UniqueGraphicsPipeline,
    sampler: UniqueSampler,
    descriptor_set: Vec<DescriptorSet>,
    pub active_skybox: u32,
}

impl Skybox {
    pub fn create(
        renderer: &VulkanRenderer,
        scene: &SceneDescription,
        engine_cfg: &EngineConfig,
    ) -> Option<Skybox> {
        let skybox_work_pkg = renderer.create_work_package()?;

        let ibl = scene
            .skyboxes
            .iter()
            .filter_map(|skybox_desc| {
                info!("Loading skybox {}", skybox_desc.tag);

                let skybox_texture_dir = Path::new(&engine_cfg.textures).join(&skybox_desc.path);

                let base_color = UniqueImageWithView::from_ktx(
                    renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.specular.ktx2"),
                )?;

                let irradiance = UniqueImageWithView::from_ktx(
                    renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.irradiance.ktx2"),
                )?;

                let brdf_lut = UniqueImageWithView::from_ktx(
                    renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.brdf.lut.ktx2"),
                )?;

                Some(SkyboxIBL {
                    specular: base_color,
                    irradiance,
                    brdf_lut,
                })
            })
            .collect::<Vec<_>>();

        renderer.push_work_package(skybox_work_pkg);

        if ibl.len() != scene.skyboxes.len() {
            error!("Failed to load all skyboxes");
            return None;
        }

        let max_lod = ibl[0].specular.info().num_levels as f32;

        let indices = [0, 3, 2, 0, 2, 1];
        let index_buffer = UniqueBuffer::gpu_only_buffer(
            renderer,
            BufferUsageFlags::INDEX_BUFFER,
            MemoryPropertyFlags::DEVICE_LOCAL,
            &[&indices],
            None,
        )?;

        let pipeline = GraphicsPipelineBuilder::new()
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(&engine_cfg.shader_path("skybox.vert.spv")),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(&engine_cfg.shader_path("skybox.frag.spv")),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .set_depth_compare_op(CompareOp::LESS_OR_EQUAL)
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .add_push_constant(
                        PushConstantRange::builder()
                            .size(size_of::<Mat4>() as u32 * renderer.max_inflight_frames())
                            .stage_flags(ShaderStageFlags::VERTEX)
                            .offset(0)
                            .build(),
                    )
                    .set(
                        0,
                        &[DescriptorSetLayoutBinding::builder()
                            .stage_flags(ShaderStageFlags::FRAGMENT)
                            .binding(0)
                            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .build()],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &SamplerCreateInfo::builder()
                .min_filter(Filter::LINEAR)
                .mag_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0f32)
                .max_lod(max_lod)
                .max_anisotropy(1f32)
                .compare_op(CompareOp::NEVER)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .build(),
        )?;

        let descriptor_set = ibl
            .iter()
            .map(|skybox_ibl| {
                let ds = unsafe {
                    renderer.graphics_device().allocate_descriptor_sets(
                        &DescriptorSetAllocateInfo::builder()
                            .set_layouts(pipeline.descriptor_layouts())
                            .descriptor_pool(renderer.descriptor_pool())
                            .build(),
                    )
                }
                .expect("Failed to allocate descriptor sets");

                assert!(ds.len() == 1);

                let dsi = [DescriptorImageInfo::builder()
                    .sampler(sampler.sampler)
                    .image_view(skybox_ibl.specular.image_view())
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build()];
                let wds = [WriteDescriptorSet::builder()
                    .dst_set(ds[0])
                    .image_info(&dsi)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .build()];

                unsafe {
                    renderer.graphics_device().update_descriptor_sets(&wds, &[]);
                }

                ds[0]
            })
            .collect::<Vec<_>>();

        Some(Skybox {
            ibl,
            index_buffer,
            pipeline,
            sampler,
            descriptor_set,
            active_skybox: 0u32,
        })
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        let graphics_device = draw_context.renderer.graphics_device();

        unsafe {
            let viewports = [draw_context.viewport];
            let scissors = [draw_context.scissor];

            graphics_device.cmd_set_viewport(draw_context.cmd_buff, 0, &viewports);
            graphics_device.cmd_set_scissor(draw_context.cmd_buff, 0, &scissors);
            graphics_device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            graphics_device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.index_buffer.buffer,
                0,
                IndexType::UINT32,
            );

            let world_view_proj = draw_context.camera.view_transform();
            let world_view_proj = from_raw_parts(
                world_view_proj.as_ptr() as *const u8,
                world_view_proj.len() * size_of::<f32>(),
            );

            let pushconst_offset = size_of::<Mat4>() as u32 * draw_context.frame_id;
            graphics_device.cmd_push_constants(
                draw_context.cmd_buff,
                self.pipeline.layout,
                ShaderStageFlags::VERTEX,
                pushconst_offset,
                world_view_proj,
            );

            let desc_sets = [self.descriptor_set[self.active_skybox as usize]];
            graphics_device.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &desc_sets,
                &[],
            );

            graphics_device.cmd_draw_indexed(draw_context.cmd_buff, 6, 1, 0, 0, 0);
        }
    }

    pub fn get_ibl_data(&self) -> &[SkyboxIBL] {
        self.ibl.as_ref()
    }
}
