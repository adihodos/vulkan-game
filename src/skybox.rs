use std::path::Path;

use ash::vk::{
    DynamicState, Filter, PipelineBindPoint,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags,
};

use crate::{
    app_config::AppConfig,
    draw_context::{DrawContext, InitContext},
    vk_renderer::{
        GraphicsPipelineBuilder, ShaderModuleDescription,
        ShaderModuleSource, UniqueGraphicsPipeline, UniqueImageWithView, VulkanRenderer,
    }, resource_system::BindlessResourceKind,
};

pub struct Skybox {
    pub id: u32,
    count: u32,
    pipeline: UniqueGraphicsPipeline,
}

impl Skybox {
    pub fn create(init_ctx: &mut InitContext) -> Option<Self> {
        let loaded = Self::load_skyboxes(init_ctx);
        if loaded == 0 {
            return None;
        }

	let (layout, descriptor_layouts) = init_ctx.rsys.pipeline_layout();

        Some(Self {
            id: 0,
            count: loaded,
            pipeline: Self::create_pipeline(
                init_ctx.cfg,
                init_ctx.renderer,
                layout,
                descriptor_layouts
            )?,
        })
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        let graphics_device = draw_context.renderer.graphics_device();

        unsafe {
            graphics_device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            graphics_device.cmd_set_viewport(draw_context.cmd_buff, 0, &[draw_context.viewport]);
            graphics_device.cmd_set_scissor(draw_context.cmd_buff, 0, &[draw_context.scissor]);
            graphics_device.cmd_draw(draw_context.cmd_buff, 6, 1, 0, 0);
        }
    }

    fn load_skyboxes(init_ctx: &mut InitContext) -> u32 {
        let scene = &init_ctx.cfg.scene;
        let skybox_work_pkg = init_ctx
            .renderer
            .create_work_package()
            .expect("Failed to create work package");
        let mut loaded = 0u32;

        scene
            .skyboxes
            .iter()
            .filter_map(|skybox_desc| {
                log::info!("Loading skybox {}", skybox_desc.tag);

                let skybox_texture_dir =
                    Path::new(&init_ctx.cfg.engine.textures.clone()).join(&skybox_desc.path);

                let base_color = UniqueImageWithView::from_ktx(
                    init_ctx.renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.specular.ktx2"),
                )?;

                let irradiance = UniqueImageWithView::from_ktx(
                    init_ctx.renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.irradiance.ktx2"),
                )?;

                let brdf_lut = UniqueImageWithView::from_ktx(
                    init_ctx.renderer,
                    &skybox_work_pkg,
                    &skybox_texture_dir.clone().join("skybox.brdf.lut.ktx2"),
                )?;

                Some((base_color, irradiance, brdf_lut))
            })
            .for_each(|(specular, irradiance, brdf_lut)| {
                loaded += 1;

                let sampler_spec = init_ctx.rsys.get_sampler(
                    &SamplerCreateInfo::builder()
                        .min_filter(Filter::LINEAR)
                        .mag_filter(Filter::LINEAR)
                        .mipmap_mode(SamplerMipmapMode::LINEAR)
                        .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0f32)
                        .max_lod(specular.info().num_levels as f32)
                        .max_anisotropy(1f32)
                        .compare_op(ash::vk::CompareOp::NEVER)
                        .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK),
                    init_ctx.renderer,
                );

                init_ctx.rsys.add_texture(
                    specular,
                    BindlessResourceKind::SamplerEnvMapPrefiltered,
                    Some(sampler_spec),
                    init_ctx.renderer,
                );

                let sampler_irradiance = init_ctx.rsys.get_sampler(
                    &SamplerCreateInfo::builder()
                        .min_filter(Filter::LINEAR)
                        .mag_filter(Filter::LINEAR)
                        .mipmap_mode(SamplerMipmapMode::LINEAR)
                        .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0f32)
                        .max_lod(irradiance.info().num_levels as f32)
                        .max_anisotropy(1f32)
                        .compare_op(ash::vk::CompareOp::NEVER)
                        .border_color(ash::vk::BorderColor::INT_OPAQUE_BLACK),
                    init_ctx.renderer,
                );

                init_ctx.rsys.add_texture(
                    irradiance,
                    BindlessResourceKind::SamplerEnvMapIrradiance,
                    Some(sampler_irradiance),
                    init_ctx.renderer,
                );
                init_ctx.rsys.add_texture(
                    brdf_lut,
                    BindlessResourceKind::SamplerEnvMapBRDFLut,
                    None,
                    init_ctx.renderer,
                );
            });

        if loaded != 0 {
            init_ctx.renderer.push_work_package(skybox_work_pkg);
        }

        loaded
    }

    fn create_pipeline(
        app_config: &AppConfig,
        renderer: &VulkanRenderer,
        pipeline_layout: std::rc::Rc<ash::vk::PipelineLayout>,
        desc_sets_layout: std::rc::Rc<Vec<ash::vk::DescriptorSetLayout>>,
    ) -> Option<UniqueGraphicsPipeline> {
        GraphicsPipelineBuilder::new()
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("skybox.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_config.engine.shader_path("skybox.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                (pipeline_layout, desc_sets_layout),
                renderer.renderpass(),
                0,
            )
    }
}
