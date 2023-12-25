use std::path::Path;

use ash::vk::{
    BufferUsageFlags, DynamicState, Filter, MemoryPropertyFlags, PipelineBindPoint,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderStageFlags,
};

use crate::{
    bindless::BindlessResourceHandle,
    draw_context::{DrawContext, InitContext},
    resource_system::SamplerDescription,
    vk_renderer::{
        BindlessPipeline, GraphicsPipelineBuilder, ShaderModuleDescription, ShaderModuleSource,
        UniqueBuffer, UniqueImageWithView,
    },
    ProgramError,
};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct SkyboxData {
    global_ubo_handle: u32,
    skybox_prefiltered: u32,
    skybox_irradiance: u32,
    skybox_brdf: u32,
}

#[derive(Copy, Clone)]
pub struct SkyboxResource {
    pub prefiltered: BindlessResourceHandle,
    pub irradiance: BindlessResourceHandle,
    pub brdf_lut: BindlessResourceHandle,
}

pub struct Skybox {
    id: u32,
    pipeline: BindlessPipeline,
    ssbo: UniqueBuffer,
    ssbo_handles: Vec<BindlessResourceHandle>,
    skyboxes: Vec<SkyboxResource>,
}

impl Skybox {
    pub fn create(init_ctx: &mut InitContext) -> Option<Self> {
        let ssbo = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            1,
            std::mem::size_of::<SkyboxData>(),
            init_ctx.renderer.max_inflight_frames(),
        )
            .expect("xxx");

	init_ctx.renderer.debug_set_object_tag("skybox/SSBO skybox data", &ssbo);

        let ssbo_handles = init_ctx.rsys.bindless.register_chunked_ssbo(
            init_ctx.renderer,
            &ssbo,
            init_ctx.renderer.max_inflight_frames() as usize,
        );

	log::info!("Skybox: registered ssbo: {:#?}", ssbo_handles);

        let skyboxes = Self::load_skyboxes(init_ctx);
        if skyboxes.len() == 0 {
            return None;
        }

        Some(Self {
            id: 0,
            ssbo,
            ssbo_handles,
            skyboxes,
            pipeline: Self::create_pipeline(init_ctx).expect("xxx"),
        })
    }

    pub fn ssbo_for_frame(&self, frame: u32) -> BindlessResourceHandle {
        self.ssbo_handles[frame as usize]
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        let graphics_device = draw_context.renderer.graphics_device();

        let skybox = self.skyboxes[self.id as usize];

        let skybox_data = SkyboxData {
            global_ubo_handle: draw_context.global_ubo_handle,
            skybox_prefiltered: skybox.prefiltered.handle(),
            skybox_irradiance: skybox.irradiance.handle(),
            skybox_brdf: skybox.brdf_lut.handle(),
        };

        self.ssbo
            .map_for_frame(draw_context.renderer, draw_context.frame_id)
            .map(|mut buf| unsafe {
                std::ptr::copy_nonoverlapping(
                    &skybox_data as *const _,
                    buf.as_mut_ptr() as *mut SkyboxData,
                    1,
                );
            })
            .expect("Failed to map/update SSBO");

        let ssbo_handle = self.ssbo_handles[draw_context.frame_id as usize].handle();

        unsafe {
            graphics_device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );
            graphics_device.cmd_push_constants(
                draw_context.cmd_buff,
                self.pipeline.layout,
                ShaderStageFlags::ALL,
                0,
                &ssbo_handle.to_le_bytes(),
            );
            graphics_device.cmd_set_viewport(draw_context.cmd_buff, 0, &[draw_context.viewport]);
            graphics_device.cmd_set_scissor(draw_context.cmd_buff, 0, &[draw_context.scissor]);
            graphics_device.cmd_draw(draw_context.cmd_buff, 6, 1, 0, 0);
        }
    }

    fn load_skyboxes(init_ctx: &mut InitContext) -> Vec<SkyboxResource> {
        let scene = &init_ctx.cfg.scene;
        let skybox_work_pkg = init_ctx
            .renderer
            .create_work_package()
            .expect("Failed to create work package");
        let mut loaded = 0u32;
        let mut skyboxes = Vec::<SkyboxResource>::new();

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

                Some((base_color, irradiance, brdf_lut, skybox_desc.tag.clone()))
            })
            .for_each(|(specular, irradiance, brdf_lut, tag)| {
                loaded += 1;

                let sampler_spec = SamplerDescription(
                    *SamplerCreateInfo::builder()
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
                );

                let prefiltered_handle = init_ctx.rsys.add_texture_bindless(
                    &format!("skybox/{tag}/prefiltered"),
                    init_ctx.renderer,
                    specular,
                    Some(sampler_spec),
                );

                let sampler_irradiance = SamplerDescription(
                    *SamplerCreateInfo::builder()
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
                );

                let irradiance_handle = init_ctx.rsys.add_texture_bindless(
                    &format!("skybox/{tag}/irradiance"),
                    init_ctx.renderer,
                    irradiance,
                    Some(sampler_irradiance),
                );

                let brdf_lut_handle = init_ctx.rsys.add_texture_bindless(
                    &format!("skybox/{tag}/brdf_lut"),
                    init_ctx.renderer,
                    brdf_lut,
                    None,
                );

                skyboxes.push(SkyboxResource {
                    prefiltered: prefiltered_handle,
                    irradiance: irradiance_handle,
                    brdf_lut: brdf_lut_handle,
                })
            });

        if loaded != 0 {
            init_ctx.renderer.push_work_package(skybox_work_pkg);
        }

        skyboxes
    }

    fn create_pipeline(init_ctx: &mut InitContext) -> Result<BindlessPipeline, ProgramError> {
        GraphicsPipelineBuilder::new()
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("skybox.bindless.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &init_ctx.cfg.engine.shader_path("skybox.bindless.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build_bindless(
                init_ctx.renderer.graphics_device(),
                init_ctx.renderer.pipeline_cache(),
                init_ctx.rsys.bindless.bindless_pipeline_layout(),
                init_ctx.renderer.renderpass(),
                0,
            )
    }
}
