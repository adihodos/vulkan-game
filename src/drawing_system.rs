use std::collections::HashMap;

use ash::vk::{
    BufferUsageFlags, DrawIndexedIndirectCommand, MemoryPropertyFlags, ShaderStageFlags,
};

use fnv::FnvHashMap;
use nalgebra_glm as glm;

use crate::{
    bindless::BindlessResourceHandle,
    color::{Hsv, Rgba32F},
    draw_context::{DrawContext, InitContext, UpdateContext},
    resource_system::{EffectType, InstanceRenderInfo, MeshId, SubmeshId},
    vk_renderer::{UniqueBuffer, UniqueImageWithView},
    ProgramError,
};

struct DrawRequest {
    effect: EffectType,
    mesh: MeshId,
    submesh: Option<SubmeshId>,
    material: Option<String>,
    obj2world: nalgebra_glm::Mat4,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct PbrRenderpassHandle {
    ubo: u32,
    instances: u32,
    materials: u32,
    skybox: u32,
}

#[derive(Copy, Clone, Debug)]
pub enum EffectParameterData {
    Float32(f32),
    Uint32(u32),
    Vec3(glm::Vec3),
    Mat4(glm::Mat4),
}

// struct EffectParameter(String, EffectParameterData);

pub struct DrawingSys {
    pbr_renderpass_buff: UniqueBuffer,
    pbr_renderpass_handles: Vec<BindlessResourceHandle>,
    g_instances_buffer: UniqueBuffer,
    g_inst_buf_handle: Vec<BindlessResourceHandle>,
    drawcalls_buffer: UniqueBuffer,
    requests: Vec<DrawRequest>,
    glow_effect: EngineGlowEffect,
    effect_parameters: std::collections::HashMap<
        EffectType,
        std::collections::HashMap<String, EffectParameterData>,
    >,
}

impl DrawingSys {
    pub const MAX_INSTANCES: usize = 4096;

    pub fn create(init_ctx: &mut InitContext) -> Result<Self, ProgramError> {
        let pbr_renderpass_buff = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            1,
            std::mem::size_of::<PbrRenderpassHandle>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("DrawSys PBR/SSBO pass setup", &pbr_renderpass_buff);

        let pbr_renderpass_handles = init_ctx.rsys.bindless.register_chunked_ssbo(
            init_ctx.renderer,
            &pbr_renderpass_buff,
            init_ctx.renderer.max_inflight_frames() as usize,
        );

        log::info!(
            "Registered renderpass setup SSBO : {:#?}",
            pbr_renderpass_handles
        );

        let g_instances_buffer = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            Self::MAX_INSTANCES,
            std::mem::size_of::<InstanceRenderInfo>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("DrawSys PBR/SSBO instance data", &g_instances_buffer);

        let g_inst_buf_handle = init_ctx.rsys.bindless.register_chunked_ssbo(
            init_ctx.renderer,
            &g_instances_buffer,
            init_ctx.renderer.max_inflight_frames() as usize,
        );

        log::info!("Registered instance data SSBO: {:#?}", g_inst_buf_handle);

        let drawcalls_buffer = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::INDIRECT_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            256,
            std::mem::size_of::<DrawIndexedIndirectCommand>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("DrawSys/drawcall buffer", &drawcalls_buffer);

        Ok(Self {
            pbr_renderpass_buff,
            pbr_renderpass_handles,
            g_instances_buffer,
            g_inst_buf_handle,
            drawcalls_buffer,
            requests: Vec::with_capacity(1024),
            glow_effect: EngineGlowEffect::new(init_ctx)?,
            effect_parameters: HashMap::new(),
        })
    }

    pub fn debug_ui(&mut self, ui: &imgui::Ui) {
        use imgui::*;
        if ui.collapsing_header("Engine glow debug", TreeNodeFlags::FRAMED) {
            let mut glow_intensity = self.glow_effect.glow_color.s;
            ui.slider("Glow intensity", 0f32, 1f32, &mut glow_intensity);
            self.glow_effect.glow_color.s = glow_intensity;

            ui.slider(
                "Glow alpha",
                1f32,
                10f32,
                &mut self.glow_effect.glow_intensity,
            );
        }
    }

    pub fn set_effect_parameter(
        &mut self,
        eff_type: EffectType,
        param: String,
        data: EffectParameterData,
    ) {
        self.effect_parameters
            .entry(eff_type)
            .and_modify(|eff_table| {
                eff_table
                    .entry(param.clone())
                    .and_modify(|e| *e = data)
                    .or_insert(data);
            })
            .or_insert_with(|| HashMap::from([(param, data)]));
    }

    pub fn add_mesh(
        &mut self,
        mesh: MeshId,
        submesh: Option<SubmeshId>,
        mat_id: Option<String>,
        transform: &nalgebra_glm::Mat4,
        effect: EffectType,
    ) {
        self.requests.push(DrawRequest {
            effect,
            mesh,
            submesh,
            material: mat_id,
            obj2world: *transform,
        });
    }

    pub fn update(&mut self, ctx: &UpdateContext) {
        let angle = (std::f32::consts::FRAC_PI_4) * ctx.elapsed_time.as_secs_f32();

        self.glow_effect.noise_rot += angle;
        if self.glow_effect.noise_rot > (2f32 * std::f32::consts::PI) {
            self.glow_effect.noise_rot -= 2f32 * std::f32::consts::PI;
        }
    }

    pub fn draw(&mut self, draw_ctx: &DrawContext) {
        let _ = self
            .pbr_renderpass_buff
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id)
            .map(|mut pass_buff| {
                let pass_setup = PbrRenderpassHandle {
                    ubo: draw_ctx.global_ubo_handle,
                    instances: self.g_inst_buf_handle[draw_ctx.frame_id as usize].handle(),
                    materials: draw_ctx.rsys.material_buffer.handle(),
                    skybox: draw_ctx.skybox_handle,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &pass_setup as *const _,
                        pass_buff.as_mut_ptr() as *mut _,
                        1,
                    );
                }
            });

        use nalgebra as na;

        use crate::color::*;
        use crate::color_palettes::StdColors;
        let c: Rgba32F = StdColors::BLUE.into();

        let _ = self
            .glow_effect
            .ssbo_glowdata
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id)
            .map(|mut glow_setup_buff| {
                let glow_intensity = self
                    .effect_parameters
                    .get(&EffectType::Glow)
                    .map(|glow_eff_params| {
                        glow_eff_params.get("glow_intensity").map(|&p| match p {
                            EffectParameterData::Float32(val) => val,
                            _ => panic!("Unexpected parameter type"),
                        })
                    })
                    .flatten()
                    .unwrap_or(1f32);

                self.glow_effect.glow_color.s = glow_intensity;

                let glow_data = EngineGlowData {
                    instance_handle: self.g_inst_buf_handle[draw_ctx.frame_id as usize].handle(),
                    glow_image: self.glow_effect.glow_img_handle.handle(),
                    noise_image: self.glow_effect.noise_img_handle.handle(),
                    glow_intensity: 1f32,
                    tex_transform: na::Rotation3::from_axis_angle(
                        &na::Vector3::z_axis(),
                        self.glow_effect.noise_rot,
                    )
                    .to_homogeneous(),
                    glow_color: (Into::<Rgba32F>::into(self.glow_effect.glow_color)).into(),
                    ubo_handle: draw_ctx.global_ubo_handle,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &glow_data as *const _,
                        glow_setup_buff.as_mut_ptr() as *mut _,
                        1,
                    );
                }
            });

        self.requests
            .sort_unstable_by_key(|req| (req.effect, req.mesh, req.submesh));

        use itertools::Itertools;

        let mut instance_offset = 0u32;
        let mut instance_data = Vec::<InstanceRenderInfo>::new();
        let mut drawcalls = Vec::<DrawIndexedIndirectCommand>::new();

        #[derive(Copy, Clone)]
        struct DrawsByEffect {
            effect: EffectType,
            calls: u32,
            offset: u32,
        }

        self.requests
            .iter()
            .group_by(|req| (req.effect, req.mesh, req.submesh))
            .into_iter()
            .for_each(|(_eff_type, grp)| {
                let grp = grp.collect_vec();

                for rq in &grp {
                    let mtl_offset = rq
                        .material
                        .as_ref()
                        .map(|rq_mtl| draw_ctx.rsys.get_material_id(&rq_mtl))
                        .unwrap_or_else(|| draw_ctx.rsys.get_mesh_material(rq.mesh));

                    let inst_render_info = InstanceRenderInfo {
                        model: rq.obj2world,
                        mtl_coll_offset: mtl_offset,
                    };

                    instance_data.push(inst_render_info);
                }

                let DrawRequest {
                    mesh: mesh_id,
                    submesh: maybe_submesh_id,
                    ..
                } = *grp[0];

                let mesh = draw_ctx.rsys.get_mesh_info(mesh_id);

                let (index_count, mesh_index_offset) = maybe_submesh_id
                    .map(|submesh_id| {
                        let si = draw_ctx.rsys.get_submesh_info(mesh_id, submesh_id);
                        (si.index_count, mesh.offset_idx + si.index_offset)
                    })
                    .unwrap_or_else(|| (mesh.indices, mesh.offset_idx));

                let drawcall = DrawIndexedIndirectCommand {
                    vertex_offset: mesh.offset_vtx as i32,
                    index_count,
                    instance_count: grp.len() as u32,
                    first_index: mesh_index_offset,
                    first_instance: instance_offset,
                };

                instance_offset += grp.len() as u32;
                drawcalls.push(drawcall);
            });

        let _ = self
            .g_instances_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id)
            .map(|mut instance_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    instance_data.as_ptr(),
                    instance_buffer.as_mut_ptr() as *mut InstanceRenderInfo,
                    instance_data.len(),
                );
            });

        let _ = self
            .drawcalls_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id)
            .map(|mut drawcalls_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    drawcalls.as_ptr(),
                    drawcalls_buffer.as_mut_ptr() as *mut DrawIndexedIndirectCommand,
                    drawcalls.len(),
                );
            });

        let mut draw_cmd_offset = 0u32;
        let draw_cmds = self
            .requests
            .iter()
            .group_by(|req| req.effect)
            .into_iter()
            .map(|(eff_type, grp)| {
                let items = grp.into_iter().unique_by(|k| (k.mesh, k.submesh)).count() as u32;

                let cmd = DrawsByEffect {
                    effect: eff_type,
                    calls: items,
                    offset: draw_cmd_offset,
                };
                draw_cmd_offset += items;
                cmd
            })
            .collect::<smallvec::SmallVec<[DrawsByEffect; 8]>>();

        let dc = draw_ctx.renderer.graphics_device();

        unsafe {
            dc.cmd_bind_vertex_buffers(
                draw_ctx.cmd_buff,
                0,
                &[draw_ctx.rsys.g_vertex_buffer.buffer],
                &[0],
            );

            dc.cmd_bind_index_buffer(
                draw_ctx.cmd_buff,
                draw_ctx.rsys.g_index_buffer.buffer,
                0,
                ash::vk::IndexType::UINT32,
            );
        }

        let ssbo_pbr_setup_handle = self.pbr_renderpass_handles[draw_ctx.frame_id as usize]
            .handle()
            .to_le_bytes();

        let ssbo_glow_handle = self.glow_effect.ssbo_glowdata_handle[draw_ctx.frame_id as usize]
            .handle()
            .to_le_bytes();

        let draw_cmd_buff_offset =
            self.drawcalls_buffer.aligned_slab_size * draw_ctx.frame_id as usize;

        draw_cmds.iter().for_each(|cmd| unsafe {
            let pipeline = draw_ctx.rsys.get_effect(cmd.effect).handle;
            draw_ctx.renderer.graphics_device().cmd_bind_pipeline(
                draw_ctx.cmd_buff,
                ash::vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            let push_const = match cmd.effect {
                EffectType::Pbr | EffectType::BasicEmissive => ssbo_pbr_setup_handle,
                EffectType::Glow => ssbo_glow_handle,
            };

            draw_ctx.renderer.graphics_device().cmd_push_constants(
                draw_ctx.cmd_buff,
                draw_ctx.rsys.bindless.bindless_pipeline_layout(),
                ShaderStageFlags::ALL,
                0,
                &push_const,
            );

            draw_ctx
                .renderer
                .graphics_device()
                .cmd_draw_indexed_indirect(
                    draw_ctx.cmd_buff,
                    self.drawcalls_buffer.buffer,
                    draw_cmd_buff_offset as u64
                        + cmd.offset as u64
                            * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
                    cmd.calls,
                    std::mem::size_of::<DrawIndexedIndirectCommand>() as u32,
                );
        });

        self.requests.clear();
    }
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
struct EngineGlowData {
    ubo_handle: u32,
    instance_handle: u32,
    glow_image: u32,
    noise_image: u32,
    glow_color: glm::Vec3,
    glow_intensity: f32,
    tex_transform: glm::Mat4,
}

struct EngineGlowEffect {
    ssbo_glowdata: UniqueBuffer,
    ssbo_glowdata_handle: Vec<BindlessResourceHandle>,
    glow_img_handle: BindlessResourceHandle,
    noise_img_handle: BindlessResourceHandle,
    noise_rot: f32,
    glow_color: Hsv,
    glow_intensity: f32,
}

impl EngineGlowEffect {
    fn new(init_ctx: &mut crate::draw_context::InitContext) -> Result<Self, ProgramError> {
        let ssbo_glowdata = UniqueBuffer::with_capacity(
            init_ctx.renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            16,
            std::mem::size_of::<EngineGlowData>(),
            init_ctx.renderer.max_inflight_frames(),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("Glow effect/SSBO", &ssbo_glowdata);

        let ssbo_glowdata_handle = init_ctx.rsys.bindless.register_chunked_ssbo(
            init_ctx.renderer,
            &ssbo_glowdata,
            init_ctx.renderer.max_inflight_frames() as usize,
        );

        let work_pkg = init_ctx
            .renderer
            .create_work_package()
            .expect("Failed to create work package");

        let glow_img = UniqueImageWithView::from_png(
            init_ctx.renderer,
            &work_pkg,
            init_ctx.cfg.engine.texture_path("engine_glow/jet_glow.png"),
        )?;

        init_ctx
            .renderer
            .debug_set_object_tag("engine_glow/jet_glow", &glow_img);

        let noise_img = UniqueImageWithView::from_png(
            init_ctx.renderer,
            &work_pkg,
            init_ctx.cfg.engine.texture_path("engine_glow/noise.png"),
        )?;

        init_ctx.renderer.push_work_package(work_pkg);

        init_ctx
            .renderer
            .debug_set_object_tag("engine_glow/noise", &noise_img);

        let glow_img_handle = init_ctx.rsys.add_texture_bindless(
            "engine_glow/jet_glow",
            init_ctx.renderer,
            glow_img,
            None,
        );

        use crate::resource_system::SamplerDescription;
        use ash::vk::*;

        let sampler_noise = *SamplerCreateInfo::builder()
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .address_mode_u(SamplerAddressMode::REPEAT)
            .address_mode_v(SamplerAddressMode::REPEAT)
            .address_mode_w(SamplerAddressMode::REPEAT)
            .min_lod(0f32)
            .max_lod(1f32)
            .border_color(BorderColor::INT_OPAQUE_BLACK);

        let noise_img_handle = init_ctx.rsys.add_texture_bindless(
            "engine_glow/noise",
            init_ctx.renderer,
            noise_img,
            Some(SamplerDescription(sampler_noise)),
        );

        use crate::color_palettes;
        log::info!(
            "HSV blue: {:#?}",
            Hsv::from(Into::<Rgba32F>::into(color_palettes::StdColors::BLUE))
        );

        Ok(Self {
            ssbo_glowdata,
            ssbo_glowdata_handle,
            glow_img_handle,
            noise_img_handle,
            noise_rot: 0f32,
            glow_color: Rgba32F {
                r: 0f32,
                g: 0f32,
                b: 1f32,
                a: 1f32,
            }
            .into(),
            glow_intensity: 1f32,
        })
    }
}
