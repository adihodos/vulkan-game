use ash::vk::{DeviceSize, DrawIndexedIndirectCommand};

use crate::{resource_system::{EffectType, MeshId, SubmeshId, GlobalLightingData, InstanceRenderInfo}, vk_renderer::{Cpu2GpuBuffer, VulkanRenderer}, draw_context::DrawContext};
use crate::resource_system::GlobalTransforms;

struct DrawRequest {
    effect: EffectType,
    mesh: MeshId,
    submesh: Option<SubmeshId>,
    material: Option<String>,
    obj2world: nalgebra_glm::Mat4,
}

pub struct DrawingSys {
    drawcalls_buffer: Cpu2GpuBuffer<ash::vk::DrawIndexedIndirectCommand>,
    requests: Vec<DrawRequest>,
}

impl DrawingSys {
    pub fn create(renderer: &VulkanRenderer) -> Option<Self> {
        let drawcalls_buffer = Cpu2GpuBuffer::<ash::vk::DrawIndexedIndirectCommand>::create(
            renderer,
            ash::vk::BufferUsageFlags::INDIRECT_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            128,
            renderer.max_inflight_frames() as ash::vk::DeviceSize,
        )?;

        Some(Self {
            drawcalls_buffer,
            requests: Vec::with_capacity(1024),
        })
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

    pub fn setup_bindless(&self, skybox_id: u32, draw_ctx: &DrawContext) {
        draw_ctx
            .rsys
            .g_transforms_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut transforms_buffer| {
                let transforms = GlobalTransforms {
                    projection_view: draw_ctx.projection_view,
                    view: draw_ctx.view_matrix,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &transforms as *const _,
                        transforms_buffer.as_mut_ptr() as *mut GlobalTransforms,
                        1,
                    );
                }
            });

        draw_ctx
            .rsys
            .g_lighting_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut light_buffer| {
                let light_data = GlobalLightingData {
                    eye_pos: draw_ctx.cam_position,
                    skybox: skybox_id,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &light_data as *const _,
                        light_buffer.as_mut_ptr() as *mut GlobalLightingData,
                        1,
                    );
                }
            });

        // 4 -> ubo tf globals, 0 0 0
        // 64
        // 4 -> ubo light , 0, 0, 0
        use std::iter::{once, repeat};
        let bindig_offsets = once(
            draw_ctx
                .rsys
                .g_transforms_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        )
        .chain(repeat(0u32).take(3))
        .chain(once(
            draw_ctx
                .rsys
                .g_instances_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        ))
        .chain(repeat(0u32).take(15))
        .chain(once(
            draw_ctx
                .rsys
                .g_lighting_buffer
                .offset_for_frame(draw_ctx.frame_id as DeviceSize) as u32,
        ))
        .chain(repeat(0u32).take(3))
        .collect::<Vec<_>>();

        let (p_layout, _) = draw_ctx.rsys.pipeline_layout();

        unsafe {
            draw_ctx
                .renderer
                .graphics_device()
                .cmd_bind_descriptor_sets(
                    draw_ctx.cmd_buff,
                    ash::vk::PipelineBindPoint::GRAPHICS,
                    p_layout,
                    0,
                    &draw_ctx.rsys.descriptor_sets,
                    &bindig_offsets,
                );
        }
    }

    pub fn draw(&mut self, draw_ctx: &DrawContext) {
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

        draw_ctx
            .rsys
            .g_instances_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
            .map(|mut instance_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    instance_data.as_ptr(),
                    instance_buffer.as_mut_ptr() as *mut InstanceRenderInfo,
                    instance_data.len(),
                );
            });

        self.drawcalls_buffer
            .map_for_frame(draw_ctx.renderer, draw_ctx.frame_id as DeviceSize)
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

        let draw_cmd_buff_offset = self
            .drawcalls_buffer
            .offset_for_frame(draw_ctx.frame_id as DeviceSize);
        draw_cmds.iter().for_each(|cmd| unsafe {
            let pipeline = draw_ctx.rsys.get_effect(cmd.effect).pipeline;
            draw_ctx.renderer.graphics_device().cmd_bind_pipeline(
                draw_ctx.cmd_buff,
                ash::vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            draw_ctx
                .renderer
                .graphics_device()
                .cmd_draw_indexed_indirect(
                    draw_ctx.cmd_buff,
                    self.drawcalls_buffer.buffer.buffer,
                    draw_cmd_buff_offset
                        + cmd.offset as u64
                            * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
                    cmd.calls,
                    std::mem::size_of::<DrawIndexedIndirectCommand>() as u32,
                );
        });

        self.requests.clear();
    }
}
