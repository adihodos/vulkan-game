use std::collections::HashMap;

use crate::{
    app_config::AppConfig,
    draw_context::{DrawContext, UpdateContext},
    math::AABB3,
    physics_engine::PhysicsEngine,
    resource_cache::ResourceHolder,
    vk_renderer::{Cpu2GpuBuffer, UniqueImageWithView, UniqueSampler, VulkanRenderer},
};
use ash::vk::{BufferUsageFlags, DeviceSize, DrawIndexedIndirectCommand};
use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};
use smallvec::SmallVec;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum MissileState {
    Inactive,
    Fired,
    Active,
}

#[derive(
    Copy,
    Clone,
    Debug,
    strum_macros::EnumProperty,
    strum_macros::EnumIter,
    strum_macros::Display,
    Hash,
    Eq,
    PartialEq,
)]
#[repr(u8)]
pub enum MissileKind {
    #[strum(props(kind = "R73"))]
    R73,
    #[strum(props(kind = "R27"))]
    R27,
}

#[derive(Copy, Clone, Debug)]
pub struct Missile {
    pub kind: MissileKind,
    pub state: MissileState,
    pub transform: nalgebra_glm::Mat4,
    // booster_time: f32,
    // booster_life: f32,
}

struct MissileClassSheet {
    mass: f32,
    aabb: AABB3,
    booster_life: f32,
    thrust: f32,
}

pub struct LiveMissile {
    kind: MissileKind,
    orientation: nalgebra::Isometry3<f32>,
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
    booster_time: f32,
    thrust: f32,
}

pub struct MissileSys {
    missiles_gpu: Cpu2GpuBuffer<glm::Mat4>,
    placeholder_mtl: UniqueImageWithView,
    sampler: UniqueSampler,
    ubo_globals: Cpu2GpuBuffer<TransformDataMultiInstanceUBO>,
    buffer_vertices: ash::vk::Buffer,
    buffer_indices: ash::vk::Buffer,
    pipeline: ash::vk::Pipeline,
    pipeline_layout: ash::vk::PipelineLayout,
    descriptor_sets: [ash::vk::DescriptorSet; 2],
    draw_indirect_calls_data: std::collections::HashMap<MissileKind, DrawIndexedIndirectCommand>,
    missiles_cpu_by_type: std::collections::HashMap<MissileKind, Vec<Missile>>,
    live_missiles: Vec<LiveMissile>,
    buffer_draw_indirect: Cpu2GpuBuffer<DrawIndexedIndirectCommand>,
    missile_classes: HashMap<MissileKind, MissileClassSheet>,
    missiles_count: u32,
}

impl MissileSys {
    const MAX_MISSILES: u32 = 1024;
    const MAX_DRAW_INDIRECT_CMDS: u32 = 32;

    pub fn new(
        renderer: &VulkanRenderer,
        resource_cache: &ResourceHolder,
        cfg: &AppConfig,
    ) -> Option<MissileSys> {
        let missiles_instances = Cpu2GpuBuffer::<glm::Mat4>::create(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            Self::MAX_MISSILES as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let work_pkg = renderer.create_work_package()?;
        let placeholder_mtl = UniqueImageWithView::from_ktx(
            renderer,
            &work_pkg,
            cfg.engine.texture_path("uv_grids/ash_uvgrid01.ktx2"),
        )?;
        renderer.push_work_package(work_pkg);

        use ash::vk::{Filter, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
        let sampler = UniqueSampler::new(
            renderer.graphics_device(),
            &SamplerCreateInfo::builder()
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

        let ubo_globals = Cpu2GpuBuffer::<TransformDataMultiInstanceUBO>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let ds_layouts = resource_cache
            .non_pbr_pipeline_instanced()
            .descriptor_layouts();

        let descriptor_sets = unsafe {
            use ash::vk::DescriptorSetAllocateInfo;
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(renderer.descriptor_pool())
                    .set_layouts(ds_layouts),
            )
        }
        .map_err(|e| log::error!("Failed to allocate descriptor sets: {e}"))
        .ok()?;

        use ash::vk::{
            DescriptorBufferInfo, DescriptorImageInfo, DescriptorType, ImageLayout,
            WriteDescriptorSet,
        };

        unsafe {
            renderer.graphics_device().update_descriptor_sets(
                &[
                    *WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(ubo_globals.buffer.buffer)
                            .offset(0)
                            .range(ubo_globals.bytes_one_frame)]),
                    *WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .buffer_info(&[*DescriptorBufferInfo::builder()
                            .buffer(missiles_instances.buffer.buffer)
                            .offset(0)
                            .range(missiles_instances.bytes_one_frame)]),
                    *WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[1])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[*DescriptorImageInfo::builder()
                            .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(placeholder_mtl.image_view())
                            .sampler(sampler.sampler)]),
                ],
                &[],
            );
        }

        let missile_mesh_handle = resource_cache.get_non_pbr_geometry_handle(&"r73r27");
        let missile_mesh = resource_cache.get_non_pbr_geometry_info(missile_mesh_handle);

        use strum::IntoEnumIterator;
        let draw_indirect_calls_data = MissileKind::iter()
            .map(|missile_kind| {
                use strum::EnumProperty;
                let missile_node =
                    missile_mesh.get_node_by_name(missile_kind.get_str("kind").unwrap());
                (
                    missile_kind,
                    DrawIndexedIndirectCommand {
                        index_count: missile_node.index_count,
                        instance_count: 0,
                        first_index: missile_node.index_offset + missile_mesh.index_offset,
                        vertex_offset: missile_mesh.vertex_offset as i32,
                        first_instance: 0,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();

        let missile_classes = MissileKind::iter()
            .map(|msl_kind| {
                use strum::EnumProperty;
                let missile_node = missile_mesh.get_node_by_name(msl_kind.get_str("kind").unwrap());

                (
                    msl_kind,
                    MissileClassSheet {
                        mass: 500f32,
                        aabb: missile_node.aabb,
                        booster_life: 25f32,
                        thrust: 550f32,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();

        let buffer_draw_indirect = Cpu2GpuBuffer::<ash::vk::DrawIndexedIndirectCommand>::create(
            renderer,
            BufferUsageFlags::INDIRECT_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            Self::MAX_DRAW_INDIRECT_CMDS as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        Some(MissileSys {
            ubo_globals,
            missiles_gpu: missiles_instances,
            placeholder_mtl,
            descriptor_sets: [descriptor_sets[0], descriptor_sets[1]],
            sampler,
            draw_indirect_calls_data,
            buffer_draw_indirect,
            buffer_vertices: resource_cache.vertex_buffer_non_pbr(),
            buffer_indices: resource_cache.index_buffer_non_pbr(),
            pipeline: resource_cache.non_pbr_pipeline_instanced().pipeline,
            pipeline_layout: resource_cache.non_pbr_pipeline_instanced().layout,
            missiles_cpu_by_type: std::collections::HashMap::new(),
            missiles_count: 0,
            live_missiles: Vec::new(),
            missile_classes,
        })
    }

    pub fn draw_inert_missile(&mut self, missile: Missile) {
        if self.missiles_count >= Self::MAX_MISSILES {
            log::error!("Cannot spawn missile, limit reached!");
            return;
        }
        self.missiles_cpu_by_type
            .entry(missile.kind)
            .and_modify(|e| e.push(missile))
            .or_insert(vec![missile]);
        self.missiles_count += 1;
    }

    pub fn draw(&mut self, draw_context: &DrawContext) {
        if self.missiles_count == 0 {
            return;
        }

        self.missiles_cpu_by_type
            .iter()
            .for_each(|(missile_kind, missiles)| {
                self.draw_indirect_calls_data
                    .entry(*missile_kind)
                    .and_modify(|draw_call_data| {
                        draw_call_data.instance_count += missiles.len() as u32;
                    });
            });

        let mut instance_offset = 0u32;
        let indirect_draw_calls = self
            .draw_indirect_calls_data
            .values()
            .filter_map(|indirect_draw_call| {
                if indirect_draw_call.instance_count != 0 {
                    let draw_call = DrawIndexedIndirectCommand {
                        first_instance: instance_offset,
                        ..*indirect_draw_call
                    };
                    instance_offset += indirect_draw_call.instance_count;

                    Some(draw_call)
                } else {
                    None
                }
            })
            .collect::<SmallVec<[DrawIndexedIndirectCommand; 8]>>();

        self.buffer_draw_indirect
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mut drawcall_buffer| unsafe {
                std::ptr::copy_nonoverlapping(
                    indirect_draw_calls.as_ptr(),
                    drawcall_buffer.as_mut_ptr() as *mut DrawIndexedIndirectCommand,
                    indirect_draw_calls.len(),
                );
            });

        self.missiles_gpu
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mut missiles_gpu_buf| {
                let dst_ptr = missiles_gpu_buf.as_mut_ptr() as *mut glm::Mat4;

                let mut offset: isize = 0;
                self.missiles_cpu_by_type.iter().for_each(|(_, v)| {
                    v.iter().for_each(|ms| {
                        unsafe {
                            dst_ptr.offset(offset).write(ms.transform);
                        }
                        offset += 1;
                    });
                });
            });

        self.ubo_globals
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mut ubo_globals| {
                let ptr = ubo_globals.as_mut_ptr() as *mut TransformDataMultiInstanceUBO;
                unsafe {
                    (*ptr).projection = draw_context.projection;
                    (*ptr).view = draw_context.view_matrix;
                }
            });

        let dc = draw_context.renderer.graphics_device();
        unsafe {
            dc.cmd_bind_pipeline(
                draw_context.cmd_buff,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            dc.cmd_bind_vertex_buffers(draw_context.cmd_buff, 0, &[self.buffer_vertices], &[0]);
            dc.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.buffer_indices,
                0,
                ash::vk::IndexType::UINT32,
            );
            dc.cmd_set_viewport(draw_context.cmd_buff, 0, &[draw_context.viewport]);
            dc.cmd_set_scissor(draw_context.cmd_buff, 0, &[draw_context.scissor]);
            dc.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[
                    self.ubo_globals
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                    self.missiles_gpu
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                ],
            );
            dc.cmd_draw_indexed_indirect(
                draw_context.cmd_buff,
                self.buffer_draw_indirect.buffer.buffer,
                self.buffer_draw_indirect
                    .offset_for_frame(draw_context.frame_id as DeviceSize),
                indirect_draw_calls.len() as u32,
                std::mem::size_of::<DrawIndexedIndirectCommand>() as u32,
            );
        }

        self.missiles_cpu_by_type.iter_mut().for_each(|(_, v)| {
            v.clear();
        });

        self.draw_indirect_calls_data.values_mut().for_each(|val| {
            val.instance_count = 0;
        });
        self.missiles_count = 0;
    }

    pub fn add_live_missile(
        &mut self,
        kind: MissileKind,
        initial_orientation: &nalgebra::Isometry3<f32>,
        physics_engine: &mut PhysicsEngine,
    ) {
        if self.missiles_count >= Self::MAX_MISSILES {
            log::error!("Cannot spawn anymore missiles, limit reached");
            return;
        }

        let msl_class_sheet = self
            .missile_classes
            .get(&kind)
            .expect(&format!("Missing data sheet for missile class {kind}"));

        use crate::physics_engine::{ColliderUserData, PhysicsObjectCollisionGroups};
        use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
        let body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .position(*initial_orientation)
            .build();

        let body_handle = physics_engine.rigid_body_set.insert(body);

        let bbox_half_extents = msl_class_sheet.aabb.extents();
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(msl_class_sheet.mass)
        .active_events(rapier3d::prelude::ActiveEvents::COLLISION_EVENTS)
        .collision_groups(PhysicsObjectCollisionGroups::missiles())
        .sensor(true)
        .user_data(ColliderUserData::new(body_handle).into())
        .build();

        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut physics_engine.rigid_body_set,
        );

        self.live_missiles.push(LiveMissile {
            kind,
            orientation: *initial_orientation,
            booster_time: msl_class_sheet.booster_life,
            rigid_body: body_handle,
            collider: collider_handle,
            thrust: msl_class_sheet.thrust,
        });
    }

    pub fn update(&mut self, context: &mut UpdateContext) {
        self.live_missiles.retain_mut(|msl| {
            let msl_phys_body = context.physics_engine.get_rigid_body_mut(msl.rigid_body);

            msl.orientation = *msl_phys_body.position();

            if msl.booster_time > 0f32 {
                msl.booster_time = (msl.booster_time - context.frame_time as f32).max(0f32);
            }

            if msl.booster_time > 0f32 {
                //
                // apply force
                msl_phys_body.apply_impulse(msl.orientation * glm::Vec3::z() * msl.thrust, true);
                true
            } else {
                //
                // dead missile, remove it
                context.physics_engine.remove_rigid_body(msl.rigid_body);
                false
            }
        });

        //
        // add live missiles to draw list
        self.live_missiles.iter().for_each(|msl| {
            let m = Missile {
                kind: msl.kind,
                state: MissileState::Inactive,
                transform: msl.orientation.to_matrix(),
            };

            self.missiles_cpu_by_type
                .entry(msl.kind)
                .and_modify(|e| e.push(m))
                .or_insert(vec![m]);

            self.missiles_count += 1;
        });
    }
}

#[repr(C, align(16))]
struct TransformDataMultiInstanceUBO {
    projection: glm::Mat4,
    view: glm::Mat4,
}
