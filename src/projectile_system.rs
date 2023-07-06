use ash::vk::{
    BufferUsageFlags, DescriptorBufferInfo, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding,
    DescriptorType, DeviceSize, DynamicState, IndexType, PipelineBindPoint, ShaderStageFlags,
    WriteDescriptorSet,
};
use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

use crate::{
    app_config::AppConfig,
    draw_context::{DrawContext, UpdateContext},
    physics_engine::{PhysicsEngine, PhysicsObjectCollisionGroups},
    resource_cache::ResourceHolder,
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder,
        ShaderModuleDescription, ShaderModuleSource, UniqueGraphicsPipeline, VulkanRenderer,
    },
};

#[derive(Copy, Clone, Debug)]
pub struct ProjectileSpawnData {
    pub origin: nalgebra::Point3<f32>,
    pub emitter: RigidBodyHandle,
    pub speed: f32,
    pub mass: f32,
    pub life: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Projectile {
    pub data: ProjectileSpawnData,
    pub direction: glm::Vec3,
    pub rigid_body_handle: RigidBodyHandle,
    pub collider_handle: ColliderHandle,
}

impl Projectile {
    fn new(
        projectile_data: ProjectileSpawnData,
        physics_engine: &mut PhysicsEngine,
        collider_size: glm::Vec3,
    ) -> Projectile {
        let shooter = physics_engine.get_rigid_body(projectile_data.emitter);

        let projectile_origin = *shooter.position() * projectile_data.origin;

        let projectile_isometry = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(projectile_origin),
            *shooter.rotation(),
        );

        let direction = (projectile_isometry * glm::Vec3::z_axis()).xyz();
        let velocity = direction * projectile_data.speed;

        let mut rigid_body = rapier3d::prelude::RigidBodyBuilder::dynamic()
            .position(projectile_isometry)
            .lock_rotations()
            .build();

        rigid_body.add_force(velocity, true);
        let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);

        let collider = rapier3d::prelude::ColliderBuilder::cuboid(
            collider_size.x,
            collider_size.y,
            collider_size.z,
        )
        .active_events(rapier3d::prelude::ActiveEvents::COLLISION_EVENTS)
        .collision_groups(PhysicsObjectCollisionGroups::projectiles())
        .sensor(true)
        .build();

        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_engine.rigid_body_set,
        );

        Projectile {
            data: projectile_data,
            direction,
            rigid_body_handle,
            collider_handle,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct GpuProjectileInstance {
    model_matrix: glm::Mat4,
    inner_color: glm::Vec4,
    outer_color: glm::Vec4,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
struct UBOProjectileTransformData {
    projection_view: glm::Mat4,
}

pub struct ProjectileSystem {
    projectiles: Vec<Projectile>,
    ubo_transforms: Cpu2GpuBuffer<UBOProjectileTransformData>,
    gpu_instances: Cpu2GpuBuffer<GpuProjectileInstance>,
    descriptor_sets: Vec<ash::vk::DescriptorSet>,
    pipeline: UniqueGraphicsPipeline,
    plasmabolt_collider_size: glm::Vec3,
    vtx_buffer: ash::vk::Buffer,
    idx_buffer: ash::vk::Buffer,
    idx_offset: u32,
    idx_count: u32,
}

impl ProjectileSystem {
    const MAX_PROJECTILES: usize = 1024;

    pub fn new(
        renderer: &VulkanRenderer,
        resource_cache: &ResourceHolder,
        app_cfg: &AppConfig,
    ) -> Option<Self> {
        use crate::imported_geometry::GeometryVertex;
        use ash::vk::{
            Format, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
        };
        use memoffset::offset_of;
        use std::mem::size_of;

        let pipeline = GraphicsPipelineBuilder::new()
            .add_vertex_input_attribute_descriptions(&[
                VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: Format::R32G32B32_SFLOAT,
                    offset: offset_of!(GeometryVertex, pos) as u32,
                },
                VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: Format::R32G32_SFLOAT,
                    offset: offset_of!(GeometryVertex, uv) as u32,
                },
            ])
            .add_vertex_input_attribute_binding(
                VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(size_of::<GeometryVertex>() as u32)
                    .input_rate(VertexInputRate::VERTEX)
                    .build(),
            )
            .shader_stages(&[
                ShaderModuleDescription {
                    stage: ShaderStageFlags::VERTEX,
                    source: ShaderModuleSource::File(
                        &app_cfg.engine.shader_path("projectile.vert.spv"),
                    ),
                    entry_point: "main",
                },
                ShaderModuleDescription {
                    stage: ShaderStageFlags::FRAGMENT,
                    source: ShaderModuleSource::File(
                        &app_cfg.engine.shader_path("projectile.frag.spv"),
                    ),
                    entry_point: "main",
                },
            ])
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build(
                renderer.graphics_device(),
                renderer.pipeline_cache(),
                GraphicsPipelineLayoutBuilder::new()
                    .set(
                        0,
                        &[
                            *DescriptorSetLayoutBinding::builder()
                                .binding(0)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .descriptor_count(1),
                            *DescriptorSetLayoutBinding::builder()
                                .binding(1)
                                .stage_flags(ShaderStageFlags::VERTEX)
                                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                                .descriptor_count(1),
                        ],
                    )
                    .build(renderer.graphics_device())?,
                renderer.renderpass(),
                0,
            )?;

        let gpu_instances = Cpu2GpuBuffer::<GpuProjectileInstance>::create(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            Self::MAX_PROJECTILES as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let ubo_transforms = Cpu2GpuBuffer::<UBOProjectileTransformData>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
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

        let descriptor_buffers = [
            DescriptorBufferInfo::builder()
                .buffer(ubo_transforms.buffer.buffer)
                .range(ubo_transforms.bytes_one_frame)
                .offset(0)
                .build(),
            DescriptorBufferInfo::builder()
                .buffer(gpu_instances.buffer.buffer)
                .range(gpu_instances.bytes_one_frame)
                .offset(0)
                .build(),
        ];

        let write_descriptor_set = [
            WriteDescriptorSet::builder()
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_set(descriptor_sets[0])
                .dst_binding(0)
                .buffer_info(&descriptor_buffers[..1])
                .dst_array_element(0)
                .build(),
            WriteDescriptorSet::builder()
                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .dst_set(descriptor_sets[0])
                .dst_binding(1)
                .buffer_info(&descriptor_buffers[1..])
                .dst_array_element(0)
                .build(),
        ];

        unsafe {
            renderer
                .graphics_device()
                .update_descriptor_sets(&write_descriptor_set, &[]);
        }

        let proj_mesh_handle = resource_cache.get_non_pbr_geometry_handle("r73r27");
        let proj_mesh = resource_cache
            .get_non_pbr_geometry_info(proj_mesh_handle)
            .get_node_by_name("plasmabolt");

        Some(Self {
            projectiles: Vec::with_capacity(Self::MAX_PROJECTILES),
            ubo_transforms,
            gpu_instances,
            descriptor_sets,
            pipeline,
            vtx_buffer: resource_cache.vertex_buffer_non_pbr(),
            idx_buffer: resource_cache.index_buffer_non_pbr(),
            idx_offset: proj_mesh.index_offset,
            idx_count: proj_mesh.index_count,
            plasmabolt_collider_size: proj_mesh.aabb.extents(),
        })
    }

    pub fn update(&mut self, update_context: &mut UpdateContext) {
        self.projectiles.retain_mut(|proj| {
            proj.data.life -= update_context.frame_time as f32;
            if proj.data.life > 0f32 {
                true
            } else {
                update_context
                    .physics_engine
                    .remove_rigid_body(proj.rigid_body_handle);
                false
            }
        });
    }

    pub fn spawn_projectile(&mut self, data: ProjectileSpawnData, phys_engine: &mut PhysicsEngine) {
        if self.projectiles.len() > Self::MAX_PROJECTILES {
            log::info!("Discarding projectile, max limit reached");
        }
        self.projectiles.push(Projectile::new(
            data,
            phys_engine,
            self.plasmabolt_collider_size,
        ));
    }

    pub fn despawn_projectile(&mut self, proj_body: RigidBodyHandle) {
        self.projectiles
            .iter()
            .position(|projectile| projectile.rigid_body_handle == proj_body)
            .map(|proj_pos| {
                self.projectiles.swap_remove(proj_pos);
            });
    }

    pub fn render(&self, draw_context: &DrawContext, phys_engine: &PhysicsEngine) {
        if self.projectiles.is_empty() {
            return;
        }

        self.ubo_transforms
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|ubo_mapping| {
                let transforms = UBOProjectileTransformData {
                    projection_view: draw_context.projection_view,
                };

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &transforms as *const _,
                        ubo_mapping.memptr() as *mut UBOProjectileTransformData,
                        1,
                    );
                }
            });

	//
	// TODO: maybe don't draw anything beyond a certain distance to the camera ??
        self.gpu_instances
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|gpu_proj_buff| {
                let cpu_instances = self
                    .projectiles
                    .iter()
                    .filter_map(|cpu_proj| {
                        phys_engine
                            .rigid_body_set
                            .get(cpu_proj.rigid_body_handle)
                            .map(|proj_body| GpuProjectileInstance {
                                model_matrix: proj_body.position().to_homogeneous(),
                                inner_color: glm::vec4(1f32, 1f32, 1f32, 1f32),
                                outer_color: glm::vec4(1f32, 0f32, 0f32, 1f32),
                            })
                    })
                    .collect::<smallvec::SmallVec<[GpuProjectileInstance; 16]>>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cpu_instances.as_ptr(),
                        gpu_proj_buff.memptr() as *mut GpuProjectileInstance,
                        cpu_instances.len(),
                    );
                }
            });

        let dc = draw_context.renderer.graphics_device();
        unsafe {
            dc.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            dc.cmd_bind_vertex_buffers(draw_context.cmd_buff, 0, &[self.vtx_buffer], &[0]);
            dc.cmd_bind_index_buffer(draw_context.cmd_buff, self.idx_buffer, 0, IndexType::UINT32);

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
                .cmd_bind_descriptor_sets(
                    draw_context.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &self.descriptor_sets,
                    &[
                        self.ubo_transforms
                            .offset_for_frame(draw_context.frame_id as DeviceSize)
                            as u32,
                        self.gpu_instances
                            .offset_for_frame(draw_context.frame_id as DeviceSize)
                            as u32,
                    ],
                );

            draw_context.renderer.graphics_device().cmd_draw_indexed(
                draw_context.cmd_buff,
                self.idx_count,
                self.projectiles.len() as u32,
                self.idx_offset,
                0,
                0,
            );
        }
    }
}
