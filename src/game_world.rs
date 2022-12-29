use std::{
    cell::{Cell, RefCell},
    mem::size_of,
    rc::Rc,
};

use ash::vk::{
    BorderColor, BufferUsageFlags, CullModeFlags, DescriptorBufferInfo, DescriptorImageInfo,
    DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType,
    DeviceSize, DynamicState, Filter, FrontFace, ImageLayout, ImageTiling, ImageUsageFlags,
    IndexType, MemoryPropertyFlags, PipelineBindPoint, PipelineRasterizationStateCreateInfo,
    PolygonMode, PrimitiveTopology, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    ShaderStageFlags, WriteDescriptorSet, WriteDescriptorSetBuilder,
};
use glm::{Mat4, Vec3};
use nalgebra::{Isometry3, Translation3};
use nalgebra_glm::Vec4;

use nalgebra_glm as glm;
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, RigidBodyBuilder, RigidBodyHandle, RigidBodyType,
};
use smallvec::SmallVec;

use crate::{
    app_config::AppConfig,
    camera::Camera,
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::{DrawContext, FrameRenderContext, UpdateContext},
    flight_cam::FlightCamera,
    game_object::GameObjectRenderState,
    math,
    physics_engine::PhysicsEngine,
    projectile_system::{ProjectileSpawnData, ProjectileSystem},
    resource_cache::{PbrDescriptorType, PbrRenderableHandle, ResourceHolder},
    shadow_swarm::ShadowFighterSwarm,
    skybox::Skybox,
    starfury::Starfury,
    vk_renderer::{
        Cpu2GpuBuffer, GraphicsPipelineBuilder, GraphicsPipelineLayoutBuilder, ScopedBufferMapping,
        ShaderModuleDescription, ShaderModuleSource, UniqueBuffer, UniqueGraphicsPipeline,
        UniqueImage, UniqueSampler, VulkanRenderer,
    },
    window::InputState,
};

#[derive(Copy, Clone, Debug)]
pub enum QueuedCommand {
    SpawnProjectile(ProjectileSpawnData),
}

#[derive(Copy, Clone, Debug)]
pub struct GameObjectCommonData {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub phys_rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub phys_collider_handle: rapier3d::prelude::ColliderHandle,
}

#[derive(Copy, Clone, Debug)]
struct DebugDrawOptions {
    wireframe_color: Vec4,
    draw_normals: bool,
    normals_color: Vec4,
    debug_draw_physics: bool,
    debug_draw_nodes_bounding: bool,
    debug_draw_mesh: bool,
    debug_draw_world_axis: bool,
    world_axis_length: f32,
}

impl std::default::Default for DebugDrawOptions {
    fn default() -> Self {
        Self {
            wireframe_color: glm::vec4(1f32, 0f32, 0f32, 1f32),
            draw_normals: false,
            normals_color: glm::vec4(0f32, 1f32, 0f32, 1f32),
            debug_draw_physics: false,
            debug_draw_nodes_bounding: false,
            debug_draw_mesh: false,
            debug_draw_world_axis: false,
            world_axis_length: 1f32,
        }
    }
}

#[repr(C)]
pub struct PbrLightingData {
    pub eye_pos: glm::Vec3,
}

#[repr(C)]
pub struct PbrTransformDataSingleInstanceUBO {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub world: glm::Mat4,
}

#[repr(C, align(16))]
struct PbrTransformDataMultiInstanceUBO {
    projection: glm::Mat4,
    view: glm::Mat4,
}

#[repr(C)]
struct PbrTransformDataInstanceEntry {
    model: glm::Mat4,
}

struct InstancedRenderingData {
    ubo_vtx_transforms: Cpu2GpuBuffer<PbrTransformDataMultiInstanceUBO>,
    stb_instances: Cpu2GpuBuffer<PbrTransformDataInstanceEntry>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl InstancedRenderingData {
    fn create(
        renderer: &VulkanRenderer,
        resource_cache: &ResourceHolder,
        instances: u32,
    ) -> Option<InstancedRenderingData> {
        let ubo_vtx_transforms = Cpu2GpuBuffer::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let stb_instances = Cpu2GpuBuffer::create(
            renderer,
            BufferUsageFlags::STORAGE_BUFFER,
            renderer.device_properties().limits.non_coherent_atom_size,
            instances as DeviceSize,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let descriptor_sets = unsafe {
            let descriptor_set_layouts =
                [resource_cache.pbr_pipeline_instanced().descriptor_layouts()[0]];

            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(&descriptor_set_layouts)
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .map_err(|e| log::error!("Failed to allocate descriptor sets: {}", e))
        .ok()?;

        let descriptor_buffers_info = [
            DescriptorBufferInfo::builder()
                .buffer(ubo_vtx_transforms.buffer.buffer)
                .offset(0)
                .range(ubo_vtx_transforms.bytes_one_frame)
                .build(),
            DescriptorBufferInfo::builder()
                .buffer(stb_instances.buffer.buffer)
                .offset(0)
                .range(stb_instances.bytes_one_frame)
                .build(),
        ];

        let write_descriptor_set = [
            WriteDescriptorSet::builder()
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_set(descriptor_sets[0])
                .dst_binding(0)
                .buffer_info(&descriptor_buffers_info[..1])
                .dst_array_element(0)
                .build(),
            WriteDescriptorSet::builder()
                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .dst_set(descriptor_sets[0])
                .dst_binding(1)
                .dst_array_element(0)
                .buffer_info(&descriptor_buffers_info[1..])
                .build(),
        ];

        unsafe {
            renderer
                .graphics_device()
                .update_descriptor_sets(&write_descriptor_set, &[]);
        }

        Some(InstancedRenderingData {
            ubo_vtx_transforms,
            stb_instances,
            descriptor_sets,
        })
    }
}

struct SingleInstanceRenderingData {
    vs_ubo_transforms: Cpu2GpuBuffer<PbrTransformDataSingleInstanceUBO>,
    fs_ubo_lights: Cpu2GpuBuffer<PbrLightingData>,
    object_descriptor_sets: Vec<DescriptorSet>,
    ibl_descriptor_sets: Vec<DescriptorSet>,
    samplers: Vec<UniqueSampler>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GameObjectHandle(u32);

struct GameObjectData {
    handle: GameObjectHandle,
    renderable: PbrRenderableHandle,
}

pub struct GameWorld {
    draw_opts: RefCell<DebugDrawOptions>,
    resource_cache: ResourceHolder,
    skybox: Skybox,
    single_inst_renderdata: SingleInstanceRenderingData,
    objects: Vec<GameObjectData>,
    starfury: Starfury,
    shadows_swarm: ShadowFighterSwarm,
    shadows_swarm_inst_render_data: InstancedRenderingData,
    frame_times: RefCell<Vec<f32>>,
    physics_engine: RefCell<PhysicsEngine>,
    camera: RefCell<FlightCamera>,
    debug_draw_overlay: Rc<RefCell<DebugDrawOverlay>>,
    projectiles_sys: RefCell<ProjectileSystem>,
}

impl GameWorld {
    const PHYSICS_TIME_STEP: f64 = 1f64 / 240f64;
    const MAX_HISTOGRAM_VALUES: usize = 32;

    fn draw_options(&self) -> std::cell::Ref<DebugDrawOptions> {
        self.draw_opts.borrow()
    }

    fn draw_options_mut(&self) -> std::cell::RefMut<DebugDrawOptions> {
        self.draw_opts.borrow_mut()
    }

    pub fn new(renderer: &VulkanRenderer, app_cfg: &AppConfig) -> Option<GameWorld> {
        ShadowFighterSwarm::write_default_config();

        let debug_draw_overlay = DebugDrawOverlay::create(renderer)?;
        let skybox = Skybox::create(renderer, &app_cfg.scene, &app_cfg.engine)?;

        let resource_cache = ResourceHolder::create(renderer, app_cfg)?;

        let vs_ubo_transforms = Cpu2GpuBuffer::<PbrTransformDataSingleInstanceUBO>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            VulkanRenderer::aligned_size_of_type::<PbrTransformDataSingleInstanceUBO>(
                renderer
                    .device_properties()
                    .limits
                    .min_uniform_buffer_offset_alignment,
            ),
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let fs_ubo_lights = Cpu2GpuBuffer::<PbrLightingData>::create(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            VulkanRenderer::aligned_size_of_type::<PbrLightingData>(
                renderer
                    .device_properties()
                    .limits
                    .min_uniform_buffer_offset_alignment,
            ),
            1,
            renderer.max_inflight_frames() as DeviceSize,
        )?;

        let pbr_descriptor_layouts = resource_cache.pbr_pipeline().descriptor_layouts();

        let per_object_ds_layouts = [
            pbr_descriptor_layouts[PbrDescriptorType::VsTransformsUbo as usize],
            pbr_descriptor_layouts[PbrDescriptorType::FsLightingData as usize],
        ];

        let object_pbr_descriptor_sets = unsafe {
            renderer.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(&per_object_ds_layouts)
                    .descriptor_pool(renderer.descriptor_pool())
                    .build(),
            )
        }
        .expect("Papali, papali sukyyyyyyyyyyyyyy");

        let desc_buff_info = [
            DescriptorBufferInfo::builder()
                .buffer(vs_ubo_transforms.buffer.buffer)
                .offset(0)
                .range(vs_ubo_transforms.bytes_one_frame)
                .build(),
            DescriptorBufferInfo::builder()
                .buffer(fs_ubo_lights.buffer.buffer)
                .offset(0)
                .range(fs_ubo_lights.bytes_one_frame)
                .build(),
        ];

        let write_descriptors_transforms_lighting = [
            WriteDescriptorSet::builder()
                .dst_set(object_pbr_descriptor_sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&desc_buff_info[0..1])
                .build(),
            WriteDescriptorSet::builder()
                .dst_set(object_pbr_descriptor_sets[1])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&desc_buff_info[1..])
                .build(),
        ];

        unsafe {
            renderer
                .graphics_device()
                .update_descriptor_sets(&write_descriptors_transforms_lighting, &[]);
        }

        let mut samplers_ibl = SmallVec::<[UniqueSampler; 8]>::new();
        let mut ibl_descriptor_sets = SmallVec::<[DescriptorSet; 4]>::new();

        let sampler_brdf_lut = UniqueSampler::new(
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
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .max_anisotropy(1f32)
                .build(),
        )?;

        skybox.get_ibl_data().iter().for_each(|ibl_data| {
            let levels_irradiance = ibl_data.irradiance.0.info.num_levels;

            let sampler_cubemaps = UniqueSampler::new(
                renderer.graphics_device(),
                &SamplerCreateInfo::builder()
                    .min_lod(0f32)
                    .max_lod(levels_irradiance as f32)
                    .min_filter(Filter::LINEAR)
                    .mag_filter(Filter::LINEAR)
                    .mipmap_mode(SamplerMipmapMode::LINEAR)
                    .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                    .border_color(BorderColor::INT_OPAQUE_BLACK)
                    .max_anisotropy(1f32)
                    .build(),
            )
            .expect("Failed to create sampler");

            let ibl_desc_img_info = [
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.irradiance.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_cubemaps.sampler)
                    .build(),
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.specular.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_cubemaps.sampler)
                    .build(),
                DescriptorImageInfo::builder()
                    .image_view(ibl_data.brdf_lut.1.view)
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(sampler_brdf_lut.sampler)
                    .build(),
            ];

            samplers_ibl.push(sampler_cubemaps);

            let dset_ibl = unsafe {
                renderer.graphics_device().allocate_descriptor_sets(
                    &DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(renderer.descriptor_pool())
                        .set_layouts(&pbr_descriptor_layouts[3..])
                        .build(),
                )
            }
            .expect("Papalyyyy cykyyyyyyyyyyyyyyy");

            let wds = [
                //
                // irradiance
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(0)
                    .image_info(&ibl_desc_img_info[0..1])
                    .dst_array_element(0)
                    .build(),
                //
                // specular
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .image_info(&ibl_desc_img_info[1..2])
                    .build(),
                //
                // BRDF lut
                WriteDescriptorSet::builder()
                    .dst_set(dset_ibl[0])
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .image_info(&ibl_desc_img_info[2..])
                    .build(),
            ];

            unsafe {
                renderer.graphics_device().update_descriptor_sets(&wds, &[]);
            }

            ibl_descriptor_sets.extend(dset_ibl);
        });

        samplers_ibl.push(sampler_brdf_lut);
        let objects = vec![GameObjectData {
            handle: GameObjectHandle(0),
            renderable: resource_cache.get_geometry_handle(&"sa23"),
        }];

        let mut physics_engine = PhysicsEngine::new();

        let starfury = Starfury::new(objects[0].handle, &mut physics_engine, &resource_cache);

        let shadows_swarm = ShadowFighterSwarm::new(&mut physics_engine, &resource_cache);

        let shadows_swarm_inst_render_data = InstancedRenderingData::create(
            renderer,
            &resource_cache,
            shadows_swarm.params.instance_count,
        )?;

        Some(GameWorld {
            draw_opts: RefCell::new(DebugDrawOptions::default()),
            resource_cache,
            skybox,
            single_inst_renderdata: SingleInstanceRenderingData {
                vs_ubo_transforms,
                fs_ubo_lights,
                object_descriptor_sets: object_pbr_descriptor_sets,
                ibl_descriptor_sets: ibl_descriptor_sets.into_vec(),
                samplers: samplers_ibl.into_vec(),
            },
            objects,
            starfury,
            shadows_swarm,
            shadows_swarm_inst_render_data,
            frame_times: RefCell::new(Vec::with_capacity(Self::MAX_HISTOGRAM_VALUES)),
            physics_engine: RefCell::new(physics_engine),
            camera: RefCell::new(FlightCamera::new()),
            debug_draw_overlay: std::rc::Rc::new(RefCell::new(
                DebugDrawOverlay::create(&renderer).expect("Failed to create debug draw overlay"),
            )),
            projectiles_sys: RefCell::new(ProjectileSystem::create(renderer, app_cfg)?),
        })
    }

    pub fn draw(&self, frame_context: &FrameRenderContext) {
        self.debug_draw_overlay.borrow_mut().clear();

        let projection = math::perspective(
            75f32,
            frame_context.framebuffer_size.x as f32 / frame_context.framebuffer_size.y as f32,
            0.1f32,
            5000f32,
        );

        {
            let cam_ref = self.camera.borrow();
            let draw_context = DrawContext {
                renderer: frame_context.renderer,
                cmd_buff: frame_context.cmd_buff,
                frame_id: frame_context.frame_id,
                viewport: frame_context.viewport,
                scissor: frame_context.scissor,
                camera: &*cam_ref,
                projection,
                projection_view: projection * cam_ref.view_transform(),
                debug_draw: self.debug_draw_overlay.clone(),
            };

            self.draw_objects(&draw_context);
        }

        if self.draw_options().debug_draw_physics {
            self.physics_engine
                .borrow_mut()
                .debug_draw(&mut self.debug_draw_overlay.borrow_mut());
        }

        self.debug_draw_overlay.borrow_mut().draw(
            frame_context.renderer,
            &(projection * self.camera.borrow().view_transform()),
        );
    }

    fn draw_objects(&self, draw_context: &DrawContext) {
        if self.draw_options().debug_draw_world_axis {
            self.debug_draw_overlay.borrow_mut().add_axes(
                Vec3::zeros(),
                self.draw_opts.borrow().world_axis_length,
                &glm::Mat3::identity(),
                None,
            );
        }

        self.skybox.draw(&draw_context);

        let device = draw_context.renderer.graphics_device();

        self.single_inst_renderdata
            .vs_ubo_transforms
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mapping| {
                self.physics_engine
                    .borrow()
                    .rigid_body_set
                    .get(self.starfury.rigid_body_handle)
                    .map(|rigid_body| {
                        let transforms = PbrTransformDataSingleInstanceUBO {
                            world: rigid_body.position().to_homogeneous(),
                            view: draw_context.camera.view_transform(),
                            projection: draw_context.projection,
                        };

                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                &transforms as *const _,
                                (mapping.memptr() as *mut u8)
                                // .offset((self.pbr_cpu_2_gpu.aligned_ubo_transforms_size) as isize)
                                as *mut PbrTransformDataSingleInstanceUBO,
                                1,
                            );
                        }
                    });
            });

        let pbr_light_data = PbrLightingData {
            eye_pos: draw_context.camera.position(),
        };

        self.single_inst_renderdata
            .fs_ubo_lights
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mapping| unsafe {
                std::ptr::copy_nonoverlapping(
                    &pbr_light_data as *const _,
                    mapping.memptr() as *mut PbrLightingData,
                    1,
                );
            });

        unsafe {
            device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.resource_cache.pbr_pipeline().pipeline,
            );
            device.cmd_set_viewport(draw_context.cmd_buff, 0, &[draw_context.viewport]);
            device.cmd_set_scissor(draw_context.cmd_buff, 0, &[draw_context.scissor]);

            let vertex_buffers = [self.resource_cache.vertex_buffer()];
            let vertex_offsets = [0u64];
            device.cmd_bind_vertex_buffers(
                draw_context.cmd_buff,
                0,
                &vertex_buffers,
                &vertex_offsets,
            );
            device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.resource_cache.index_buffer(),
                0,
                IndexType::UINT32,
            );

            self.objects.iter().for_each(|game_object| {
                let object_renderable = self
                    .resource_cache
                    .get_pbr_renderable(game_object.renderable);

                let bound_descriptor_sets = [
                    self.single_inst_renderdata.object_descriptor_sets[0],
                    object_renderable.descriptor_sets[0],
                    self.single_inst_renderdata.object_descriptor_sets[1],
                    self.single_inst_renderdata.ibl_descriptor_sets
                        [self.skybox.active_skybox as usize],
                ];

                let descriptor_set_offsets = [
                    self.single_inst_renderdata
                        .vs_ubo_transforms
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32
                        + self.single_inst_renderdata.vs_ubo_transforms.align as u32
                            * game_object.handle.0,
                    0,
                    self.single_inst_renderdata
                        .fs_ubo_lights
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                ];

                device.cmd_bind_descriptor_sets(
                    draw_context.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    self.resource_cache.pbr_pipeline().layout,
                    0,
                    &bound_descriptor_sets,
                    &descriptor_set_offsets,
                );

                device.cmd_draw_indexed(
                    draw_context.cmd_buff,
                    object_renderable.geometry.index_count,
                    1,
                    object_renderable.geometry.index_offset,
                    object_renderable.geometry.vertex_offset as i32,
                    0,
                );

                if self.draw_options().debug_draw_mesh {
                    // let aabb = self.render_state.borrow()[game_object.handle.0 as usize]
                    //     .render_pos
                    //     .to_homogeneous()
                    //     * object_renderable.geometry.aabb;
                    //
                    // draw_context.debug_draw.borrow_mut().add_aabb(
                    //     &aabb.min,
                    //     &aabb.max,
                    //     0xFF_00_00_FF,
                    // );
                }

                if self.draw_options().debug_draw_nodes_bounding {
                    // let object_transform = self.render_state.borrow()
                    //     [game_object.handle.0 as usize]
                    //     .render_pos
                    //     .to_homogeneous();
                    //
                    // object_renderable.geometry.nodes.iter().for_each(|node| {
                    //     let transformed_aabb = object_transform * node.aabb;
                    //
                    //     draw_context.debug_draw.borrow_mut().add_aabb(
                    //         &transformed_aabb.min,
                    //         &transformed_aabb.max,
                    //         0xFF_00_FF_00,
                    //     );
                    // });
                }
            });

            self.projectiles_sys
                .borrow()
                .render(draw_context, &self.physics_engine.borrow());
        }

        self.draw_instanced_objects(draw_context);
    }

    fn draw_instanced_objects(&self, draw_context: &DrawContext) {
        let device = draw_context.renderer.graphics_device();

        let global_uniforms = PbrTransformDataMultiInstanceUBO {
            view: draw_context.camera.view_transform(),
            projection: draw_context.projection,
        };

        self.shadows_swarm_inst_render_data
            .ubo_vtx_transforms
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|mapping_ubo| unsafe {
                std::ptr::copy_nonoverlapping(
                    &global_uniforms as *const _,
                    mapping_ubo.memptr() as *mut PbrTransformDataMultiInstanceUBO,
                    1,
                );
            });

        let phys_engine = self.physics_engine.borrow();

        self.shadows_swarm_inst_render_data
            .stb_instances
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|instances_storage_buffer| {
                let instance_model_transforms = self
                    .shadows_swarm
                    .instances_physics_data
                    .iter()
                    .filter_map(|instance_physics_data| {
                        phys_engine
                            .rigid_body_set
                            .get(instance_physics_data.rigid_body_handle)
                            .map(|rigid_body| PbrTransformDataInstanceEntry {
                                model: rigid_body.position().to_homogeneous(),
                            })
                    })
                    .collect::<SmallVec<[PbrTransformDataInstanceEntry; 16]>>();

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        instance_model_transforms.as_ptr(),
                        instances_storage_buffer.memptr() as *mut PbrTransformDataInstanceEntry,
                        instance_model_transforms.len(),
                    );
                }
            });

        unsafe {
            let graphics_device = draw_context.renderer.graphics_device();

            graphics_device.cmd_bind_pipeline(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.resource_cache.pbr_pipeline_instanced().pipeline,
            );

            graphics_device.cmd_set_viewport(draw_context.cmd_buff, 0, &[draw_context.viewport]);
            graphics_device.cmd_set_scissor(draw_context.cmd_buff, 0, &[draw_context.scissor]);

            graphics_device.cmd_bind_vertex_buffers(
                draw_context.cmd_buff,
                0,
                &[self.resource_cache.vertex_buffer()],
                &[0],
            );
            graphics_device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.resource_cache.index_buffer(),
                0,
                IndexType::UINT32,
            );

            let renderable = self
                .resource_cache
                .get_pbr_renderable(self.shadows_swarm.renderable);

            graphics_device.cmd_bind_descriptor_sets(
                draw_context.cmd_buff,
                PipelineBindPoint::GRAPHICS,
                self.resource_cache.pbr_pipeline_instanced().layout,
                0,
                &[
                    self.shadows_swarm_inst_render_data.descriptor_sets[0],
                    renderable.descriptor_sets[0],
                    self.single_inst_renderdata.object_descriptor_sets[1],
                    self.single_inst_renderdata.ibl_descriptor_sets
                        [self.skybox.active_skybox as usize],
                ],
                &[
                    self.shadows_swarm_inst_render_data
                        .ubo_vtx_transforms
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                    self.shadows_swarm_inst_render_data
                        .stb_instances
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                    0u32,
                    self.single_inst_renderdata
                        .fs_ubo_lights
                        .offset_for_frame(draw_context.frame_id as DeviceSize)
                        as u32,
                ],
            );

            graphics_device.cmd_draw_indexed(
                draw_context.cmd_buff,
                renderable.geometry.index_count,
                self.shadows_swarm.instances_physics_data.len() as u32,
                renderable.geometry.index_offset,
                renderable.geometry.vertex_offset as i32,
                0,
            );
        }
    }

    pub fn ui(&self, ui: &mut imgui::Ui) {
        ui.window("Options")
            .size([400.0, 110.0], imgui::Condition::FirstUseEver)
            .build(|| {
                {
                    let frames_histogram_values = self.frame_times.borrow();
                    ui.plot_histogram("Frame times", &frames_histogram_values)
                        .scale_min(0f32)
                        .scale_max(0.05f32)
                        .graph_size([400f32, 150f32])
                        .build();

                    ui.plot_lines("Frame times (lines)", &frames_histogram_values)
                        .scale_min(0f32)
                        .scale_max(0.05f32)
                        .graph_size([400f32, 150f32])
                        .build();
                }

                ui.separator();
                ui.text("Debug draw:");

                {
                    let mut dbg_draw = self.draw_options_mut();
                    ui.checkbox("World axis", &mut dbg_draw.debug_draw_world_axis);
                    ui.same_line();
                    ui.slider(
                        "World axis length",
                        0.1f32,
                        4f32,
                        &mut dbg_draw.world_axis_length,
                    );

                    ui.checkbox("Physics objects", &mut dbg_draw.debug_draw_physics);
                    ui.checkbox(
                        "Mesh nodes bounding boxes",
                        &mut dbg_draw.debug_draw_nodes_bounding,
                    );
                    ui.checkbox("Mesh bounding box", &mut dbg_draw.debug_draw_mesh);
                }

                ui.separator();
                ui.text("Starfury:");

                {
                    let phys_eng = self.physics_engine.borrow();

                    phys_eng
                        .rigid_body_set
                        .get(self.starfury.rigid_body_handle)
                        .map(|b| {
                            ui.text_colored(
                                [1f32, 0f32, 0f32, 1f32],
                                format!("Linear velocity: {}", b.linvel()),
                            );
                            ui.text_colored(
                                [1f32, 0f32, 0f32, 1f32],
                                format!("Angular velocity: {}", b.angvel()),
                            );
                            ui.text_colored(
                                [0f32, 1f32, 0f32, 1f32],
                                format!("Position: {}", b.position().translation.vector),
                            );
                        });
                }
            });
    }

    fn num_physics_steps_240hz(elapsed: f64) -> i32 {
        //
        // from https://www.gamedeveloper.com/programming/fixing-your-time-step-the-easy-way-with-the-golden-4-8537-ms-

        //
        // Our simulation frequency is 240Hz, a 4⅙  (four one sixth) ms period.
        // We will pretend our display sync rate is one of these:
        if elapsed > 7.5f64 * Self::PHYSICS_TIME_STEP {
            return 8; // 30 Hz        ( .. to 32 Hz )
        } else if elapsed > 6.5f64 * Self::PHYSICS_TIME_STEP {
            return 7; // 34.29 Hz     ( 32 Hz to 36.92 Hz )
        } else if elapsed > 5.5f64 * Self::PHYSICS_TIME_STEP {
            return 6; // 40 Hz        ( 36.92 Hz to 43.64 Hz )
        } else if elapsed > 4.5f64 * Self::PHYSICS_TIME_STEP {
            return 5; // 48 Hz        ( 43.64 Hz to 53.33 Hz )
        } else if elapsed > 3.5f64 * Self::PHYSICS_TIME_STEP {
            return 4; // 60 Hz        ( 53.33 Hz to 68.57 Hz )
        } else if elapsed > 2.5f64 * Self::PHYSICS_TIME_STEP {
            return 3; // 90 Hz        ( 68.57 Hz to 96 Hz )
        } else if elapsed > 1.5f64 * Self::PHYSICS_TIME_STEP {
            return 2; // 120 Hz       ( 96 Hz to 160 Hz )
        } else {
            return 1; // 240 Hz       ( 160 Hz to .. )
        }
    }

    pub fn update(&self, frame_time: f64) {
        // log::info!("Frame time: {}", frame_time);

        {
            let queued_commands = {
                let mut phys_engine = self.physics_engine.borrow_mut();
                let mut update_ctx = UpdateContext {
                    physics_engine: &mut &mut phys_engine,
                    queued_commands: Vec::with_capacity(32),
                    frame_time,
                };

                self.projectiles_sys.borrow_mut().update(&mut update_ctx);

                self.starfury.update(&mut update_ctx);
                // log::info!(
                //     "Queued commands for this frame: {}",
                //     update_ctx.queued_commands.len()
                // );

                update_ctx.queued_commands
            };

            {
                let mut phys_eng = self.physics_engine.borrow_mut();
                queued_commands
                    .iter()
                    .for_each(|&queued_cmd| match queued_cmd {
                        QueuedCommand::SpawnProjectile(data) => {
                            self.projectiles_sys
                                .borrow_mut()
                                .add_projectile(data, &mut phys_eng);
                        }
                        _ => {}
                    });
            }
        }

        {
            let mut frame_times = self.frame_times.borrow_mut();
            if (frame_times.len() + 1) > Self::MAX_HISTOGRAM_VALUES {
                frame_times.rotate_left(1);
                frame_times[Self::MAX_HISTOGRAM_VALUES - 1] = frame_time as f32;
            } else {
                frame_times.push(frame_time as f32);
            }
        }

        (0..Self::num_physics_steps_240hz(frame_time)).for_each(|_| {
            //
            // do physics step
            self.physics_engine.borrow_mut().update();
            //
            // update flight camera
            self.physics_engine
                .borrow()
                .rigid_body_set
                .get(self.starfury.rigid_body_handle)
                .map(|starfury_phys_obj| self.camera.borrow_mut().update(starfury_phys_obj));
        });
    }

    pub fn input_event(&self, event: &winit::event::WindowEvent) {
        use winit::event::WindowEvent;
        match event {
            WindowEvent::KeyboardInput {
                device_id: _,
                input,
                is_synthetic: _,
            } => {
                self.starfury.input_event(input);
            }
            _ => {}
        }
    }

    pub fn gamepad_input(&self, input_state: &InputState) {
        self.starfury.gamepad_input(input_state);
    }
}
