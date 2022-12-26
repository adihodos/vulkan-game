use std::{
    cell::{Cell, RefCell},
    mem::size_of,
};

use ash::vk::{
    BorderColor, BufferUsageFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorType, DeviceSize, Filter, ImageLayout, ImageTiling,
    ImageUsageFlags, IndexType, MemoryPropertyFlags, PipelineBindPoint, SamplerAddressMode,
    SamplerCreateInfo, SamplerMipmapMode, WriteDescriptorSet,
};
use glm::{Mat4, Vec3};
use nalgebra::Isometry3;
use nalgebra_glm::Vec4;

use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
use smallvec::SmallVec;

use crate::{
    app_config::AppConfig,
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::DrawContext,
    game_object::GameObjectRenderState,
    physics_engine::PhysicsEngine,
    resource_cache::{PbrDescriptorType, PbrRenderableHandle, ResourceHolder},
    shadow_swarm::ShadowFighterSwarm,
    skybox::Skybox,
    starfury::Starfury,
    vk_renderer::{
        Cpu2GpuBuffer, ScopedBufferMapping, UniqueBuffer, UniqueImage, UniqueSampler,
        VulkanRenderer,
    },
    window::InputState,
};

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

struct PbrCpu2GpuData {
    aligned_ubo_transforms_size: DeviceSize,
    aligned_ubo_lighting_size: DeviceSize,
    size_ubo_transforms_one_frame: DeviceSize,
    size_ubo_lighting_one_frame: DeviceSize,
    ubo_transforms: UniqueBuffer,
    ubo_lighting: UniqueBuffer,
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
    pbr_cpu_2_gpu: PbrCpu2GpuData,
    objects: Vec<GameObjectData>,
    starfury: Starfury,
    shadows_swarm: ShadowFighterSwarm,
    shadows_swarm_inst_render_data: InstancedRenderingData,
    accumulator: Cell<f64>,
    frame_times: RefCell<Vec<f32>>,
    physics_engine: RefCell<PhysicsEngine>,
    render_state: RefCell<Vec<GameObjectRenderState>>,
}

impl GameWorld {
    const PHYSICS_TIME_STEP: f64 = 1f64 / 60f64;
    const MAX_HISTOGRAM_VALUES: usize = 32;

    fn get_object_state(&self, handle: GameObjectHandle) -> std::cell::Ref<GameObjectRenderState> {
        let b = self.render_state.borrow();
        std::cell::Ref::map(b, |game_objs| &game_objs[handle.0 as usize])
    }

    fn get_object_state_mut(
        &self,
        handle: GameObjectHandle,
    ) -> std::cell::RefMut<GameObjectRenderState> {
        let b = self.render_state.borrow_mut();
        std::cell::RefMut::map(b, |game_objs| &mut game_objs[handle.0 as usize])
    }

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
        let aligned_ubo_transforms_size =
            VulkanRenderer::aligned_size_of_type::<PbrTransformDataSingleInstanceUBO>(
                renderer
                    .device_properties()
                    .limits
                    .min_uniform_buffer_offset_alignment,
            );
        let size_ubo_transforms_one_frame = aligned_ubo_transforms_size * 2; // 1 object for now
        let pbr_ubo_transforms = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            size_ubo_transforms_one_frame * renderer.max_inflight_frames() as DeviceSize,
        )?;

        let aligned_ubo_lighting_size = VulkanRenderer::aligned_size_of_type::<PbrLightingData>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );
        let size_ubo_lighting_one_frame = aligned_ubo_lighting_size * 1; // 1 object for now
        let pbr_ubo_lighting = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            size_ubo_lighting_one_frame * renderer.max_inflight_frames() as DeviceSize,
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
                .buffer(pbr_ubo_transforms.buffer)
                .offset(0)
                .range(size_of::<PbrTransformDataSingleInstanceUBO>() as DeviceSize)
                .build(),
            DescriptorBufferInfo::builder()
                .buffer(pbr_ubo_lighting.buffer)
                .offset(0)
                .range(size_of::<PbrLightingData>() as DeviceSize)
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

        let render_state = vec![GameObjectRenderState {
            render_pos: Isometry3::identity(),
            physics_pos: Isometry3::identity(),
        }];

        Some(GameWorld {
            draw_opts: RefCell::new(DebugDrawOptions::default()),
            resource_cache,
            skybox,
            pbr_cpu_2_gpu: PbrCpu2GpuData {
                aligned_ubo_transforms_size,
                aligned_ubo_lighting_size,
                size_ubo_transforms_one_frame,
                size_ubo_lighting_one_frame,
                ubo_transforms: pbr_ubo_transforms,
                ubo_lighting: pbr_ubo_lighting,
                object_descriptor_sets: object_pbr_descriptor_sets,
                ibl_descriptor_sets: ibl_descriptor_sets.into_vec(),
                samplers: samplers_ibl.into_vec(),
            },
            objects,
            starfury,
            shadows_swarm,
            shadows_swarm_inst_render_data,
            accumulator: Cell::new(0f64),
            frame_times: RefCell::new(Vec::with_capacity(Self::MAX_HISTOGRAM_VALUES)),
            physics_engine: RefCell::new(physics_engine),
            render_state: RefCell::new(render_state),
        })
    }

    pub fn draw(&self, draw_context: &DrawContext) {
        if self.draw_options().debug_draw_world_axis {
            draw_context.debug_draw.borrow_mut().add_axes(
                Vec3::zeros(),
                self.draw_opts.borrow().world_axis_length,
                &glm::Mat3::identity(),
                None,
            );
        }

        self.skybox.draw(draw_context);

        let device = draw_context.renderer.graphics_device();

        let viewports = [draw_context.viewport];
        let scisssors = [draw_context.scissor];

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.pbr_cpu_2_gpu.ubo_transforms,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame * draw_context.frame_id as u64,
        )
        .map(|mapping| {
            let render_state = self.render_state.borrow();
            render_state
                .iter()
                .enumerate()
                .for_each(|(idx, render_state)| {
                    let transforms = PbrTransformDataSingleInstanceUBO {
                        world: render_state.render_pos.to_homogeneous(),
                        view: draw_context.camera.view_transform(),
                        projection: draw_context.projection,
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            &transforms as *const _,
                            (mapping.memptr() as *mut u8).offset(
                                (idx as u64 * self.pbr_cpu_2_gpu.aligned_ubo_transforms_size)
                                    as isize,
                            ) as *mut PbrTransformDataSingleInstanceUBO,
                            1,
                        );
                    }
                });
        });

        let pbr_light_data = PbrLightingData {
            eye_pos: draw_context.camera.position(),
        };

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.pbr_cpu_2_gpu.ubo_lighting,
            self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame,
            self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame * draw_context.frame_id as u64,
        )
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
                    self.pbr_cpu_2_gpu.object_descriptor_sets[0],
                    object_renderable.descriptor_sets[0],
                    self.pbr_cpu_2_gpu.object_descriptor_sets[1],
                    self.pbr_cpu_2_gpu.ibl_descriptor_sets[self.skybox.active_skybox as usize],
                ];

                let descriptor_set_offsets = [
                    self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame as u32 * draw_context.frame_id
                        + self.pbr_cpu_2_gpu.aligned_ubo_transforms_size as u32
                            * game_object.handle.0,
                    0,
                    self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame as u32 * draw_context.frame_id,
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
                    let aabb = self.render_state.borrow()[game_object.handle.0 as usize]
                        .render_pos
                        .to_homogeneous()
                        * object_renderable.geometry.aabb;

                    draw_context.debug_draw.borrow_mut().add_aabb(
                        &aabb.min,
                        &aabb.max,
                        0xFF_00_00_FF,
                    );
                }

                if self.draw_options().debug_draw_nodes_bounding {
                    let object_transform = self.render_state.borrow()
                        [game_object.handle.0 as usize]
                        .render_pos
                        .to_homogeneous();

                    object_renderable.geometry.nodes.iter().for_each(|node| {
                        let transformed_aabb = object_transform * node.aabb;

                        draw_context.debug_draw.borrow_mut().add_aabb(
                            &transformed_aabb.min,
                            &transformed_aabb.max,
                            0xFF_00_FF_00,
                        );
                    });
                }
            });
        }

        self.draw_instanced_objects(draw_context);

        if self.draw_options().debug_draw_physics {
            self.physics_engine
                .borrow_mut()
                .debug_draw(&mut draw_context.debug_draw.borrow_mut());
        }
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

        self.shadows_swarm_inst_render_data
            .stb_instances
            .map_for_frame(draw_context.renderer, draw_context.frame_id as DeviceSize)
            .map(|instances_storage_buffer| {
                let instance_model_transforms = self
                    .shadows_swarm
                    .instances_render_data
                    .borrow()
                    .iter()
                    .map(|inst_render_data| PbrTransformDataInstanceEntry {
                        model: inst_render_data.render_pos.to_homogeneous(),
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
                    self.pbr_cpu_2_gpu.object_descriptor_sets[1],
                    self.pbr_cpu_2_gpu.ibl_descriptor_sets[self.skybox.active_skybox as usize],
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
                    self.pbr_cpu_2_gpu.size_ubo_lighting_one_frame as u32 * draw_context.frame_id,
                ],
            );

            graphics_device.cmd_draw_indexed(
                draw_context.cmd_buff,
                renderable.geometry.index_count,
                self.shadows_swarm.instances_render_data.borrow().len() as u32,
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
                        });
                }
            });
    }

    pub fn update(&self, frame_time: f64) {
        // log::info!("Frame time: {}", frame_time);
        self.starfury
            .physics_update(&mut self.physics_engine.borrow_mut());

        {
            let mut frame_times = self.frame_times.borrow_mut();
            if (frame_times.len() + 1) > Self::MAX_HISTOGRAM_VALUES {
                frame_times.rotate_left(1);
                frame_times[Self::MAX_HISTOGRAM_VALUES - 1] = frame_time as f32;
            } else {
                frame_times.push(frame_time as f32);
            }
        }
        self.accumulator.set(self.accumulator.get() + frame_time);

        while self.accumulator.get() >= Self::PHYSICS_TIME_STEP {
            //
            // do physics step
            self.physics_engine.borrow_mut().update();
            self.accumulator
                .set(self.accumulator.get() - Self::PHYSICS_TIME_STEP);
        }

        //
        // interpolate transforms
        let physics_time_factor = self.accumulator.get() / Self::PHYSICS_TIME_STEP;
        let phys_engine = self.physics_engine.borrow();

        let objects = [(self.starfury.rigid_body_handle, self.starfury.object_handle)];

        objects.iter().for_each(|&(body_handle, object_handle)| {
            phys_engine.rigid_body_set.get(body_handle).map(|rbody| {
                let previous_object_state = *self.get_object_state(object_handle);

                let new_object_state = GameObjectRenderState {
                    physics_pos: *rbody.position(),
                    render_pos: previous_object_state
                        .physics_pos
                        .lerp_slerp(rbody.position(), physics_time_factor as f32),
                };

                *self.get_object_state_mut(object_handle) = new_object_state;
            });
        });

        let mut instances_render_data = self.shadows_swarm.instances_render_data.borrow_mut();
        instances_render_data
            .iter_mut()
            .zip(self.shadows_swarm.instances_physics_data.iter())
            .for_each(|(mut render_data, phys_data)| {
                phys_engine
                    .rigid_body_set
                    .get(phys_data.rigid_body_handle)
                    .map(|instance_rigid_body| {
                        let prev_instance_state = *render_data;
                        *render_data = GameObjectRenderState {
                            physics_pos: *instance_rigid_body.position(),
                            render_pos: prev_instance_state.physics_pos.lerp_slerp(
                                instance_rigid_body.position(),
                                physics_time_factor as f32,
                            ),
                        };
                    });
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
        // log::info!("==========>>>>><<<<<<<<<<<<<<=============");
        self.starfury.gamepad_input(input_state);
    }
}