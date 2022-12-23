use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
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
    physics_engine::PhysicsEngine,
    resource_cache::{PbrDescriptorType, PbrRenderableHandle, ResourceHolder},
    skybox::Skybox,
    starfury::Starfury,
    vk_renderer::{ScopedBufferMapping, UniqueBuffer, UniqueImage, UniqueSampler, VulkanRenderer},
};

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
pub struct PbrTransformDataUBO {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub world: glm::Mat4,
}

#[derive(Copy, Clone, Debug)]
struct GameObjectRenderState {
    physics_pos: Isometry3<f32>,
    render_pos: Isometry3<f32>,
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
        let debug_draw_overlay = DebugDrawOverlay::create(renderer)?;
        let skybox = Skybox::create(renderer, &app_cfg.scene, &app_cfg.engine)?;

        let resource_cache = ResourceHolder::create(renderer, app_cfg)?;
        let aligned_ubo_transforms_size = VulkanRenderer::aligned_size_of_type::<PbrTransformDataUBO>(
            renderer
                .device_properties()
                .limits
                .min_uniform_buffer_offset_alignment,
        );
        let size_ubo_transforms_one_frame = aligned_ubo_transforms_size * 1; // 1 object for now
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
                .range(size_of::<PbrTransformDataUBO>() as DeviceSize)
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

        let starfury = Starfury::new(
            objects[0].renderable,
            objects[0].handle,
            &mut physics_engine,
            &resource_cache
                .get_pbr_renderable(objects[0].renderable)
                .geometry,
        );

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

        let view_matrix = draw_context.camera.view_transform();

        let perspective = draw_context.projection;

        let starfury_transform = self
            .get_object_state(self.starfury.object_handle)
            .render_pos
            .to_homogeneous();

        let transforms = PbrTransformDataUBO {
            world: starfury_transform,
            view: draw_context.camera.view_transform(),
            projection: perspective,
        };

        ScopedBufferMapping::create(
            draw_context.renderer,
            &self.pbr_cpu_2_gpu.ubo_transforms,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame,
            self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame * draw_context.frame_id as u64,
        )
        .map(|mapping| unsafe {
            std::ptr::copy_nonoverlapping(
                &transforms as *const _,
                mapping.memptr() as *mut PbrTransformDataUBO,
                1,
            );
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
            device.cmd_set_viewport(draw_context.cmd_buff, 0, &viewports);
            device.cmd_set_scissor(draw_context.cmd_buff, 0, &scisssors);

            let sa23_renderable = self
                .resource_cache
                .get_pbr_renderable(self.starfury.renderable);

            if self.draw_options().debug_draw_mesh {
                let aabb = starfury_transform * sa23_renderable.geometry.aabb;

                draw_context
                    .debug_draw
                    .borrow_mut()
                    .add_aabb(&aabb.min, &aabb.max, 0xFF_00_00_FF);
            }

            if self.draw_options().debug_draw_nodes_bounding {
                sa23_renderable.geometry.nodes.iter().for_each(|node| {
                    let transformed_aabb = starfury_transform * node.aabb;

                    draw_context.debug_draw.borrow_mut().add_aabb(
                        &transformed_aabb.min,
                        &transformed_aabb.max,
                        0xFF_00_FF_00,
                    );
                });
            }

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

            let bound_descriptor_sets = [
                self.pbr_cpu_2_gpu.object_descriptor_sets[0],
                sa23_renderable.descriptor_sets[0],
                self.pbr_cpu_2_gpu.object_descriptor_sets[1],
                self.pbr_cpu_2_gpu.ibl_descriptor_sets[self.skybox.active_skybox as usize],
            ];

            let descriptor_set_offsets = [
                self.pbr_cpu_2_gpu.size_ubo_transforms_one_frame as u32 * draw_context.frame_id,
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
                sa23_renderable.geometry.index_count,
                1,
                sa23_renderable.geometry.index_offset,
                sa23_renderable.geometry.vertex_offset as i32,
                0,
            );
        }

        if self.draw_options().debug_draw_physics {
            self.physics_engine
                .borrow_mut()
                .debug_draw(&mut draw_context.debug_draw.borrow_mut());
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
        phys_engine
            .rigid_body_set
            .get(self.starfury.rigid_body_handle)
            .map(|rbody| {
                let previous_object_state = *self.get_object_state(self.starfury.object_handle);

                let new_object_state = GameObjectRenderState {
                    physics_pos: *rbody.position(),
                    render_pos: previous_object_state
                        .physics_pos
                        .lerp_slerp(rbody.position(), physics_time_factor as f32),
                };

                *self.get_object_state_mut(self.starfury.object_handle) = new_object_state;
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
}
