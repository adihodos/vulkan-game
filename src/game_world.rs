use std::{
    borrow::Borrow,
    cell::{RefCell, RefMut},
    rc::Rc,
};

use ash::vk::{
    BorderColor, BufferUsageFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorType, DeviceSize, Filter, ImageLayout, IndexType,
    PhysicalDevicePipelineExecutablePropertiesFeaturesKHRBuilder, PipelineBindPoint,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, WriteDescriptorSet,
};
use nalgebra::Point3;
use nalgebra_glm as glm;
use nalgebra_glm::Vec4;

use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};
use smallvec::SmallVec;

use crate::{
    app_config::{AppConfig, PlayerShipConfig},
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::{DrawContext, FrameRenderContext, InitContext, UpdateContext},
    flight_cam::FlightCamera,
    fps_camera::FirstPersonCamera,
    frustrum::{is_aabb_on_frustrum, Frustrum, FrustrumPlane},
    game_object::GameObjectPhysicsData,
    math,
    missile_sys::{Missile, MissileKind, MissileSys},
    particles::{ImpactSpark, SparksSystem},
    physics_engine::{PhysicsEngine, PhysicsObjectCollisionGroups},
    projectile_system::{ProjectileSpawnData, ProjectileSystem},
    resource_cache::{
        DrawingSys, PbrDescriptorType, PbrRenderableHandle, ResourceHolder, ResourceSystem,
    },
    shadow_swarm::ShadowFighterSwarm,
    skybox::Skybox,
    sprite_batch::{SpriteBatch, TextureRegion},
    starfury::Starfury,
    ui_backend::UiBackend,
    vk_renderer::{Cpu2GpuBuffer, UniqueSampler, VulkanRenderer},
    window::{GamepadInputState, InputState},
};

#[derive(Copy, Clone, Debug)]
pub enum QueuedCommand {
    SpawnProjectile(ProjectileSpawnData),
    SpawnMissile(MissileKind, nalgebra::Isometry3<f32>, glm::Vec3, glm::Vec3),
    ProcessProjectileImpact(RigidBodyHandle),
    DrawMissile(Missile),
    DrawEngineExhaust(glm::Mat4),
}

#[derive(Copy, Clone, Debug)]
pub struct GameObjectCommonData {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub phys_rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub phys_collider_handle: rapier3d::prelude::ColliderHandle,
}

#[derive(Copy, Clone, Debug)]
struct DebugOptions {
    wireframe_color: Vec4,
    draw_normals: bool,
    normals_color: Vec4,
    debug_draw_physics: bool,
    debug_draw_nodes_bounding: bool,
    debug_draw_mesh: bool,
    debug_draw_world_axis: bool,
    world_axis_length: f32,
    frustrum_planes: enumflags2::BitFlags<FrustrumPlane>,
    debug_camera: bool,
    draw_frustrum_planes: bool,
}

impl DebugOptions {
    const WORLD_AXIS_MAX_LEN: f32 = 128f32;
}

impl std::default::Default for DebugOptions {
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
            frustrum_planes: enumflags2::BitFlags::empty(),
            debug_camera: false,
            draw_frustrum_planes: false,
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

struct PlayerShipOptions {
    spr_crosshair_normal: TextureRegion,
    spr_crosshair_hit: TextureRegion,
    spr_obj_outline: TextureRegion,
    spr_obj_centermass: TextureRegion,
    crosshair_size: f32,
    crosshair_color: u32,
    enemy_outline_color: u32,
}

impl PlayerShipOptions {
    fn new(cfg: &PlayerShipConfig, texture_atlas: &SpriteBatch) -> Self {
        Self {
            spr_crosshair_normal: texture_atlas
                .get_sprite_by_name(&cfg.crosshair_normal)
                .unwrap(),
            spr_crosshair_hit: texture_atlas
                .get_sprite_by_name(&cfg.crosshair_hit)
                .unwrap(),
            spr_obj_outline: texture_atlas
                .get_sprite_by_name(&cfg.target_outline)
                .unwrap(),
            spr_obj_centermass: texture_atlas
                .get_sprite_by_name(&cfg.target_centermass)
                .unwrap(),
            crosshair_size: cfg.crosshair_size,
            crosshair_color: cfg.crosshair_color,
            enemy_outline_color: cfg.target_color,
        }
    }
}

struct Statistics {
    total_instances: u32,
    visible_instances: u32,
}

pub struct GameWorld {
    draw_opts: RefCell<DebugOptions>,
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
    dbg_camera: RefCell<FirstPersonCamera>,
    debug_draw_overlay: Rc<RefCell<DebugDrawOverlay>>,
    projectiles_sys: RefCell<ProjectileSystem>,
    sparks_sys: RefCell<SparksSystem>,
    sprite_batch: RefCell<SpriteBatch>,
    player_opts: PlayerShipOptions,
    stats: RefCell<Statistics>,
    locked_target: RefCell<Option<(ColliderHandle, RigidBodyHandle)>>,
    missile_sys: RefCell<MissileSys>,
    rt: tokio::runtime::Runtime,
    resource_sys: ResourceSystem,
    drawing_sys: RefCell<DrawingSys>,
    ui: RefCell<UiBackend>,
}

impl GameWorld {
    const PHYSICS_TIME_STEP: f64 = 1f64 / 240f64;
    const MAX_HISTOGRAM_VALUES: usize = 32;

    fn debug_options(&self) -> std::cell::Ref<DebugOptions> {
        self.draw_opts.borrow()
    }

    fn debug_options_mut(&self) -> std::cell::RefMut<DebugOptions> {
        self.draw_opts.borrow_mut()
    }

    pub fn new(
        window: &winit::window::Window,
        renderer: &VulkanRenderer,
        cfg: &AppConfig,
    ) -> Option<GameWorld> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("vkgame-thread-pool")
            .thread_stack_size(4 * 1024 * 1024)
            .build()
            .expect("Failed to create Tokio runtime");

        let mut rsys =
            ResourceSystem::create(&renderer, &cfg).expect("Failed to create resource system");

        let resource_cache = ResourceHolder::create(renderer, cfg)?;

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
        .expect("Failed to allocate descriptor sets");

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

        // skybox.get_ibl_data().iter().for_each(|ibl_data| {
        //     let levels_irradiance = ibl_data.irradiance.info().num_levels;

        //     let sampler_cubemaps = UniqueSampler::new(
        //         renderer.graphics_device(),
        //         &SamplerCreateInfo::builder()
        //             .min_lod(0f32)
        //             .max_lod(levels_irradiance as f32)
        //             .min_filter(Filter::LINEAR)
        //             .mag_filter(Filter::LINEAR)
        //             .mipmap_mode(SamplerMipmapMode::LINEAR)
        //             .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
        //             .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
        //             .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
        //             .border_color(BorderColor::INT_OPAQUE_BLACK)
        //             .max_anisotropy(1f32)
        //             .build(),
        //     )
        //     .expect("Failed to create sampler");

        //     let ibl_desc_img_info = [
        //         DescriptorImageInfo::builder()
        //             .image_view(ibl_data.irradiance.image_view())
        //             .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        //             .sampler(sampler_cubemaps.sampler)
        //             .build(),
        //         DescriptorImageInfo::builder()
        //             .image_view(ibl_data.specular.image_view())
        //             .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        //             .sampler(sampler_cubemaps.sampler)
        //             .build(),
        //         DescriptorImageInfo::builder()
        //             .image_view(ibl_data.brdf_lut.image_view())
        //             .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        //             .sampler(sampler_brdf_lut.sampler)
        //             .build(),
        //     ];

        //     samplers_ibl.push(sampler_cubemaps);

        //     let dset_ibl = unsafe {
        //         renderer.graphics_device().allocate_descriptor_sets(
        //             &DescriptorSetAllocateInfo::builder()
        //                 .descriptor_pool(renderer.descriptor_pool())
        //                 .set_layouts(&pbr_descriptor_layouts[3..])
        //                 .build(),
        //         )
        //     }
        //     .expect("Failed to allocate descriptor sets");

        //     let wds = [
        //         //
        //         // irradiance
        //         WriteDescriptorSet::builder()
        //             .dst_set(dset_ibl[0])
        //             .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        //             .dst_binding(0)
        //             .image_info(&ibl_desc_img_info[0..1])
        //             .dst_array_element(0)
        //             .build(),
        //         //
        //         // specular
        //         WriteDescriptorSet::builder()
        //             .dst_set(dset_ibl[0])
        //             .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        //             .dst_binding(1)
        //             .dst_array_element(0)
        //             .image_info(&ibl_desc_img_info[1..2])
        //             .build(),
        //         //
        //         // BRDF lut
        //         WriteDescriptorSet::builder()
        //             .dst_set(dset_ibl[0])
        //             .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        //             .dst_binding(2)
        //             .dst_array_element(0)
        //             .image_info(&ibl_desc_img_info[2..])
        //             .build(),
        //     ];

        //     unsafe {
        //         renderer.graphics_device().update_descriptor_sets(&wds, &[]);
        //     }

        //     ibl_descriptor_sets.extend(dset_ibl);
        // });

        samplers_ibl.push(sampler_brdf_lut);
        let objects = vec![GameObjectData {
            handle: GameObjectHandle(0),
            renderable: resource_cache.get_pbr_geometry_handle(&"sa23"),
        }];

        let mut physics_engine = PhysicsEngine::new();

        let starfury = Starfury::new(objects[0].handle, &mut physics_engine, &resource_cache);
        let shadows_swarm = ShadowFighterSwarm::new(&mut physics_engine, &resource_cache);

        let shadows_swarm_inst_render_data = InstancedRenderingData::create(
            renderer,
            &resource_cache,
            shadows_swarm.params.instance_count,
        )?;

        let sprites = SpriteBatch::create(renderer, cfg)?;
        let missile_sys = MissileSys::new(renderer, &resource_cache, cfg)?;
        let projectile_sys = ProjectileSystem::new(renderer, &resource_cache, cfg)?;

        let aspect = renderer.framebuffer_extents().width as f32
            / renderer.framebuffer_extents().height as f32;

        let ui = RefCell::new(UiBackend::new(&mut InitContext {
            window,
            renderer,
            cfg,
            rsys: &mut rsys,
        })?);

        let skybox = Skybox::create(&mut InitContext {
            window,
            renderer,
            cfg,
            rsys: &mut rsys,
        })?;

        Some(GameWorld {
            draw_opts: RefCell::new(DebugOptions::default()),
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
            camera: RefCell::new(FlightCamera::new(75f32, aspect, 0.1f32, 5000f32)),
            dbg_camera: RefCell::new(FirstPersonCamera::new(75f32, aspect, 0.1f32, 5000f32)),
            debug_draw_overlay: std::rc::Rc::new(RefCell::new(
                DebugDrawOverlay::create(renderer).expect("Failed to create debug draw overlay"),
            )),
            sparks_sys: RefCell::new(SparksSystem::create(renderer, cfg)?),
            player_opts: PlayerShipOptions::new(&cfg.player, &sprites),
            sprite_batch: RefCell::new(sprites),
            stats: RefCell::new(Statistics {
                total_instances: 0,
                visible_instances: 0,
            }),
            locked_target: RefCell::new(None),
            rt,
            projectiles_sys: RefCell::new(projectile_sys),
            missile_sys: RefCell::new(missile_sys),
            resource_sys: rsys,
            drawing_sys: RefCell::new(DrawingSys::create(renderer)?),
            ui,
        })
    }

    fn object_visibility_check(
        &self,
    ) -> tokio::task::JoinHandle<Vec<(GameObjectPhysicsData, nalgebra::Isometry3<f32>)>> {
        let frustrum = Frustrum::from_flight_cam(&self.camera.borrow());

        if self.debug_options().debug_camera {
            use crate::color_palettes::StdColors;
            let cam = self.camera.borrow();

            if self.debug_options().draw_frustrum_planes {
                self.debug_draw_overlay.borrow_mut().add_frustrum(
                    &frustrum,
                    &cam.position,
                    self.debug_options().frustrum_planes,
                );
            } else {
                self.debug_draw_overlay.borrow_mut().add_frustrum_pyramid(
                    cam.fovy,
                    cam.near,
                    500f32,
                    cam.aspect,
                    cam.right_up_dir(),
                    cam.position,
                    StdColors::SEA_GREEN,
                );
            }
        }

        let inst_aabb = self
            .resource_cache
            .get_pbr_geometry_info(self.shadows_swarm.renderable)
            .aabb;

        let all_instances = self
            .shadows_swarm
            .instances()
            .iter()
            .filter_map(|i| {
                self.physics_engine
                    .borrow()
                    .rigid_body_set
                    .get(i.rigid_body_handle)
                    .and_then(|rbody| {
                        let inst_transform = *rbody.position();
                        Some((*i, inst_transform))
                    })
            })
            .collect::<Vec<_>>();

        self.rt.spawn(async move {
            all_instances
                .iter()
                .filter_map(|(i, mtx)| {
                    if is_aabb_on_frustrum(&frustrum, &inst_aabb, &mtx) {
                        Some((*i, *mtx))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
    }

    pub fn draw(&self, frame_context: &FrameRenderContext) {
        //
        // start a visibility check early

        let (projection, inverse_projection) = math::perspective(
            75f32,
            frame_context.framebuffer_size.x as f32 / frame_context.framebuffer_size.y as f32,
            0.1f32,
            5000f32,
        );

        self.camera.borrow_mut().projection_matrix = projection;
        self.camera.borrow_mut().inverse_projection = inverse_projection;
        self.camera.borrow_mut().aspect =
            frame_context.framebuffer_size.x as f32 / frame_context.framebuffer_size.y as f32;

        if self.debug_options().debug_camera {
            self.dbg_camera.borrow_mut().set_lens(
                75f32,
                frame_context.framebuffer_size.x as f32 / frame_context.framebuffer_size.y as f32,
                0.1f32,
                5000f32,
            );
        }

        let (view_matrix, cam_position) = {
            let flight_cam = self.camera.borrow();

            if self.debug_options().debug_camera {
                (
                    self.dbg_camera.borrow().view_matrix,
                    self.dbg_camera.borrow().position,
                )
            } else {
                (flight_cam.view_matrix, flight_cam.position)
            }
        };

        let rcache = self.resource_cache.borrow();

        let draw_context = DrawContext {
            rcache: &rcache,
            rsys: &self.resource_sys,
            renderer: frame_context.renderer,
            cmd_buff: frame_context.cmd_buff,
            frame_id: frame_context.frame_id,
            viewport: frame_context.viewport,
            scissor: frame_context.scissor,
            view_matrix,
            cam_position,
            projection,
            inverse_projection,
            projection_view: projection * view_matrix,
            debug_draw: self.debug_draw_overlay.clone(),
        };

        use nalgebra::{Isometry3, Rotation, Translation3};
        let sf0 = Isometry3::from_parts(
            Translation3::new(0f32, 0f32, 10f32),
            Rotation::identity().into(),
        );
        let sf1 = Isometry3::from_parts(
            Translation3::new(0f32, 10f32, 5f32),
            Rotation::identity().into(),
        );

        self.drawing_sys
            .borrow_mut()
            .add_mesh("sa23".into(), None, None, &sf0.to_matrix());
        self.drawing_sys
            .borrow_mut()
            .add_mesh("sa23".into(), None, None, &sf1.to_matrix());

        self.drawing_sys.borrow_mut().setup_bindless(self.skybox.id, &draw_context);

        self.skybox.draw(&draw_context);
        self.drawing_sys.borrow_mut().draw(&draw_context);

        // self.draw_objects(&draw_context);
        // self.draw_crosshair(&draw_context);
        // self.draw_locked_target_indicator(&draw_context);
        // self.sprite_batch.borrow_mut().render(&draw_context);

        // if self.debug_options().debug_draw_physics {
        //     self.physics_engine
        //         .borrow_mut()
        //         .debug_draw(&mut self.debug_draw_overlay.borrow_mut());
        // }

        // self.debug_draw_overlay
        //     .borrow_mut()
        //     .draw(frame_context.renderer, &draw_context.projection_view);

        // self.debug_draw_overlay.borrow_mut().clear();

        {
            let mut u = RefMut::map(self.ui.borrow_mut(), |ui| {
                ui.new_frame(frame_context.window)
            });

            self.ui(&mut u);
        }
        {
            let mut ui = self.ui.borrow_mut();
            ui.apply_cursor_before_render(frame_context.window);
            ui.draw_frame(&draw_context);
        }
    }

    fn draw_objects(&self, draw_context: &DrawContext) {
        let visible_objects_future = self.object_visibility_check();

        if self.debug_options().debug_draw_world_axis {
            self.debug_draw_overlay
                .borrow_mut()
                .world_space_coord_sys(self.draw_opts.borrow().world_axis_length);
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
                            view: draw_context.view_matrix,
                            projection: draw_context.projection,
                        };

                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                &transforms as *const _,
                                (mapping.memptr() as *mut u8)
                                    as *mut PbrTransformDataSingleInstanceUBO,
                                1,
                            );
                        }
                    });
            });

        let pbr_light_data = PbrLightingData {
            eye_pos: draw_context.cam_position,
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

            device.cmd_bind_vertex_buffers(
                draw_context.cmd_buff,
                0,
                &[self.resource_cache.vertex_buffer_pbr()],
                &[0u64],
            );
            device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.resource_cache.index_buffer_pbr(),
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
                        [
			    // self.skybox.active_skybox as usize
			    0
			],
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

                if self.debug_options().debug_draw_mesh {
                    // let aabb = self.render_state.borrow()[game_object.handle.0 as usize]
                    //     .render_pos
                    //     .to_homogeneous()
                    //     * object_renderable.geometry.aabb;

                    // draw_context.debug_draw.borrow_mut().add_aabb(
                    //     &aabb.min,
                    //     &aabb.max,
                    //     0xFF_00_00_FF,
                    // );
                }

                if self.debug_options().debug_draw_nodes_bounding {
                    let geometry = self
                        .resource_cache
                        .get_pbr_geometry_info(self.starfury.renderable);

                    let transform = self
                        .physics_engine
                        .borrow()
                        .get_rigid_body(self.starfury.rigid_body_handle)
                        .position()
                        .to_homogeneous();

                    geometry.nodes.iter().for_each(|node| {
                        let aabb = transform * node.aabb;

                        use crate::color_palettes::StdColors;
                        draw_context.debug_draw.borrow_mut().add_aabb(
                            &aabb.min,
                            &aabb.max,
                            StdColors::RED,
                        );
                    });
                }
            });
        }

        self.draw_instanced_objects(visible_objects_future, draw_context);
        self.missile_sys.borrow_mut().draw(draw_context);

        self.projectiles_sys
            .borrow()
            .render(draw_context, &self.physics_engine.borrow());
        self.sparks_sys.borrow().render(draw_context);
    }

    fn draw_instanced_objects(
        &self,
        vis_objects_future: tokio::task::JoinHandle<
            Vec<(GameObjectPhysicsData, nalgebra::Isometry3<f32>)>,
        >,
        draw_context: &DrawContext,
    ) {
        let visible_instances = self
            .rt
            .block_on(vis_objects_future)
            .expect("Failed to wait async visibility check task");

        *self.stats.borrow_mut() = Statistics {
            visible_instances: visible_instances.len() as u32,
            total_instances: self.shadows_swarm.instances().len() as u32,
        };

        if visible_instances.is_empty() {
            return;
        }

        let global_uniforms = PbrTransformDataMultiInstanceUBO {
            view: draw_context.view_matrix,
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
                let instance_model_transforms = visible_instances
                    .iter()
                    .map(|(_, inst_transform)| PbrTransformDataInstanceEntry {
                        model: inst_transform.to_homogeneous(),
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
                &[self.resource_cache.vertex_buffer_pbr()],
                &[0],
            );
            graphics_device.cmd_bind_index_buffer(
                draw_context.cmd_buff,
                self.resource_cache.index_buffer_pbr(),
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
                        [
			    // self.skybox.active_skybox as usize
			    0
			],
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
                visible_instances.len() as u32,
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
                if ui.collapsing_header("Debug draw:", imgui::TreeNodeFlags::FRAMED) {
                    let mut dbg_draw = self.debug_options_mut();
                    ui.checkbox("World axis", &mut dbg_draw.debug_draw_world_axis);
                    ui.same_line();
                    ui.slider(
                        "World axis length",
                        0.1f32,
                        DebugOptions::WORLD_AXIS_MAX_LEN,
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
                if ui.collapsing_header("Starfury:", imgui::TreeNodeFlags::FRAMED) {
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

                ui.separator();
                if ui.collapsing_header("Camera", imgui::TreeNodeFlags::FRAMED) {
                    if ui.checkbox(
                        "Activate debug camera",
                        &mut self.debug_options_mut().debug_camera,
                    ) {
                        let camera_frame = self.camera.borrow().view_matrix;
                        let camera_origin = self.camera.borrow().position;
                        self.dbg_camera
                            .borrow_mut()
                            .set_frame(&camera_frame, camera_origin);
                    }

                    ui.checkbox(
                        "Draw frustrum as planes/pyramid",
                        &mut self.debug_options_mut().draw_frustrum_planes,
                    );
                    use enumflags2::BitFlags;

                    BitFlags::<FrustrumPlane>::all().iter().for_each(|f| {
                        let mut value = self.debug_options().frustrum_planes.intersects(f);
                        if ui.checkbox(format!("{:?}", f), &mut value) {
                            self.debug_options_mut().frustrum_planes.toggle(f);
                        }
                    });

                    if ui.collapsing_header("Camera frame", imgui::TreeNodeFlags::FRAMED) {
                        let (right, up, dir) = self.camera.borrow().right_up_dir();
                        ui.text(format!("Position: {}", self.camera.borrow().position));
                        ui.text(format!("X: {}", right));
                        ui.text(format!("Y: {}", up));
                        ui.text(format!("Z: {}", dir));
                    }
                }

                ui.separator();
                ui.text("Instancing:");
                ui.text(format!(
                    "Total instances: {}",
                    self.stats.borrow().total_instances
                ));
                ui.text(format!(
                    "Visible (sent to GPU) instances: {}",
                    self.stats.borrow().visible_instances
                ));
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

    pub fn update(&mut self, frame_time: f64) {
        {
            let mut frame_times = self.frame_times.borrow_mut();
            if (frame_times.len() + 1) > Self::MAX_HISTOGRAM_VALUES {
                frame_times.rotate_left(1);
                frame_times[Self::MAX_HISTOGRAM_VALUES - 1] = frame_time as f32;
            } else {
                frame_times.push(frame_time as f32);
            }
        }

        let mut cmds = Vec::<QueuedCommand>::with_capacity(16);
        let mut removed_bodies: Vec<RigidBodyHandle> = Vec::new();

        (0..Self::num_physics_steps_240hz(frame_time)).for_each(|_| {
            cmds.clear();
            removed_bodies.clear();

            //
            // do physics step
            self.physics_engine.borrow_mut().update(&mut cmds);

            cmds.iter().for_each(|&cmd| match cmd {
                QueuedCommand::ProcessProjectileImpact(cdata) => {
                    self.projectile_impacted_event(cdata);
                    removed_bodies.push(cdata);
                }
                _ => {}
            });

            //
            // update flight camera
            self.physics_engine
                .borrow()
                .rigid_body_set
                .get(self.starfury.rigid_body_handle)
                .map(|starfury_phys_obj| {
                    self.camera
                        .borrow_mut()
                        .update(starfury_phys_obj.position())
                });

            //
            // remove impacted bullets/missiles
            {
                let mut pe = self.physics_engine.borrow_mut();

                removed_bodies.iter().for_each(|rbody| {
                    self.projectiles_sys.borrow_mut().despawn_projectile(*rbody);
                    self.projectiles_sys.borrow_mut().despawn_projectile(*rbody);
                    pe.remove_rigid_body(*rbody);
                });
            }
        });

        {
            let queued_commands = {
                let mut phys_engine = self.physics_engine.borrow_mut();
                let mut update_ctx = UpdateContext {
                    physics_engine: &mut phys_engine,
                    queued_commands: Vec::with_capacity(32),
                    frame_time,
                    camera_pos: self.camera.borrow().position,
                };

                // self.projectiles_sys.borrow_mut().update(&mut update_ctx);
                // self.missile_sys.borrow_mut().update(&mut update_ctx);
                // self.sparks_sys.borrow_mut().update(&mut update_ctx);
                self.starfury.update(&mut update_ctx);

                update_ctx.queued_commands
            };

            {
                let mut phys_eng = self.physics_engine.borrow_mut();
                queued_commands
                    .iter()
                    .for_each(|&queued_cmd| match queued_cmd {
                        // QueuedCommand::SpawnProjectile(data) => {
                        //     self.projectiles_sys
                        //         .borrow_mut()
                        //         .spawn_projectile(data, &mut phys_eng);
                        // }

                        // QueuedCommand::SpawnMissile(
                        //     msl_kind,
                        //     msl_orientation,
                        //     linear_vel,
                        //     angular_val,
                        // ) => {
                        //     self.missile_sys.borrow_mut().add_live_missile(
                        //         msl_kind,
                        //         &msl_orientation,
                        //         linear_vel,
                        //         angular_val,
                        //         &mut phys_eng,
                        //     );
                        // }

                        // QueuedCommand::DrawMissile(msl) => {
                        //     self.missile_sys.borrow_mut().draw_inert_missile(msl);
                        // }
                        _ => {}
                    });
            }
        }

        if self.debug_options().debug_camera {
            self.dbg_camera.borrow_mut().update_view_matrix();
        }
    }

    fn projectile_impacted_event(&self, proj_handle: RigidBodyHandle) {
        log::info!("Impact for {:?}", proj_handle);
        let projectile_isometry = *self
            .physics_engine
            .borrow()
            .get_rigid_body(proj_handle)
            .position();

        self.sparks_sys.borrow_mut().spawn_sparks(ImpactSpark {
            pos: Point3::from_slice(projectile_isometry.translation.vector.as_slice()),
            dir: projectile_isometry * glm::Vec3::z(),
            color: glm::vec3(1f32, 0f32, 0f32),
            speed: 2.0f32,
            life: 2f32,
        });

        log::info!("Removed {:?}", proj_handle);
    }

    pub fn handle_winit_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        self.ui.borrow_mut().handle_event(window, event);
    }

    pub fn gamepad_input(&mut self, input_state: &InputState) {
        if input_state.gamepad.btn_lock_target {
            self.physics_engine
                .borrow()
                .rigid_body_set
                .get(self.starfury.rigid_body_handle)
                .map(|rigid_body| {
                    let ship_isometry = *rigid_body.position();

                    let (ray_origin, ray_dir) = (
                        ship_isometry.translation.vector,
                        (ship_isometry.rotation * glm::Vec3::z_axis()).xyz(),
                    );

                    let query_filter = rapier3d::prelude::QueryFilter::new()
                        .exclude_sensors()
                        .exclude_rigid_body(self.starfury.rigid_body_handle)
                        .groups(PhysicsObjectCollisionGroups::ships());

                    const MAX_RAY_DIST: f32 = 1000f32;

                    if let Some(target_info) = self
                        .physics_engine
                        .borrow()
                        .cast_ray(ray_origin.into(), ray_dir, MAX_RAY_DIST, query_filter)
                        .and_then(|(collider_handle, _)| {
                            self.physics_engine
                                .borrow()
                                .collider_set
                                .get(collider_handle)
                                .and_then(|collider| {
                                    collider.parent().map(|body| (collider_handle, body))
                                })
                        })
                    {
                        *self.locked_target.borrow_mut() = Some(target_info);
                    } else {
                        *self.locked_target.borrow_mut() = None;
                    }
                });
        }

        if self.debug_options().debug_camera {
            Self::dbg_cam_gamepad_input(&mut self.dbg_camera.borrow_mut(), &input_state.gamepad);
        } else {
            self.starfury.gamepad_input(input_state);
        }
    }

    fn draw_crosshair(&self, draw_context: &DrawContext) {
        let player_ship_transform = *self
            .physics_engine
            .borrow()
            .rigid_body_set
            .get(self.starfury.rigid_body_handle)
            .unwrap()
            .position();

        let ray_dir = (player_ship_transform * glm::Vec3::z_axis())
            .to_homogeneous()
            .xyz();

        let query_filter = rapier3d::prelude::QueryFilter::new()
            .exclude_sensors()
            .exclude_rigid_body(self.starfury.rigid_body_handle)
            .groups(PhysicsObjectCollisionGroups::ships());

        const MAX_RAY_DIST: f32 = 1000f32;

        let left_gun_origin = player_ship_transform * self.starfury.lower_left_gun();
        self.physics_engine
            .borrow()
            .cast_ray(left_gun_origin, ray_dir, MAX_RAY_DIST, query_filter)
            .or_else(|| {
                let right_gun_origin = player_ship_transform * self.starfury.lower_right_gun();
                self.physics_engine.borrow().cast_ray(
                    right_gun_origin,
                    ray_dir,
                    MAX_RAY_DIST,
                    query_filter,
                )
            })
            .map(|(_, t)| {
                //
                // impact from guns is possible, draw full crosshair
                let ray_end = left_gun_origin.xyz() + ray_dir * t;
                let clip_space_pos = draw_context.projection_view * ray_end.to_homogeneous();
                let ndc_pos = clip_space_pos.xyz() / clip_space_pos.w;
                let window_space_pos = glm::vec2(
                    ((ndc_pos.x + 1f32) * 0.5f32) * draw_context.viewport.width,
                    ((ndc_pos.y + 1f32) * 0.5f32) * draw_context.viewport.height,
                );
                self.sprite_batch.borrow_mut().draw_with_origin(
                    window_space_pos.x,
                    window_space_pos.y,
                    self.player_opts.crosshair_size,
                    self.player_opts.crosshair_size,
                    self.player_opts.spr_crosshair_normal,
                    Some(self.player_opts.crosshair_color),
                );
                self.sprite_batch.borrow_mut().draw_with_origin(
                    window_space_pos.x,
                    window_space_pos.y,
                    self.player_opts.crosshair_size,
                    self.player_opts.crosshair_size,
                    self.player_opts.spr_crosshair_hit,
                    Some(self.player_opts.crosshair_color),
                );
            })
            .or_else(|| {
                //
                // no impact possible, draw empty crosshair cirle

                let ray_start =
                    Point3::from_slice(player_ship_transform.translation.vector.as_slice());
                let ray_end = ray_start + ray_dir * MAX_RAY_DIST;

                let window_space_pos = math::world_coords_to_screen_coords(
                    ray_end,
                    &draw_context.projection_view,
                    draw_context.viewport.width,
                    draw_context.viewport.height,
                );

                self.sprite_batch.borrow_mut().draw_with_origin(
                    window_space_pos.x,
                    window_space_pos.y,
                    self.player_opts.crosshair_size,
                    self.player_opts.crosshair_size,
                    self.player_opts.spr_crosshair_normal,
                    Some(self.player_opts.crosshair_color),
                );
                Some(())
            });
    }

    fn draw_locked_target_indicator(&self, draw_context: &DrawContext) {
        let physics_engine = self.physics_engine.borrow();
        let target_is_out_of_view = self
            .locked_target
            .borrow()
            .and_then(|(collider_handle, locked_target_phys_handle)| {
                physics_engine
                    .rigid_body_set
                    .get(locked_target_phys_handle)
                    .map(|phys_body| (collider_handle, phys_body))
            })
            .map(|(collider_handle, locked_target)| {
                //
                // if target not in field of view clear lock indicator
                let ship_frame = physics_engine
                    .get_rigid_body(self.starfury.rigid_body_handle)
                    .position()
                    .to_matrix();
                let ship_dir = ship_frame.column(2).xyz();
                let target_vec =
                    locked_target.position().translation.vector.xyz() - ship_frame.column(3).xyz();

                const MAX_ANGLE: f32 = 1.3089969389957472f32; // 75 degrees
                let angle = glm::angle(&ship_dir, &target_vec);

                if angle > MAX_ANGLE {
                    return true;
                }

                let current_position = *locked_target.position();

                let ship_centermass_world = current_position.translation.vector;

                let predicted_pos = locked_target.predict_position_using_velocity_and_forces(1f32);

                let position_vec = predicted_pos.translation.vector - ship_centermass_world;

                let lead_ind_circle_pos = if position_vec.norm_squared() > 1.0e-4f32 {
                    //
                    // also draw a line from the centermass to the predicted position
                    math::world_coords_to_screen_coords(
                        Point3::from_slice(predicted_pos.translation.vector.as_slice()),
                        &draw_context.projection_view,
                        draw_context.viewport.width,
                        draw_context.viewport.height,
                    )
                } else {
                    math::world_coords_to_screen_coords(
                        Point3::from_slice(ship_centermass_world.as_slice()),
                        &draw_context.projection_view,
                        draw_context.viewport.width,
                        draw_context.viewport.height,
                    )
                };

                self.sprite_batch.borrow_mut().draw_with_origin(
                    lead_ind_circle_pos.x,
                    lead_ind_circle_pos.y,
                    64f32,
                    64f32,
                    self.player_opts.spr_obj_centermass,
                    Some(self.player_opts.enemy_outline_color),
                );

                physics_engine
                    .collider_set
                    .get(collider_handle)
                    .map(|collider| {
                        let aabb = collider.compute_aabb();

                        let (pmin, pmax) = aabb
                            .vertices()
                            .iter()
                            .map(|&aabb_vertex| {
                                math::world_coords_to_screen_coords(
                                    aabb_vertex,
                                    &draw_context.projection_view,
                                    draw_context.viewport.width,
                                    draw_context.viewport.height,
                                )
                            })
                            .fold(
                                (
                                    glm::vec2(std::f32::MAX, std::f32::MAX),
                                    glm::vec2(std::f32::MIN, std::f32::MIN),
                                ),
                                |(min_p, max_p), pt| {
                                    (glm::min2(&min_p, &pt), glm::max2(&max_p, &pt))
                                },
                            );

                        let size = (pmax - pmin).abs();

                        self.sprite_batch.borrow_mut().draw(
                            pmin.x,
                            pmin.y,
                            size.x,
                            size.y,
                            self.player_opts.spr_obj_outline,
                            Some(self.player_opts.enemy_outline_color),
                        );
                    });
                return false;
            })
            .unwrap_or_else(|| false);

        if target_is_out_of_view {
            self.locked_target.borrow_mut().take();
        }
    }

    fn dbg_cam_gamepad_input(cam: &mut FirstPersonCamera, input: &GamepadInputState) {
        const CAM_SPD: f32 = 0.025f32;
        const CAM_ROT_SPD: f32 = 0.5f32;

        input.left_stick_x.axis_data.map(|data| {
            if data.value().abs() > input.left_stick_x.deadzone {
                cam.strafe(CAM_SPD * data.value());
            }
        });

        input.left_stick_y.axis_data.map(|data| {
            if data.value().abs() > input.left_stick_y.deadzone {
                cam.walk(CAM_SPD * data.value());
            }
        });

        input.right_stick_x.axis_data.map(|data| {
            if data.value().abs() > input.right_stick_x.deadzone {
                cam.yaw(CAM_ROT_SPD * data.value());
            }
        });

        input.right_stick_y.axis_data.map(|data| {
            if data.value().abs() > input.right_stick_y.deadzone {
                cam.pitch(CAM_ROT_SPD * data.value());
            }
        });

        input.rtrigger.data.map(|data| {
            if data.is_pressed() {
                cam.jump(CAM_SPD);
            }
        });

        input.ltrigger.data.map(|data| {
            if data.is_pressed() {
                cam.jump(-CAM_SPD);
            }
        });

        if input.btn_lock_target {
            cam.reset();
        }
    }
}
