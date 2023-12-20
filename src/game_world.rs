use std::{
    borrow::Borrow,
    cell::{RefCell, RefMut},
    rc::Rc,
};

use ash::vk::{BufferUsageFlags, MemoryPropertyFlags, PipelineBindPoint};
use itertools::UniqueBy;
use nalgebra::Point3;
use nalgebra_glm as glm;
use nalgebra_glm::Vec4;

use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

use crate::{
    app_config::{AppConfig, PlayerShipConfig},
    bindless::BindlessResourceHandle2,
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::{DrawContext, FrameRenderContext, InitContext, UpdateContext},
    drawing_system::DrawingSys,
    flight_cam::FlightCamera,
    fps_camera::FirstPersonCamera,
    frustrum::{is_aabb_on_frustrum, Frustrum, FrustrumPlane},
    game_object::GameObjectPhysicsData,
    math,
    missile_sys::{MissileSpawnData, MissileSys, ProjectileSpawnData},
    particles::{ImpactSpark, SparksSystem},
    physics_engine::{PhysicsEngine, PhysicsObjectCollisionGroups},
    resource_system::{EffectType, ResourceSystem},
    shadow_swarm::ShadowFighterSwarm,
    skybox::Skybox,
    sprite_batch::{SpriteBatch, TextureRegion},
    starfury::Starfury,
    ui_backend::UiBackend,
    vk_renderer::{RendererWorkPackage, UniqueBuffer, VulkanRenderer},
    window::{GamepadInputState, InputState},
};

#[derive(Copy, Clone)]
pub enum QueuedCommand {
    SpawnProjectile(ProjectileSpawnData),
    SpawnMissile(MissileSpawnData),
    ProcessProjectileImpact(RigidBodyHandle),
    DrawEngineExhaust(glm::Mat4),
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

#[derive(Copy, Clone)]
#[repr(C)]
struct GlobalUniformData {
    projection_view: glm::Mat4,
    view: glm::Mat4,
    frame_id: u32,
}

struct Statistics {
    total_instances: u32,
    visible_instances: u32,
}

pub struct GameWorld {
    draw_opts: RefCell<DebugOptions>,
    skybox: Skybox,
    starfury: Starfury,
    shadows_swarm: ShadowFighterSwarm,
    frame_times: RefCell<Vec<f32>>,
    physics_engine: RefCell<PhysicsEngine>,
    camera: RefCell<FlightCamera>,
    dbg_camera: RefCell<FirstPersonCamera>,
    debug_draw_overlay: Rc<RefCell<DebugDrawOverlay>>,
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
    ubo_bindless: UniqueBuffer,
    ubo_bindless_handles: Vec<BindlessResourceHandle2>,
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

        let mut physics_engine = PhysicsEngine::new();

        let starfury = Starfury::new(&rsys, &mut physics_engine);
        let shadows_swarm = ShadowFighterSwarm::new(&mut physics_engine, &rsys);

        let sprites = SpriteBatch::create(renderer, cfg)?;
        let missile_sys = MissileSys::new(&InitContext {
            window,
            renderer,
            cfg,
            rsys: &mut rsys,
        })?;

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

        let ubo_bindless = UniqueBuffer::with_capacity(
            renderer,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_VISIBLE,
            1,
            std::mem::size_of::<GlobalUniformData>(),
            renderer.max_inflight_frames(),
        )
        .expect("xxx");

        let ubo_bindless_handles = rsys.bindless.register_chunked_uniform(
            renderer,
            &ubo_bindless,
            renderer.max_inflight_frames() as usize,
        );

        Some(GameWorld {
            draw_opts: RefCell::new(DebugOptions::default()),
            skybox,
            starfury,
            shadows_swarm,
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
            missile_sys: RefCell::new(missile_sys),
            resource_sys: rsys,
            drawing_sys: RefCell::new(DrawingSys::create(renderer)?),
            ui,
            ubo_bindless,
            ubo_bindless_handles,
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

        let inst_aabb = self.shadows_swarm.bounds;

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

        //
        // start a visibility check early
        let visible_objects_future = self.object_visibility_check();

        let physics = self.physics_engine.borrow();

        let draw_context = DrawContext {
            physics: &physics,
            rsys: &self.resource_sys,
            renderer: frame_context.renderer,
            cmd_buff: frame_context.cmd_buff,
            frame_id: frame_context.frame_id,
            viewport: frame_context.viewport,
            scissor: frame_context.scissor,
	    global_ubo_handle: self.ubo_bindless_handles[frame_context.frame_id as usize].handle(),
            view_matrix,
            cam_position,
            projection,
            inverse_projection,
            projection_view: projection * view_matrix,
            debug_draw: self.debug_draw_overlay.clone(),
        };

        let globals = GlobalUniformData {
            projection_view: projection * view_matrix,
            view: view_matrix,
	    frame_id: frame_context.frame_id
        };

        self.ubo_bindless
            .map_for_frame(frame_context.renderer, frame_context.frame_id)
            .map(|mut b| unsafe {
                std::ptr::copy_nonoverlapping(
                    &globals as *const _,
                    b.as_mut_ptr() as *mut GlobalUniformData,
                    1,
                );
            })
            .expect("xxxx");

        //
        // setup bindless
        unsafe {
            let bindless = self.resource_sys.bindless_setup();
            frame_context
                .renderer
                .graphics_device()
                .cmd_bind_descriptor_sets(
                    frame_context.cmd_buff,
                    PipelineBindPoint::GRAPHICS,
                    bindless.pipeline_layout,
                    0,
                    bindless.descriptor_set,
                    &[],
                );
        }

        // self.drawing_sys
        //     .borrow_mut()
        //     .setup_bindless(self.skybox.id, &draw_context);
        //
        self.skybox.draw(&draw_context);
        //
        // self.starfury
        //     .borrow()
        //     .draw(&draw_context, &mut self.drawing_sys.borrow_mut());

        //
        // draw ze enemies Hans, ja ja wunderbar
        let visible_instances = self
            .rt
            .block_on(visible_objects_future)
            .expect("Failed to wait async visibility check task");

        *self.stats.borrow_mut() = Statistics {
            visible_instances: visible_instances.len() as u32,
            total_instances: self.shadows_swarm.instances().len() as u32,
        };

        {
            // let mut draw_sys = self.drawing_sys.borrow_mut();
            // visible_instances.iter().for_each(|(_, inst_transform)| {
            //     draw_sys.add_mesh(
            //         self.shadows_swarm.mesh_id,
            //         None,
            //         None,
            //         &inst_transform.to_matrix(),
            //         EffectType::Pbr,
            //     );
            // });
            // self.missile_sys.borrow().draw(&draw_context, &mut draw_sys);
        }

        // self.drawing_sys.borrow_mut().draw(&draw_context);

        // self.draw_objects(&draw_context);

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

        // self.draw_crosshair(&draw_context);
        // self.draw_locked_target_indicator(&draw_context);
        // self.sprite_batch.borrow_mut().render(&draw_context);
    }

    fn draw_objects(&self, draw_context: &DrawContext) {
        // if self.debug_options().debug_draw_world_axis {
        //     self.debug_draw_overlay
        //         .borrow_mut()
        //         .world_space_coord_sys(self.draw_opts.borrow().world_axis_length);
        // }

        //         if self.debug_options().debug_draw_mesh {
        //             // let aabb = self.render_state.borrow()[game_object.handle.0 as usize]
        //             //     .render_pos
        //             //     .to_homogeneous()
        //             //     * object_renderable.geometry.aabb;

        //             // draw_context.debug_draw.borrow_mut().add_aabb(
        //             //     &aabb.min,
        //             //     &aabb.max,
        //             //     0xFF_00_00_FF,
        //             // );
        //         }

        //         if self.debug_options().debug_draw_nodes_bounding {
        //             let geometry = self
        //                 .resource_cache
        //                 .get_pbr_geometry_info(self.starfury.renderable);

        //             let transform = self
        //                 .physics_engine
        //                 .borrow()
        //                 .get_rigid_body(self.starfury.rigid_body_handle)
        //                 .position()
        //                 .to_homogeneous();

        //             geometry.nodes.iter().for_each(|node| {
        //                 let aabb = transform * node.aabb;

        //                 use crate::color_palettes::StdColors;
        //                 draw_context.debug_draw.borrow_mut().add_aabb(
        //                     &aabb.min,
        //                     &aabb.max,
        //                     StdColors::RED,
        //                 );
        //             });
        //         }
        //     });
        // }
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
                    self.missile_sys.borrow_mut().despawn_projectile(*rbody);
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

                self.missile_sys.borrow_mut().update(&mut update_ctx);
                // self.sparks_sys.borrow_mut().update(&mut update_ctx);
                self.starfury.update(&mut update_ctx);

                update_ctx.queued_commands
            };

            {
                let mut phys_eng = self.physics_engine.borrow_mut();
                queued_commands
                    .iter()
                    .for_each(|&queued_cmd| match queued_cmd {
                        QueuedCommand::SpawnProjectile(data) => {
                            self.missile_sys
                                .borrow_mut()
                                .spawn_projectile(&data, &mut phys_eng);
                        }

                        QueuedCommand::SpawnMissile(msl_data) => {
                            self.missile_sys
                                .borrow_mut()
                                .spawn_missile(&msl_data, &mut phys_eng);
                        }

                        _ => {}
                    });
            }
        }

        if self.debug_options().debug_camera {
            self.dbg_camera.borrow_mut().update_view_matrix();
        }
    }

    fn projectile_impacted_event(&self, proj_handle: RigidBodyHandle) {
        // log::info!("Impact for {:?}", proj_handle);
        // let projectile_isometry = *self
        //     .physics_engine
        //     .borrow()
        //     .get_rigid_body(proj_handle)
        //     .position();

        // self.sparks_sys.borrow_mut().spawn_sparks(ImpactSpark {
        //     pos: Point3::from_slice(projectile_isometry.translation.vector.as_slice()),
        //     dir: projectile_isometry * glm::Vec3::z(),
        //     color: glm::vec3(1f32, 0f32, 0f32),
        //     speed: 2.0f32,
        //     life: 2f32,
        // });

        // log::info!("Removed {:?}", proj_handle);
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
            .get_rigid_body(self.starfury.rigid_body_handle)
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
