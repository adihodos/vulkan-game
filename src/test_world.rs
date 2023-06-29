use std::{cell::RefCell, rc::Rc};

use ash::vk::{
    BorderColor, BufferUsageFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet,
    DescriptorSetAllocateInfo, DescriptorType, DeviceSize, Filter, ImageLayout, IndexType,
    PipelineBindPoint, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    WriteDescriptorSet,
};
use glm::Vec3;
use nalgebra::Point3;
use nalgebra_glm::Vec4;

use nalgebra_glm as glm;

use smallvec::SmallVec;

use crate::{
    app_config::{AppConfig, PlayerShipConfig},
    camera::Camera,
    color_palettes::StdColors,
    debug_draw_overlay::DebugDrawOverlay,
    draw_context::{DrawContext, FrameRenderContext, UpdateContext},
    flight_cam::FlightCamera,
    fps_camera::FirstPersonCamera,
    frustrum::{is_aabb_on_frustrum, Frustrum},
    math::{self, AABB3},
    particles::{ImpactSpark, SparksSystem},
    physics_engine::{ColliderUserData, PhysicsEngine, PhysicsObjectCollisionGroups},
    plane::Plane,
    projectile_system::{ProjectileSpawnData, ProjectileSystem},
    resource_cache::{PbrDescriptorType, PbrRenderableHandle, ResourceHolder},
    shadow_swarm::ShadowFighterSwarm,
    skybox::Skybox,
    sprite_batch::{SpriteBatch, TextureRegion},
    starfury::Starfury,
    vk_renderer::{Cpu2GpuBuffer, UniqueSampler, VulkanRenderer},
    window::InputState,
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

pub struct TestWorld {
    camera: RefCell<FirstPersonCamera>,
    debug_draw_overlay: Rc<RefCell<DebugDrawOverlay>>,
    f: Frustrum,
    org: glm::Vec3,
    boxes: Vec<AABB3>,
    tfs: Vec<nalgebra::Isometry3<f32>>,
}

impl TestWorld {
    const PHYSICS_TIME_STEP: f64 = 1f64 / 240f64;
    const MAX_HISTOGRAM_VALUES: usize = 32;

    pub fn new(renderer: &VulkanRenderer, app_cfg: &AppConfig) -> Option<TestWorld> {
        use nalgebra::Isometry3;

        let a = Isometry3::new(glm::vec3(-5f32, -5f32, -5f32), glm::Vec3::y());
        let b = Isometry3::new(glm::vec3(5f32, 5f32, 5f32), glm::Vec3::y());

        let camera = FirstPersonCamera::new(65f32, 1.333f32, 0.1f32, 1000f32);
        Some(TestWorld {
            camera: RefCell::new(camera),
            debug_draw_overlay: std::rc::Rc::new(RefCell::new(
                DebugDrawOverlay::create(&renderer).expect("Failed to create debug draw overlay"),
            )),
            f: Frustrum::from_fpscam(&camera),
            org: camera.position,
            tfs: vec![a, b],
            boxes: vec![AABB3::unit(), AABB3::unit()],
        })
    }

    pub fn draw(&self, frame_context: &FrameRenderContext) {
        self.debug_draw_overlay.borrow_mut().clear();

        let projection = self.camera.borrow().projection_matrix;
        // let view = self.camera.borrow().view_matrix;

        // {
        //     let cam_ref = self.camera.borrow();
        //     let draw_context = DrawContext {
        //         renderer: frame_context.renderer,
        //         cmd_buff: frame_context.cmd_buff,
        //         frame_id: frame_context.frame_id,
        //         viewport: frame_context.viewport,
        //         scissor: frame_context.scissor,
        //         camera: &*cam_ref,
        //         projection,
        //         inverse_projection: projection,
        //         projection_view: projection * cam_ref.view_transform(),
        //         debug_draw: self.debug_draw_overlay.clone(),
        //     };

        //     self.draw_objects(&draw_context);
        // }

        {
            const PLANE_SIZE: f32 = 5f32;
            let org = self.org;
            let mut dbg_overlay = self.debug_draw_overlay.borrow_mut();
            // dbg_overlay.add_plane(&self.f.near_face, &org, PLANE_SIZE, StdColors::GREEN);
            // dbg_overlay.add_plane(&self.f.top_face, &org, PLANE_SIZE, StdColors::CORNFLOWER_BLUE);
            // dbg_overlay.add_plane(&self.f.bottom_face, &org, PLANE_SIZE, StdColors::DARK_ORANGE);
            // dbg_overlay.add_plane(&self.f.left_face, &org, PLANE_SIZE, StdColors::BLUE_VIOLET);
            // dbg_overlay.add_plane(&self.f.right_face, &org, PLANE_SIZE, StdColors::INDIAN_RED);
        }

        // let p = Plane::from_normal_and_origin(Vec3::new(0f32, 1f32, 0f32), Vec3::zeros());

        // use crate::color_palettes::StdColors;

        // self.debug_draw_overlay.borrow_mut().add_plane(
        //     &p,
        //     &Vec3::new(0f32, 0f32, 0f32),
        //     30f32,
        //     StdColors::RED,
        // );

        {
            let f = Frustrum::from_fpscam(&self.camera.borrow());

            self.boxes
                .iter()
                .zip(self.tfs.iter())
                .for_each(|(bbox, tf)| {
                    if is_aabb_on_frustrum(&f, bbox, tf) {
                        let mtx = tf.to_matrix();
                        let aabbtf = mtx * (*bbox);
                        self.debug_draw_overlay.borrow_mut().add_aabb(
                            &aabbtf.min,
                            &aabbtf.max,
                            StdColors::RED,
                        );
                    }
                    // log::info!("Box {} drawn: {}", bbox, drawn);
                });
        }

        self.debug_draw_overlay.borrow_mut().draw(
            frame_context.renderer,
            &(projection * self.camera.borrow().view_transform()),
        );
    }

    fn draw_objects(&self, draw_context: &DrawContext) {
        // let device = draw_context.renderer.graphics_device();
    }

    pub fn ui(&self, ui: &mut imgui::Ui) {}

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
        self.camera.borrow_mut().update_view_matrix();
    }

    pub fn input_event(&self, event: &winit::event::WindowEvent) {
        {
            let mut cam = self.camera.borrow_mut();
            let mut controller_cam = FpsCameraController { cam: &mut cam };
            controller_cam.handle_input_event(event);
        }
    }

    pub fn gamepad_input(&self, input_state: &InputState) {}
}

struct FpsCameraController<'a> {
    cam: &'a mut FirstPersonCamera,
}

impl<'a> FpsCameraController<'a> {
    pub fn handle_input_event(&mut self, event: &winit::event::WindowEvent) {
        const CAM_SPEED: f32 = 1f32;

        use winit::event::WindowEvent;
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                use winit::event::ElementState;
                use winit::event::KeyboardInput;
                use winit::event::VirtualKeyCode;

                match input {
                    KeyboardInput {
                        state,
                        virtual_keycode,
                        ..
                    } => {
                        if *state == ElementState::Pressed {
                            match virtual_keycode {
                                Some(VirtualKeyCode::W) => self.cam.walk(CAM_SPEED),
                                Some(VirtualKeyCode::S) => self.cam.walk(-CAM_SPEED),
                                Some(VirtualKeyCode::A) => self.cam.strafe(-CAM_SPEED),
                                Some(VirtualKeyCode::D) => self.cam.strafe(CAM_SPEED),
                                Some(VirtualKeyCode::Up) => self.cam.pitch(CAM_SPEED),
                                Some(VirtualKeyCode::Down) => self.cam.pitch(-CAM_SPEED),
                                Some(VirtualKeyCode::Left) => self.cam.yaw(-CAM_SPEED),
                                Some(VirtualKeyCode::Right) => self.cam.yaw(CAM_SPEED),
                                Some(VirtualKeyCode::Back) => self.cam.reset(),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
