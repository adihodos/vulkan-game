use std::cell::RefCell;

use crate::{
    draw_context::DrawContext,
    game_world::GameObjectHandle,
    math::AABB3,
    physics_engine::PhysicsEngine,
    resource_cache::{GeometryRenderInfo, PbrRenderable, PbrRenderableHandle},
    window::InputState,
};

use glm::Vec3;
use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
use serde::{Deserialize, Serialize};
use strum_macros;

#[derive(
    Copy, Clone, Debug, strum_macros::EnumIter, strum_macros::EnumProperty, Serialize, Deserialize,
)]
#[repr(u8)]
enum EngineThrusterId {
    #[strum(props(node_id = "engine.thruster.upper.left.front",))]
    UpperLeftFront,

    #[strum(props(node_id = "engine.thruster.upper.left.up",))]
    UpperLeftUp,

    #[strum(props(node_id = "engine.thruster.upper.left.left",))]
    UpperLeftLeft,

    #[strum(props(node_id = "engine.thruster.upper.left.back"))]
    UpperLeftBack,

    #[strum(props(node_id = "engine.thruster.upper.right.front"))]
    UpperRightFront,

    #[strum(props(node_id = "engine.thruster.upper.right.up"))]
    UpperRightUp,

    #[strum(props(node_id = "engine.thruster.upper.right.right"))]
    UpperRightRight,

    #[strum(props(node_id = "engine.thruster.upper.right.back"))]
    UpperRightBack,

    #[strum(props(node_id = "engine.thruster.lower.left.front"))]
    LowerLeftFront,

    #[strum(props(node_id = "engine.thruster.lower.left.down"))]
    LowerLeftDown,

    #[strum(props(node_id = "engine.thruster.lower.left.left"))]
    LowerLeftLeft,

    #[strum(props(node_id = "engine.thruster.lower.left.back"))]
    LowerLeftBack,

    #[strum(props(node_id = "engine.thruster.lower.right.front"))]
    LowerRightFront,

    #[strum(props(node_id = "engine.thruster.lower.right.down"))]
    LowerRightDown,

    #[strum(props(node_id = "engine.thruster.lower.right.right"))]
    LowerRightRight,

    #[strum(props(node_id = "engine.thruster.lower.right.back"))]
    LowerRightBack,
}

#[derive(Clone, Serialize, Deserialize)]
struct Roll {
    left: Vec<EngineThrusterId>,
    right: Vec<EngineThrusterId>,
}

impl std::default::Default for Roll {
    fn default() -> Self {
        Roll {
            left: vec![EngineThrusterId::UpperLeftFront],
            right: vec![EngineThrusterId::LowerRightBack],
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Pitch {
    up: Vec<EngineThrusterId>,
    down: Vec<EngineThrusterId>,
}

impl std::default::Default for Pitch {
    fn default() -> Self {
        Pitch {
            up: vec![EngineThrusterId::LowerRightRight],
            down: vec![EngineThrusterId::UpperLeftLeft],
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Yaw {
    left: Vec<EngineThrusterId>,
    right: Vec<EngineThrusterId>,
}

impl std::default::Default for Yaw {
    fn default() -> Self {
        Yaw {
            left: vec![EngineThrusterId::LowerRightDown],
            right: vec![EngineThrusterId::UpperLeftUp],
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Movement {
    forward: EngineThrusterId,
    backward: EngineThrusterId,
    left: EngineThrusterId,
    right: EngineThrusterId,
}

impl std::default::Default for Movement {
    fn default() -> Self {
        Movement {
            forward: EngineThrusterId::UpperRightBack,
            backward: EngineThrusterId::UpperLeftUp,
            left: EngineThrusterId::UpperRightRight,
            right: EngineThrusterId::UpperLeftLeft,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct Maneuver {
    roll: Roll,
    pitch: Pitch,
    yaw: Yaw,
    movement: Movement,
}

#[derive(Serialize, Deserialize)]
struct FlightModel {
    mass: f32,
    linear_damping: f32,
    angular_damping: f32,
    throttle_sensitivity: f32,
    thruster_force_primary: f32,
    thruster_force_secondary: f32,
    thruster_force_vectors: [glm::Vec3; 16],
    maneuver: Maneuver,
}

struct EngineThruster {
    name: String,
    transform: glm::Mat4,
    aabb: AABB3,
}

impl FlightModel {
    fn write_default_config() {
        use ron::ser::{to_writer_pretty, PrettyConfig};

        let cfg_opts = PrettyConfig::new()
            .depth_limit(8)
            .separate_tuple_members(true);

        to_writer_pretty(
            std::fs::File::create("config/starfury.flightmodel.default.cfg.ron").expect("cykaaaaa"),
            &FlightModel::default(),
            cfg_opts.clone(),
        )
        .expect("Failed to write default flight model config");
    }

    fn thruster_force_vector(&self, thruster_id: EngineThrusterId) -> glm::Vec3 {
        self.thruster_force_vectors[thruster_id as usize]
    }
}

impl std::default::Default for FlightModel {
    fn default() -> Self {
        Self {
            mass: 23_000f32,
            linear_damping: 1f32,
            angular_damping: 1f32,
            thruster_force_primary: 10_000f32,
            thruster_force_secondary: 2500f32,
            throttle_sensitivity: 0.5f32,
            thruster_force_vectors: [
                //
                //upper left
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, -1f32, 0f32),
                glm::vec3(1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                //upper right
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, -1f32, 0f32),
                glm::vec3(-1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                // Lower left thruster
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, 1f32, 0f32),
                glm::vec3(1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                // Lower right thruster
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, 1f32, 0f32),
                glm::vec3(-1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
            ],
            maneuver: Maneuver::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum PhysicsOp {
    ApplyForce(glm::Vec3),
    ApplyTorque(glm::Vec3),
    Reset,
}

pub struct Starfury {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub collider_handle: rapier3d::prelude::ColliderHandle,
    flight_model: FlightModel,
    thrusters: Vec<EngineThruster>,
    physics_ops_queue: RefCell<Vec<PhysicsOp>>,
}

impl Starfury {
    pub fn new(
        renderable: PbrRenderableHandle,
        object_handle: GameObjectHandle,
        physics_engine: &mut PhysicsEngine,
        geometry: &GeometryRenderInfo,
    ) -> Starfury {
        FlightModel::write_default_config();

        let flight_model: FlightModel = ron::de::from_reader(
            std::fs::File::open("config/starfury.flightmodel.cfg.ron")
                .expect("Failed to read Starfury flight model configuration file."),
        )
        .expect("Invalid configuration file");

        use strum::{EnumProperty, IntoEnumIterator};
        let thrusters = EngineThrusterId::iter()
            .map(|thruster_id| {
                let node = geometry
                    .nodes
                    .iter()
                    .find(|node| node.name == thruster_id.get_str("node_id").unwrap())
                    .unwrap();

                EngineThruster {
                    name: node.name.clone(),
                    transform: node.transform,
                    aabb: node.aabb,
                }
            })
            .collect::<Vec<_>>();

        let body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .translation(glm::vec3(0f32, 0f32, 0f32))
            .linear_damping(flight_model.linear_damping)
            .angular_damping(flight_model.angular_damping)
            .build();

        let bbox_half_extents = geometry.aabb.extents() * 0.5f32;
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(flight_model.mass)
        .build();

        let body_handle = physics_engine.rigid_body_set.insert(body);
        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut physics_engine.rigid_body_set,
        );

        Starfury {
            renderable,
            object_handle,
            rigid_body_handle: body_handle,
            collider_handle,
            flight_model,
            thrusters,
            physics_ops_queue: RefCell::new(Vec::new()),
        }
    }

    pub fn input_event(&self, event: &winit::event::KeyboardInput) {
        use winit::event::VirtualKeyCode;

        let physics_op = event.virtual_keycode.and_then(|key_code| match key_code {
            VirtualKeyCode::F10 => Some(PhysicsOp::Reset),

            VirtualKeyCode::Q => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self.flight_model.thruster_force_vectors
                        [self.flight_model.maneuver.roll.left[0] as usize],
            )),

            VirtualKeyCode::E => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self.flight_model.thruster_force_vectors
                        [self.flight_model.maneuver.roll.right[0] as usize],
            )),

            VirtualKeyCode::W => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self.flight_model.thruster_force_vectors
                        [self.flight_model.maneuver.pitch.down[0] as usize],
            )),

            VirtualKeyCode::S => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self.flight_model.thruster_force_vectors
                        [self.flight_model.maneuver.pitch.up[0] as usize],
            )),

            VirtualKeyCode::A => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self
                        .flight_model
                        .thruster_force_vector(self.flight_model.maneuver.yaw.left[0]),
            )),

            VirtualKeyCode::D => Some(PhysicsOp::ApplyTorque(
                self.flight_model.thruster_force_secondary
                    * self
                        .flight_model
                        .thruster_force_vector(self.flight_model.maneuver.yaw.right[0]),
            )),
            _ => None,
        });

        physics_op.map(|i| {
            self.physics_ops_queue.borrow_mut().push(i);
        });
    }

    pub fn physics_update(&self, phys_engine: &mut PhysicsEngine) {
        if self.physics_ops_queue.borrow().is_empty() {
            return;
        }

        let rigid_body = phys_engine
            .rigid_body_set
            .get_mut(self.rigid_body_handle)
            .unwrap();

        let isometry = *rigid_body.position();

        self.physics_ops_queue
            .borrow()
            .iter()
            .for_each(|&impulse| match impulse {
                PhysicsOp::ApplyForce(f) => {
                    rigid_body.apply_impulse(isometry * f, true);
                }

                PhysicsOp::ApplyTorque(t) => {
                    rigid_body.apply_torque_impulse(isometry * t, true);
                }

                PhysicsOp::Reset => {
                    rigid_body.reset_forces(true);
                    rigid_body.reset_torques(true);
                    rigid_body.set_linvel(Vec3::zeros(), true);
                    rigid_body.set_angvel(Vec3::zeros(), true);
                    rigid_body.set_position(nalgebra::Isometry::identity(), true);
                }
            });

        self.physics_ops_queue.borrow_mut().clear();
    }

    pub fn gamepad_input(&self, input_state: &InputState) {
        use gilrs::{Axis, Gamepad};

        let movement = [
            (
                &input_state.gamepad.left_stick_x,
                self.flight_model.maneuver.movement.right,
                self.flight_model.maneuver.movement.left,
            ),
            (
                &input_state.gamepad.left_stick_y,
                self.flight_model.maneuver.movement.forward,
                self.flight_model.maneuver.movement.backward,
            ),
        ];

        movement
            .iter()
            .for_each(|&(gamepad_stick, thruster_id_pos, thruster_id_neg)| {
                gamepad_stick.axis_data.map(|axis_data| {
                    if axis_data.value().abs() <= gamepad_stick.deadzone {
                        return;
                    }

                    let throttle = axis_data.value() * self.flight_model.throttle_sensitivity;

                    let phys_op = if throttle > 0f32 {
                        PhysicsOp::ApplyForce(
                            throttle.abs()
                                * self.flight_model.thruster_force_primary
                                * self.flight_model.thruster_force_vector(thruster_id_pos),
                        )
                    } else {
                        PhysicsOp::ApplyForce(
                            throttle.abs()
                                * self.flight_model.thruster_force_primary
                                * self.flight_model.thruster_force_vector(thruster_id_neg),
                        )
                    };

                    self.physics_ops_queue.borrow_mut().push(phys_op);
                });
            });

        let roll_pitch = [
            (
                &input_state.gamepad.right_stick_x,
                self.flight_model.maneuver.roll.left[0],
                self.flight_model.maneuver.roll.right[0],
            ),
            (
                &input_state.gamepad.right_stick_y,
                self.flight_model.maneuver.pitch.up[0],
                self.flight_model.maneuver.pitch.down[0],
            ),
        ];

        roll_pitch
            .iter()
            .for_each(|&(gamepad_stick, thruster_id_pos, thruster_id_neg)| {
                gamepad_stick.axis_data.map(|axis_data| {
                    if axis_data.value().abs() <= gamepad_stick.deadzone {
                        return;
                    }

                    let throttle = axis_data.value() * self.flight_model.throttle_sensitivity;
                    // log::info!("Throttle: {}", throttle);

                    let phys_op = if throttle > 0f32 {
                        PhysicsOp::ApplyTorque(
                            throttle.abs()
                                * self.flight_model.thruster_force_secondary
                                * self.flight_model.thruster_force_vector(thruster_id_pos),
                        )
                    } else {
                        PhysicsOp::ApplyTorque(
                            throttle.abs()
                                * self.flight_model.thruster_force_secondary
                                * self.flight_model.thruster_force_vector(thruster_id_neg),
                        )
                    };

                    self.physics_ops_queue.borrow_mut().push(phys_op);
                });
            });

        let yaws = [
            (
                &input_state.gamepad.right_z,
                self.flight_model.maneuver.yaw.right[0],
            ),
            (
                &input_state.gamepad.left_z,
                self.flight_model.maneuver.yaw.left[0],
            ),
        ];

        yaws.iter().for_each(|&(gamepad_btn, thruster_id)| {
            gamepad_btn.data.map(|button_data| {
                if button_data.value().abs() <= gamepad_btn.deadzone {
                    return;
                }

                let throttle_factor = button_data.value() * self.flight_model.throttle_sensitivity;

                self.physics_ops_queue
                    .borrow_mut()
                    .push(PhysicsOp::ApplyTorque(
                        throttle_factor
                            * self.flight_model.thruster_force_secondary
                            * self.flight_model.thruster_force_vector(thruster_id),
                    ));
            });
        });
    }
}
