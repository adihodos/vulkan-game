use crate::{
    draw_context::UpdateContext,
    game_world::{GameObjectHandle, QueuedCommand},
    math::AABB3,
    missile_sys::{Missile, MissileKind, MissileState},
    physics_engine::PhysicsEngine,
    projectile_system::ProjectileSpawnData,
    resource_cache::{PbrRenderableHandle, ResourceHolder},
    window::InputState,
};

use glm::Vec3;
use nalgebra::{Isometry3, Point3};
use nalgebra_glm as glm;
use rand_distr::num_traits::Zero;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
use serde::{Deserialize, Serialize};
use strum_macros;

#[derive(
    Copy,
    Clone,
    Debug,
    strum_macros::EnumIter,
    strum_macros::EnumProperty,
    strum_macros::Display,
    Serialize,
    Deserialize,
)]
#[repr(u8)]
pub enum WeaponPylonId {
    #[strum(props(pylon_id = "pylon.upper.left.0",))]
    UpperWingLeft0,

    #[strum(props(pylon_id = "pylon.upper.left.1",))]
    UpperWingLeft1,

    #[strum(props(pylon_id = "pylon.upper.left.2",))]
    UpperWingLeft2,

    #[strum(props(pylon_id = "pylon.upper.right.0",))]
    UpperWingRight0,

    #[strum(props(pylon_id = "pylon.upper.right.1",))]
    UpperWingRight1,

    #[strum(props(pylon_id = "pylon.upper.right.2",))]
    UpperWingRight2,
}

const PYLON_ATTACHMENT_POINTS: [[f32; 3]; 6] = [
    [0.940288f32, 0.307043f32, 0.065791f32],
    [0.694851f32, 0.185932f32, 0.21051f32],
    [0.48396f32, 0.07659f32, 0.293167f32],
    [-0.940288f32, 0.307043f32, 0.065791f32],
    [-0.694851f32, 0.185932f32, 0.21051f32],
    [-0.48396f32, 0.07659f32, 0.293167f32],
];
const Y_OFFSET_BY_MISSILE: [f32; 2] = [-0.03f32, -0.03f32];

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

#[derive(Copy, Clone, Serialize, Deserialize)]
struct GunPorts {
    lower_left: nalgebra::Point3<f32>,
    lower_right: nalgebra::Point3<f32>,
}

impl std::default::Default for GunPorts {
    fn default() -> Self {
        GunPorts {
            lower_left: nalgebra::Point3::new(0.19122f32, -0.3282f32, 1.111f32),
            lower_right: nalgebra::Point3::new(-0.19122f32, -0.3282f32, 1.111f32),
        }
    }
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

impl FlightModel {
    fn thruster_force_vector(&self, thruster_id: EngineThrusterId) -> glm::Vec3 {
        self.thruster_force_vectors[thruster_id as usize]
    }
}

struct EngineThruster {
    name: String,
    transform: glm::Mat4,
    aabb: AABB3,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct LaserParams {
    speed: f32,
    mass: f32,
    lifetime: f32,
}

impl std::default::Default for LaserParams {
    fn default() -> Self {
        Self {
            speed: 0.1f32,
            mass: 1f32,
            lifetime: 5f32,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Weapons {
    guns_cooldown: f32,
    gun_ports: GunPorts,
    laser: LaserParams,
}

impl std::default::Default for Weapons {
    fn default() -> Self {
        Self {
            guns_cooldown: 1f32 / 8f32,
            gun_ports: GunPorts::default(),
            laser: LaserParams::default(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct StarfuryParameters {
    fm: FlightModel,
    weapons: Weapons,
}

#[derive(Clone, Copy)]
enum QueuedOp {
    ApplyForce(glm::Vec3),
    ApplyTorque(glm::Vec3),
    FireGuns,
    FireMissile(WeaponPylonId),
    Reset,
}

struct SuspendedWeaponry {
    id: WeaponPylonId,
    kind: MissileKind,
    weapon_attached: bool,
}

pub struct Starfury {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub collider_handle: rapier3d::prelude::ColliderHandle,
    params: StarfuryParameters,
    thrusters: Vec<EngineThruster>,
    pylons_weaponry: Vec<SuspendedWeaponry>,
    queued_ops: Vec<QueuedOp>,
    guns_cooldown: f32,
    missile_respawn_cooldown: f32,
    missile_fire_cooldown: f32,
}

impl Starfury {
    pub fn new(
        object_handle: GameObjectHandle,
        physics_engine: &mut PhysicsEngine,
        resource_cache: &ResourceHolder,
    ) -> Starfury {
        let params: StarfuryParameters = ron::de::from_reader(
            std::fs::File::open("config/starfury.flightmodel.cfg.ron")
                .expect("Failed to read Starfury flight model configuration file."),
        )
        .expect("Invalid configuration file");

        let geometry_handle = resource_cache.get_pbr_geometry_handle(&"sa23");
        let geometry = resource_cache.get_pbr_geometry_info(geometry_handle);

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
            .linear_damping(params.fm.linear_damping)
            .angular_damping(params.fm.angular_damping)
            .build();

        let bbox_half_extents = geometry.aabb.extents();
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(params.fm.mass)
        .build();

        let body_handle = physics_engine.rigid_body_set.insert(body);
        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut physics_engine.rigid_body_set,
        );

        log::info!("Starfury collider {:?}", collider_handle);

        Starfury {
            renderable: geometry_handle,
            object_handle,
            rigid_body_handle: body_handle,
            collider_handle,
            params,
            thrusters,
            queued_ops: Vec::new(),
            guns_cooldown: 0f32,
            missile_respawn_cooldown: 0f32,
            missile_fire_cooldown: 0f32,
            pylons_weaponry: WeaponPylonId::iter()
                .map(|pylon_id| {
                    let kind = if (pylon_id as u8) % 2 == 0 {
                        MissileKind::R27
                    } else {
                        MissileKind::R73
                    };

                    SuspendedWeaponry {
                        id: pylon_id,
                        kind,
                        weapon_attached: true,
                    }
                })
                .collect(),
        }
    }

    // pub fn input_event(&self, event: &winit::event::KeyboardInput) {
    //     use winit::event::VirtualKeyCode;

    //     let physics_op = event.virtual_keycode.and_then(|key_code| match key_code {
    //         VirtualKeyCode::F10 => Some(QueuedOp::Reset),

    //         VirtualKeyCode::Q => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self.params.fm.thruster_force_vectors
    //                     [self.params.fm.maneuver.roll.left[0] as usize],
    //         )),

    //         VirtualKeyCode::E => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self.params.fm.thruster_force_vectors
    //                     [self.params.fm.maneuver.roll.right[0] as usize],
    //         )),

    //         VirtualKeyCode::W => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self.params.fm.thruster_force_vectors
    //                     [self.params.fm.maneuver.pitch.down[0] as usize],
    //         )),

    //         VirtualKeyCode::S => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self.params.fm.thruster_force_vectors
    //                     [self.params.fm.maneuver.pitch.up[0] as usize],
    //         )),

    //         VirtualKeyCode::A => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self
    //                     .params
    //                     .fm
    //                     .thruster_force_vector(self.params.fm.maneuver.yaw.left[0]),
    //         )),

    //         VirtualKeyCode::D => Some(QueuedOp::ApplyTorque(
    //             self.params.fm.thruster_force_secondary
    //                 * self
    //                     .params
    //                     .fm
    //                     .thruster_force_vector(self.params.fm.maneuver.yaw.right[0]),
    //         )),
    //         _ => None,
    //     });

    //     physics_op.map(|i| {
    //         self.queued_ops.borrow_mut().push(i);
    //     });
    // }

    pub fn update(&mut self, update_context: &mut UpdateContext) {
        self.physics_update(&mut update_context.physics_engine);

        self.guns_cooldown = (self.guns_cooldown - update_context.frame_time as f32).max(0f32);

        self.missile_fire_cooldown =
            (self.missile_fire_cooldown - update_context.frame_time as f32).max(0f32);

        if self.missile_respawn_cooldown > 0f32 {
            self.missile_respawn_cooldown =
                (self.missile_respawn_cooldown - update_context.frame_time as f32).max(0f32);
            //
            // cooldown expired, respawn pylon attachments
            if self.missile_respawn_cooldown.is_zero() {
                self.pylons_weaponry.iter_mut().for_each(|p| {
                    p.weapon_attached = true;
                });
                self.missile_fire_cooldown = 0f32;
            }
        }

        {
            let _rigid_body = update_context
                .physics_engine
                .rigid_body_set
                .get_mut(self.rigid_body_handle)
                .unwrap();

            self.queued_ops.iter().for_each(|&op| match op {
                QueuedOp::FireMissile(pylon_id) => {
                    //
                    // if out of missiles start the cooldown timer
                    if self.missile_respawn_cooldown.is_zero()
                        && self
                            .pylons_weaponry
                            .iter()
                            .all(|p| p.weapon_attached == false)
                    {
                        self.missile_respawn_cooldown = 10f32;
                    }

		    
                    let (object2world, linear_vel, angular_vel) = {
			let rigid_body = update_context
                        .physics_engine
                            .get_rigid_body(self.rigid_body_handle);
			
			let angular_vel = *rigid_body.angvel();
			let linear_vel = *rigid_body.linvel();
			
                        (*rigid_body.position(), linear_vel, angular_vel)
		    };

                    let msl_kind = self.pylons_weaponry[pylon_id as usize].kind;

                    let pylon_attachment_point = PYLON_ATTACHMENT_POINTS[pylon_id as usize];
                    let pylon_attachment_point = [
                        pylon_attachment_point[0],
                        pylon_attachment_point[1] + Y_OFFSET_BY_MISSILE[msl_kind as usize],
                        pylon_attachment_point[2],
                    ];

                    //
                    // rotate missile 45 degrees then move it to pylon center
                    use nalgebra::{Rotation3, Translation3};
                    let missile2pylon = Isometry3::from_parts(
                        Translation3::from(pylon_attachment_point),
                        Rotation3::from_euler_angles(0f32, 0f32, 45f32.to_radians()).into(),
                    );
                    let iso = object2world * missile2pylon;

                    update_context
                        .queued_commands
                        .push(QueuedCommand::SpawnMissile(msl_kind, iso, linear_vel, angular_vel));
                }

                QueuedOp::FireGuns => {
                    update_context.queued_commands.extend(
                        [
                            QueuedCommand::SpawnProjectile(ProjectileSpawnData {
                                origin: self.params.weapons.gun_ports.lower_left,
                                speed: self.params.weapons.laser.speed,
                                mass: self.params.weapons.laser.mass,
                                emitter: self.rigid_body_handle,
                                life: self.params.weapons.laser.lifetime,
                            }),
                            QueuedCommand::SpawnProjectile(ProjectileSpawnData {
                                origin: self.params.weapons.gun_ports.lower_right,
                                speed: self.params.weapons.laser.speed,
                                mass: self.params.weapons.laser.mass,
                                emitter: self.rigid_body_handle,
                                life: self.params.weapons.laser.lifetime,
                            }),
                        ]
                        .iter(),
                    );
                }
                _ => {}
            });
        }

        //
        // add remaining missiles to draw sys
        self.draw_suspended_weaponry(update_context);
        self.queued_ops.clear();
    }

    pub fn physics_update(&mut self, phys_engine: &mut PhysicsEngine) {
        if self.queued_ops.is_empty() {
            return;
        }

        let rigid_body = phys_engine
            .rigid_body_set
            .get_mut(self.rigid_body_handle)
            .unwrap();

        let isometry = *rigid_body.position();

        self.queued_ops.iter().for_each(|&impulse| match impulse {
            QueuedOp::ApplyForce(f) => {
                rigid_body.apply_impulse(isometry * f, true);
            }

            QueuedOp::ApplyTorque(t) => {
                rigid_body.apply_torque_impulse(isometry * t, true);
            }

            QueuedOp::Reset => {
                rigid_body.reset_forces(true);
                rigid_body.reset_torques(true);
                rigid_body.set_linvel(Vec3::zeros(), true);
                rigid_body.set_angvel(Vec3::zeros(), true);
                rigid_body.set_position(nalgebra::Isometry::identity(), true);
            }

            _ => (),
        });
    }

    pub fn gamepad_input(&mut self, input_state: &InputState) {
        input_state.gamepad.ltrigger.data.map(|btn| {
            if btn.is_pressed() && self.guns_cooldown.is_zero() {
                self.queued_ops.push(QueuedOp::FireGuns);
                self.guns_cooldown = self.params.weapons.guns_cooldown;
            }
        });

        input_state.gamepad.rtrigger.data.map(|btn| {
            if btn.is_pressed()
                && self.missile_respawn_cooldown.is_zero()
                && self.missile_fire_cooldown.is_zero()
            {
                const FIRING_SEQUENCE: [WeaponPylonId; 6] = [
                    WeaponPylonId::UpperWingLeft0,
                    WeaponPylonId::UpperWingRight0,
                    WeaponPylonId::UpperWingLeft1,
                    WeaponPylonId::UpperWingRight1,
                    WeaponPylonId::UpperWingLeft2,
                    WeaponPylonId::UpperWingRight2,
                ];

                for pylon in FIRING_SEQUENCE {
                    if self.pylons_weaponry[pylon as usize].weapon_attached {
                        self.pylons_weaponry[pylon as usize].weapon_attached = false;
                        self.queued_ops.push(QueuedOp::FireMissile(pylon));
                        break;
                    }
                }

                self.missile_fire_cooldown = 5f32;
            }
        });

        let movement = [
            (
                &input_state.gamepad.left_stick_x,
                self.params.fm.maneuver.movement.left,
                self.params.fm.maneuver.movement.right,
                true,
            ),
            (
                &input_state.gamepad.left_stick_y,
                self.params.fm.maneuver.movement.forward,
                self.params.fm.maneuver.movement.backward,
                true,
            ),
        ];

        movement.iter().for_each(
            |&(gamepad_stick, thruster_id_pos, thruster_id_neg, primary)| {
                gamepad_stick.axis_data.map(|axis_data| {
                    if axis_data.value().abs() <= gamepad_stick.deadzone {
                        return;
                    }

                    let throttle = axis_data.value() * self.params.fm.throttle_sensitivity;

                    let force_multiplier = if primary {
                        self.params.fm.thruster_force_primary
                    } else {
                        self.params.fm.thruster_force_secondary
                    };

                    let phys_op = if throttle > 0f32 {
                        QueuedOp::ApplyForce(
                            throttle.abs()
                                * force_multiplier
                                * self.params.fm.thruster_force_vector(thruster_id_pos),
                        )
                    } else {
                        QueuedOp::ApplyForce(
                            throttle.abs()
                                * force_multiplier
                                * self.params.fm.thruster_force_vector(thruster_id_neg),
                        )
                    };

                    self.queued_ops.push(phys_op);
                });
            },
        );

        let roll_pitch = [
            (
                &input_state.gamepad.right_stick_x,
                self.params.fm.maneuver.roll.left[0],
                self.params.fm.maneuver.roll.right[0],
            ),
            (
                &input_state.gamepad.right_stick_y,
                self.params.fm.maneuver.pitch.up[0],
                self.params.fm.maneuver.pitch.down[0],
            ),
        ];

        roll_pitch
            .iter()
            .for_each(|&(gamepad_stick, thruster_id_pos, thruster_id_neg)| {
                gamepad_stick.axis_data.map(|axis_data| {
                    if axis_data.value().abs() <= gamepad_stick.deadzone {
                        return;
                    }

                    let throttle = axis_data.value() * self.params.fm.throttle_sensitivity;

                    let phys_op = if throttle > 0f32 {
                        QueuedOp::ApplyTorque(
                            throttle.abs()
                                * self.params.fm.thruster_force_secondary
                                * self.params.fm.thruster_force_vector(thruster_id_pos),
                        )
                    } else {
                        QueuedOp::ApplyTorque(
                            throttle.abs()
                                * self.params.fm.thruster_force_secondary
                                * self.params.fm.thruster_force_vector(thruster_id_neg),
                        )
                    };

                    self.queued_ops.push(phys_op);
                });
            });

        let yaws = [
            (
                &input_state.gamepad.right_z,
                self.params.fm.maneuver.yaw.left[0],
            ),
            (
                &input_state.gamepad.left_z,
                self.params.fm.maneuver.yaw.right[0],
            ),
        ];

        yaws.iter().for_each(|&(gamepad_btn, thruster_id)| {
            gamepad_btn.data.map(|button_data| {
                if button_data.value().abs() <= gamepad_btn.deadzone {
                    return;
                }

                let throttle_factor = button_data.value() * self.params.fm.throttle_sensitivity;

                self.queued_ops.push(QueuedOp::ApplyTorque(
                    throttle_factor
                        * self.params.fm.thruster_force_secondary
                        * self.params.fm.thruster_force_vector(thruster_id),
                ));
            });
        });
    }

    pub fn lower_left_gun(&self) -> Point3<f32> {
        self.params.weapons.gun_ports.lower_left
    }

    pub fn lower_right_gun(&self) -> Point3<f32> {
        self.params.weapons.gun_ports.lower_right
    }

    fn draw_suspended_weaponry(&self, update_ctx: &mut UpdateContext) {
        let object2world = update_ctx
            .physics_engine
            .get_rigid_body(self.rigid_body_handle)
            .position();

        self.pylons_weaponry
            .iter()
            .filter(|p| p.weapon_attached)
            .for_each(|pylon| {
                let pylon_attachment_point = PYLON_ATTACHMENT_POINTS[pylon.id as usize];
                let pylon_attachment_point = [
                    pylon_attachment_point[0],
                    pylon_attachment_point[1] + Y_OFFSET_BY_MISSILE[pylon.kind as usize],
                    pylon_attachment_point[2],
                ];

                //
                // rotate missile 45 degrees then move it to pylon center
                use nalgebra::{Rotation3, Translation3};
                let missile2pylon = Isometry3::from_parts(
                    Translation3::from(pylon_attachment_point),
                    Rotation3::from_euler_angles(0f32, 0f32, 45f32.to_radians()).into(),
                );
                let iso = object2world * missile2pylon;

                update_ctx
                    .queued_commands
                    .push(QueuedCommand::DrawMissile(Missile {
                        kind: pylon.kind,
                        state: MissileState::Inactive,
                        transform: iso.to_matrix(),
                    }));
            });
    }
}
